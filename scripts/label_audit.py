"""Audit the training labels of a directory of episodes, as the dataset reads them.

Reads every episode's usable frames at the training frame rate through the schema adapter
and writes OUT_DIR/index.html, a self-contained report, and OUT_DIR/audit.json with its
numbers: statistics per town, per navigation command (as derived from the path, which
training uses, and against the recorded one) and per driving mode; findings, where labels
contradict each other or the driving, with sample frames of each.

usage: python scripts/label_audit.py DATA_DIR OUT_DIR [--workers 8] [--samples 6]
"""

from __future__ import annotations

import argparse
import base64
import html
import io
import json
from multiprocessing import Pool
from pathlib import Path
from typing import Any

import h5py
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from src.data import hdf5_schema as S  # noqa: E402
from src.data import labels, schema_adapter  # noqa: E402
from src.data.driving_dataset import FPS, WAYPOINT_OFFSETS, DrivingSequenceDataset  # noqa: E402
from src.eval.metrics import TURN_THRESHOLD  # noqa: E402

LEFT, RIGHT, STRAIGHT, LANEFOLLOW = 1, 2, 3, 4
MOVING_MODES = {1, 2, 3}  # FollowLane, ApproachingTurn, Turning
STOP_MODES = {4, 6, 7, 8}  # TrafficLightStop, PedestrianStop, EmergencyStop, StopSignStop
STANDING = DrivingSequenceDataset.STATIONARY_SPEED_THRESH  # m/s
STYLE = (
    "body{font:14px system-ui,sans-serif;max-width:1250px;margin:24px auto;padding:0 16px}"
    "table{border-collapse:collapse;margin:8px 0 20px}"
    "td,th{border:1px solid #ccc;padding:4px 8px;text-align:right}"
    "td:first-child,th:first-child{text-align:left}img{max-width:100%}"
)
MOVING = 1.0  # m/s: below this, steering barely changes the path
COLUMNS = (
    "steer",
    "throttle",
    "brake",
    "nav",
    "recorded",
    "mode",
    "speed",
    "target",
    "wp_x",
    "wp_y",
    "wp_ok",
    "situation",
    "point",
)
TURNING = labels.SITUATIONS.index("turning")


def text(value: Any) -> str:
    return value.decode() if isinstance(value, bytes) else str(value)


def read_episode(path: Path) -> dict[str, Any]:
    """Labels of one episode's usable frames, or the error that stopped reading it."""
    record: dict[str, Any] = {"file": path.name}
    try:
        with h5py.File(path, "r") as f:
            meta = f[S.METADATA_GROUP].attrs
            episode = schema_adapter.read_labels(f, FPS)
            nav = episode.command
            n = len(nav)
            w = schema_adapter.read_window(f, 0, n, max(WAYPOINT_OFFSETS), fps=FPS, images=False)
            far = labels.ego_waypoints(w.position, w.yaw, WAYPOINT_OFFSETS)[:, -1]
            record.update(
                town=text(meta.get(S.META_TOWN, "unknown")),
                status=text(meta.get(S.META_STATUS, "unknown")),
                recorded_s=f[S.POSITION].shape[0] / float(meta[S.META_FPS]),
                steer=w.steering,
                throttle=w.throttle,
                brake=w.brake,
                nav=nav,
                recorded=w.nav_command,
                mode=w.driving_mode,
                speed=np.linalg.norm(w.velocity[:, :2], axis=1),
                target=w.target_speed,
                wp_x=far[:, 0],
                wp_y=far[:, 1],
                wp_ok=np.arange(n) + max(WAYPOINT_OFFSETS) < len(w.position),
                situation=episode.situation,
                point=episode.target_point,
            )
    except Exception as e:  # noqa: BLE001 — an unreadable episode is itself a finding
        record["error"] = f"{type(e).__name__}: {e}"
    return record


def runs(values: np.ndarray, episode: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Starts and ends of the runs of equal values within each episode."""
    cut = np.flatnonzero((np.diff(values) != 0) | (np.diff(episode) != 0)) + 1
    return np.r_[0, cut], np.r_[cut, len(values)]


def mark(n: int, starts: np.ndarray, ends: np.ndarray) -> np.ndarray:
    marked = np.zeros(n, bool)
    for s, e in zip(starts, ends):
        marked[s:e] = True
    return marked


def lasting(mask: np.ndarray, episode: np.ndarray, frames: int) -> np.ndarray:
    """`mask`, kept only where it holds for at least `frames` frames in a row."""
    starts, ends = runs(mask.astype(np.int8), episode)
    keep = mask[starts] & (ends - starts >= frames)
    return mark(len(mask), starts[keep], ends[keep])


def findings(t: dict[str, np.ndarray]) -> list[tuple[str, str, np.ndarray]]:
    """(name, what it shows, frames) for every check."""
    n, ep, nav, steer = len(t["nav"]), t["episode"], t["nav"], t["steer"]
    starts, ends = runs(nav, ep)
    value, length = nav[starts], ends - starts
    same_episode = np.r_[False, ep[starts[1:]] == ep[starts[:-1]]]
    before = np.r_[0, value[:-1]]
    after = np.r_[value[1:], 0]
    inside = same_episode & np.r_[same_episode[1:], False]  # neither first nor last of its episode
    flicker = inside & (length < FPS) & (before == after) & (before != value)
    toward = np.where(value == LEFT, -1.0, 1.0)
    turned = np.maximum.reduceat(toward[np.repeat(np.arange(len(starts)), length)] * steer, starts)
    dry = inside & np.isin(value, (LEFT, RIGHT)) & (turned <= TURN_THRESHOLD)
    moving_mode = np.isin(t["mode"], list(MOVING_MODES))
    moving = t["speed"] >= MOVING
    return [
        (
            "LaneFollow at more than half lock",
            "LANEFOLLOW while moving at 1 m/s or more with |steering| ≥ 0.5, not on a sharp bend",
            moving & (nav == LANEFOLLOW) & (np.abs(steer) >= 0.5) & ~t["situation"][:, TURNING],
        ),
        (
            "Turn command, hard steering the other way",
            "LEFT with steering > 0.3, or RIGHT with steering < −0.3, while moving",
            moving & (((nav == LEFT) & (steer > 0.3)) | ((nav == RIGHT) & (steer < -0.3))),
        ),
        (
            "Turn command that never turns",
            "A LEFT or RIGHT run, not at either end of its episode, in which steering never "
            "passes 0.1 towards that side",
            mark(n, starts[dry], ends[dry]),
        ),
        (
            "Command flicker",
            "A command lasting under 1 s between two runs of the same other command",
            mark(n, starts[flicker], ends[flicker]),
        ),
        (
            "Throttle in a stop mode",
            "Throttle > 0.1 in TrafficLightStop, PedestrianStop, EmergencyStop or StopSignStop",
            np.isin(t["mode"], list(STOP_MODES)) & (t["throttle"] > 0.1),
        ),
        (
            "Throttle and brake together",
            "Throttle > 0.1 and brake > 0.1 on the same frame",
            (t["throttle"] > 0.1) & (t["brake"] > 0.1),
        ),
        (
            "Standing still in a driving mode",
            "FollowLane, ApproachingTurn or Turning below 0.5 km/h for 2 s or longer",
            lasting(moving_mode & (t["speed"] < STANDING), ep, 2 * FPS),
        ),
        (
            "Faster than the target speed",
            "Speed more than 5 km/h above a positive target speed",
            (t["target"] > 0) & (t["speed"] * 3.6 > t["target"] + 5),
        ),
    ]


def share(mask: np.ndarray, of: np.ndarray | None = None) -> str:
    base = mask.size if of is None else int(of.sum())
    return f"{(mask if of is None else mask & of).sum() / max(base, 1):.1%}"


def group_rows(t: dict[str, np.ndarray], key: str, names: dict[Any, str]) -> list[list[str]]:
    total, rows = len(t["nav"]), []
    turning = np.abs(t["steer"]) > TURN_THRESHOLD
    standing = t["speed"] < STANDING
    for value, name in names.items():
        m = t[key] == value
        if not m.any():
            continue
        moving = m & ~standing
        rows.append(
            [
                name,
                f"{m.sum() / FPS / 3600:,.1f}",
                f"{m.sum() / total:.1%}",
                share(standing, m),
                share(turning, m),
                f"{np.median(t['speed'][moving]) * 3.6:.0f}" if moving.any() else "–",
                share(t["brake"] > 0.1, m),
                " · ".join(share(t["nav"] == c, m) for c in (LEFT, RIGHT, STRAIGHT, LANEFOLLOW)),
            ]
        )
    return rows


def command_rows(t: dict[str, np.ndarray]) -> list[list[str]]:
    starts, ends = runs(t["nav"], t["episode"])
    rows = []
    for c in sorted(set(t["nav"].tolist())):
        m = t["nav"] == c
        turning = m & (np.abs(t["steer"]) > TURN_THRESHOLD)
        lengths = (ends - starts)[t["nav"][starts] == c] / FPS
        side = {LEFT: -1, RIGHT: 1}.get(c)
        rows.append(
            [
                S.NAV_COMMAND_NAMES[c],
                f"{m.mean():.1%}",
                share(turning, m),
                "–" if side is None else share(np.sign(t["steer"]) == side, turning),
                "–" if side is None else share(np.sign(t["wp_y"]) == side, turning & t["wp_ok"]),
                f"{len(lengths):,}",
                f"{np.median(lengths):.1f}",
                f"{(lengths < 1).mean():.1%}",
            ]
        )
    return rows


def recorded_rows(t: dict[str, np.ndarray]) -> list[list[str]]:
    """For each recorded command, the share of its frames under each derived command."""
    rows = []
    for c in sorted(set(t["recorded"].tolist())):
        m = t["recorded"] == c
        cells = [share(t["nav"] == d, m) for d in (LEFT, RIGHT, STRAIGHT, LANEFOLLOW)]
        rows.append([S.NAV_COMMAND_NAMES[c], f"{m.mean():.1%}", *cells])
    return rows


def situation_rows(t: dict[str, np.ndarray]) -> list[list[str]]:
    standing = t["speed"] < STANDING
    rows = []
    for i, name in enumerate(labels.SITUATIONS):
        m = t["situation"][:, i]
        moving = m & ~standing
        rows.append(
            [
                name,
                f"{m.sum() / FPS / 3600:,.1f}",
                f"{m.mean():.1%}",
                share(standing, m),
                f"{np.median(t['speed'][moving]) * 3.6:.0f}" if moving.any() else "–",
                share(t["brake"] > 0.1, m),
                str(len(np.unique(t["episode"][m]))),
            ]
        )
    return rows


def table(head: list[str], rows: list[list[str]]) -> str:
    cells = "".join(f"<th>{html.escape(h)}</th>" for h in head)
    cells = f"<tr>{cells}</tr>" if head else ""
    body = "".join(
        "<tr>" + "".join(f"<td>{html.escape(str(c))}</td>" for c in r) + "</tr>" for r in rows
    )
    return f"<table>{cells}{body}</table>"


def png(fig: Any) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=90, bbox_inches="tight")
    plt.close(fig)
    return f'<img src="data:image/png;base64,{base64.b64encode(buf.getvalue()).decode()}">'


def sample_frames(
    t: dict[str, np.ndarray], files: list[Path], mask: np.ndarray, n: int, rng: np.random.Generator
) -> str:
    picked, seen = [], set()
    for i in rng.permutation(np.flatnonzero(mask)):
        if t["episode"][i] not in seen:
            seen.add(t["episode"][i])
            picked.append(int(i))
        if len(picked) == n:
            break
    if not picked:
        return ""
    fig, axes = plt.subplots(
        len(picked),
        2,
        figsize=(13, 2.3 * len(picked)),
        squeeze=False,
        gridspec_kw={"width_ratios": [6, 1]},
    )
    for (image_ax, wp_ax), i in zip(axes, picked):
        path, k = files[t["episode"][i]], int(t["frame"][i])
        with h5py.File(path, "r") as f:
            w = schema_adapter.read_window(
                f, k, k + 1, max(WAYPOINT_OFFSETS), fps=FPS, size=(160, 960)
            )
        points = labels.ego_waypoints(w.position, w.yaw, WAYPOINT_OFFSETS)[0]
        image_ax.imshow(w.images[0])
        image_ax.set_axis_off()
        active = [n for n, on in zip(labels.SITUATIONS, t["situation"][i]) if on]
        image_ax.set_title(
            f"{path.stem} frame {k} · {S.NAV_COMMAND_NAMES[int(t['nav'][i])]} · "
            f"{S.DRIVING_MODE_NAMES[int(t['mode'][i])]} · steer {t['steer'][i]:+.2f} · "
            f"throttle {t['throttle'][i]:.2f} brake {t['brake'][i]:.2f} · "
            f"{t['speed'][i] * 3.6:.0f} km/h (target {t['target'][i]:.0f})\n"
            f"situations: {', '.join(active) or 'none'}",
            fontsize=8,
            loc="left",
        )
        wp_ax.plot(points[:, 1], points[:, 0], "o-", color="tab:blue")
        wp_ax.plot(t["point"][i, 1], t["point"][i, 0], "*", color="tab:red", markersize=10)
        wp_ax.plot(0, 0, "k^")
        wp_ax.set(xlim=(-20, 20), ylim=(-2, 35), aspect="equal")
        wp_ax.tick_params(labelsize=7)
        wp_ax.set_title("waypoints, target ★, m", fontsize=7)
    return png(fig)


def distributions(t: dict[str, np.ndarray]) -> str:
    fig, (a, b, c) = plt.subplots(1, 3, figsize=(15, 3.6))
    for cmd in (LEFT, RIGHT, STRAIGHT, LANEFOLLOW):
        m = t["nav"] == cmd
        if m.any():
            a.hist(
                t["steer"][m],
                bins=80,
                range=(-1, 1),
                histtype="step",
                label=S.NAV_COMMAND_NAMES[cmd],
            )
    a.set(yscale="log", xlabel="steering", title="Steering by command")
    a.legend(fontsize=7)
    modes = [m for m in S.DRIVING_MODE_NAMES if (t["mode"] == m).any()]
    b.boxplot([t["speed"][t["mode"] == m] * 3.6 for m in modes], showfliers=False)
    b.set_xticks(
        range(1, len(modes) + 1), [S.DRIVING_MODE_NAMES[m] for m in modes], rotation=40, fontsize=7
    )
    b.set(ylabel="km/h", title="Speed by driving mode")
    pick = np.random.default_rng(0).permutation(np.flatnonzero(t["wp_ok"]))[:30000]
    for cmd, colour in ((LANEFOLLOW, "0.6"), (LEFT, "tab:blue"), (RIGHT, "tab:red")):
        m = pick[t["nav"][pick] == cmd]
        c.scatter(t["steer"][m], t["wp_y"][m], s=1, c=colour, label=S.NAV_COMMAND_NAMES[cmd])
    c.set(
        xlabel="steering",
        ylabel="waypoint at 2 s, y right (m)",
        title="Steering against the 2 s waypoint",
    )
    c.legend(fontsize=7, markerscale=6)
    return png(fig)


def main() -> None:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("data_dir", type=Path)
    p.add_argument("out_dir", type=Path)
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--samples", type=int, default=6, help="Sample frames shown per finding")
    args = p.parse_args()

    paths = sorted(args.data_dir.glob("episode_*.h5"))
    with Pool(args.workers) as pool:
        records = pool.map(read_episode, paths)
    errors = [f"{r['file']}: {r['error']}" for r in records if "error" in r]
    ok = [(path, r) for path, r in zip(paths, records) if "error" not in r]
    files = [path for path, _ in ok]
    t = {c: np.concatenate([r[c] for _, r in ok]) for c in COLUMNS}
    t["episode"] = np.concatenate([np.full(len(r["nav"]), i) for i, (_, r) in enumerate(ok)])
    t["frame"] = np.concatenate([np.arange(len(r["nav"])) for _, r in ok])
    town = np.concatenate([np.full(len(r["nav"]), r["town"]) for _, r in ok])
    total, recorded_s = len(t["nav"]), sum(r["recorded_s"] for _, r in ok)

    statuses = sorted({r["status"] for _, r in ok})
    towns = sorted(set(town.tolist()))
    overview = [
        ["Episodes", f"{len(paths):,} ({len(errors)} unreadable)"],
        ["Usable frames", f"{total:,} at {FPS} Hz, {total / FPS / 3600:,.1f} h"],
        [
            "Dropped by the label rules",
            f"{1 - total / FPS / max(recorded_s, 1e-9):.1%} of recorded time",
        ],
        ["Outcomes", ", ".join(f"{sum(r['status'] == s for _, r in ok)} {s}" for s in statuses)],
    ]
    head = [
        "",
        "hours",
        "share",
        "standing",
        "turning",
        "median km/h moving",
        "braking",
        "Left · Right · Straight · LaneFollow",
    ]
    t["town"] = np.searchsorted(towns, town)
    town_rows = group_rows(t, "town", dict(enumerate(towns)))
    mode_rows = group_rows(t, "mode", {m: n for m, n in S.DRIVING_MODE_NAMES.items()})
    cmd_head = [
        "",
        "frames",
        "turning",
        "steering towards the command",
        "2 s waypoint on that side",
        "runs",
        "median run, s",
        "runs under 1 s",
    ]
    cmd_rows = command_rows(t)
    rng = np.random.default_rng(0)
    found: list[dict[str, Any]] = []
    for name, what, mask in findings(t):
        found.append(
            {
                "name": name,
                "what": what,
                "frames": int(mask.sum()),
                "share": float(mask.mean()),
                "episodes": int(len(np.unique(t["episode"][mask]))),
                "figure": sample_frames(t, files, mask, args.samples, rng),
            }
        )

    sit_rows = situation_rows(t)
    sit_figures = [
        sample_frames(t, files, t["situation"][:, i], args.samples, rng)
        for i in range(len(labels.SITUATIONS))
    ]
    args.out_dir.mkdir(parents=True, exist_ok=True)
    finding_rows = [
        [f["name"], f["what"], f"{f['frames']:,}", f"{f['share']:.2%}", str(f["episodes"])]
        for f in found
    ]
    sections = "".join(
        f"<h3>{html.escape(f['name'])}</h3><p>{html.escape(f['what'])}: {f['frames']:,} frames "
        f"({f['share']:.2%}) in {f['episodes']} episodes.</p>{f['figure']}"
        for f in found
        if f["frames"]
    )
    unreadable = "<br>".join(map(html.escape, errors))
    page = "\n".join(
        [
            "<!doctype html><meta charset='utf-8'><title>Label audit</title>",
            f"<style>{STYLE}</style>",
            f"<h1>Label audit</h1><p>{html.escape(str(args.data_dir))}: every usable frame at "
            f"{FPS} Hz, read as training reads it. Standing: below {STANDING * 3.6:.1f} km/h. "
            f"Turning: |steering| &gt; {TURN_THRESHOLD}.</p>",
            table([], overview),
            f"<h2>Per town</h2>{table(head, town_rows)}",
            "<h2>Per command</h2><p>Commands are derived from the path driven and the junction "
            "outlines, as training uses them.</p>",
            table(cmd_head, cmd_rows),
            "<p>The last two shares are of turning frames; the waypoint share counts only frames "
            "whose 2 s waypoint lies inside the episode. The recorded command against the "
            "derived one:</p>",
            table(
                ["recorded", "frames", "Left", "Right", "Straight", "LaneFollow"], recorded_rows(t)
            ),
            "<h2>Situations</h2><p>Several can hold at once; sample frames are at the end.</p>",
            table(
                ["", "hours", "share", "standing", "median km/h moving", "braking", "episodes"],
                sit_rows,
            ),
            f"<h2>Per driving mode (the expert's own)</h2>{table(head, mode_rows)}",
            f"<h2>Distributions</h2>{distributions(t)}",
            f"<h2>Findings</h2>{table(['', 'what', 'frames', 'share', 'episodes'], finding_rows)}",
            f"<h3>Unreadable episodes</h3><p>{unreadable}</p>" if errors else "",
            sections,
            "".join(
                f"<h3>Situation: {html.escape(row[0])}</h3>{figure}"
                for row, figure in zip(sit_rows, sit_figures)
                if figure
            ),
        ]
    )
    (args.out_dir / "index.html").write_text(page, encoding="utf-8")
    numbers = {
        "fps": FPS,
        "episodes": len(paths),
        "errors": errors,
        "usable_frames": total,
        "recorded_s": recorded_s,
        "overview": dict(overview),
        "towns": {r[0]: r[1:] for r in town_rows},
        "commands": {r[0]: r[1:] for r in cmd_rows},
        "recorded_commands": {r[0]: r[1:] for r in recorded_rows(t)},
        "situations": {r[0]: r[1:] for r in sit_rows},
        "modes": {r[0]: r[1:] for r in mode_rows},
        "findings": [{k: v for k, v in f.items() if k != "figure"} for f in found],
    }
    (args.out_dir / "audit.json").write_text(json.dumps(numbers, indent=2), encoding="utf-8")
    print(f"{len(paths)} episodes, {total:,} frames, {len(errors)} unreadable")
    for f in found:
        print(
            f"  {f['name']}: {f['frames']:,} frames ({f['share']:.2%}) in {f['episodes']} episodes"
        )
    raise SystemExit(1 if errors else 0)


if __name__ == "__main__":
    main()
