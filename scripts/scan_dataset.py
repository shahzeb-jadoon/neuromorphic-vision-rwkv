"""Inventory a directory of driving episodes from their signals (images are not decoded).

Writes one JSON record per episode (size, schema, frame counts, metadata, non-finite values,
nav-command runs and counts, driving-mode counts, steering statistics) and a summary.

usage: python scripts/scan_dataset.py DATA_DIR OUT.json [--workers N]
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from multiprocessing import Pool
from pathlib import Path
from typing import Any

import h5py
import numpy as np

from src.data import hdf5_schema as S
from src.data import schema_adapter
from src.eval.metrics import TURN_THRESHOLD


def plain(value: Any) -> Any:
    if isinstance(value, bytes):
        return value.decode(errors="replace")
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    return value


def runs(values: np.ndarray) -> list[list[int]]:
    """Run-length encoding as [value, start, length] triples."""
    if values.size == 0:
        return []
    edges = np.flatnonzero(np.diff(values)) + 1
    starts = np.concatenate(([0], edges))
    lengths = np.diff(np.concatenate((starts, [values.size])))
    return [[int(values[s]), int(s), int(n)] for s, n in zip(starts, lengths)]


def scan_episode(path: Path) -> dict[str, Any]:
    record: dict[str, Any] = {"file": path.name, "bytes": path.stat().st_size}
    try:
        with h5py.File(path, "r") as f:
            images = f[S.IMAGES]
            n_usable = schema_adapter.usable_frames(f)
            steering = schema_adapter.read_window(f, 0, n_usable, 0, images=False).steering
            nav = schema_adapter.nav_commands(f)
            nonfinite = 0
            for name, ds in f.items():
                if isinstance(ds, h5py.Dataset) and name != S.IMAGES and ds.dtype.kind == "f":
                    nonfinite += int((~np.isfinite(ds[()])).sum())
            speed = np.linalg.norm(f[S.VELOCITY][:n_usable, :2], axis=1) * 3.6
            record.update(
                schema=schema_adapter.schema_version(f),
                frames=int(images.shape[0]),
                usable_frames=n_usable,
                image_shape=list(images.shape[1:]),
                image_compression=images.compression,
                fields=sorted(k for k, v in f.items() if isinstance(v, h5py.Dataset)),
                meta={k: plain(v) for k, v in f[S.METADATA_GROUP].attrs.items()}
                if S.METADATA_GROUP in f
                else {},
                nonfinite_values=nonfinite,
                nav_runs=runs(nav),
                nav_counts={int(k): int(v) for k, v in Counter(nav.tolist()).items()},
                mode_counts={
                    int(k): int(v)
                    for k, v in Counter(f[S.DRIVING_MODE][:n_usable].tolist()).items()
                },
                steer_mean_abs=float(np.abs(steering).mean()) if n_usable else 0.0,
                turn_frame_share=float((np.abs(steering) > TURN_THRESHOLD).mean())
                if n_usable
                else 0.0,
                speed_max_kmh=float(speed.max()) if n_usable else 0.0,
            )
    except Exception as e:  # noqa: BLE001 — an unreadable file is itself a finding
        record["error"] = f"{type(e).__name__}: {e}"
    return record


def summarize(records: list[dict[str, Any]]) -> dict[str, Any]:
    ok = [r for r in records if "error" not in r]
    total = Counter[str]()
    for key in ("nav_counts", "mode_counts"):
        for r in ok:
            total.update({f"{key}:{k}": v for k, v in r[key].items()})
    return {
        "episodes": len(records),
        "unreadable": [r["file"] for r in records if "error" in r],
        "bytes": sum(r["bytes"] for r in records),
        "frames": sum(r["frames"] for r in ok),
        "usable_frames": sum(r["usable_frames"] for r in ok),
        "schemas": dict(Counter(str(r["schema"]) for r in ok)),
        "statuses": dict(Counter(str(r["meta"].get(S.META_STATUS)) for r in ok)),
        "towns": dict(Counter(str(r["meta"].get(S.META_TOWN)) for r in ok)),
        "weathers": dict(Counter(str(r["meta"].get(S.META_WEATHER)) for r in ok)),
        "frame_counts": dict(Counter(str(r["frames"]) for r in ok).most_common(5)),
        "nonfinite_values": sum(r["nonfinite_values"] for r in ok),
        "nav_counts": {k.split(":")[1]: v for k, v in total.items() if k.startswith("nav")},
        "mode_counts": {k.split(":")[1]: v for k, v in total.items() if k.startswith("mode")},
        "field_sets": dict(Counter(",".join(r["fields"]) for r in ok)),
    }


def main() -> None:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("data_dir", type=Path)
    p.add_argument("out", type=Path)
    p.add_argument("--workers", type=int, default=8)
    args = p.parse_args()

    paths = sorted(args.data_dir.glob("episode_*.h5"))
    with Pool(args.workers) as pool:
        records = pool.map(scan_episode, paths, chunksize=8)
    report = {"data_dir": args.data_dir.name, "summary": summarize(records), "episodes": records}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=1))
    s = report["summary"]
    print(
        f"{s['episodes']} episodes, {s['frames']:,} frames ({s['usable_frames']:,} usable), "
        f"{s['bytes'] / 1e12:.2f} TB, schemas {s['schemas']}, unreadable {len(s['unreadable'])}, "
        f"non-finite values {s['nonfinite_values']}"
    )
    print(f"statuses {s['statuses']}")
    print(f"towns {s['towns']}")


if __name__ == "__main__":
    main()
