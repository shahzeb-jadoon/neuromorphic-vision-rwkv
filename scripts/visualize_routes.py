"""Visualize recorded vehicle trajectories from HDF5 episode files.

Reads position and nav_command data from collected episodes and produces
bird's-eye-view plots showing the ego-vehicle's trajectory, colour-coded
by navigation command (Left / Right / Straight / LaneFollow).

Usage:
    python scripts/visualize_routes.py data/test_collection/
    python scripts/visualize_routes.py data/test_collection/ --save route_plots/
"""

from __future__ import annotations

import argparse
import glob
import sys
from pathlib import Path
from typing import Any

import h5py
import numpy as np

try:
    import matplotlib

    matplotlib.use("Agg")  # Non-interactive backend (no display needed)
    import matplotlib.pyplot as plt

    MPL_AVAILABLE = True
except ImportError:
    MPL_AVAILABLE = False

from src.data import hdf5_schema as S

# Colour map for nav commands
NAV_COLOURS: dict[int, str] = {
    1: "#e74c3c",  # Left  — red
    2: "#2ecc71",  # Right — green
    3: "#3498db",  # Straight — blue
    4: "#95a5a6",  # LaneFollow — grey
}

NAV_NAMES: dict[int, str] = {
    1: "Left",
    2: "Right",
    3: "Straight",
    4: "LaneFollow",
}


def load_episode(path: str | Path) -> dict[str, Any]:
    """Load trajectory data from a single HDF5 episode file."""
    with h5py.File(path, "r") as h:
        data: dict[str, Any] = {
            "path": str(path),
            "position": h[S.POSITION][:],  # (N, 3)
            "nav_command": h[S.NAV_COMMAND][:],  # (N,)
        }
        if S.VELOCITY in h:
            data["velocity"] = h[S.VELOCITY][:]  # (N, 3)
        if S.ROTATION in h:
            data["rotation"] = h[S.ROTATION][:]  # (N, 3)

        meta = h[S.METADATA_GROUP]
        data["town"] = meta.attrs.get(S.META_TOWN, "?")
        data["weather"] = meta.attrs.get(S.META_WEATHER, "?")
        data["status"] = meta.attrs.get(S.META_STATUS, "?")
        data["n_frames"] = meta.attrs.get(S.META_N_FRAMES, len(data["position"]))
        if S.META_COLLISION_FRAME in meta.attrs:
            data["collision_frame"] = int(meta.attrs[S.META_COLLISION_FRAME])
        else:
            data["collision_frame"] = None

    return data


def compute_speed_kmh(velocity: np.ndarray) -> np.ndarray:
    """Compute scalar speed in km/h from (N, 3) velocity array."""
    speed_ms: np.ndarray = np.linalg.norm(velocity, axis=1)
    return speed_ms * 3.6


def plot_single_episode(
    data: dict[str, Any],
    ax: Any,
    title: str | None = None,
) -> None:
    """Plot one episode trajectory on the given axes."""
    pos = data["position"]  # (N, 3) — x, y, z in CARLA world coords
    nav = data["nav_command"]  # (N,)

    # CARLA uses left-handed coordinates; x=forward, y=right
    # For bird's-eye view: plot x vs -y (so "up" = north, "right" = east)
    x = pos[:, 0]
    y = -pos[:, 1]  # Flip y for standard map orientation

    # Plot trajectory segments colour-coded by nav command
    for cmd_id, colour in NAV_COLOURS.items():
        mask = nav == cmd_id
        if not mask.any():
            continue
        ax.scatter(
            x[mask],
            y[mask],
            c=colour,
            s=2,
            alpha=0.6,
            label=NAV_NAMES[cmd_id],
            zorder=2,
        )

    # Mark start and end
    ax.plot(x[0], y[0], "o", color="#27ae60", markersize=10, zorder=5, label="Start")
    ax.plot(x[-1], y[-1], "X", color="#c0392b", markersize=10, zorder=5, label="End")

    # Mark collision frame if any
    if data.get("collision_frame") is not None:
        cf = data["collision_frame"]
        if cf < len(x):
            ax.plot(
                x[cf],
                y[cf],
                "*",
                color="#f39c12",
                markersize=15,
                zorder=6,
                label=f"Collision (frame {cf})",
            )

    ax.set_aspect("equal")
    ax.grid(True, alpha=0.3)
    ax.set_xlabel("X (metres)")
    ax.set_ylabel("Y (metres, flipped)")

    if title:
        ax.set_title(title, fontsize=10)

    # Remove duplicate legend entries
    handles, labels = ax.get_legend_handles_labels()
    unique = dict(zip(labels, handles))
    ax.legend(unique.values(), unique.keys(), fontsize=7, loc="best")


def plot_speed_profile(
    data: dict[str, Any],
    ax: Any,
) -> None:
    """Plot speed over time for one episode."""
    if "velocity" not in data:
        ax.text(0.5, 0.5, "No velocity data", ha="center", va="center")
        return

    speed = compute_speed_kmh(data["velocity"])
    frames = np.arange(len(speed))

    ax.plot(frames, speed, linewidth=0.8, color="#2c3e50")
    ax.set_xlabel("Frame")
    ax.set_ylabel("Speed (km/h)")
    ax.set_title("Speed Profile", fontsize=10)
    ax.grid(True, alpha=0.3)

    # Mark collision
    if data.get("collision_frame") is not None:
        cf = data["collision_frame"]
        if cf < len(speed):
            ax.axvline(cf, color="#f39c12", linestyle="--", label=f"Collision (f{cf})")
            ax.legend(fontsize=7)


def plot_overview(
    episodes: list[dict[str, Any]],
    save_path: Path | None = None,
) -> None:
    """Plot all episodes on a single map overview."""
    if not MPL_AVAILABLE:
        print("matplotlib not installed — skipping plots")
        return

    fig, ax = plt.subplots(1, 1, figsize=(14, 10))

    colours = plt.cm.tab10.colors  # type: ignore[attr-defined]

    for i, ep in enumerate(episodes):
        pos = ep["position"]
        x = pos[:, 0]
        y = -pos[:, 1]
        colour = colours[i % len(colours)]
        name = Path(ep["path"]).stem

        ax.plot(x, y, linewidth=1.2, color=colour, alpha=0.7, label=name)
        ax.plot(x[0], y[0], "o", color=colour, markersize=8)
        ax.plot(x[-1], y[-1], "X", color=colour, markersize=8)

        if ep.get("collision_frame") is not None:
            cf = ep["collision_frame"]
            if cf < len(x):
                ax.plot(x[cf], y[cf], "*", color="#f39c12", markersize=12)

    ax.set_aspect("equal")
    ax.grid(True, alpha=0.3)
    ax.set_xlabel("X (metres)")
    ax.set_ylabel("Y (metres, flipped)")
    ax.set_title(
        f"All Episodes Overview — {episodes[0].get('town', '?')}",
        fontsize=12,
    )
    ax.legend(fontsize=7, loc="best")

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path / "overview.png", dpi=150)
        print(f"  Saved: {save_path / 'overview.png'}")
    else:
        plt.show()
    plt.close(fig)


def plot_episode_detail(
    data: dict[str, Any],
    idx: int,
    save_path: Path | None = None,
) -> None:
    """Plot detailed view: trajectory + speed profile for one episode."""
    if not MPL_AVAILABLE:
        return

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    name = Path(data["path"]).stem
    status = data.get("status", "?")
    n_frames = data.get("n_frames", len(data["position"]))
    title = f"{name} — {status} — {n_frames} frames"

    plot_single_episode(data, axes[0], title=title)
    plot_speed_profile(data, axes[1])

    plt.tight_layout()
    if save_path:
        fname = save_path / f"episode_{idx:03d}_detail.png"
        fig.savefig(fname, dpi=150)
        print(f"  Saved: {fname}")
    else:
        plt.show()
    plt.close(fig)


def print_trajectory_summary(data: dict[str, Any]) -> None:
    """Print a text summary of the trajectory."""
    pos = data["position"]
    nav = data["nav_command"]

    start = pos[0]
    end = pos[-1]
    distance = np.sum(np.linalg.norm(np.diff(pos, axis=0), axis=1))

    if "velocity" in data:
        speed = compute_speed_kmh(data["velocity"])
        avg_speed = float(np.mean(speed))
        max_speed = float(np.max(speed))
    else:
        avg_speed = max_speed = 0.0

    name = Path(data["path"]).stem
    print(f"\n  {name}:")
    print(f"    Status: {data.get('status', '?')}")
    print(f"    Frames: {data.get('n_frames', len(pos))}")
    print(f"    Start:  ({start[0]:.1f}, {start[1]:.1f}, {start[2]:.1f})")
    print(f"    End:    ({end[0]:.1f}, {end[1]:.1f}, {end[2]:.1f})")
    print(f"    Distance: {distance:.1f}m")
    print(f"    Speed:  avg={avg_speed:.1f} km/h, max={max_speed:.1f} km/h")

    # Nav distribution
    total = len(nav)
    for cmd_id in sorted(NAV_NAMES):
        count = int(np.sum(nav == cmd_id))
        pct = count / total * 100 if total else 0
        print(f"    {NAV_NAMES[cmd_id]:>10s}: {count:4d}/{total} ({pct:5.1f}%)")

    if data.get("collision_frame") is not None:
        print(f"    Collision at frame: {data['collision_frame']}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Visualize collected episode trajectories")
    parser.add_argument("data_dir", help="Directory containing episode_*.h5 files")
    parser.add_argument("--save", help="Directory to save plot PNGs (created if needed)")
    parser.add_argument(
        "--no-detail",
        action="store_true",
        help="Skip per-episode detail plots (overview only)",
    )
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    files = sorted(glob.glob(str(data_dir / "episode_*.h5")))

    if not files:
        print(f"ERROR: No episode_*.h5 files found in {data_dir}")
        sys.exit(1)

    print(f"Found {len(files)} episode(s) in {data_dir}")

    # Load all episodes
    episodes = [load_episode(f) for f in files]

    # Print text summaries
    print("\n" + "=" * 60)
    print("Trajectory Summaries")
    print("=" * 60)
    for ep in episodes:
        print_trajectory_summary(ep)

    # Save path
    save_path: Path | None = None
    if args.save:
        save_path = Path(args.save)
        save_path.mkdir(parents=True, exist_ok=True)

    if not MPL_AVAILABLE:
        print("\nWARNING: matplotlib not installed — text summaries only")
        return

    # Plot overview
    print("\n" + "=" * 60)
    print("Generating plots...")
    print("=" * 60)
    plot_overview(episodes, save_path)

    # Plot per-episode detail
    if not args.no_detail:
        for i, ep in enumerate(episodes):
            plot_episode_detail(ep, i, save_path)

    if save_path:
        print(f"\nAll plots saved to: {save_path}")
    print("Done.")


if __name__ == "__main__":
    main()
