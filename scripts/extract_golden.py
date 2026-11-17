"""Extract a small, fixed set of real episode windows ("golden subset") for regression tests.

Selection is deterministic: per town, one window from a completed episode and one from a
collision or timeout episode (alternating by town), each WINDOW frames long and holding at
least MIN_TURN frames of both LEFT and RIGHT commands. A collision window ends at the
collision and needs only one of them: a crash usually happens within one manoeuvre.
Images are copied as their original compressed chunks, so pixels are unchanged.

usage: python scripts/extract_golden.py SCAN.json DATA_DIR OUT_DIR [--seed 0]
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import h5py
import numpy as np

from src.data import hdf5_schema as S

WINDOW = 800
MIN_TURN = 50
LEFT, RIGHT = 1, 2


def nav_array(runs: list[list[int]], n: int) -> np.ndarray:
    nav = np.zeros(n, dtype=np.int32)
    for value, start, length in runs:
        nav[start : start + length] = value
    return nav


def window_start(episode: dict[str, Any]) -> int | None:
    """Start of the first qualifying window, or None. Collision windows end at the crash."""
    n = episode["usable_frames"]
    if n < WINDOW:
        return None
    nav = nav_array(episode["nav_runs"], n)
    left, right = np.cumsum(nav == LEFT), np.cumsum(nav == RIGHT)

    def turns(s: int, both: bool = True) -> bool:
        end = s + WINDOW - 1
        l_n = left[end] - (left[s - 1] if s else 0)
        r_n = right[end] - (right[s - 1] if s else 0)
        ok = (l_n >= MIN_TURN, r_n >= MIN_TURN)
        return all(ok) if both else any(ok)

    if episode["meta"].get(S.META_STATUS) == "collision":
        s = n - WINDOW
        return s if turns(s, both=False) else None
    return next((s for s in range(0, n - WINDOW + 1, 50) if turns(s)), None)


def select(episodes: list[dict[str, Any]], seed: int) -> list[tuple[dict[str, Any], int]]:
    rng = np.random.default_rng(seed)
    towns = sorted({e["meta"][S.META_TOWN] for e in episodes})
    chosen = []
    for i, town in enumerate(towns):
        for status in ("completed", ("collision", "timeout")[i % 2]):
            pool = [
                (e, s)
                for e in episodes
                if e["meta"][S.META_TOWN] == town and e["meta"].get(S.META_STATUS) == status
                for s in [window_start(e)]
                if s is not None
            ]
            if not pool:
                raise SystemExit(f"no qualifying {status} window in {town}")
            chosen.append(pool[int(rng.integers(len(pool)))])
    return chosen


def copy_window(src: Path, dst: Path, start: int) -> None:
    end = start + WINDOW
    with h5py.File(src, "r") as f, h5py.File(dst, "w") as g:
        for name, ds in f.items():
            if not isinstance(ds, h5py.Dataset):
                continue
            if name == S.IMAGES:
                out = g.create_dataset(
                    name,
                    shape=(WINDOW, *ds.shape[1:]),
                    dtype=ds.dtype,
                    chunks=ds.chunks,
                    compression=ds.compression,
                    compression_opts=ds.compression_opts,
                )
                if ds.chunks is not None and ds.chunks == (1, *ds.shape[1:]):
                    for i in range(WINDOW):
                        filter_mask, chunk = ds.id.read_direct_chunk((start + i, 0, 0, 0))
                        out.id.write_direct_chunk((i, 0, 0, 0), chunk, filter_mask)
                else:
                    out[...] = ds[start:end]
            else:
                g.create_dataset(
                    name, data=ds[start:end], dtype=ds.dtype, compression=ds.compression
                )
        meta = g.create_group(S.METADATA_GROUP)
        for k, v in f[S.METADATA_GROUP].attrs.items():
            meta.attrs[k] = v
        meta.attrs[S.META_N_FRAMES] = WINDOW
        if S.META_COLLISION_FRAME in meta.attrs:
            del meta.attrs[S.META_COLLISION_FRAME]
        meta.attrs[S.META_GOLDEN_SOURCE_FILE] = src.name
        meta.attrs[S.META_GOLDEN_SOURCE_START] = start


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(1 << 22), b""):
            h.update(block)
    return h.hexdigest()


def main() -> None:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("scan", type=Path)
    p.add_argument("data_dir", type=Path)
    p.add_argument("out_dir", type=Path)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    episodes = [e for e in json.loads(args.scan.read_text())["episodes"] if "error" not in e]
    args.out_dir.mkdir(parents=True, exist_ok=True)
    index = []
    for e, start in select(episodes, args.seed):
        src = args.data_dir / e["file"]
        dst = args.out_dir / f"{src.stem}_{start:04d}.h5"
        copy_window(src, dst, start)
        index.append(
            {
                "file": dst.name,
                "source": src.name,
                "start": start,
                "frames": WINDOW,
                "town": e["meta"][S.META_TOWN],
                "weather": e["meta"].get(S.META_WEATHER),
                "status": e["meta"].get(S.META_STATUS),
                "sha256": sha256(dst),
            }
        )
        print(
            f"{dst.name}: {index[-1]['town']} {index[-1]['status']} frames {start}-{start + WINDOW}"
        )
    (args.out_dir / "index.json").write_text(
        json.dumps({"seed": args.seed, "windows": index}, indent=1)
    )


if __name__ == "__main__":
    main()
