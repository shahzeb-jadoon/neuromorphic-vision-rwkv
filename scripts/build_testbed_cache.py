"""Build a training cache from driving episodes: JPEG images at a lower rate, signals raw.

Each source episode becomes one HDF5 file holding every non-image dataset unchanged (full
rate, lossless) and one JPEG every `stride` source frames, resized the way training
resizes (bilinear, no corner alignment). src.data.schema_adapter reads a cache as an
episode at the image rate, so labels are computed at load time from the raw signals.
Episodes are written under a temporary name and renamed when complete; finished episodes
are skipped, so an interrupted build resumes where it stopped.

usage: python scripts/build_testbed_cache.py SRC_DIR OUT_DIR [--workers 6]
"""

from __future__ import annotations

import argparse
import io
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from src.data import hdf5_schema as S

CHUNK = 32  # source images decoded at a time


def build_episode(
    src: Path, dst: Path, height: int, width: int, quality: int, stride: int
) -> dict[str, Any]:
    torch.set_num_threads(1)
    tmp = dst.with_suffix(".tmp")
    with h5py.File(src, "r") as f, h5py.File(tmp, "w") as g:
        images = f[S.IMAGES]
        n_images = -(-images.shape[0] // stride)
        jpeg = g.create_dataset(S.IMAGES_JPEG, shape=(n_images,), dtype=h5py.vlen_dtype(np.uint8))
        for k0 in range(0, n_images, CHUNK):
            ks = range(k0, min(k0 + CHUNK, n_images))
            frames = torch.from_numpy(np.stack([images[k * stride] for k in ks]))
            small = F.interpolate(
                frames.permute(0, 3, 1, 2).float(),
                size=(height, width),
                mode="bilinear",
                align_corners=False,
            )
            pixels = small.round().clamp(0, 255).to(torch.uint8).permute(0, 2, 3, 1).numpy()
            for k, img in zip(ks, pixels):
                buf = io.BytesIO()
                Image.fromarray(img).save(buf, format="JPEG", quality=quality)
                jpeg[k] = np.frombuffer(buf.getvalue(), dtype=np.uint8)
        for name, ds in f.items():
            if isinstance(ds, h5py.Dataset) and name != S.IMAGES:
                g.create_dataset(name, data=ds[()], dtype=ds.dtype, compression=ds.compression)
        meta = g.create_group(S.METADATA_GROUP)
        for key, value in f[S.METADATA_GROUP].attrs.items():
            meta.attrs[key] = value
        meta.attrs[S.META_CACHE_FORMAT] = f"jpeg-q{quality}-{width}x{height}-stride{stride}"
        meta.attrs[S.META_IMAGE_STRIDE] = stride
        meta.attrs[S.META_SOURCE_FILE] = src.name
    tmp.rename(dst)
    return {"file": dst.name, "images": n_images, "bytes": dst.stat().st_size}


def main() -> None:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("src_dir", type=Path)
    p.add_argument("out_dir", type=Path)
    p.add_argument("--height", type=int, default=160)
    p.add_argument("--width", type=int, default=960)
    p.add_argument("--quality", type=int, default=90)
    p.add_argument("--stride", type=int, default=2)
    p.add_argument("--workers", type=int, default=6)
    args = p.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    for stale in args.out_dir.glob("*.tmp"):
        stale.unlink()
    sources = sorted(args.src_dir.glob("episode_*.h5"))
    todo = [s for s in sources if not (args.out_dir / s.name).exists()]
    print(f"{len(sources)} episodes, {len(sources) - len(todo)} already built", flush=True)

    t0, done, total, failed = time.time(), 0, 0, []
    with ProcessPoolExecutor(args.workers) as pool:
        jobs = {
            pool.submit(
                build_episode,
                s,
                args.out_dir / s.name,
                args.height,
                args.width,
                args.quality,
                args.stride,
            ): s
            for s in todo
        }
        for job in as_completed(jobs):
            try:
                total += job.result()["bytes"]
            except Exception as e:  # noqa: BLE001 — report every failed episode, keep going
                failed.append(f"{jobs[job].name}: {type(e).__name__}: {e}")
            done += 1
            if done % 10 == 0 or done == len(todo):
                eta = (len(todo) - done) / (done / (time.time() - t0)) / 60
                print(
                    f"{done}/{len(todo)} episodes, {total / 1e9:.1f} GB written, ETA {eta:.0f} min",
                    flush=True,
                )
    for line in failed:
        print("FAILED", line)
    raise SystemExit(1 if failed else 0)


if __name__ == "__main__":
    main()
