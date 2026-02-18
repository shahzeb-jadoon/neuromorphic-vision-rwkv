"""Benchmark HDF5 compression strategies for DataLoader throughput.

Tests different compression levels to find the best trade-off between
storage size and read speed for 32-frame clip random access patterns.

Usage:
    python scripts/benchmark_hdf5.py
"""

import os
import tempfile
import time

import h5py
import numpy as np


def main() -> None:
    print("HDF5 Compression Benchmark")
    print("=" * 65)
    print(f"Reading pattern: 100 random 32-frame clips from 500 frames")
    print(f"Image size: 224×224×3 (uint8)")
    print()

    rng = np.random.default_rng(42)
    images = rng.integers(0, 255, size=(500, 224, 224, 3), dtype=np.uint8)

    configs = [
        ("uncompressed", {}),
        ("gzip-1", {"compression": "gzip", "compression_opts": 1}),
        ("gzip-4 (current)", {"compression": "gzip", "compression_opts": 4}),
        ("gzip-9", {"compression": "gzip", "compression_opts": 9}),
    ]

    header = f"{'Config':<20} {'Size MB':>10} {'Write (s)':>10} {'Read (s)':>10} {'Read MB/s':>10}"
    print(header)
    print("-" * 65)

    best_read = None
    best_name = None

    for name, opts in configs:
        tmpf = tempfile.NamedTemporaryFile(suffix=".h5", delete=False)
        tmppath = tmpf.name
        tmpf.close()

        # Write
        t0 = time.time()
        with h5py.File(tmppath, "w") as f:
            f.create_dataset("images", data=images, chunks=(1, 224, 224, 3), **opts)
        write_time = time.time() - t0
        file_size = os.path.getsize(tmppath) / 1e6

        # Read (simulate DataLoader random clip access)
        t0 = time.time()
        with h5py.File(tmppath, "r") as f:
            for _ in range(100):
                start = int(rng.integers(0, 500 - 32))
                _ = f["images"][start : start + 32]  # 32-frame clip read
        read_time = time.time() - t0
        read_mb_per_sec = (100 * 32 * 224 * 224 * 3 / 1e6) / read_time

        print(
            f"{name:<20} {file_size:>10.1f} {write_time:>10.2f} {read_time:>10.2f} {read_mb_per_sec:>10.1f}"
        )

        if best_read is None or read_time < best_read:
            best_read = read_time
            best_name = name

        os.unlink(tmppath)

    print()
    print(f"Fastest read: {best_name}")
    print()

    # Also benchmark with real CARLA data if available
    import glob

    real_episodes = glob.glob("data/campaign_test/episode_*.h5")
    if real_episodes:
        ep_path = real_episodes[0]
        print(f"Real CARLA data benchmark: {os.path.basename(ep_path)}")
        print("-" * 65)

        with h5py.File(ep_path, "r") as f:
            n_frames = f["images"].shape[0]
            comp = f["images"].compression
            comp_opts = f["images"].compression_opts
            print(f"  Frames: {n_frames}, Compression: {comp} (level {comp_opts})")
            print(f"  File size: {os.path.getsize(ep_path) / 1e6:.1f} MB")

            t0 = time.time()
            n_clips = min(50, n_frames // 32)
            for _ in range(n_clips):
                start = int(rng.integers(0, n_frames - 32))
                _ = f["images"][start : start + 32]
            read_time = time.time() - t0
            rate = n_clips / read_time
            print(
                f"  Random clip reads: {n_clips} clips in {read_time:.2f}s = {rate:.1f} clips/sec"
            )
    else:
        print("No real CARLA data found in data/campaign_test/")


if __name__ == "__main__":
    main()
