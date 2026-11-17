#!/usr/bin/env python3
"""Diagnose HDF5 write behaviour: page cache vs direct disk I/O.

This script writes HDF5 data in the same pattern as the real data collector
(_StreamingEpisodeWriter) and monitors:
  1. Page cache growth (from /proc/meminfo)
  2. Python RSS (from /proc/self/status)
  3. Actual disk bytes written (from /proc/self/io)
  4. Whether data sits in page cache or hits disk

Usage:
    conda run -n rwkv python scripts/diagnose_hdf5_cache.py

The test writes ~4 GB of dummy panoramic frames (1000 frames × 1920×320×3)
with gzip-1 compression, exactly like the real writer, then measures cache.
"""

import gc
import os
import shutil
import subprocess
import time
from pathlib import Path

import h5py
import numpy as np


def get_meminfo() -> dict[str, int]:
    """Parse /proc/meminfo, return values in kB."""
    info = {}
    with open("/proc/meminfo") as f:
        for line in f:
            parts = line.split()
            if len(parts) >= 2:
                key = parts[0].rstrip(":")
                info[key] = int(parts[1])
    return info


def get_cache_mb() -> float:
    """Return Linux page cache size in MB (Cached + Buffers)."""
    m = get_meminfo()
    return (m.get("Cached", 0) + m.get("Buffers", 0)) / 1024.0


def get_mem_available_mb() -> float:
    m = get_meminfo()
    return m.get("MemAvailable", 0) / 1024.0


def get_rss_mb() -> float:
    with open("/proc/self/status") as f:
        for line in f:
            if line.startswith("VmRSS:"):
                return int(line.split()[1]) / 1024.0
    return 0.0


def get_io_write_bytes() -> int:
    """Return total bytes written by this process (from /proc/self/io)."""
    try:
        with open("/proc/self/io") as f:
            for line in f:
                if line.startswith("write_bytes:"):
                    return int(line.split()[1])
    except PermissionError:
        pass
    return -1


def drop_caches() -> None:
    """Try to drop page cache. Requires root or CAP_SYS_ADMIN."""
    try:
        subprocess.run(
            ["sudo", "-n", "sh", "-c", "sync; echo 3 > /proc/sys/vm/drop_caches"],
            timeout=5,
            capture_output=True,
        )
    except Exception:
        pass


def test_streaming_write(
    out_dir: Path,
    n_frames: int = 1000,
    h: int = 320,
    w: int = 1920,
    c: int = 3,
    compression: int = 1,
    flush_interval: int = 0,
) -> dict:
    """Write n_frames of dummy panoramic images, mimicking the real writer.

    Args:
        flush_interval: If >0, call h5.flush() every N frames. 0 = never flush.
    """
    filepath = out_dir / "test_streaming.h5"

    # Baseline measurements
    gc.collect()
    cache_before = get_cache_mb()
    rss_before = get_rss_mb()
    avail_before = get_mem_available_mb()
    io_before = get_io_write_bytes()

    print(f"\n{'=' * 60}")
    print("  Streaming HDF5 Write Test")
    print(f"  Frames: {n_frames}, Resolution: {w}x{h}x{c}")
    print(f"  Raw frame size: {h * w * c / 1024 / 1024:.2f} MB")
    print(f"  Total raw data: {n_frames * h * w * c / 1024 / 1024 / 1024:.2f} GB")
    print(f"  Compression: gzip level {compression}")
    print(
        f"  Flush interval: {'never' if not flush_interval else f'every {flush_interval} frames'}"
    )
    print(f"  Output: {filepath}")
    print(f"{'=' * 60}")
    print("\n  Baseline:")
    print(f"    Page cache:   {cache_before:.0f} MB")
    print(f"    MemAvailable: {avail_before:.0f} MB")
    print(f"    Python RSS:   {rss_before:.0f} MB")
    if io_before >= 0:
        print(f"    Disk written: {io_before / 1024 / 1024:.0f} MB")

    # Create HDF5 file exactly like _StreamingEpisodeWriter
    f = h5py.File(filepath, "w")
    ds = f.create_dataset(
        "images",
        shape=(n_frames, h, w, c),
        dtype=np.uint8,
        maxshape=(None, h, w, c),
        chunks=(1, h, w, c),
        compression="gzip",
        compression_opts=compression,
    )

    t0 = time.time()
    print(f"\n  Writing {n_frames} frames...")
    print(f"  {'Frame':>8} {'Elapsed':>8} {'Cache':>8} {'RSS':>8} {'DiskWr':>10} {'Rate':>10}")

    for i in range(n_frames):
        # Random data (incompressible) to simulate worst-case
        frame = np.random.randint(0, 256, (h, w, c), dtype=np.uint8)
        ds[i] = frame

        if flush_interval and (i + 1) % flush_interval == 0:
            f.flush()

        if (i + 1) % 100 == 0 or i == 0:
            elapsed = time.time() - t0
            cache_now = get_cache_mb()
            rss_now = get_rss_mb()
            io_now = get_io_write_bytes()
            disk_wr = (io_now - io_before) / 1024 / 1024 if io_before >= 0 else -1
            rate = (cache_now - cache_before) / elapsed if elapsed > 0 else 0
            print(
                f"  {i + 1:>8} {elapsed:>7.1f}s {cache_now:>7.0f}M {rss_now:>7.0f}M "
                f"{disk_wr:>9.0f}M {rate:>8.1f}M/s"
            )

    # After all writes, before close
    elapsed = time.time() - t0
    cache_after_write = get_cache_mb()
    rss_after_write = get_rss_mb()
    io_after_write = get_io_write_bytes()

    print("\n  After writing (before close):")
    print(
        f"    Page cache:   {cache_after_write:.0f} MB (delta: +{cache_after_write - cache_before:.0f} MB)"  # noqa: E501
    )
    print(
        f"    MemAvailable: {get_mem_available_mb():.0f} MB (delta: {get_mem_available_mb() - avail_before:.0f} MB)"  # noqa: E501
    )
    print(
        f"    Python RSS:   {rss_after_write:.0f} MB (delta: +{rss_after_write - rss_before:.0f} MB)"  # noqa: E501
    )
    if io_before >= 0:
        disk_wr = (io_after_write - io_before) / 1024 / 1024
        print(f"    Disk written: {disk_wr:.0f} MB")
    print(f"    Write time:   {elapsed:.1f}s")

    # Close the file
    f.close()
    cache_after_close = get_cache_mb()
    io_after_close = get_io_write_bytes()

    print("\n  After close:")
    print(
        f"    Page cache:   {cache_after_close:.0f} MB (delta: +{cache_after_close - cache_before:.0f} MB)"  # noqa: E501
    )
    if io_before >= 0:
        disk_wr = (io_after_close - io_before) / 1024 / 1024
        print(f"    Disk written: {disk_wr:.0f} MB")

    # File size on disk
    file_size = filepath.stat().st_size
    print(f"    File on disk: {file_size / 1024 / 1024:.0f} MB")

    # Try to drop cache and see what happens
    print("\n  Attempting to drop page cache (sync + drop_caches)...")
    os.sync()
    time.sleep(1)
    cache_after_sync = get_cache_mb()
    print(
        f"    After sync:   {cache_after_sync:.0f} MB (delta: +{cache_after_sync - cache_before:.0f} MB)"  # noqa: E501
    )

    drop_caches()
    time.sleep(1)
    cache_after_drop = get_cache_mb()
    avail_after_drop = get_mem_available_mb()
    print(
        f"    After drop:   {cache_after_drop:.0f} MB (delta: +{cache_after_drop - cache_before:.0f} MB)"  # noqa: E501
    )
    print(
        f"    MemAvailable: {avail_after_drop:.0f} MB (recovered: +{avail_after_drop - get_mem_available_mb():.0f} MB)"  # noqa: E501
    )

    return {
        "n_frames": n_frames,
        "raw_gb": n_frames * h * w * c / 1024**3,
        "file_mb": file_size / 1024**2,
        "cache_growth_mb": cache_after_write - cache_before,
        "rss_growth_mb": rss_after_write - rss_before,
        "cache_after_drop_mb": cache_after_drop - cache_before,
        "write_time_s": elapsed,
    }


def test_write_with_periodic_sync(
    out_dir: Path,
    n_frames: int = 1000,
    h: int = 320,
    w: int = 1920,
    c: int = 3,
    sync_interval: int = 100,
) -> dict:
    """Same as streaming write but calls os.sync() periodically.

    This forces dirty pages to be written to disk, potentially freeing
    page cache entries for WSL2 to release back to the Windows host.
    """
    filepath = out_dir / "test_sync.h5"

    gc.collect()
    cache_before = get_cache_mb()

    print(f"\n{'=' * 60}")
    print(f"  Streaming Write + Periodic sync (every {sync_interval} frames)")
    print(f"{'=' * 60}")

    f = h5py.File(filepath, "w")
    ds = f.create_dataset(
        "images",
        shape=(n_frames, h, w, c),
        dtype=np.uint8,
        maxshape=(None, h, w, c),
        chunks=(1, h, w, c),
        compression="gzip",
        compression_opts=1,
    )

    t0 = time.time()
    peak_cache = cache_before
    print(f"  {'Frame':>8} {'Elapsed':>8} {'Cache':>8} {'CacheDelta':>12} {'Action':>10}")

    for i in range(n_frames):
        frame = np.random.randint(0, 256, (h, w, c), dtype=np.uint8)
        ds[i] = frame

        if (i + 1) % sync_interval == 0:
            f.flush()
            os.sync()
            cache_now = get_cache_mb()
            delta = cache_now - cache_before
            if cache_now > peak_cache:
                peak_cache = cache_now
            print(
                f"  {i + 1:>8} {time.time() - t0:>7.1f}s {cache_now:>7.0f}M {delta:>+10.0f}M   sync"
            )

    elapsed = time.time() - t0
    cache_after = get_cache_mb()
    f.close()

    file_size = filepath.stat().st_size
    print("\n  Results:")
    print(f"    Peak cache delta: +{peak_cache - cache_before:.0f} MB")
    print(f"    Final cache delta: +{cache_after - cache_before:.0f} MB")
    print(f"    File on disk: {file_size / 1024 / 1024:.0f} MB")
    print(f"    Write time: {elapsed:.1f}s")

    return {
        "peak_cache_growth_mb": peak_cache - cache_before,
        "final_cache_growth_mb": cache_after - cache_before,
        "file_mb": file_size / 1024**2,
    }


def test_write_with_posix_fadvise(
    out_dir: Path,
    n_frames: int = 1000,
    h: int = 320,
    w: int = 1920,
    c: int = 3,
    advise_interval: int = 100,
) -> dict:
    """Write HDF5 then use posix_fadvise(DONTNEED) to evict pages.

    This tells the kernel we don't need the written pages anymore,
    allowing it to drop them from page cache immediately.
    """
    filepath = out_dir / "test_fadvise.h5"

    gc.collect()
    cache_before = get_cache_mb()

    print(f"\n{'=' * 60}")
    print("  Streaming Write + posix_fadvise(DONTNEED)")
    print(f"{'=' * 60}")

    f = h5py.File(filepath, "w")
    ds = f.create_dataset(
        "images",
        shape=(n_frames, h, w, c),
        dtype=np.uint8,
        maxshape=(None, h, w, c),
        chunks=(1, h, w, c),
        compression="gzip",
        compression_opts=1,
    )

    t0 = time.time()
    peak_cache = cache_before
    POSIX_FADV_DONTNEED = 4  # from <fcntl.h>

    print(f"  {'Frame':>8} {'Elapsed':>8} {'Cache':>8} {'CacheDelta':>12} {'Action':>10}")

    for i in range(n_frames):
        frame = np.random.randint(0, 256, (h, w, c), dtype=np.uint8)
        ds[i] = frame

        if (i + 1) % advise_interval == 0:
            f.flush()
            # Get the file descriptor
            fd = f.id.get_vfd_handle()
            if isinstance(fd, int) and fd > 0:
                try:
                    import ctypes
                    import ctypes.util

                    libc = ctypes.CDLL(ctypes.util.find_library("c"), use_errno=True)
                    # fsync first to ensure data is on disk
                    os.fsync(fd)
                    # Then tell kernel we don't need these pages
                    libc.posix_fadvise(fd, 0, 0, POSIX_FADV_DONTNEED)
                except Exception as e:
                    print(f"    fadvise failed: {e}")

            cache_now = get_cache_mb()
            delta = cache_now - cache_before
            if cache_now > peak_cache:
                peak_cache = cache_now
            elapsed = time.time() - t0
            print(f"  {i + 1:>8} {elapsed:>7.1f}s {cache_now:>7.0f}M {delta:>+10.0f}M   fadvise")

    elapsed = time.time() - t0
    cache_after = get_cache_mb()
    f.close()

    file_size = filepath.stat().st_size
    print("\n  Results:")
    print(f"    Peak cache delta: +{peak_cache - cache_before:.0f} MB")
    print(f"    Final cache delta: +{cache_after - cache_before:.0f} MB")
    print(f"    File on disk: {file_size / 1024 / 1024:.0f} MB")
    print(f"    Write time: {elapsed:.1f}s")

    return {
        "peak_cache_growth_mb": peak_cache - cache_before,
        "final_cache_growth_mb": cache_after - cache_before,
        "file_mb": file_size / 1024**2,
    }


def main() -> None:
    print("=" * 60)
    print("  HDF5 Write + Page Cache Diagnostic")
    print("  " + time.strftime("%Y-%m-%d %H:%M:%S"))
    print("=" * 60)

    m = get_meminfo()
    print("\n  System memory:")
    print(f"    Total:     {m['MemTotal'] / 1024:.0f} MB")
    print(f"    Free:      {m['MemFree'] / 1024:.0f} MB")
    print(f"    Available: {m['MemAvailable'] / 1024:.0f} MB")
    print(f"    Cached:    {m['Cached'] / 1024:.0f} MB")
    print(f"    Buffers:   {m['Buffers'] / 1024:.0f} MB")

    out_dir = Path("data/test_hdf5_cache")
    out_dir.mkdir(parents=True, exist_ok=True)

    # Test 1: Baseline — no flush, no sync (current behavior)
    r1 = test_streaming_write(out_dir, n_frames=1000, flush_interval=0)

    # Clean up for next test
    for f in out_dir.glob("*.h5"):
        f.unlink()
    drop_caches()
    time.sleep(2)

    # Test 2: With periodic sync
    r2 = test_write_with_periodic_sync(out_dir, n_frames=1000, sync_interval=100)

    # Clean up for next test
    for f in out_dir.glob("*.h5"):
        f.unlink()
    drop_caches()
    time.sleep(2)

    # Test 3: With posix_fadvise
    r3 = test_write_with_posix_fadvise(out_dir, n_frames=1000, advise_interval=100)

    # Summary
    print(f"\n{'=' * 60}")
    print("  SUMMARY")
    print(f"{'=' * 60}")
    print(f"\n  {'Test':<35} {'Cache Growth':>14} {'File Size':>10}")
    print(f"  {'-' * 35} {'-' * 14} {'-' * 10}")
    print(
        f"  {'No flush (current behavior)':<35} {r1['cache_growth_mb']:>+12.0f} MB {r1['file_mb']:>8.0f} MB"  # noqa: E501
    )
    print(
        f"  {'Periodic sync (100 frames)':<35} {r2['peak_cache_growth_mb']:>+12.0f} MB {r2['file_mb']:>8.0f} MB"  # noqa: E501
    )
    print(
        f"  {'posix_fadvise(DONTNEED)':<35} {r3['peak_cache_growth_mb']:>+12.0f} MB {r3['file_mb']:>8.0f} MB"  # noqa: E501
    )

    print(f"\n  Raw data written: {r1['raw_gb']:.2f} GB")
    print(f"  Compressed file:  {r1['file_mb']:.0f} MB")
    print(f"  Compression ratio: {r1['raw_gb'] * 1024 / r1['file_mb']:.2f}x")

    if r1["cache_growth_mb"] > 500:
        print("\n  *** CONFIRMED: HDF5 writes accumulate in Linux page cache ***")
        print(f"  The {r1['cache_growth_mb']:.0f} MB of page cache is what VmmemWSL")
        print("  reports as memory usage on the Windows side.")
        print("  This is NOT a memory leak — it's the Linux kernel caching writes.")
        if r2["peak_cache_growth_mb"] < r1["cache_growth_mb"] * 0.5:
            print(
                f"\n  Periodic sync REDUCES cache growth by {(1 - r2['peak_cache_growth_mb'] / r1['cache_growth_mb']) * 100:.0f}%"  # noqa: E501
            )
        if r3["peak_cache_growth_mb"] < r1["cache_growth_mb"] * 0.5:
            print(
                f"  posix_fadvise REDUCES cache growth by {(1 - r3['peak_cache_growth_mb'] / r1['cache_growth_mb']) * 100:.0f}%"  # noqa: E501
            )

    # Clean up
    shutil.rmtree(out_dir, ignore_errors=True)


if __name__ == "__main__":
    main()
