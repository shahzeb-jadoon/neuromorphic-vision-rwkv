#!/usr/bin/env python3
"""Quick script to monitor system memory during world.tick() on a given map.

Tracks BOTH process RSS and system MemAvailable to detect server-side
memory leaks (CARLA server on Windows consuming shared WSL2 memory).

Usage:
    python scripts/diagnose_server_leak.py --town Town03 --ticks 1000
    python scripts/diagnose_server_leak.py --town Town04 --ticks 1000
"""
import argparse
import gc
import time
import carla


def get_rss_mb() -> int:
    try:
        with open("/proc/self/status") as f:
            for line in f:
                if line.startswith("VmRSS:"):
                    return int(line.split()[1]) // 1024
    except Exception:
        return 0
    return 0


def get_mem_avail_mb() -> int:
    try:
        with open("/proc/meminfo") as f:
            for line in f:
                if line.startswith("MemAvailable:"):
                    return int(line.split()[1]) // 1024
    except Exception:
        return -1
    return -1


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="172.20.0.1")
    parser.add_argument("--port", type=int, default=2000)
    parser.add_argument("--town", default="Town03")
    parser.add_argument("--ticks", type=int, default=1000)
    args = parser.parse_args()

    print("Server-side memory leak test")
    print(f"Town: {args.town}, Ticks: {args.ticks}")
    print(f"{'='*70}")

    client = carla.Client(args.host, args.port)
    client.set_timeout(60.0)
    world = client.get_world()

    # Clean up any orphaned actors
    actors = world.get_actors()
    n_destroyed = 0
    for s in actors.filter("sensor.*"):
        try:
            s.stop()
            s.destroy()
            n_destroyed += 1
        except Exception:
            pass
    for v in actors.filter("vehicle.*"):
        try:
            v.set_autopilot(False)
            v.destroy()
            n_destroyed += 1
        except Exception:
            pass
    for c in actors.filter("controller.*"):
        try:
            c.stop()
            c.destroy()
            n_destroyed += 1
        except Exception:
            pass
    for w in actors.filter("walker.*"):
        try:
            w.destroy()
            n_destroyed += 1
        except Exception:
            pass
    if n_destroyed:
        print(f"  Cleaned up {n_destroyed} orphaned actors")
        time.sleep(2)

    # Load map if needed
    if args.town not in world.get_map().name:
        print(f"  Loading {args.town}...")
        client.load_world(args.town)
        time.sleep(5)
        world = client.get_world()

    # Sync mode
    settings = world.get_settings()
    settings.synchronous_mode = True
    settings.fixed_delta_seconds = 0.05
    world.apply_settings(settings)

    # Spawn ego
    bp = world.get_blueprint_library().find("vehicle.tesla.model3")
    sp = world.get_map().get_spawn_points()
    vehicle = world.spawn_actor(bp, sp[0])

    # Warm up
    for _ in range(10):
        world.tick()
    gc.collect()

    rss0 = get_rss_mb()
    avail0 = get_mem_avail_mb()
    t0 = time.time()
    print(f"  Baseline: RSS={rss0} MB, MemAvail={avail0} MB")
    print(f"  {'tick':>6} {'RSS':>6} {'MemAvail':>9} {'dRSS':>6} {'dAvail':>7} {'elapsed':>8}")
    print(f"  {'-'*50}")

    for i in range(args.ticks):
        world.tick()

        if i % 20 == 0:
            rss = get_rss_mb()
            avail = get_mem_avail_mb()
            elapsed = time.time() - t0
            print(
                f"  {i:6d} {rss:5d}MB {avail:8d}MB {rss-rss0:+5d}MB {avail-avail0:+6d}MB {elapsed:7.1f}s"  # noqa: E501
            )

            if avail < 1500:
                print(f"  *** SAFETY ABORT: MemAvail={avail} MB < 1500 MB ***")
                break

    rss_end = get_rss_mb()
    avail_end = get_mem_avail_mb()
    elapsed = time.time() - t0
    print(
        f"\n  Final:  RSS={rss_end} MB (+{rss_end-rss0} MB), MemAvail={avail_end} MB ({avail_end-avail0:+d} MB)"  # noqa: E501
    )
    print(f"  Elapsed: {elapsed:.1f}s")

    rss_rate = (rss_end - rss0) / elapsed if elapsed > 0 else 0
    avail_rate = (avail0 - avail_end) / elapsed if elapsed > 0 else 0
    print(f"  RSS growth rate:     {rss_rate:.1f} MB/s")
    print(f"  MemAvail drain rate: {avail_rate:.1f} MB/s")

    if avail_rate > 5.0:
        print(f"\n  *** SERVER-SIDE LEAK: system memory draining at {avail_rate:.0f} MB/s")
        print(f"      while Python RSS is {'stable' if rss_rate < 1.0 else 'growing'}")
        print("      This means the CARLA SERVER (Windows) is leaking memory,")
        print("      which reduces WSL2's dynamic allocation and triggers OOM. ***")
    elif rss_rate > 1.0:
        print(f"\n  *** CLIENT-SIDE LEAK: Python RSS growing at {rss_rate:.1f} MB/s ***")
    else:
        print("\n  No significant leak detected.")

    # Cleanup
    vehicle.destroy()
    settings.synchronous_mode = False
    settings.fixed_delta_seconds = None
    world.apply_settings(settings)


if __name__ == "__main__":
    main()
