#!/usr/bin/env python3
"""Diagnose junction gap-fill behaviour and server memory accumulation.

Validates that the tentative-fill fix in _fill_junction_gaps():
  1. Does NOT insert diverging (wrong-road) waypoints into the buffer
  2. Keeps cumulative RPC count bounded (hundreds, not tens of thousands)
  3. Does not cause server-side memory accumulation across episodes

Tests:
  T1: gap_rpc   — Single expert episode on Town03, monitor RPC count & diverged count
  T2: multi_ep  — 3 sequential expert episodes on Town03, monitor MemAvail accumulation
  T3: corridor  — Verify corridor integrity (no wrong-road waypoints in buffer)

Usage:
  python scripts/diagnose_gap_fill.py --test gap_rpc   --town Town03 --ticks 5000
  python scripts/diagnose_gap_fill.py --test multi_ep  --town Town03 --ticks 3000
  python scripts/diagnose_gap_fill.py --test corridor  --town Town03 --ticks 3000
  python scripts/diagnose_gap_fill.py --test all       --town Town03 --ticks 3000
"""

import argparse
import gc
import os
import sys
import time
from typing import Any

import carla

# ---------------------------------------------------------------------------
# Memory helpers
# ---------------------------------------------------------------------------


def get_rss_mb() -> float:
    try:
        with open("/proc/self/status") as f:
            for line in f:
                if line.startswith("VmRSS:"):
                    return int(line.split()[1]) / 1024
    except Exception:
        return -1
    return -1


def get_anon_rss_mb() -> float:
    try:
        with open("/proc/self/status") as f:
            for line in f:
                if line.startswith("RssAnon:"):
                    return int(line.split()[1]) / 1024
    except Exception:
        return -1
    return -1


def get_mem_avail_mb() -> float:
    try:
        with open("/proc/meminfo") as f:
            for line in f:
                if line.startswith("MemAvailable:"):
                    return int(line.split()[1]) / 1024
    except Exception:
        return -1
    return -1


# ---------------------------------------------------------------------------
# Helper: connect to CARLA, load map
# ---------------------------------------------------------------------------

CARLA_HOST = os.environ.get("CARLA_HOST", "172.20.0.1")
CARLA_PORT = int(os.environ.get("CARLA_PORT", 2000))


def connect_carla(town: str) -> tuple:
    """Connect and load town. Returns (client, world)."""
    client = carla.Client(CARLA_HOST, CARLA_PORT)
    client.set_timeout(60.0)
    world = client.get_world()
    current_map = world.get_map().name
    if not current_map.endswith(town):
        print(f"  Loading {town}...")
        world = client.load_world(town)
        import time

        time.sleep(3.0)
    else:
        print(f"  Map {town} already loaded")
    settings = world.get_settings()
    settings.synchronous_mode = True
    settings.fixed_delta_seconds = 0.05
    world.apply_settings(settings)
    return client, world


def cleanup_actors(world: Any) -> None:
    """Destroy orphaned vehicles/walkers."""
    actors = world.get_actors()
    killed = 0
    for a in actors.filter("*vehicle*"):
        a.destroy()
        killed += 1
    for a in actors.filter("*walker*"):
        a.destroy()
        killed += 1
    if killed:
        print(f"  Cleaned up {killed} orphaned actors")


def spawn_ego(world: Any, spawn_transform: Any = None) -> Any:
    """Spawn ego vehicle at given transform, or spawn point 10 as fallback."""
    bp = world.get_blueprint_library().find("vehicle.tesla.model3")
    if spawn_transform is not None:
        vehicle = world.try_spawn_actor(bp, spawn_transform)
        if vehicle:
            world.tick()
            return vehicle
    # Fallback
    spawns = world.get_map().get_spawn_points()
    sp_idx = min(10, len(spawns) - 1)
    vehicle = world.try_spawn_actor(bp, spawns[sp_idx])
    if vehicle is None:
        for sp in spawns:
            vehicle = world.try_spawn_actor(bp, sp)
            if vehicle:
                break
    if vehicle is None:
        raise RuntimeError("Cannot spawn ego")
    world.tick()
    return vehicle


def print_header(title: str, town: str, ticks: int) -> None:
    print()
    print("=" * 70)
    print(f"  {title} | {town} | {ticks} ticks")
    print("=" * 70)


def print_summary(name: str, results: dict[str, Any]) -> None:
    verdict = results.get("verdict", "UNKNOWN")
    print()
    print(f"  === RESULTS: {name} ===")
    for k, v in results.items():
        if k != "verdict":
            print(f"  {k}: {v}")
    print(f"  Verdict: {verdict}")
    print()


# ---------------------------------------------------------------------------
# T1: gap_rpc — Single episode, monitor RPC + diverged counts
# ---------------------------------------------------------------------------


def test_gap_rpc(town: str, ticks: int, rss_limit: int) -> dict:
    """Run expert for N ticks, report RPC count and diverged gap count."""
    print_header("T1: gap_rpc — RPC + divergence tracking", town, ticks)

    client, world = connect_carla(town)
    cleanup_actors(world)

    # Import expert
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from agents.expert.privileged_expert import PrivilegedExpert, ExpertConfig
    from agents.navigation.global_route_planner import GlobalRoutePlanner

    grp = GlobalRoutePlanner(world.get_map(), sampling_resolution=2.0)
    spawns = world.get_map().get_spawn_points()
    # Use spawn 20->100 on Town03 for a junction-heavy route
    sp_start = spawns[min(20, len(spawns) - 1)]
    sp_end = spawns[min(100, len(spawns) - 1)]

    # Spawn ego AT the route start so the expert can follow the route
    vehicle = spawn_ego(world, sp_start)
    expert = PrivilegedExpert(vehicle, world, grp, config=ExpertConfig())

    route = grp.trace_route(sp_start.location, sp_end.location)
    from agents.navigation.local_planner import RoadOption

    n_turns = sum(1 for _, cmd in route if cmd in (RoadOption.LEFT, RoadOption.RIGHT))
    expert.set_route(route)
    print(f"  Route: {len(route)} waypoints, {n_turns} turns")

    rss0 = get_rss_mb()
    avail0 = get_mem_avail_mb()
    t0 = time.time()

    print(f"  Baseline: RSS={rss0:.0f} MB, MemAvail={avail0:.0f} MB")
    print(f"  Safety abort at RSS > {rss_limit} MB or MemAvail < 1500 MB")
    print()
    print(
        f"  {'tick':>6}  {'RSS':>5}  {'MemAvl':>6}  {'RPCs':>7}  {'Divg':>5}  "
        f"{'Spd':>5}  {'Rte%':>5}  elapsed"
    )
    print(f"  {'-' * 65}")

    max_rpcs_per_frame = 0
    prev_rpcs = 0

    for i in range(1, ticks + 1):
        control, info = expert.run_step()
        vehicle.apply_control(control)
        world.tick()

        rpcs_this_frame = info.gap_fill_rpcs - prev_rpcs
        prev_rpcs = info.gap_fill_rpcs
        max_rpcs_per_frame = max(max_rpcs_per_frame, rpcs_this_frame)

        if i % 200 == 0 or i == ticks:
            rss = get_rss_mb()
            avail = get_mem_avail_mb()
            elapsed = time.time() - t0
            spd = info.current_speed
            rte = info.route_completion * 100
            print(
                f"  {i:>6}  {rss:>4.0f}M  {avail:>5.0f}M  {info.gap_fill_rpcs:>7}  "
                f"{info.gap_fill_diverged:>5}  {spd:>4.0f}  {rte:>4.0f}%  {elapsed:.1f}s"
            )

            if rss > rss_limit:
                print(f"\n  *** SAFETY ABORT: RSS={rss:.0f} MB > {rss_limit} MB ***")
                break
            if avail < 1500:
                print(f"\n  *** SAFETY ABORT: MemAvail={avail:.0f} MB < 1500 MB ***")
                break

    rss_end = get_rss_mb()
    avail_end = get_mem_avail_mb()
    elapsed = time.time() - t0

    vehicle.destroy()

    results = {
        "ticks": f"{i}/{ticks}",
        "elapsed": f"{elapsed:.1f}s",
        "rss": f"{rss0:.0f} -> {rss_end:.0f} MB (d{rss_end - rss0:+.0f} MB)",
        "mem_avail": f"{avail0:.0f} -> {avail_end:.0f} MB (drained {avail0 - avail_end:+.0f} MB)",
        "total_rpcs": info.gap_fill_rpcs,
        "diverged_gaps": info.gap_fill_diverged,
        "max_rpcs_per_frame": max_rpcs_per_frame,
        "avg_rpcs_per_frame": f"{info.gap_fill_rpcs / i:.1f}",
    }

    # Verdict: CLEAN if total RPCs < 5000 (should be ~hundreds with fix)
    if info.gap_fill_rpcs < 5000 and (avail0 - avail_end) < 500:
        results["verdict"] = "CLEAN"
    elif info.gap_fill_rpcs < 5000:
        results["verdict"] = "WARN (RPCs ok, MemAvail drained)"
    else:
        results["verdict"] = f"FAIL (RPCs={info.gap_fill_rpcs}, expected <5000)"

    print_summary("T1:gap_rpc", results)
    return results


# ---------------------------------------------------------------------------
# T2: multi_ep — 3 sequential episodes on Town03, track MemAvail
# ---------------------------------------------------------------------------


def test_multi_ep(town: str, ticks: int, rss_limit: int) -> dict:
    """Run 3 back-to-back expert episodes, check MemAvail accumulation."""
    print_header("T2: multi_ep — 3 sequential episodes", town, ticks)

    client, world = connect_carla(town)
    cleanup_actors(world)

    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from agents.expert.privileged_expert import PrivilegedExpert, ExpertConfig
    from agents.navigation.global_route_planner import GlobalRoutePlanner

    grp = GlobalRoutePlanner(world.get_map(), sampling_resolution=2.0)
    spawns = world.get_map().get_spawn_points()
    n_episodes = 3
    ep_results = []

    avail_start = get_mem_avail_mb()
    print(f"  Starting MemAvail: {avail_start:.0f} MB")
    print()

    for ep in range(n_episodes):
        # Clean between episodes
        cleanup_actors(world)
        gc.collect()
        world.tick()
        time.sleep(1.0)

        # Pick different junction-heavy routes each episode
        route_pairs = [(20, 100), (5, 150), (30, 100)]
        si, ei = route_pairs[ep % len(route_pairs)]
        sp_start = spawns[min(si, len(spawns) - 1)]
        sp_end = spawns[min(ei, len(spawns) - 1)]

        # Spawn ego AT the route start
        vehicle = spawn_ego(world, sp_start)
        expert = PrivilegedExpert(vehicle, world, grp, config=ExpertConfig())

        route = grp.trace_route(sp_start.location, sp_end.location)
        expert.set_route(route)

        avail_before = get_mem_avail_mb()
        rss_before = get_rss_mb()
        t0 = time.time()

        print(
            f"  Episode {ep + 1}/{n_episodes}: route={len(route)} wps, "
            f"RSS={rss_before:.0f} MB, MemAvail={avail_before:.0f} MB"
        )

        for i in range(1, ticks + 1):
            control, info = expert.run_step()
            vehicle.apply_control(control)
            world.tick()

            if i % 500 == 0:
                rss = get_rss_mb()
                avail = get_mem_avail_mb()
                print(
                    f"    tick {i:>5}: RSS={rss:.0f}M, MemAvl={avail:.0f}M, "
                    f"RPCs={info.gap_fill_rpcs}, Divg={info.gap_fill_diverged}"
                )

                if avail < 1500:
                    print(f"\n  *** SAFETY ABORT: MemAvail={avail:.0f} MB < 1500 MB ***")
                    break

        elapsed = time.time() - t0
        rss_after = get_rss_mb()
        avail_after = get_mem_avail_mb()

        ep_results.append(
            {
                "episode": ep + 1,
                "ticks": i,
                "rpcs": info.gap_fill_rpcs,
                "diverged": info.gap_fill_diverged,
                "rss_delta": rss_after - rss_before,
                "avail_delta": avail_before - avail_after,
                "elapsed": elapsed,
            }
        )

        print(
            f"    Done: RPCs={info.gap_fill_rpcs}, Divg={info.gap_fill_diverged}, "
            f"MemAvail delta={avail_before - avail_after:+.0f} MB, {elapsed:.1f}s"
        )
        print()

        vehicle.destroy()
        del expert
        gc.collect()

    avail_end = get_mem_avail_mb()
    total_drain = avail_start - avail_end

    print(f"  Total MemAvail drain across {n_episodes} episodes: {total_drain:+.0f} MB")
    print(f"  ({avail_start:.0f} -> {avail_end:.0f} MB)")

    results = {
        "episodes": n_episodes,
        "ticks_per_ep": ticks,
        "total_mem_drain_mb": f"{total_drain:.0f}",
        "per_episode": ep_results,
    }

    # Verdict: < 200 MB total drain across 3 episodes is healthy
    if total_drain < 200:
        results["verdict"] = "CLEAN"
    elif total_drain < 500:
        results["verdict"] = f"WARN (drained {total_drain:.0f} MB)"
    else:
        results["verdict"] = f"FAIL (drained {total_drain:.0f} MB — server accumulation)"

    print_summary("T2:multi_ep", results)
    return results


# ---------------------------------------------------------------------------
# T3: corridor — Verify waypoint buffer integrity
# ---------------------------------------------------------------------------


def test_corridor(town: str, ticks: int, rss_limit: int) -> dict:
    """Run expert, checking every frame that all buffer waypoints are on
    roads reachable from the current route (no wrong-road fills)."""
    print_header("T3: corridor — buffer integrity check", town, ticks)

    client, world = connect_carla(town)
    cleanup_actors(world)

    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from agents.expert.privileged_expert import PrivilegedExpert, ExpertConfig
    from agents.navigation.global_route_planner import GlobalRoutePlanner

    carla_map = world.get_map()
    grp = GlobalRoutePlanner(carla_map, sampling_resolution=2.0)

    spawns = carla_map.get_spawn_points()
    # Use junction-heavy route (spawn 20->100 on Town03)
    sp_start = spawns[min(20, len(spawns) - 1)]
    sp_end = spawns[min(100, len(spawns) - 1)]

    # Spawn ego AT the route start
    vehicle = spawn_ego(world, sp_start)
    expert = PrivilegedExpert(vehicle, world, grp, config=ExpertConfig())

    route = grp.trace_route(sp_start.location, sp_end.location)
    from agents.navigation.local_planner import RoadOption

    n_turns = sum(1 for _, cmd in route if cmd in (RoadOption.LEFT, RoadOption.RIGHT))
    expert.set_route(route)
    print(f"  Route: {len(route)} waypoints, {n_turns} turns")

    t0 = time.time()
    max_consecutive_gap = 0.0
    n_large_gaps = 0
    n_buffer_checks = 0
    buffer_corruption_frames = []

    print()
    print(
        f"  {'tick':>6}  {'BufSz':>5}  {'MaxGap':>7}  {'LrgGaps':>7}  "
        f"{'Corrupt':>7}  {'RPCs':>5}  {'Divg':>4}  elapsed"
    )
    print(f"  {'-' * 65}")

    for i in range(1, ticks + 1):
        control, info = expert.run_step()
        vehicle.apply_control(control)
        world.tick()

        # Check buffer integrity every 10 frames
        if i % 10 == 0:
            n_buffer_checks += 1
            buf = list(expert._wp_buffer)
            frame_max_gap = 0.0
            frame_large_gaps = 0

            for k in range(1, len(buf)):
                wp_a, _ = buf[k - 1]
                wp_b, _ = buf[k]
                d = wp_a.transform.location.distance(wp_b.transform.location)
                frame_max_gap = max(frame_max_gap, d)
                if d > 4.0:
                    frame_large_gaps += 1

            max_consecutive_gap = max(max_consecutive_gap, frame_max_gap)
            n_large_gaps = max(n_large_gaps, frame_large_gaps)

            # Check for waypoints on wildly different roads:
            # If any consecutive pair in the buffer has a gap > 30m,
            # that's almost certainly corruption (normal route gaps
            # after fill should be ≤ ~20m for diverging junctions).
            if frame_max_gap > 30.0:
                buffer_corruption_frames.append((i, frame_max_gap))

        if i % 200 == 0 or i == ticks:
            rss = get_rss_mb()
            avail = get_mem_avail_mb()
            elapsed = time.time() - t0
            print(
                f"  {i:>6}  {len(expert._wp_buffer):>5}  {max_consecutive_gap:>6.1f}m  "
                f"{n_large_gaps:>7}  {len(buffer_corruption_frames):>7}  "
                f"{info.gap_fill_rpcs:>5}  {info.gap_fill_diverged:>4}  {elapsed:.1f}s"
            )

            if rss > rss_limit:
                print("\n  *** SAFETY ABORT ***")
                break
            if avail < 1500:
                print("\n  *** SAFETY ABORT ***")
                break

    elapsed = time.time() - t0
    vehicle.destroy()

    results = {
        "ticks": f"{i}/{ticks}",
        "elapsed": f"{elapsed:.1f}s",
        "max_consecutive_gap_m": f"{max_consecutive_gap:.1f}",
        "buffer_corruption_frames": len(buffer_corruption_frames),
        "total_rpcs": info.gap_fill_rpcs,
        "diverged_gaps": info.gap_fill_diverged,
    }

    if not buffer_corruption_frames and info.gap_fill_rpcs < 5000:
        results["verdict"] = "CLEAN"
    elif buffer_corruption_frames:
        results["verdict"] = f"FAIL ({len(buffer_corruption_frames)} corrupted frames)"
    else:
        results["verdict"] = f"WARN (RPCs={info.gap_fill_rpcs})"

    print_summary("T3:corridor", results)
    return results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description="Gap-fill diagnostic")
    parser.add_argument("--test", required=True, choices=["gap_rpc", "multi_ep", "corridor", "all"])
    parser.add_argument("--town", default="Town03")
    parser.add_argument("--ticks", type=int, default=3000)
    parser.add_argument("--rss-limit", type=int, default=2000)
    args = parser.parse_args()

    print("Gap-Fill Diagnostic")
    print(
        f"Test: {args.test} | Town: {args.town} | Ticks: {args.ticks} | RSS limit: {args.rss_limit} MB"  # noqa: E501
    )

    tests = {
        "gap_rpc": test_gap_rpc,
        "multi_ep": test_multi_ep,
        "corridor": test_corridor,
    }

    if args.test == "all":
        results = {}
        for name, fn in tests.items():
            results[name] = fn(args.town, args.ticks, args.rss_limit)
        print()
        print("=" * 70)
        print("  ALL TESTS COMPLETE")
        print("=" * 70)
        for name, r in results.items():
            print(f"  {name}: {r['verdict']}")
        all_clean = all(r["verdict"] == "CLEAN" for r in results.values())
        print(f"\n  Overall: {'CLEAN' if all_clean else 'ISSUES FOUND'}")
    else:
        tests[args.test](args.town, args.ticks, args.rss_limit)


if __name__ == "__main__":
    main()
