#!/usr/bin/env python3
"""Incremental memory leak diagnostic for CARLA world.tick().

Introduces ONE variable at a time to isolate what causes
VmmemWSL to balloon to 20 GB and trigger OOM on Town03.

Each test variant runs the same tick loop with RSS + MemAvail
monitoring. Safety aborts if RSS exceeds --rss-limit.

Tested so far:
  T1 tick-only Town03 5000 ticks -> CLEAN (RSS=97MB stable, MemAvail~18000MB, 44s)

Usage:
    python scripts/diagnose_tick_leak.py --test tick --town Town03 --ticks 5000
    python scripts/diagnose_tick_leak.py --test cam1 --town Town03 --ticks 5000
    python scripts/diagnose_tick_leak.py --test cam3 --town Town03 --ticks 5000
    python scripts/diagnose_tick_leak.py --test npcs --town Town03 --ticks 5000
    python scripts/diagnose_tick_leak.py --test get_actors --town Town03 --ticks 5000
    python scripts/diagnose_tick_leak.py --test get_waypoints --town Town03 --ticks 5000
    python scripts/diagnose_tick_leak.py --test full --town Town03 --ticks 5000
"""

import argparse
import gc
import math
import os
import time
from typing import Any
import carla

# ---------- Memory helpers ----------


def get_memory_info() -> dict[str, Any]:
    """Read VmRSS, VmSize, VmData, RssAnon from /proc/self/status."""
    info = {}
    try:
        with open("/proc/self/status") as f:
            for line in f:
                if line.startswith("VmRSS:"):
                    info["rss_mb"] = int(line.split()[1]) / 1024
                elif line.startswith("VmSize:"):
                    info["vsize_mb"] = int(line.split()[1]) / 1024
                elif line.startswith("VmData:"):
                    info["data_mb"] = int(line.split()[1]) / 1024
                elif line.startswith("RssAnon:"):
                    info["anon_mb"] = int(line.split()[1]) / 1024
    except Exception:
        pass
    return info


def get_mem_avail_mb() -> float:
    try:
        with open("/proc/meminfo") as f:
            for line in f:
                if line.startswith("MemAvailable:"):
                    return int(line.split()[1]) / 1024
    except Exception:
        return -1
    return -1


# ---------- Actor cleanup ----------


def cleanup_actors(world: Any) -> int:
    """Destroy ALL actors from previous crashed runs."""
    actors = world.get_actors()
    n = 0
    for s in actors.filter("sensor.*"):
        try:
            s.stop()
            s.destroy()
            n += 1
        except Exception:
            pass
    for v in actors.filter("vehicle.*"):
        try:
            v.set_autopilot(False)
            v.destroy()
            n += 1
        except Exception:
            pass
    for c in actors.filter("controller.*"):
        try:
            c.stop()
            c.destroy()
            n += 1
        except Exception:
            pass
    for w in actors.filter("walker.*"):
        try:
            w.destroy()
            n += 1
        except Exception:
            pass
    return n


# ---------- Core tick loop ----------


def run_tick_loop(
    world: Any, ticks: int, rss_limit: int, per_tick_fn: Any = None, label: str = ""
) -> dict[str, Any]:
    """Core tick loop with memory monitoring.

    per_tick_fn: optional callable(world, i) invoked AFTER each world.tick().
                 This is where we add the incremental variable being tested.

    Returns dict with results.
    """
    mem0 = get_memory_info()
    avail0 = get_mem_avail_mb()
    t0 = time.time()

    print(
        f"  Baseline: RSS={mem0.get('rss_mb',0):.0f} MB, "
        f"AnonRSS={mem0.get('anon_mb',0):.0f} MB, "
        f"MemAvail={avail0:.0f} MB"
    )
    print(f"  Safety abort at RSS > {rss_limit} MB or MemAvail < 1500 MB")
    print()
    print(
        f"  {'tick':>6} {'RSS':>7} {'AnonRSS':>8} {'MemAvl':>7} "
        f"{'dRSS':>7} {'dAvail':>8} {'RSS/s':>7} {'Avl/s':>7} {'elapsed':>8}"
    )
    print(f"  {'-'*80}")

    last_print = 0
    aborted = False
    abort_reason = ""
    tick_done = 0

    for i in range(1, ticks + 1):
        world.tick()
        tick_done = i

        # Run the per-tick function (the variable under test)
        if per_tick_fn is not None:
            per_tick_fn(world, i)

        # Check memory every 5 ticks
        if i % 5 == 0:
            mem = get_memory_info()
            rss = mem.get("rss_mb", 0)
            avail = get_mem_avail_mb()

            # Safety checks
            if rss > rss_limit:
                elapsed = time.time() - t0
                drss = rss - mem0.get("rss_mb", 0)
                davail = avail0 - avail
                rate_r = drss / elapsed if elapsed > 0 else 0
                rate_a = davail / elapsed if elapsed > 0 else 0
                print(
                    f"  {i:6d} {rss:6.0f}M {mem.get('anon_mb',0):7.0f}M "
                    f"{avail:6.0f}M {drss:+6.0f}M {-davail:+7.0f}M "
                    f"{rate_r:+6.1f} {rate_a:+6.1f} {elapsed:7.1f}s"
                )
                print(f"\n  *** SAFETY ABORT: RSS={rss:.0f} MB > {rss_limit} MB ***")
                aborted = True
                abort_reason = f"RSS={rss:.0f}>{rss_limit}"
                break

            if avail < 1500:
                elapsed = time.time() - t0
                drss = rss - mem0.get("rss_mb", 0)
                davail = avail0 - avail
                rate_r = drss / elapsed if elapsed > 0 else 0
                rate_a = davail / elapsed if elapsed > 0 else 0
                print(
                    f"  {i:6d} {rss:6.0f}M {mem.get('anon_mb',0):7.0f}M "
                    f"{avail:6.0f}M {drss:+6.0f}M {-davail:+7.0f}M "
                    f"{rate_r:+6.1f} {rate_a:+6.1f} {elapsed:7.1f}s"
                )
                print(f"\n  *** SAFETY ABORT: MemAvail={avail:.0f} MB < 1500 MB ***")
                aborted = True
                abort_reason = f"MemAvail={avail:.0f}<1500"
                break

            # Print progress -- adaptive interval
            drss = rss - mem0.get("rss_mb", 0)
            davail = avail0 - avail
            interval = 50 if abs(drss) > 10 or abs(davail) > 200 else 100
            if i - last_print >= interval:
                elapsed = time.time() - t0
                rate_r = drss / elapsed if elapsed > 0 else 0
                rate_a = davail / elapsed if elapsed > 0 else 0
                print(
                    f"  {i:6d} {rss:6.0f}M {mem.get('anon_mb',0):7.0f}M "
                    f"{avail:6.0f}M {drss:+6.0f}M {-davail:+7.0f}M "
                    f"{rate_r:+6.1f} {rate_a:+6.1f} {elapsed:7.1f}s"
                )
                last_print = i

    # -- Final measurement --
    gc.collect()
    time.sleep(0.5)
    mem_end = get_memory_info()
    avail_end = get_mem_avail_mb()
    elapsed = time.time() - t0

    rss_end = mem_end.get("rss_mb", 0)
    drss = rss_end - mem0.get("rss_mb", 0)
    davail = avail0 - avail_end
    rss_rate = drss / elapsed if elapsed > 0 else 0
    avail_rate = davail / elapsed if elapsed > 0 else 0

    print(f"\n  === RESULTS: {label} ===")
    print(f"  Ticks:    {tick_done}/{ticks} {'(ABORTED: '+abort_reason+')' if aborted else ''}")
    print(f"  Elapsed:  {elapsed:.1f}s")
    print(
        f"  RSS:      {mem0.get('rss_mb',0):.0f} -> {rss_end:.0f} MB  "
        f"(d{drss:+.0f} MB, {rss_rate:+.1f} MB/s)"
    )
    print(
        f"  MemAvail: {avail0:.0f} -> {avail_end:.0f} MB  "
        f"(drained {davail:+.0f} MB, {avail_rate:+.1f} MB/s)"
    )

    # Verdict
    if abs(rss_rate) < 1.0 and abs(avail_rate) < 5.0:
        verdict = "CLEAN"
    elif avail_rate > 5.0 and rss_rate < 1.0:
        verdict = f"SERVER-SIDE LEAK (MemAvail draining {avail_rate:.0f} MB/s, RSS stable)"
    elif rss_rate > 1.0:
        verdict = f"CLIENT LEAK (RSS growing {rss_rate:.1f} MB/s)"
    else:
        verdict = "MARGINAL"
    print(f"  Verdict:  {verdict}")

    return {
        "label": label,
        "ticks": tick_done,
        "elapsed": elapsed,
        "drss": drss,
        "rss_rate": rss_rate,
        "davail": davail,
        "avail_rate": avail_rate,
        "aborted": aborted,
        "verdict": verdict,
    }


# ---------- Setup / teardown ----------


def setup(client: Any, world: Any, town: str) -> tuple[Any, Any, Any]:
    """Load map, sync mode, spawn ego. Returns (world, vehicle, settings)."""
    current_map = world.get_map().name
    if town not in current_map:
        print(f"  Loading {town}...")
        client.load_world(town)
        time.sleep(5)
        world = client.get_world()

    settings = world.get_settings()
    settings.synchronous_mode = True
    settings.fixed_delta_seconds = 0.05
    world.apply_settings(settings)

    bp = world.get_blueprint_library().find("vehicle.tesla.model3")
    sp = world.get_map().get_spawn_points()
    vehicle = world.spawn_actor(bp, sp[0])

    # Warm up
    for _ in range(20):
        world.tick()
    gc.collect()
    time.sleep(1)

    return world, vehicle, settings


def teardown(
    world: Any, vehicle: Any, settings: Any, extra_actors: Any = None, tm: Any = None
) -> None:
    """Cleanup everything."""
    if extra_actors:
        for a in extra_actors:
            try:
                if hasattr(a, "stop"):
                    a.stop()
                a.destroy()
            except Exception:
                pass
    if tm is not None:
        try:
            tm.set_synchronous_mode(False)
        except Exception:
            pass
    vehicle.destroy()
    settings.synchronous_mode = False
    settings.fixed_delta_seconds = None
    world.apply_settings(settings)
    gc.collect()


# ---------- Test variants ----------


def test_tick(client: Any, world: Any, town: str, ticks: int, rss_limit: int) -> dict[str, Any]:
    """T1: Bare world.tick(), no sensors, no NPCs."""
    print(f"\n{'='*70}")
    print(f"T1: tick-only | {town} | {ticks} ticks")
    print(f"{'='*70}")
    world, vehicle, settings = setup(client, world, town)
    r = run_tick_loop(world, ticks, rss_limit, per_tick_fn=None, label=f"T1:tick-only/{town}")
    teardown(world, vehicle, settings)
    return r


def test_cam1(client: Any, world: Any, town: str, ticks: int, rss_limit: int) -> dict[str, Any]:
    """T2: world.tick() + 1 RGB camera listening (discard data)."""
    print(f"\n{'='*70}")
    print(f"T2: tick + 1 camera | {town} | {ticks} ticks")
    print(f"{'='*70}")
    world, vehicle, settings = setup(client, world, town)

    cam_bp = world.get_blueprint_library().find("sensor.camera.rgb")
    cam_bp.set_attribute("image_size_x", "640")
    cam_bp.set_attribute("image_size_y", "480")
    cam_bp.set_attribute("fov", "90")
    cam = world.spawn_actor(
        cam_bp, carla.Transform(carla.Location(x=1.5, z=2.4)), attach_to=vehicle
    )
    frame_count = [0]
    cam.listen(lambda img: frame_count.__setitem__(0, frame_count[0] + 1))

    # Extra warm up with camera
    for _ in range(20):
        world.tick()
    gc.collect()
    time.sleep(1)
    frame_count[0] = 0

    r = run_tick_loop(world, ticks, rss_limit, per_tick_fn=None, label=f"T2:1cam/{town}")
    print(f"  Frames received: {frame_count[0]}")
    teardown(world, vehicle, settings, extra_actors=[cam])
    return r


def test_cam3(client: Any, world: Any, town: str, ticks: int, rss_limit: int) -> dict[str, Any]:
    """T3: world.tick() + 3 RGB cameras listening (discard data)."""
    print(f"\n{'='*70}")
    print(f"T3: tick + 3 cameras | {town} | {ticks} ticks")
    print(f"{'='*70}")
    world, vehicle, settings = setup(client, world, town)

    cam_bp = world.get_blueprint_library().find("sensor.camera.rgb")
    cam_bp.set_attribute("image_size_x", "640")
    cam_bp.set_attribute("image_size_y", "480")
    cam_bp.set_attribute("fov", "90")

    cams = []
    frame_count = [0]
    transforms = [
        carla.Transform(carla.Location(x=1.5, z=2.4)),
        carla.Transform(carla.Location(x=0.0, y=-0.8, z=2.4), carla.Rotation(yaw=-60)),
        carla.Transform(carla.Location(x=0.0, y=0.8, z=2.4), carla.Rotation(yaw=60)),
    ]
    for t in transforms:
        c = world.spawn_actor(cam_bp, t, attach_to=vehicle)
        c.listen(lambda img: frame_count.__setitem__(0, frame_count[0] + 1))
        cams.append(c)

    for _ in range(20):
        world.tick()
    gc.collect()
    time.sleep(1)
    frame_count[0] = 0

    r = run_tick_loop(world, ticks, rss_limit, per_tick_fn=None, label=f"T3:3cams/{town}")
    print(f"  Frames received: {frame_count[0]}")
    teardown(world, vehicle, settings, extra_actors=cams)
    return r


def test_npcs(client: Any, world: Any, town: str, ticks: int, rss_limit: int) -> dict[str, Any]:
    """T4: world.tick() + 20 NPC vehicles with TrafficManager."""
    print(f"\n{'='*70}")
    print(f"T4: tick + 20 NPCs + TM | {town} | {ticks} ticks")
    print(f"{'='*70}")
    world, vehicle, settings = setup(client, world, town)

    bp_lib = world.get_blueprint_library()
    spawn_points = world.get_map().get_spawn_points()
    npcs = []
    veh_bps = bp_lib.filter("vehicle.*")
    for i, sp in enumerate(spawn_points[1:21]):
        try:
            npc = world.spawn_actor(veh_bps[i % len(veh_bps)], sp)
            npcs.append(npc)
        except Exception:
            pass

    tm = client.get_trafficmanager(8000)
    tm.set_synchronous_mode(True)
    for npc in npcs:
        npc.set_autopilot(True, 8000)
    print(f"  Spawned {len(npcs)} NPC vehicles")

    for _ in range(20):
        world.tick()
    gc.collect()
    time.sleep(1)

    r = run_tick_loop(world, ticks, rss_limit, per_tick_fn=None, label=f"T4:npcs/{town}")
    teardown(world, vehicle, settings, extra_actors=npcs, tm=tm)
    return r


def test_get_actors(
    client: Any, world: Any, town: str, ticks: int, rss_limit: int
) -> dict[str, Any]:
    """T5: world.tick() + world.get_actors() x4 per frame.
    Tests if repeated get_actors() causes server-side caching."""
    print(f"\n{'='*70}")
    print(f"T5: tick + get_actors()x4/frame | {town} | {ticks} ticks")
    print(f"{'='*70}")
    world, vehicle, settings = setup(client, world, town)

    def per_tick(w: Any, i: int) -> None:
        w.get_actors().filter("vehicle.*")
        w.get_actors().filter("walker.*")
        w.get_actors().filter("traffic_light.*")
        w.get_actors().filter("traffic.stop")

    for _ in range(20):
        world.tick()
    gc.collect()
    time.sleep(1)

    r = run_tick_loop(world, ticks, rss_limit, per_tick_fn=per_tick, label=f"T5:get_actors/{town}")
    teardown(world, vehicle, settings)
    return r


def test_get_waypoints(
    client: Any, world: Any, town: str, ticks: int, rss_limit: int
) -> dict[str, Any]:
    """T6: world.tick() + map.get_waypoint() per frame.
    Tests if repeated waypoint generation causes server caching."""
    print(f"\n{'='*70}")
    print(f"T6: tick + get_waypoint()x5/frame | {town} | {ticks} ticks")
    print(f"{'='*70}")
    world, vehicle, settings = setup(client, world, town)
    carla_map = world.get_map()

    def per_tick(w: Any, i: int) -> None:
        loc = vehicle.get_location()
        # Generate waypoints ahead -- like the expert does
        wp = carla_map.get_waypoint(loc)
        if wp:
            _ = wp.next(2.0)
            _ = wp.next(5.0)
            _ = wp.next(10.0)
            _ = wp.previous(2.0)

    for _ in range(20):
        world.tick()
    gc.collect()
    time.sleep(1)

    r = run_tick_loop(world, ticks, rss_limit, per_tick_fn=per_tick, label=f"T6:waypoints/{town}")
    teardown(world, vehicle, settings)
    return r


def test_full(client: Any, world: Any, town: str, ticks: int, rss_limit: int) -> dict[str, Any]:
    """T7: Cameras + NPCs + get_actors + get_waypoint.
    Full data-collection-like scenario."""
    print(f"\n{'='*70}")
    print(f"T7: cameras + NPCs + get_actors + waypoints | {town} | {ticks} ticks")
    print(f"{'='*70}")
    world, vehicle, settings = setup(client, world, town)

    # 3 cameras
    cam_bp = world.get_blueprint_library().find("sensor.camera.rgb")
    cam_bp.set_attribute("image_size_x", "640")
    cam_bp.set_attribute("image_size_y", "480")
    cam_bp.set_attribute("fov", "90")
    cams = []
    frame_count = [0]
    transforms = [
        carla.Transform(carla.Location(x=1.5, z=2.4)),
        carla.Transform(carla.Location(x=0.0, y=-0.8, z=2.4), carla.Rotation(yaw=-60)),
        carla.Transform(carla.Location(x=0.0, y=0.8, z=2.4), carla.Rotation(yaw=60)),
    ]
    for t in transforms:
        c = world.spawn_actor(cam_bp, t, attach_to=vehicle)
        c.listen(lambda img: frame_count.__setitem__(0, frame_count[0] + 1))
        cams.append(c)

    # 20 NPCs
    bp_lib = world.get_blueprint_library()
    spawn_points = world.get_map().get_spawn_points()
    npcs = []
    veh_bps = bp_lib.filter("vehicle.*")
    for i, sp in enumerate(spawn_points[1:21]):
        try:
            npc = world.spawn_actor(veh_bps[i % len(veh_bps)], sp)
            npcs.append(npc)
        except Exception:
            pass

    tm = client.get_trafficmanager(8000)
    tm.set_synchronous_mode(True)
    for npc in npcs:
        npc.set_autopilot(True, 8000)
    print(f"  Spawned 3 cameras + {len(npcs)} NPCs")

    carla_map = world.get_map()

    def per_tick(w: Any, i: int) -> None:
        w.get_actors().filter("vehicle.*")
        w.get_actors().filter("walker.*")
        w.get_actors().filter("traffic_light.*")
        w.get_actors().filter("traffic.stop")
        loc = vehicle.get_location()
        wp = carla_map.get_waypoint(loc)
        if wp:
            _ = wp.next(2.0)
            _ = wp.next(5.0)

    for _ in range(20):
        world.tick()
    gc.collect()
    time.sleep(1)
    frame_count[0] = 0

    r = run_tick_loop(world, ticks, rss_limit, per_tick_fn=per_tick, label=f"T7:full/{town}")
    print(f"  Frames received: {frame_count[0]}")
    all_actors = cams + npcs
    teardown(world, vehicle, settings, extra_actors=all_actors, tm=tm)
    return r


def test_expert(client: Any, world: Any, town: str, ticks: int, rss_limit: int) -> dict[str, Any]:
    """T8: world.tick() + PrivilegedExpert.run_step() per frame.
    No cameras, no NPCs. Just the expert agent logic.
    RESULT: OOM killed on Town03 before first 100-tick line."""
    print(f"\n{'='*70}")
    print(f"T8: tick + PrivilegedExpert | {town} | {ticks} ticks")
    print(f"{'='*70}")

    from agents.expert.privileged_expert import PrivilegedExpert, ExpertConfig
    from agents.navigation.global_route_planner import GlobalRoutePlanner

    world, vehicle, settings = setup(client, world, town)
    carla_map = world.get_map()
    spawn_points = carla_map.get_spawn_points()

    route_planner = GlobalRoutePlanner(carla_map, sampling_resolution=2.0)
    expert = PrivilegedExpert(vehicle, world, route_planner, ExpertConfig())

    dest_idx = 10 if len(spawn_points) > 10 else len(spawn_points) - 1
    expert.set_destination(vehicle.get_location(), spawn_points[dest_idx].location)
    print(f"  Expert route set to spawn point {dest_idx}")

    for _ in range(20):
        world.tick()
    gc.collect()
    time.sleep(1)

    def per_tick(w: Any, i: int) -> None:
        if expert.done():
            return
        control, info = expert.run_step(0.05)
        vehicle.apply_control(control)
        del control, info

    r = run_tick_loop(world, ticks, rss_limit, per_tick_fn=per_tick, label=f"T8:expert/{town}")
    teardown(world, vehicle, settings)
    return r


def test_expert_init_only(
    client: Any, world: Any, town: str, ticks: int, rss_limit: int
) -> dict[str, Any]:
    """T8a: Import + init PrivilegedExpert + set_destination, but
    do NOT call run_step(). Tests if just having the expert alive leaks."""
    print(f"\n{'='*70}")
    print(f"T8a: tick + expert init only (no run_step) | {town} | {ticks} ticks")
    print(f"{'='*70}")

    from agents.expert.privileged_expert import PrivilegedExpert, ExpertConfig
    from agents.navigation.global_route_planner import GlobalRoutePlanner

    world, vehicle, settings = setup(client, world, town)
    carla_map = world.get_map()
    spawn_points = carla_map.get_spawn_points()

    route_planner = GlobalRoutePlanner(carla_map, sampling_resolution=2.0)
    expert = PrivilegedExpert(vehicle, world, route_planner, ExpertConfig())

    dest_idx = 10 if len(spawn_points) > 10 else len(spawn_points) - 1
    expert.set_destination(vehicle.get_location(), spawn_points[dest_idx].location)
    print("  Expert initialized, route set, but run_step NOT called")

    for _ in range(20):
        world.tick()
    gc.collect()
    time.sleep(1)

    # Just tick, expert exists but is never called
    r = run_tick_loop(world, ticks, rss_limit, per_tick_fn=None, label=f"T8a:expert_init/{town}")
    teardown(world, vehicle, settings)
    return r


def test_tl_scan(client: Any, world: Any, town: str, ticks: int, rss_limit: int) -> dict[str, Any]:
    """T8b: tick + traffic-light scan per frame.
    Calls get_actors().filter('*traffic_light*') + get_affected_lane_waypoints()
    exactly like _check_traffic_light does."""
    print(f"\n{'='*70}")
    print(
        f"T8b: tick + TL scan (get_actors + get_affected_lane_waypoints) | {town} | {ticks} ticks"
    )
    print(f"{'='*70}")

    world, vehicle, settings = setup(client, world, town)
    carla_map = world.get_map()
    tl_count = [0]
    awp_count = [0]

    def per_tick(w: Any, i: int) -> None:
        ego_loc = vehicle.get_transform().location
        _ = carla_map.get_waypoint(ego_loc)
        for tl in w.get_actors().filter("*traffic_light*"):
            tl_count[0] += 1
            try:
                for awp in tl.get_affected_lane_waypoints():
                    awp_count[0] += 1
            except (AttributeError, RuntimeError):
                pass

    for _ in range(20):
        world.tick()
    gc.collect()
    time.sleep(1)

    r = run_tick_loop(world, ticks, rss_limit, per_tick_fn=per_tick, label=f"T8b:tl_scan/{town}")
    print(f"  TL actors checked: {tl_count[0]}, AWPs iterated: {awp_count[0]}")
    teardown(world, vehicle, settings)
    return r


def test_tl_no_awp(
    client: Any, world: Any, town: str, ticks: int, rss_limit: int
) -> dict[str, Any]:
    """T8c: tick + traffic-light scan WITHOUT get_affected_lane_waypoints.
    Calls get_actors().filter('*traffic_light*') + get_transform() only.
    If T8b leaks but T8c is clean, the leak is in get_affected_lane_waypoints."""
    print(f"\n{'='*70}")
    print(f"T8c: tick + TL scan (get_actors only, NO AWP) | {town} | {ticks} ticks")
    print(f"{'='*70}")

    world, vehicle, settings = setup(client, world, town)
    tl_count = [0]

    def per_tick(w: Any, i: int) -> None:
        for tl in w.get_actors().filter("*traffic_light*"):
            tl_count[0] += 1
            _ = tl.get_transform().location
            _ = tl.state

    for _ in range(20):
        world.tick()
    gc.collect()
    time.sleep(1)

    r = run_tick_loop(world, ticks, rss_limit, per_tick_fn=per_tick, label=f"T8c:tl_no_awp/{town}")
    print(f"  TL actors checked: {tl_count[0]}")
    teardown(world, vehicle, settings)
    return r


def test_detect_vehicles(
    client: Any, world: Any, town: str, ticks: int, rss_limit: int
) -> dict[str, Any]:
    """T8d: tick + vehicle detection logic per frame.
    Calls get_actors().filter('*vehicle*') + get_transform + get_velocity
    + OBB computation, like _detect_vehicles does."""
    print(f"\n{'='*70}")
    print(f"T8d: tick + vehicle detection (OBB) | {town} | {ticks} ticks")
    print(f"{'='*70}")

    import numpy as np

    world, vehicle, settings = setup(client, world, town)

    def per_tick(w: Any, i: int) -> None:
        ego_loc = vehicle.get_transform().location
        for v in w.get_actors().filter("*vehicle*"):
            if v.id == vehicle.id:
                continue
            vt = v.get_transform()
            _ = v.get_velocity()
            dist = ego_loc.distance(vt.location)
            if dist < 50.0:
                bb = v.bounding_box
                ext = bb.extent
                # Compute 2D OBB corners like _get_actor_obb_2d
                yaw = math.radians(vt.rotation.yaw)
                cos_y, sin_y = math.cos(yaw), math.sin(yaw)
                cx, cy = vt.location.x, vt.location.y
                dx, dy = ext.x, ext.y
                _ = np.array(
                    [
                        [cx + cos_y * dx - sin_y * dy, cy + sin_y * dx + cos_y * dy],
                        [cx - cos_y * dx - sin_y * dy, cy - sin_y * dx + cos_y * dy],
                        [cx - cos_y * dx + sin_y * dy, cy - sin_y * dx - cos_y * dy],
                        [cx + cos_y * dx + sin_y * dy, cy + sin_y * dx - cos_y * dy],
                    ]
                )

    for _ in range(20):
        world.tick()
    gc.collect()
    time.sleep(1)

    r = run_tick_loop(world, ticks, rss_limit, per_tick_fn=per_tick, label=f"T8d:detect_veh/{town}")
    teardown(world, vehicle, settings)
    return r


def test_stop_signs(
    client: Any, world: Any, town: str, ticks: int, rss_limit: int
) -> dict[str, Any]:
    """T8e: tick + stop sign scan per frame.
    Calls get_actors().filter('*stop*') + get_transform + get_waypoint."""
    print(f"\n{'='*70}")
    print(f"T8e: tick + stop sign scan | {town} | {ticks} ticks")
    print(f"{'='*70}")

    world, vehicle, settings = setup(client, world, town)
    carla_map = world.get_map()
    ss_count = [0]

    def per_tick(w: Any, i: int) -> None:
        ego_loc = vehicle.get_transform().location
        _ = carla_map.get_waypoint(ego_loc)
        for ss in w.get_actors().filter("*stop*"):
            ss_count[0] += 1
            ss_loc = ss.get_transform().location
            dist = ego_loc.distance(ss_loc)
            if dist < 50.0:
                _ = carla_map.get_waypoint(ss_loc)

    for _ in range(20):
        world.tick()
    gc.collect()
    time.sleep(1)

    r = run_tick_loop(world, ticks, rss_limit, per_tick_fn=per_tick, label=f"T8e:stop_signs/{town}")
    print(f"  Stop sign actors checked: {ss_count[0]}")
    teardown(world, vehicle, settings)
    return r


def test_expert_npcs(
    client: Any, world: Any, town: str, ticks: int, rss_limit: int
) -> dict[str, Any]:
    """T9: world.tick() + PrivilegedExpert + 50 NPCs + 30 walkers.
    No cameras. This is the heaviest non-camera scenario."""
    print(f"\n{'='*70}")
    print(f"T9: tick + Expert + 50 NPCs + 30 walkers | {town} | {ticks} ticks")
    print(f"{'='*70}")

    from agents.expert.privileged_expert import PrivilegedExpert, ExpertConfig
    from agents.navigation.global_route_planner import GlobalRoutePlanner
    import numpy as np

    world, vehicle, settings = setup(client, world, town)
    carla_map = world.get_map()
    spawn_points = carla_map.get_spawn_points()
    bp_lib = world.get_blueprint_library()

    # NPCs
    npc_bps = bp_lib.filter("vehicle.*")
    npc_bps = [bp for bp in npc_bps if int(bp.get_attribute("number_of_wheels")) == 4]
    npcs = []
    ego_loc = vehicle.get_location()
    available = [sp for sp in spawn_points if sp.location.distance(ego_loc) > 50.0]
    np.random.shuffle(available)
    for i in range(min(50, len(available))):
        npc = world.try_spawn_actor(np.random.choice(npc_bps), available[i])
        if npc is not None:
            npcs.append(npc)
    tm = client.get_trafficmanager(8000)
    tm.set_synchronous_mode(True)
    for npc in npcs:
        npc.set_autopilot(True, tm.get_port())
    print(f"  Spawned {len(npcs)} NPC vehicles")

    # Walkers
    walkers = []
    walker_controllers = []
    walker_bps = bp_lib.filter("walker.pedestrian.*")
    for _ in range(60):
        loc = world.get_random_location_from_navigation()
        if loc is not None:
            w_bp = np.random.choice(walker_bps)
            walker = world.try_spawn_actor(w_bp, carla.Transform(location=loc))
            if walker is not None:
                walkers.append(walker)
        if len(walkers) >= 30:
            break
    world.tick()
    controller_bp = bp_lib.find("controller.ai.walker")
    for walker in walkers:
        ctrl = world.spawn_actor(controller_bp, carla.Transform(), attach_to=walker)
        walker_controllers.append(ctrl)
    world.tick()
    for ctrl in walker_controllers:
        dest = world.get_random_location_from_navigation()
        if dest is not None:
            ctrl.start()
            ctrl.go_to_location(dest)
            ctrl.set_max_speed(1.0 + np.random.random() * 1.5)
    print(f"  Spawned {len(walkers)} walkers")

    # Expert
    route_planner = GlobalRoutePlanner(carla_map, sampling_resolution=2.0)
    expert = PrivilegedExpert(vehicle, world, route_planner, ExpertConfig())
    dest_idx = 10 if len(spawn_points) > 10 else len(spawn_points) - 1
    expert.set_destination(vehicle.get_location(), spawn_points[dest_idx].location)
    print(f"  Expert route set to spawn point {dest_idx}")

    for _ in range(20):
        world.tick()
    gc.collect()
    time.sleep(1)

    def per_tick(w: Any, i: int) -> None:
        if expert.done():
            return
        control, info = expert.run_step(0.05)
        vehicle.apply_control(control)
        del control, info

    all_actors = npcs + walkers + walker_controllers
    r = run_tick_loop(world, ticks, rss_limit, per_tick_fn=per_tick, label=f"T9:expert_npcs/{town}")
    teardown(world, vehicle, settings, extra_actors=all_actors, tm=tm)
    return r


def test_cam_queue(
    client: Any, world: Any, town: str, ticks: int, rss_limit: int
) -> dict[str, Any]:
    """T10: 3 cameras with queue.Queue + np.frombuffer + reshape + copy.
    Tests if the image processing path (vs just listen/discard) leaks."""
    print(f"\n{'='*70}")
    print(f"T10: tick + 3 cameras (queue + frombuffer) | {town} | {ticks} ticks")
    print(f"{'='*70}")

    import queue as queue_mod
    import numpy as np

    world, vehicle, settings = setup(client, world, town)

    cam_bp = world.get_blueprint_library().find("sensor.camera.rgb")
    cam_bp.set_attribute("image_size_x", "640")
    cam_bp.set_attribute("image_size_y", "320")
    cam_bp.set_attribute("fov", "90")

    cams = []
    cam_queues = []
    transforms = [
        carla.Transform(carla.Location(x=1.5, z=2.4)),
        carla.Transform(carla.Location(x=0.0, y=-0.8, z=2.4), carla.Rotation(yaw=-60)),
        carla.Transform(carla.Location(x=0.0, y=0.8, z=2.4), carla.Rotation(yaw=60)),
    ]
    for t in transforms:
        c = world.spawn_actor(cam_bp, t, attach_to=vehicle)
        q: queue_mod.Queue[Any] = queue_mod.Queue()
        c.listen(lambda img, qq=q: qq.put(img))
        cams.append(c)
        cam_queues.append(q)

    for _ in range(20):
        world.tick()
    # Drain warm-up frames
    for q in cam_queues:
        while not q.empty():
            try:
                q.get_nowait()
            except Exception:
                break
    gc.collect()
    time.sleep(1)

    frame_count = [0]

    def per_tick(w: Any, i: int) -> None:
        for q in cam_queues:
            try:
                img = q.get(timeout=2.0)
                raw = np.frombuffer(img.raw_data, dtype=np.uint8)
                raw = raw.reshape(320, 640, 4)
                rgb = raw[:, :, :3][:, :, ::-1].copy()
                frame_count[0] += 1
                del img, raw, rgb
            except Exception:
                pass

    r = run_tick_loop(world, ticks, rss_limit, per_tick_fn=per_tick, label=f"T10:cam_queue/{town}")
    print(f"  Frames processed: {frame_count[0]}")
    teardown(world, vehicle, settings, extra_actors=cams)
    return r


def test_expert_fine(
    client: Any, world: Any, town: str, ticks: int, rss_limit: int
) -> dict[str, Any]:
    """T11: expert.run_step() with PER-TICK monitoring.
    Monitors RSS + MemAvail at EVERY tick for fine-grained leak detection.
    Aborts if MemAvail < 4000 MB (aggressive) to avoid OOM kill."""
    print(f"\n{'='*70}")
    print(f"T11: expert.run_step() FINE-GRAINED | {town} | {ticks} ticks")
    print(f"{'='*70}")

    from agents.expert.privileged_expert import PrivilegedExpert, ExpertConfig
    from agents.navigation.global_route_planner import GlobalRoutePlanner

    world, vehicle, settings = setup(client, world, town)
    carla_map = world.get_map()
    spawn_points = carla_map.get_spawn_points()

    route_planner = GlobalRoutePlanner(carla_map, sampling_resolution=2.0)
    expert = PrivilegedExpert(vehicle, world, route_planner, ExpertConfig())
    dest_idx = 10 if len(spawn_points) > 10 else len(spawn_points) - 1
    expert.set_destination(vehicle.get_location(), spawn_points[dest_idx].location)
    print(f"  Expert route set to spawn point {dest_idx}")

    for _ in range(20):
        world.tick()
    gc.collect()
    time.sleep(1)

    mem0 = get_memory_info()
    avail0 = get_mem_avail_mb()
    t0 = time.time()
    print(
        f"  Baseline: RSS={mem0.get('rss_mb',0):.0f} MB, "
        f"AnonRSS={mem0.get('anon_mb',0):.0f} MB, "
        f"MemAvail={avail0:.0f} MB"
    )
    print("  Aggressive abort at MemAvail < 4000 MB")
    print()
    print(
        f"  {'tick':>4} {'RSS':>6} {'Anon':>6} {'Avail':>7} "
        f"{'dRSS':>6} {'dAvl':>7} {'elapsed':>8}"
    )
    print(f"  {'-'*55}")

    aborted = False
    tick_done = 0
    for i in range(1, ticks + 1):
        world.tick()
        tick_done = i

        if not expert.done():
            control, info = expert.run_step(0.05)
            vehicle.apply_control(control)
            del control, info

        # Monitor EVERY tick
        mem = get_memory_info()
        rss = mem.get("rss_mb", 0)
        anon = mem.get("anon_mb", 0)
        avail = get_mem_avail_mb()
        drss = rss - mem0.get("rss_mb", 0)
        davail = avail0 - avail
        elapsed = time.time() - t0

        # Print every tick for first 20, then every 10
        if i <= 20 or i % 10 == 0:
            print(
                f"  {i:4d} {rss:5.0f}M {anon:5.0f}M {avail:6.0f}M "
                f"{drss:+5.0f}M {-davail:+6.0f}M {elapsed:7.1f}s"
            )

        # Aggressive abort
        if avail < 4000:
            print(f"\n  *** ABORT: MemAvail={avail:.0f} MB < 4000 MB ***")
            aborted = True
            break
        if rss > rss_limit:
            print(f"\n  *** ABORT: RSS={rss:.0f} MB > {rss_limit} MB ***")
            aborted = True
            break

    gc.collect()
    time.sleep(0.5)
    mem_end = get_memory_info()
    avail_end = get_mem_avail_mb()
    elapsed = time.time() - t0
    drss = mem_end.get("rss_mb", 0) - mem0.get("rss_mb", 0)
    davail = avail0 - avail_end

    verdict = "ABORT" if aborted else ("LEAK" if drss > 50 else "CLEAN")
    print(f"\n  === RESULTS: T11:expert_fine/{town} ===")
    print(f"  Ticks:    {tick_done}/{ticks}")
    print(f"  Elapsed:  {elapsed:.1f}s")
    print(
        f"  RSS:      {mem0.get('rss_mb',0):.0f} -> {mem_end.get('rss_mb',0):.0f} MB "
        f" (d{drss:+.0f} MB, {drss/elapsed:+.1f} MB/s)"
    )
    print(
        f"  MemAvail: {avail0:.0f} -> {avail_end:.0f} MB "
        f" (drained {-davail:+.0f} MB, {davail/elapsed:+.1f} MB/s)"
    )
    print(f"  Verdict:  {verdict}")

    teardown(world, vehicle, settings)
    return {
        "verdict": verdict,
        "drss": drss,
        "davail": davail,
        "elapsed": elapsed,
        "ticks_done": tick_done,
    }


def test_expert_drive(
    client: Any, world: Any, town: str, ticks: int, rss_limit: int
) -> dict[str, Any]:
    """T12: tick + constant throttle (ego drives) + same CARLA queries
    that the expert makes, but WITHOUT the expert object.
    Tests if vehicle movement + queries causes the leak."""
    print(f"\n{'='*70}")
    print(f"T12: tick + drive + queries (no expert) | {town} | {ticks} ticks")
    print(f"{'='*70}")

    world, vehicle, settings = setup(client, world, town)
    carla_map = world.get_map()

    for _ in range(20):
        world.tick()
    gc.collect()
    time.sleep(1)

    def per_tick(w: Any, i: int) -> None:
        # Drive the vehicle with constant throttle
        vehicle.apply_control(carla.VehicleControl(throttle=0.4, steer=0.0))

        # Replicate the expert's CARLA API calls (no processing)
        ego_loc = vehicle.get_transform().location
        _ = vehicle.get_velocity()
        _ = vehicle.get_speed_limit()
        _ = carla_map.get_waypoint(ego_loc)

        # Same 4 get_actors filters the expert does
        for f in ("*traffic_light*", "*stop*", "*vehicle*", "*walker*"):
            for a in w.get_actors().filter(f):
                _ = a.get_transform()

        # get_affected_lane_waypoints on traffic lights (like expert)
        for tl in w.get_actors().filter("*traffic_light*"):
            try:
                for awp in tl.get_affected_lane_waypoints():
                    pass
            except Exception:
                pass

    r = run_tick_loop(
        world, ticks, rss_limit, per_tick_fn=per_tick, label=f"T12:drive_queries/{town}"
    )
    teardown(world, vehicle, settings)
    return r


def test_expert_nodetect(
    client: Any, world: Any, town: str, ticks: int, rss_limit: int
) -> dict[str, Any]:
    """T13: Expert with ALL detection methods disabled (monkey-patched).
    run_step() still does route advancement + control computation +
    apply_control, but _check_traffic_light, _check_stop_signs,
    _detect_vehicles, _detect_pedestrians are no-ops.
    Isolates: is the leak in route/control or in detection?"""
    print(f"\n{'='*70}")
    print(f"T13: expert (detection disabled) | {town} | {ticks} ticks")
    print(f"{'='*70}")

    from agents.expert.privileged_expert import PrivilegedExpert, ExpertConfig
    from agents.navigation.global_route_planner import GlobalRoutePlanner

    world, vehicle, settings = setup(client, world, town)
    carla_map = world.get_map()
    spawn_points = carla_map.get_spawn_points()

    route_planner = GlobalRoutePlanner(carla_map, sampling_resolution=2.0)
    expert = PrivilegedExpert(vehicle, world, route_planner, ExpertConfig())
    dest_idx = 10 if len(spawn_points) > 10 else len(spawn_points) - 1
    expert.set_destination(vehicle.get_location(), spawn_points[dest_idx].location)
    print(f"  Expert route set to spawn point {dest_idx}")

    # Monkey-patch detection methods to no-ops
    expert._check_traffic_light = lambda t: (None, None)
    expert._check_stop_signs = lambda t, s: None
    expert._detect_vehicles = lambda t: []
    expert._detect_pedestrians = lambda t: []
    print(
        "  Patched: _check_traffic_light, _check_stop_signs, "
        "_detect_vehicles, _detect_pedestrians -> no-op"
    )

    for _ in range(20):
        world.tick()
    gc.collect()
    time.sleep(1)

    def per_tick(w: Any, i: int) -> None:
        if expert.done():
            return
        control, info = expert.run_step(0.05)
        vehicle.apply_control(control)
        del control, info

    r = run_tick_loop(
        world, ticks, rss_limit, per_tick_fn=per_tick, label=f"T13:expert_nodetect/{town}"
    )
    teardown(world, vehicle, settings)
    return r


def test_expert_noctrl(
    client: Any, world: Any, town: str, ticks: int, rss_limit: int
) -> dict[str, Any]:
    """T14: Expert run_step() computes everything but vehicle.apply_control()
    is NOT called. Vehicle sits still.
    Isolates: does the vehicle need to be MOVING for the leak?"""
    print(f"\n{'='*70}")
    print(f"T14: expert (no apply_control) | {town} | {ticks} ticks")
    print(f"{'='*70}")

    from agents.expert.privileged_expert import PrivilegedExpert, ExpertConfig
    from agents.navigation.global_route_planner import GlobalRoutePlanner

    world, vehicle, settings = setup(client, world, town)
    carla_map = world.get_map()
    spawn_points = carla_map.get_spawn_points()

    route_planner = GlobalRoutePlanner(carla_map, sampling_resolution=2.0)
    expert = PrivilegedExpert(vehicle, world, route_planner, ExpertConfig())
    dest_idx = 10 if len(spawn_points) > 10 else len(spawn_points) - 1
    expert.set_destination(vehicle.get_location(), spawn_points[dest_idx].location)
    print(f"  Expert route set to spawn point {dest_idx}")

    for _ in range(20):
        world.tick()
    gc.collect()
    time.sleep(1)

    def per_tick(w: Any, i: int) -> None:
        if expert.done():
            return
        control, info = expert.run_step(0.05)
        # Deliberately NOT calling vehicle.apply_control()
        del control, info

    r = run_tick_loop(
        world, ticks, rss_limit, per_tick_fn=per_tick, label=f"T14:expert_noctrl/{town}"
    )
    teardown(world, vehicle, settings)
    return r


def test_drive_only(
    client: Any, world: Any, town: str, ticks: int, rss_limit: int
) -> dict[str, Any]:
    """T15: tick + vehicle.apply_control(throttle=0.4) per frame.
    No expert, no queries, no NPCs. Just the ego vehicle driving.
    Isolates: does pure ego driving cause the leak?"""
    print(f"\n{'='*70}")
    print(f"T15: tick + pure ego driving (apply_control only) | {town} | {ticks} ticks")
    print(f"{'='*70}")

    world, vehicle, settings = setup(client, world, town)

    for _ in range(20):
        world.tick()
    gc.collect()
    time.sleep(1)

    def per_tick(w: Any, i: int) -> None:
        vehicle.apply_control(carla.VehicleControl(throttle=0.4, steer=0.0))

    r = run_tick_loop(world, ticks, rss_limit, per_tick_fn=per_tick, label=f"T15:drive_only/{town}")
    teardown(world, vehicle, settings)
    return r


def test_drive_route(
    client: Any, world: Any, town: str, ticks: int, rss_limit: int
) -> dict[str, Any]:
    """T16: expert + apply_control but _fill_junction_gaps disabled.
    Vehicle follows the route but without gap filling.
    If this is CLEAN but T13 OOMs, the leak is in _fill_junction_gaps."""
    print(f"\n{'='*70}")
    print(f"T16: expert + drive (NO junction gap fill) | {town} | {ticks} ticks")
    print(f"{'='*70}")

    from agents.expert.privileged_expert import PrivilegedExpert, ExpertConfig
    from agents.navigation.global_route_planner import GlobalRoutePlanner

    world, vehicle, settings = setup(client, world, town)
    carla_map = world.get_map()
    spawn_points = carla_map.get_spawn_points()

    route_planner = GlobalRoutePlanner(carla_map, sampling_resolution=2.0)
    expert = PrivilegedExpert(vehicle, world, route_planner, ExpertConfig())
    dest_idx = 10 if len(spawn_points) > 10 else len(spawn_points) - 1
    expert.set_destination(vehicle.get_location(), spawn_points[dest_idx].location)
    print(f"  Expert route set to spawn point {dest_idx}")

    # Disable detection AND _fill_junction_gaps
    expert._check_traffic_light = lambda t: (None, None)
    expert._check_stop_signs = lambda t, s: None
    expert._detect_vehicles = lambda t: []
    expert._detect_pedestrians = lambda t: []
    expert._fill_junction_gaps = lambda: None  # <-- KEY: disable gap filling
    print("  Detection + _fill_junction_gaps disabled; route + control active")

    for _ in range(20):
        world.tick()
    gc.collect()
    time.sleep(1)

    def per_tick(w: Any, i: int) -> None:
        if expert.done():
            return
        control, info = expert.run_step(0.05)
        vehicle.apply_control(control)
        del control, info

    r = run_tick_loop(
        world, ticks, rss_limit, per_tick_fn=per_tick, label=f"T16:drive_nojgap/{town}"
    )
    teardown(world, vehicle, settings)
    return r


def test_drive_route_noext(
    client: Any, world: Any, town: str, ticks: int, rss_limit: int
) -> dict[str, Any]:
    """T17: expert + apply_control, _fill_junction_gaps disabled,
    AND route-end extension in _fill_buffer disabled.
    If T16 still OOMs but T17 is clean, the leak is in _fill_buffer's
    route-end extension (the `last_wp.next(2.0)` while loop)."""
    print(f"\n{'='*70}")
    print(f"T17: expert + drive (NO gap fill, NO route extension) | {town} | {ticks} ticks")
    print(f"{'='*70}")

    from agents.expert.privileged_expert import PrivilegedExpert, ExpertConfig
    from agents.navigation.global_route_planner import GlobalRoutePlanner

    world, vehicle, settings = setup(client, world, town)
    carla_map = world.get_map()
    spawn_points = carla_map.get_spawn_points()

    route_planner = GlobalRoutePlanner(carla_map, sampling_resolution=2.0)
    expert = PrivilegedExpert(vehicle, world, route_planner, ExpertConfig())
    dest_idx = 10 if len(spawn_points) > 10 else len(spawn_points) - 1
    expert.set_destination(vehicle.get_location(), spawn_points[dest_idx].location)
    print(f"  Expert route set to spawn point {dest_idx}")

    # Disable detection, gap filling, AND route-end extension
    expert._check_traffic_light = lambda t: (None, None)
    expert._check_stop_signs = lambda t, s: None
    expert._detect_vehicles = lambda t: []
    expert._detect_pedestrians = lambda t: []
    expert._fill_junction_gaps = lambda: None

    # Patch _fill_buffer to skip route-end extension (the wp.next() loop)
    _ = expert._fill_buffer.__func__

    def _fill_buffer_no_ext(self_inner: Any) -> None:
        while (
            len(self_inner._wp_buffer) < self_inner._wp_buffer.maxlen
            and self_inner._route_index < self_inner._route_length
        ):
            self_inner._wp_buffer.append(self_inner._route[self_inner._route_index])
            self_inner._route_index += 1
        # Skip _fill_junction_gaps (already patched)
        # Skip route-end extension entirely

    import types

    expert._fill_buffer = types.MethodType(_fill_buffer_no_ext, expert)
    print("  Detection + gap fill + route extension disabled")

    for _ in range(20):
        world.tick()
    gc.collect()
    time.sleep(1)

    def per_tick(w: Any, i: int) -> None:
        if expert.done():
            return
        control, info = expert.run_step(0.05)
        vehicle.apply_control(control)
        del control, info

    r = run_tick_loop(
        world, ticks, rss_limit, per_tick_fn=per_tick, label=f"T17:drive_noext/{town}"
    )
    teardown(world, vehicle, settings)
    return r


# ---------- Main ----------

TEST_MAP = {
    "tick": test_tick,  # T1
    "cam1": test_cam1,  # T2
    "cam3": test_cam3,  # T3
    "npcs": test_npcs,  # T4
    "get_actors": test_get_actors,  # T5
    "get_waypoints": test_get_waypoints,  # T6
    "full": test_full,  # T7
    "expert": test_expert,  # T8
    "expert_init": test_expert_init_only,  # T8a
    "tl_scan": test_tl_scan,  # T8b
    "tl_no_awp": test_tl_no_awp,  # T8c
    "detect_veh": test_detect_vehicles,  # T8d
    "stop_signs": test_stop_signs,  # T8e
    "expert_npcs": test_expert_npcs,  # T9
    "cam_queue": test_cam_queue,  # T10
    "expert_fine": test_expert_fine,  # T11
    "drive_queries": test_expert_drive,  # T12
    "expert_nodetect": test_expert_nodetect,  # T13
    "expert_noctrl": test_expert_noctrl,  # T14
    "drive_only": test_drive_only,  # T15
    "drive_route": test_drive_route,  # T16
    "drive_noext": test_drive_route_noext,  # T17
}


def main() -> None:
    parser = argparse.ArgumentParser(description="Incremental CARLA memory leak diagnostic")
    parser.add_argument("--host", default="172.20.0.1")
    parser.add_argument("--port", type=int, default=2000)
    parser.add_argument("--town", default="Town03")
    parser.add_argument("--ticks", type=int, default=5000)
    parser.add_argument(
        "--rss-limit", type=int, default=2000, help="Abort if RSS > this (MB). Default 2000."
    )
    parser.add_argument(
        "--test", required=True, choices=list(TEST_MAP.keys()), help="Which single test to run"
    )
    args = parser.parse_args()

    print("CARLA Memory Leak Diagnostic -- Incremental")
    print(
        f"Test: {args.test} | Town: {args.town} | "
        f"Ticks: {args.ticks} | RSS limit: {args.rss_limit} MB"
    )
    print(f"PID: {os.getpid()}")
    print(f"{'='*70}")

    client = carla.Client(args.host, args.port)
    client.set_timeout(60.0)
    world = client.get_world()

    n = cleanup_actors(world)
    if n:
        print(f"Cleaned up {n} orphaned actors")
        time.sleep(2)
        world = client.get_world()

    test_fn = TEST_MAP[args.test]
    result = test_fn(client, world, args.town, args.ticks, args.rss_limit)

    print(f"\n{'='*70}")
    print(f"DONE -- {result['verdict']}")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
