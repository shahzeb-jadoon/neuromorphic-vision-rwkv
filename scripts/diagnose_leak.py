#!/usr/bin/env python3
"""Comprehensive CARLA memory leak diagnostic.

Runs a battery of isolation tests to determine exactly where
the memory leak occurs in libcarla's data pipeline.

Tests:
  T1: tick() only, NO sensors            → baseline C++ overhead per tick
  T2: sensors spawned, NO listen()        → server sends data, client ignores
  T3: sensors + listen, discard images    → full pipeline, images consumed & freed
  T4: sensors + listen, accumulate images → images held in Python (queue grows)
  T5: T3 on Town01 (simple map)           → compare leak rate to Town04

Usage:
  python scripts/diagnose_leak.py [--town Town04] [--ticks 600] [--host 172.20.0.1]
"""

import argparse
import gc
import queue
import time
from typing import Any

import carla
import numpy as np


def get_rss_mb() -> int:
    """Read VmRSS from /proc/self/status (Linux only)."""
    try:
        with open("/proc/self/status") as f:
            for line in f:
                if line.startswith("VmRSS:"):
                    return int(line.split()[1]) // 1024
    except Exception:
        pass
    return -1


def get_mem_avail_mb() -> int:
    """Read MemAvailable from /proc/meminfo."""
    try:
        with open("/proc/meminfo") as f:
            for line in f:
                if line.startswith("MemAvailable:"):
                    return int(line.split()[1]) // 1024
    except Exception:
        pass
    return -1


def setup_carla(host: str, port: int, town: str) -> tuple:
    """Connect to CARLA and load the specified map in synchronous mode."""
    client = carla.Client(host, port)
    client.set_timeout(60.0)

    world = client.get_world()
    current_map = world.get_map().name

    # ALWAYS destroy orphaned actors from previous (possibly killed) runs.
    # Without this, a killed test leaves NPCs/cameras alive, consuming
    # memory and corrupting subsequent tests.
    actors = world.get_actors()
    for s in actors.filter("sensor.*"):
        try:
            s.stop()
            s.destroy()
        except Exception:
            pass
    for v in actors.filter("vehicle.*"):
        try:
            v.set_autopilot(False)
            v.destroy()
        except Exception:
            pass
    for c in actors.filter("controller.*"):
        try:
            c.stop()
            c.destroy()
        except Exception:
            pass
    for w in actors.filter("walker.*"):
        try:
            w.destroy()
        except Exception:
            pass

    # Force a few ticks in async mode to let CARLA process the destroys
    settings = world.get_settings()
    settings.synchronous_mode = False
    settings.fixed_delta_seconds = None
    world.apply_settings(settings)
    time.sleep(2.0)
    gc.collect()

    if town not in current_map:
        print(f"  Loading map {town}...")
        client.load_world(town)
        time.sleep(5.0)
        world = client.get_world()

    # Set synchronous mode
    settings = world.get_settings()
    settings.synchronous_mode = True
    settings.fixed_delta_seconds = 0.05  # 20 FPS
    world.apply_settings(settings)

    # Spawn ego vehicle
    bp_lib = world.get_blueprint_library()
    vehicle_bp = bp_lib.find("vehicle.tesla.model3")
    spawn_points = world.get_map().get_spawn_points()
    vehicle = world.spawn_actor(vehicle_bp, spawn_points[0])

    return client, world, vehicle


def spawn_cameras(
    world: Any, vehicle: Any, n_cameras: int = 3, width: int = 640, height: int = 320
) -> list[Any]:
    """Spawn cameras identical to the user's collection config."""
    bp_lib = world.get_blueprint_library()
    cam_bp = bp_lib.find("sensor.camera.rgb")
    cam_bp.set_attribute("image_size_x", str(width))
    cam_bp.set_attribute("image_size_y", str(height))
    cam_bp.set_attribute("fov", "60")

    yaws = [0.0, -60.0, 60.0][:n_cameras]
    cameras = []
    for yaw in yaws:
        transform = carla.Transform(
            carla.Location(x=1.6, z=1.7),
            carla.Rotation(yaw=yaw),
        )
        cam = world.spawn_actor(cam_bp, transform, attach_to=vehicle)
        cameras.append(cam)
    return cameras


def cleanup(world: Any, vehicle: Any, cameras: Any, original_settings: Any = None) -> None:
    """Destroy actors and restore async mode."""
    for cam in cameras:
        try:
            cam.stop()
        except Exception:
            pass
        try:
            cam.destroy()
        except Exception:
            pass
    try:
        vehicle.destroy()
    except Exception:
        pass
    if original_settings:
        world.apply_settings(original_settings)


def run_test(
    test_name: str, host: str, port: int, town: str, n_ticks: int, test_fn: Any, **kwargs: Any
) -> dict[str, Any]:
    """Run a single diagnostic test with RSS tracking."""
    print(f"\n{'=' * 60}")
    print(f"TEST: {test_name}")
    print(f"Town: {town}, Ticks: {n_ticks}")
    print(f"{'=' * 60}")

    client, world, vehicle = setup_carla(host, port, town)

    # Do a few warm-up ticks to let CARLA settle
    for _ in range(10):
        world.tick()

    gc.collect()
    rss_start = get_rss_mb()
    mem_start = get_mem_avail_mb()
    t_start = time.time()

    print(f"  Baseline: RSS={rss_start} MB, MemAvail={mem_start} MB")

    measurements: list[Any] = []

    try:
        test_fn(
            world=world,
            vehicle=vehicle,
            client=client,
            n_ticks=n_ticks,
            measurements=measurements,
            town=town,
            **kwargs,
        )
    except KeyboardInterrupt:
        print("  Interrupted!")
    except Exception as e:
        print(f"  ERROR: {e}")
    finally:
        gc.collect()
        rss_end = get_rss_mb()
        mem_end = get_mem_avail_mb()
        t_elapsed = time.time() - t_start

        rss_growth = rss_end - rss_start
        rate = rss_growth / t_elapsed if t_elapsed > 0 else 0

        print(f"  Final:    RSS={rss_end} MB, MemAvail={mem_end} MB")
        print(f"  Growth:   {rss_growth} MB in {t_elapsed:.1f}s = {rate:.1f} MB/s")
        print(f"  {'LEAK DETECTED' if rate > 1.0 else 'NO SIGNIFICANT LEAK'}")

        # Cleanup
        cameras = kwargs.get("_cameras_ref", [])
        cleanup(world, vehicle, cameras)

        # Restore async mode
        settings = world.get_settings()
        settings.synchronous_mode = False
        settings.fixed_delta_seconds = None
        try:
            world.apply_settings(settings)
        except Exception:
            pass

        return {
            "test": test_name,
            "town": town,
            "ticks": n_ticks,
            "rss_start": rss_start,
            "rss_end": rss_end,
            "rss_growth": rss_growth,
            "elapsed": t_elapsed,
            "rate_mb_s": rate,
            "measurements": measurements,
        }


# ==========================================================================
# Test implementations
# ==========================================================================


def test_tick_only(
    world: Any, vehicle: Any, n_ticks: int, measurements: list[Any], **kwargs: Any
) -> None:
    """T1: Just tick, no sensors at all."""
    log_interval = max(1, n_ticks // 10)
    for i in range(n_ticks):
        world.tick()
        if i % log_interval == 0:
            rss = get_rss_mb()
            measurements.append((i, rss, time.time()))
            print(f"    tick {i:5d}: RSS={rss} MB")

        # Safety: abort if MemAvail drops below 3 GB
        if i % 100 == 0:
            avail = get_mem_avail_mb()
            if avail < 3000:
                print(f"    SAFETY ABORT: MemAvail={avail} MB < 3000 MB")
                break


def test_sensors_no_listen(
    world: Any, vehicle: Any, n_ticks: int, measurements: list[Any], **kwargs: Any
) -> None:
    """T2: Cameras spawned but listen() NOT called."""
    cameras = spawn_cameras(world, vehicle)
    kwargs["_cameras_re"] = cameras

    log_interval = max(1, n_ticks // 10)
    for i in range(n_ticks):
        world.tick()
        if i % log_interval == 0:
            rss = get_rss_mb()
            measurements.append((i, rss, time.time()))
            print(f"    tick {i:5d}: RSS={rss} MB")
        if i % 100 == 0:
            avail = get_mem_avail_mb()
            if avail < 3000:
                print(f"    SAFETY ABORT: MemAvail={avail} MB < 3000 MB")
                break

    for cam in cameras:
        cam.destroy()
    cameras.clear()


def test_sensors_listen_discard(
    world: Any, vehicle: Any, n_ticks: int, measurements: list[Any], **kwargs: Any
) -> None:
    """T3: Cameras with listen(), images consumed and discarded each tick."""
    cameras_raw = spawn_cameras(world, vehicle)
    cam_queues = []
    for cam in cameras_raw:
        q: queue.Queue[Any] = queue.Queue()
        cam.listen(lambda img, qq=q: qq.put(img))
        cam_queues.append((cam, q))
    kwargs["_cameras_re"] = cameras_raw

    log_interval = max(1, n_ticks // 10)
    for i in range(n_ticks):
        world.tick()

        # Consume all images from queues (like the user's code does)
        for _cam, q in cam_queues:
            try:
                img = q.get(timeout=2.0)
                # Mimic user code: np.frombuffer + copy
                raw = np.frombuffer(img.raw_data, dtype=np.uint8)
                raw = raw.reshape(320, 640, 4)
                rgb = raw[:, :, :3][:, :, ::-1].copy()
                # rgb goes out of scope here → freed
                del img, raw, rgb
            except queue.Empty:
                pass

        if i % log_interval == 0:
            gc.collect()
            rss = get_rss_mb()
            measurements.append((i, rss, time.time()))
            print(f"    tick {i:5d}: RSS={rss} MB")
        if i % 100 == 0:
            avail = get_mem_avail_mb()
            if avail < 3000:
                print(f"    SAFETY ABORT: MemAvail={avail} MB < 3000 MB")
                break

    for cam in cameras_raw:
        cam.stop()
        cam.destroy()
    cameras_raw.clear()


def test_sensors_listen_accumulate(
    world: Any, vehicle: Any, n_ticks: int, measurements: list[Any], **kwargs: Any
) -> None:
    """T4: Cameras with listen(), images kept in list (never freed)."""
    cameras_raw = spawn_cameras(world, vehicle)
    cam_queues = []
    for cam in cameras_raw:
        q: queue.Queue[Any] = queue.Queue()
        cam.listen(lambda img, qq=q: qq.put(img))
        cam_queues.append((cam, q))
    kwargs["_cameras_re"] = cameras_raw

    all_images = []  # intentionally accumulate

    log_interval = max(1, n_ticks // 10)
    for i in range(n_ticks):
        world.tick()

        for _cam, q in cam_queues:
            try:
                img = q.get(timeout=2.0)
                all_images.append(img)  # KEEP reference → should grow linearly
            except queue.Empty:
                pass

        if i % log_interval == 0:
            rss = get_rss_mb()
            measurements.append((i, rss, time.time()))
            print(f"    tick {i:5d}: RSS={rss} MB, images_held={len(all_images)}")
        if i % 100 == 0:
            avail = get_mem_avail_mb()
            if avail < 3000:
                print(f"    SAFETY ABORT: MemAvail={avail} MB < 3000 MB")
                break

    print(f"    Clearing {len(all_images)} accumulated images...")
    all_images.clear()
    gc.collect()
    rss_after_clear = get_rss_mb()
    print(f"    RSS after clear+gc: {rss_after_clear} MB")

    for cam in cameras_raw:
        cam.stop()
        cam.destroy()
    cameras_raw.clear()


def test_sensors_listen_discard_simple_town(
    world: Any,
    vehicle: Any,
    client: Any,
    n_ticks: int,
    measurements: list[Any],
    town: str,
    **kwargs: Any,
) -> None:
    """T5: Same as T3 but on Town01 (simple map) for comparison."""
    # This test is called with town=Town01 from the runner
    test_sensors_listen_discard(
        world=world,
        vehicle=vehicle,
        client=client,
        n_ticks=n_ticks,
        measurements=measurements,
        town=town,
        **kwargs,
    )


def test_no_sensors_only_tick_count(
    world: Any, vehicle: Any, n_ticks: int, measurements: list[Any], **kwargs: Any
) -> None:
    """T6: Verify tick returns quickly — measure per-tick time."""
    times = []
    for i in range(min(n_ticks, 100)):
        t0 = time.monotonic()
        world.tick()
        dt = time.monotonic() - t0
        times.append(dt)

    avg = sum(times) / len(times)
    mx = max(times)
    mn = min(times)
    print(
        f"    Tick timing ({len(times)} ticks): avg={avg * 1000:.1f}ms, min={mn * 1000:.1f}ms, max={mx * 1000:.1f}ms"  # noqa: E501
    )
    measurements.append(("tick_avg_ms", avg * 1000))
    measurements.append(("tick_max_ms", mx * 1000))


def test_buffer_reuse(
    world: Any, vehicle: Any, n_ticks: int, measurements: list[Any], **kwargs: Any
) -> None:
    """T7: Check if buffers are reused via pool (RSS should plateau)."""
    cameras_raw = spawn_cameras(world, vehicle, n_cameras=1)
    q: queue.Queue[Any] = queue.Queue()
    cameras_raw[0].listen(lambda img, qq=q: qq.put(img))
    kwargs["_cameras_re"] = cameras_raw

    log_interval = max(1, n_ticks // 20)
    for i in range(n_ticks):
        world.tick()
        try:
            img = q.get(timeout=2.0)
            # Access raw_data to force buffer access, then immediately discard
            _ = len(img.raw_data)
            del img
        except queue.Empty:
            pass

        if i % log_interval == 0:
            gc.collect()
            rss = get_rss_mb()
            measurements.append((i, rss, time.time()))
            print(f"    tick {i:5d}: RSS={rss} MB")
        if i % 100 == 0:
            avail = get_mem_avail_mb()
            if avail < 3000:
                print(f"    SAFETY ABORT: MemAvail={avail} MB < 3000 MB")
                break

    cameras_raw[0].stop()
    cameras_raw[0].destroy()
    cameras_raw.clear()


def test_with_npcs(
    world: Any, vehicle: Any, client: Any, n_ticks: int, measurements: list[Any], **kwargs: Any
) -> None:
    """T8: Full pipeline with NPC traffic (TrafficManager) — matches user scenario."""
    bp_lib = world.get_blueprint_library()
    spawn_points = world.get_map().get_spawn_points()

    # --- Spawn NPC vehicles with TrafficManager (matching user code) ---
    tm = client.get_trafficmanager(8000)
    tm.set_synchronous_mode(True)
    tm.set_global_distance_to_leading_vehicle(2.5)

    npc_bps = bp_lib.filter("vehicle.*")
    npc_bps = [bp for bp in npc_bps if int(bp.get_attribute("number_of_wheels")) == 4]

    npcs = []
    ego_loc = vehicle.get_location()
    available = [sp for sp in spawn_points if sp.location.distance(ego_loc) > 50.0]
    np.random.shuffle(available)
    n_npcs = min(20, len(available))

    for i in range(n_npcs):
        npc_bp = np.random.choice(npc_bps)
        npc = world.try_spawn_actor(npc_bp, available[i])
        if npc is not None:
            npc.set_autopilot(True, tm.get_port())
            npcs.append(npc)
    print(f"    Spawned {len(npcs)} NPC vehicles with TrafficManager")

    # --- Spawn cameras + listen ---
    cameras_raw = spawn_cameras(world, vehicle)
    cam_queues = []
    for cam in cameras_raw:
        q: queue.Queue[Any] = queue.Queue()
        cam.listen(lambda img, qq=q: qq.put(img))
        cam_queues.append((cam, q))
    kwargs["_cameras_re"] = cameras_raw

    log_interval = max(1, n_ticks // 10)
    for i in range(n_ticks):
        world.tick()

        # Consume images (same as user code)
        for _cam, q in cam_queues:
            try:
                img = q.get(timeout=2.0)
                raw = np.frombuffer(img.raw_data, dtype=np.uint8)
                raw = raw.reshape(320, 640, 4)
                rgb = raw[:, :, :3][:, :, ::-1].copy()
                del img, raw, rgb
            except queue.Empty:
                pass

        if i % log_interval == 0:
            gc.collect()
            rss = get_rss_mb()
            measurements.append((i, rss, time.time()))
            print(f"    tick {i:5d}: RSS={rss} MB")
        if i % 100 == 0:
            avail = get_mem_avail_mb()
            if avail < 3000:
                print(f"    SAFETY ABORT: MemAvail={avail} MB < 3000 MB")
                break

    # Cleanup
    for cam in cameras_raw:
        cam.stop()
        cam.destroy()
    cameras_raw.clear()

    for npc in npcs:
        npc.set_autopilot(False)
        npc.destroy()
    npcs.clear()
    tm.set_synchronous_mode(False)


def test_with_npcs_no_cameras(
    world: Any, vehicle: Any, client: Any, n_ticks: int, measurements: list[Any], **kwargs: Any
) -> None:
    """T9: NPCs with TrafficManager but NO cameras — isolate TM leak."""
    bp_lib = world.get_blueprint_library()
    spawn_points = world.get_map().get_spawn_points()

    tm = client.get_trafficmanager(8000)
    tm.set_synchronous_mode(True)
    tm.set_global_distance_to_leading_vehicle(2.5)

    npc_bps = bp_lib.filter("vehicle.*")
    npc_bps = [bp for bp in npc_bps if int(bp.get_attribute("number_of_wheels")) == 4]

    npcs = []
    ego_loc = vehicle.get_location()
    available = [sp for sp in spawn_points if sp.location.distance(ego_loc) > 50.0]
    np.random.shuffle(available)
    n_npcs = min(20, len(available))

    for i in range(n_npcs):
        npc_bp = np.random.choice(npc_bps)
        npc = world.try_spawn_actor(npc_bp, available[i])
        if npc is not None:
            npc.set_autopilot(True, tm.get_port())
            npcs.append(npc)
    print(f"    Spawned {len(npcs)} NPC vehicles (NO cameras)")

    log_interval = max(1, n_ticks // 10)
    for i in range(n_ticks):
        world.tick()

        if i % log_interval == 0:
            rss = get_rss_mb()
            measurements.append((i, rss, time.time()))
            print(f"    tick {i:5d}: RSS={rss} MB")
        if i % 100 == 0:
            avail = get_mem_avail_mb()
            if avail < 3000:
                print(f"    SAFETY ABORT: MemAvail={avail} MB < 3000 MB")
                break

    for npc in npcs:
        npc.set_autopilot(False)
        npc.destroy()
    npcs.clear()
    tm.set_synchronous_mode(False)


def test_get_actors_calls(
    world: Any, vehicle: Any, client: Any, n_ticks: int, measurements: list[Any], **kwargs: Any
) -> None:
    """T10: Tick + world.get_actors() x4 per frame (like PrivilegedExpert).

    The expert calls get_actors().filter("*vehicle*"), filter("*walker*"),
    filter("*traffic_light*"), filter("*stop*") every single frame.
    Also spawn 50 NPCs + 30 walkers to match the actual collection config.
    """
    bp_lib = world.get_blueprint_library()
    spawn_points = world.get_map().get_spawn_points()
    carla_map = world.get_map()

    tm = client.get_trafficmanager(8000)
    tm.set_synchronous_mode(True)
    tm.set_global_distance_to_leading_vehicle(2.5)

    npc_bps = bp_lib.filter("vehicle.*")
    npc_bps = [bp for bp in npc_bps if int(bp.get_attribute("number_of_wheels")) == 4]

    npcs = []
    ego_loc = vehicle.get_location()
    available = [sp for sp in spawn_points if sp.location.distance(ego_loc) > 50.0]
    np.random.shuffle(available)
    n_npcs = min(50, len(available))

    for i in range(n_npcs):
        npc_bp = np.random.choice(npc_bps)
        npc = world.try_spawn_actor(npc_bp, available[i])
        if npc is not None:
            npc.set_autopilot(True, tm.get_port())
            npcs.append(npc)
    print(f"    Spawned {len(npcs)} NPC vehicles")

    # Spawn walkers
    walkers = []
    walker_controllers = []
    walker_bps = bp_lib.filter("walker.pedestrian.*")
    for _ in range(60):
        loc = world.get_random_location_from_navigation()
        if loc is not None:
            w_bp = np.random.choice(walker_bps)
            import carla

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
    print(f"    Spawned {len(walkers)} walkers")

    log_interval = max(1, n_ticks // 10)
    for i in range(n_ticks):
        world.tick()

        # Mimic PrivilegedExpert's 4x get_actors() per frame
        actors_all = world.get_actors()
        vehicles_list = actors_all.filter("*vehicle*")
        walkers_list = actors_all.filter("*walker*")
        tl_list = actors_all.filter("*traffic_light*")
        stop_list = actors_all.filter("*stop*")

        # Iterate over them like the expert does (access attributes)
        for v in vehicles_list:
            if v.id == vehicle.id:
                continue
            _ = v.get_transform()
            _ = v.get_velocity()
            _ = v.bounding_box

        for w in walkers_list:
            _ = w.get_transform()
            _ = w.get_velocity()

        for tl in tl_list:
            _ = tl.get_transform()
            _ = tl.state

        # Also mimic get_waypoint calls
        ego_wp = carla_map.get_waypoint(vehicle.get_location())
        _ = ego_wp

        del actors_all, vehicles_list, walkers_list, tl_list, stop_list

        if i % log_interval == 0:
            gc.collect()
            rss = get_rss_mb()
            measurements.append((i, rss, time.time()))
            print(f"    tick {i:5d}: RSS={rss} MB")
        if i % 100 == 0:
            avail = get_mem_avail_mb()
            if avail < 3000:
                print(f"    SAFETY ABORT: MemAvail={avail} MB < 3000 MB")
                break

    for ctrl in walker_controllers:
        try:
            ctrl.stop()
            ctrl.destroy()
        except Exception:
            pass
    for w in walkers:
        try:
            w.destroy()
        except Exception:
            pass
    for npc in npcs:
        npc.set_autopilot(False)
        npc.destroy()
    npcs.clear()
    tm.set_synchronous_mode(False)


def test_full_expert_pipeline(
    world: Any, vehicle: Any, client: Any, n_ticks: int, measurements: list[Any], **kwargs: Any
) -> None:
    """T11: Full PrivilegedExpert + cameras + NPCs + walkers (no HDF5).

    This mirrors the actual collect_episode() pipeline minus the HDF5
    writing, to confirm whether the expert agent logic causes the leak.
    """
    from agents.expert.privileged_expert import PrivilegedExpert, ExpertConfig
    from agents.navigation.global_route_planner import GlobalRoutePlanner
    import carla

    bp_lib = world.get_blueprint_library()
    carla_map = world.get_map()
    spawn_points = carla_map.get_spawn_points()

    tm = client.get_trafficmanager(8000)
    tm.set_synchronous_mode(True)
    tm.set_global_distance_to_leading_vehicle(2.5)

    # Spawn NPCs
    npc_bps = bp_lib.filter("vehicle.*")
    npc_bps = [bp for bp in npc_bps if int(bp.get_attribute("number_of_wheels")) == 4]
    npcs = []
    ego_loc = vehicle.get_location()
    available = [sp for sp in spawn_points if sp.location.distance(ego_loc) > 50.0]
    np.random.shuffle(available)
    n_npcs = min(50, len(available))
    for i in range(n_npcs):
        npc_bp = np.random.choice(npc_bps)
        npc = world.try_spawn_actor(npc_bp, available[i])
        if npc is not None:
            npc.set_autopilot(True, tm.get_port())
            npcs.append(npc)
    print(f"    Spawned {len(npcs)} NPC vehicles")

    # Spawn walkers
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
    print(f"    Spawned {len(walkers)} walkers")

    # Cameras
    cameras_raw = spawn_cameras(world, vehicle)
    cam_queues = []
    for cam in cameras_raw:
        q: queue.Queue[Any] = queue.Queue()
        cam.listen(lambda img, qq=q: qq.put(img))
        cam_queues.append((cam, q))
    kwargs["_cameras_re"] = cameras_raw

    # Expert agent
    route_planner = GlobalRoutePlanner(carla_map, sampling_resolution=2.0)
    expert = PrivilegedExpert(vehicle, world, route_planner, ExpertConfig())
    # Random destination
    dest_idx = np.random.randint(len(spawn_points))
    expert.set_destination(vehicle.get_location(), spawn_points[dest_idx].location)
    print(f"    Expert route set to spawn point {dest_idx}")

    # Warm up
    for _ in range(10):
        world.tick()
    for _cam, img_q in cam_queues:
        while not img_q.empty():
            try:
                img_q.get_nowait()
            except queue.Empty:
                break

    log_interval = max(1, n_ticks // 10)
    for i in range(n_ticks):
        if expert.done():
            print(f"    Expert finished route at tick {i}")
            break

        world.tick()

        # Expert step (the key suspect!)
        control, info = expert.run_step(0.05)
        vehicle.apply_control(control)

        # Capture + process images (same as collection code)
        for _cam, q in cam_queues:
            try:
                img = q.get(timeout=2.0)
                raw = np.frombuffer(img.raw_data, dtype=np.uint8)
                raw = raw.reshape(320, 640, 4)
                rgb = raw[:, :, :3][:, :, ::-1].copy()
                del img, raw, rgb
            except queue.Empty:
                pass

        del control, info

        if i % log_interval == 0:
            gc.collect()
            rss = get_rss_mb()
            measurements.append((i, rss, time.time()))
            print(f"    tick {i:5d}: RSS={rss} MB")
        if i % 100 == 0:
            avail = get_mem_avail_mb()
            if avail < 3000:
                print(f"    SAFETY ABORT: MemAvail={avail} MB < 3000 MB")
                break

    # Cleanup
    for cam in cameras_raw:
        cam.stop()
        cam.destroy()
    cameras_raw.clear()
    for ctrl in walker_controllers:
        try:
            ctrl.stop()
            ctrl.destroy()
        except Exception:
            pass
    for w in walkers:
        try:
            w.destroy()
        except Exception:
            pass
    for npc in npcs:
        npc.set_autopilot(False)
        npc.destroy()
    npcs.clear()
    tm.set_synchronous_mode(False)


def test_expert_no_cameras_no_npcs(
    world: Any, vehicle: Any, client: Any, n_ticks: int, measurements: list[Any], **kwargs: Any
) -> None:
    """T12: Expert agent ONLY — no cameras, no NPCs, no walkers.

    Isolates whether the PrivilegedExpert logic + GlobalRoutePlanner
    causes the memory leak independently of sensor data or NPC traffic.
    """
    from agents.expert.privileged_expert import PrivilegedExpert, ExpertConfig
    from agents.navigation.global_route_planner import GlobalRoutePlanner

    carla_map = world.get_map()
    spawn_points = carla_map.get_spawn_points()

    rss_before_grp = get_rss_mb()
    print(f"    RSS before GlobalRoutePlanner: {rss_before_grp} MB")

    route_planner = GlobalRoutePlanner(carla_map, sampling_resolution=2.0)

    rss_after_grp = get_rss_mb()
    print(
        f"    RSS after GlobalRoutePlanner:  {rss_after_grp} MB (+{rss_after_grp - rss_before_grp} MB)"  # noqa: E501
    )

    expert = PrivilegedExpert(vehicle, world, route_planner, ExpertConfig())
    dest_idx = np.random.randint(len(spawn_points))
    expert.set_destination(vehicle.get_location(), spawn_points[dest_idx].location)

    rss_after_expert = get_rss_mb()
    print(
        f"    RSS after expert.set_destination: {rss_after_expert} MB (+{rss_after_expert - rss_after_grp} MB)"  # noqa: E501
    )

    # Warm up
    for _ in range(5):
        world.tick()

    log_interval = max(1, n_ticks // 20)
    for i in range(n_ticks):
        if expert.done():
            print(f"    Expert finished route at tick {i}")
            break

        world.tick()
        control, info = expert.run_step(0.05)
        vehicle.apply_control(control)
        del control, info

        if i % log_interval == 0:
            gc.collect()
            rss = get_rss_mb()
            measurements.append((i, rss, time.time()))
            print(f"    tick {i:5d}: RSS={rss} MB")
        if i % 50 == 0:
            avail = get_mem_avail_mb()
            if avail < 2000:
                print(f"    SAFETY ABORT: MemAvail={avail} MB < 2000 MB")
                break


def test_expert_with_npcs_no_cameras(
    world: Any, vehicle: Any, client: Any, n_ticks: int, measurements: list[Any], **kwargs: Any
) -> None:
    """T13: Expert agent + 50 NPCs + 30 walkers, NO cameras.

    Isolates expert+NPC interaction without sensor data overhead.
    """
    from agents.expert.privileged_expert import PrivilegedExpert, ExpertConfig
    from agents.navigation.global_route_planner import GlobalRoutePlanner
    import carla

    bp_lib = world.get_blueprint_library()
    carla_map = world.get_map()
    spawn_points = carla_map.get_spawn_points()

    tm = client.get_trafficmanager(8000)
    tm.set_synchronous_mode(True)
    tm.set_global_distance_to_leading_vehicle(2.5)

    # Spawn NPCs
    npc_bps = bp_lib.filter("vehicle.*")
    npc_bps = [bp for bp in npc_bps if int(bp.get_attribute("number_of_wheels")) == 4]
    npcs = []
    ego_loc = vehicle.get_location()
    available = [sp for sp in spawn_points if sp.location.distance(ego_loc) > 50.0]
    np.random.shuffle(available)
    n_npcs = min(50, len(available))
    for i in range(n_npcs):
        npc_bp = np.random.choice(npc_bps)
        npc = world.try_spawn_actor(npc_bp, available[i])
        if npc is not None:
            npc.set_autopilot(True, tm.get_port())
            npcs.append(npc)
    print(f"    Spawned {len(npcs)} NPC vehicles")

    # Spawn walkers
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
    print(f"    Spawned {len(walkers)} walkers")

    rss_before_grp = get_rss_mb()
    print(f"    RSS before GlobalRoutePlanner: {rss_before_grp} MB")

    route_planner = GlobalRoutePlanner(carla_map, sampling_resolution=2.0)

    rss_after_grp = get_rss_mb()
    print(
        f"    RSS after GlobalRoutePlanner:  {rss_after_grp} MB (+{rss_after_grp - rss_before_grp} MB)"  # noqa: E501
    )

    expert = PrivilegedExpert(vehicle, world, route_planner, ExpertConfig())
    dest_idx = np.random.randint(len(spawn_points))
    expert.set_destination(vehicle.get_location(), spawn_points[dest_idx].location)
    print(f"    Expert route set to spawn point {dest_idx}")

    for _ in range(5):
        world.tick()

    log_interval = max(1, n_ticks // 20)
    for i in range(n_ticks):
        if expert.done():
            print(f"    Expert finished route at tick {i}")
            break

        world.tick()
        control, info = expert.run_step(0.05)
        vehicle.apply_control(control)
        del control, info

        if i % log_interval == 0:
            gc.collect()
            rss = get_rss_mb()
            measurements.append((i, rss, time.time()))
            print(f"    tick {i:5d}: RSS={rss} MB")
        if i % 50 == 0:
            avail = get_mem_avail_mb()
            if avail < 2000:
                print(f"    SAFETY ABORT: MemAvail={avail} MB < 2000 MB")
                break

    # Cleanup
    for ctrl in walker_controllers:
        try:
            ctrl.stop()
            ctrl.destroy()
        except Exception:
            pass
    for w in walkers:
        try:
            w.destroy()
        except Exception:
            pass
    for npc in npcs:
        npc.set_autopilot(False)
        npc.destroy()
    npcs.clear()
    tm.set_synchronous_mode(False)


def main() -> None:
    parser = argparse.ArgumentParser(description="CARLA memory leak diagnostic")
    parser.add_argument("--host", default="172.20.0.1")
    parser.add_argument("--port", type=int, default=2000)
    parser.add_argument("--town", default="Town04")
    parser.add_argument(
        "--ticks", type=int, default=600, help="Ticks per test (600 = 30s at 20 FPS)"
    )
    parser.add_argument(
        "--test", default="all", help="Run specific test: T1,T2,T3,T4,T5,T6,T7 or 'all'"
    )
    args = parser.parse_args()

    print("CARLA Memory Leak Diagnostic")
    print(f"{'=' * 60}")
    print(f"Host: {args.host}:{args.port}")
    print(f"Town: {args.town}")
    print(f"Ticks per test: {args.ticks}")
    print(f"System RSS: {get_rss_mb()} MB, MemAvail: {get_mem_avail_mb()} MB")

    results = []
    tests_to_run = (
        args.test.upper().split(",")
        if args.test != "all"
        else ["T1", "T2", "T3", "T6", "T7", "T8", "T9", "T10", "T11"]
    )

    if "T6" in tests_to_run:
        r = run_test(
            "T6: Tick timing",
            args.host,
            args.port,
            args.town,
            args.ticks,
            test_no_sensors_only_tick_count,
        )
        results.append(r)

    if "T1" in tests_to_run:
        r = run_test(
            "T1: tick only, no sensors", args.host, args.port, args.town, args.ticks, test_tick_only
        )
        results.append(r)

    if "T2" in tests_to_run:
        r = run_test(
            "T2: sensors spawned, no listen()",
            args.host,
            args.port,
            args.town,
            args.ticks,
            test_sensors_no_listen,
        )
        results.append(r)

    if "T3" in tests_to_run:
        r = run_test(
            "T3: sensors+listen, discard images",
            args.host,
            args.port,
            args.town,
            args.ticks,
            test_sensors_listen_discard,
        )
        results.append(r)

    if "T4" in tests_to_run:
        r = run_test(
            "T4: sensors+listen, ACCUMULATE images",
            args.host,
            args.port,
            args.town,
            min(args.ticks, 300),
            test_sensors_listen_accumulate,
        )
        results.append(r)

    if "T5" in tests_to_run:
        r = run_test(
            "T5: sensors+listen+discard on Town01",
            args.host,
            args.port,
            "Town01",
            args.ticks,
            test_sensors_listen_discard,
        )
        results.append(r)

    if "T7" in tests_to_run:
        r = run_test(
            "T7: single camera buffer reuse",
            args.host,
            args.port,
            args.town,
            args.ticks,
            test_buffer_reuse,
        )
        results.append(r)

    if "T8" in tests_to_run:
        r = run_test(
            "T8: cameras + NPCs + TrafficManager",
            args.host,
            args.port,
            args.town,
            args.ticks,
            test_with_npcs,
        )
        results.append(r)

    if "T9" in tests_to_run:
        r = run_test(
            "T9: NPCs + TrafficManager, NO cameras",
            args.host,
            args.port,
            args.town,
            args.ticks,
            test_with_npcs_no_cameras,
        )
        results.append(r)

    if "T10" in tests_to_run:
        r = run_test(
            "T10: get_actors() x4/frame + 50 NPCs + 30 walkers",
            args.host,
            args.port,
            args.town,
            args.ticks,
            test_get_actors_calls,
        )
        results.append(r)

    if "T11" in tests_to_run:
        r = run_test(
            "T11: full expert pipeline (no HDF5)",
            args.host,
            args.port,
            args.town,
            args.ticks,
            test_full_expert_pipeline,
        )
        results.append(r)

    if "T12" in tests_to_run:
        r = run_test(
            "T12: expert only, no cameras/NPCs",
            args.host,
            args.port,
            args.town,
            args.ticks,
            test_expert_no_cameras_no_npcs,
        )
        results.append(r)

    if "T13" in tests_to_run:
        r = run_test(
            "T13: expert + 50 NPCs + 30 walkers, no cameras",
            args.host,
            args.port,
            args.town,
            args.ticks,
            test_expert_with_npcs_no_cameras,
        )
        results.append(r)

    # Summary
    print(f"\n{'=' * 60}")
    print("SUMMARY")
    print(f"{'=' * 60}")
    print(f"{'Test':<45} {'Growth':>8} {'Rate':>10} {'Verdict':<20}")
    print("-" * 85)
    for r in results:
        if r is None:
            continue
        verdict = "LEAK" if r["rate_mb_s"] > 1.0 else "OK"
        print(f"{r['test']:<45} {r['rss_growth']:>6} MB {r['rate_mb_s']:>8.1f} MB/s  {verdict}")

    print("\nDiagnostic Interpretation:")
    # Build interpretation
    rates = {r["test"].split(":")[0].strip(): r["rate_mb_s"] for r in results if r}

    if "T1" in rates and rates["T1"] > 1.0:
        print("  → T1 leaks: world.tick() itself leaks (C++ client state)")
    elif "T1" in rates:
        print("  → T1 clean: world.tick() alone does NOT leak")

    if "T2" in rates and rates["T2"] > 1.0:
        print("  → T2 leaks: spawning sensors causes leak even without listen()")
        print("    This means the SERVER is streaming data and CLIENT receives/buffers it")
    elif "T2" in rates:
        print("  → T2 clean: server data streaming alone doesn't leak")

    if "T3" in rates and rates["T3"] > 1.0:
        if "T2" in rates and rates["T2"] <= 1.0:
            print("  → T3 leaks but T2 clean: leak is in sensor callback/deserialization")
        else:
            print("  → T3 leaks: full pipeline leaks (sensor data reception + processing)")
    elif "T3" in rates:
        print("  → T3 clean: full pipeline with proper image disposal doesn't leak!")
        print("    If user observes leaks, the issue is likely in their collection code")

    if "T8" in rates and rates["T8"] > 1.0:
        if "T9" in rates and rates["T9"] > 1.0:
            print("  → T8+T9 both leak: TrafficManager / NPC autopilot leaks memory in C++ client!")
            print("    Leak is in TrafficManager::Tick() or NPC state updates, NOT cameras")
        elif "T9" in rates and rates["T9"] <= 1.0:
            print("  → T8 leaks but T9 clean: leak requires cameras + NPCs INTERACTION")
            print("    More sensor streams + TM tick compete for C++ memory")
        else:
            print("  → T8 leaks: NPCs + cameras cause leak")
    elif "T8" in rates:
        print("  → T8 clean: cameras + NPCs + TrafficManager does NOT leak!")
        print("    Leak is likely in user's collection code (expert agent, route planner, etc.)")

    if "T9" in rates and rates["T9"] > 1.0 and ("T8" not in rates or rates["T8"] <= 1.0):
        print("  → T9 leaks alone: pure TrafficManager/NPC memory leak (no cameras needed)")


if __name__ == "__main__":
    main()
