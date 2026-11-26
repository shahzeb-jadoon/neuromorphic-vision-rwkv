"""Run the trained VisionRWKVDriver in a live CARLA simulation.

Requires a running CARLA server (0.9.16) on the specified host/port. The world ticks
synchronously at 20 Hz, the rate the expert recorded at; the agent steps the model at the
rate it trained at.

Usage (auto-detect Windows host, Town01, 500 ticks, per-tick log):
    python scripts/run_learned_agent.py \
        --checkpoint runs/short/short_rwkv7/checkpoints/best.pt \
        --town Town01 --max-steps 500 --seed 0 --log runs/drive/town01.csv

The script reports per episode the route completion (share of the planned route's length
driven), collisions, average speed, ticks and the time `run_step` took on the ticks where the
model stepped. `--log` writes one row per tick.
"""

from __future__ import annotations

import argparse
import csv
import random
import subprocess
import sys
import time
from pathlib import Path
from typing import List, Tuple

import numpy as np

try:
    import carla
except ImportError:
    print("ERROR: 'carla' package not found.")
    print("Install it or run: pip install carla==0.9.16")
    sys.exit(1)

from agents.learned.learned_agent import LearnedAgent
from src.data.labels import CAUSES

SETTLE_TICKS = 10  # ticks the expert's collector lets a new car settle before it records


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def get_windows_host_ip() -> str | None:
    """Return the Windows host IP visible from WSL2 (default gateway)."""
    try:
        result = subprocess.run(
            ["ip", "route", "show", "default"],
            capture_output=True,
            text=True,
            check=True,
        )
        return result.stdout.strip().split()[2]
    except (subprocess.CalledProcessError, IndexError, FileNotFoundError):
        return None


def _safe_destroy(actor: "carla.Actor") -> None:
    """Destroy a CARLA actor, ignoring errors (actor may already be gone)."""
    try:
        actor.destroy()
    except Exception:
        pass


def _setup_world(
    client: carla.Client,
    town: str,
    n_vehicles: int,
    n_walkers: int,
    seed: int,
) -> Tuple["carla.World", "carla.WorldSettings", List["carla.Actor"]]:
    """Load the town, enable sync mode, and spawn NPC traffic.

    Returns:
        (world, original_settings, npc_actors)
        Caller is responsible for destroying npc_actors and restoring
        original_settings on exit.
    """
    world = client.load_world(town)
    time.sleep(2.0)  # let the map finish loading
    world = client.get_world()

    original_settings = world.get_settings()

    settings = world.get_settings()
    settings.synchronous_mode = True
    settings.fixed_delta_seconds = 1.0 / 20.0  # 20 Hz, as the expert recorded
    world.apply_settings(settings)

    traffic_manager = client.get_trafficmanager()
    traffic_manager.set_synchronous_mode(True)
    traffic_manager.set_random_device_seed(seed)

    bp_lib = world.get_blueprint_library()
    npc_actors: List[carla.Actor] = []

    # Spawn NPC vehicles
    vehicle_bps = bp_lib.filter("vehicle.*")
    vehicle_bps = [b for b in vehicle_bps if int(b.get_attribute("number_of_wheels")) == 4]
    spawn_points = world.get_map().get_spawn_points()
    random.shuffle(spawn_points)

    for sp in spawn_points[:n_vehicles]:
        bp = random.choice(vehicle_bps)
        actor = world.try_spawn_actor(bp, sp)
        if actor:
            actor.set_autopilot(True, traffic_manager.get_port())
            npc_actors.append(actor)

    # Spawn NPC walkers (simplified — no AI controller for demo)
    walker_bps = bp_lib.filter("walker.pedestrian.*")
    for _ in range(n_walkers):
        sp = carla.Transform(world.get_random_location_from_navigation())
        bp = random.choice(walker_bps)
        if bp.has_attribute("is_invincible"):
            bp.set_attribute("is_invincible", "false")
        actor = world.try_spawn_actor(bp, sp)
        if actor:
            npc_actors.append(actor)

    print(f"[setup] Spawned {len(npc_actors)} NPC actors in {town}.")
    return world, original_settings, npc_actors


def _spawn_ego(world: carla.World) -> carla.Vehicle:
    """Spawn the ego vehicle at a random spawn point: the model the expert recorded with
    (src/data/expert_collector.py), whose throttle and steering responses the model learned."""
    ego_bp = world.get_blueprint_library().find("vehicle.tesla.model3")
    ego_bp.set_attribute("role_name", "hero")

    spawn_points = world.get_map().get_spawn_points()
    for sp in random.sample(spawn_points, len(spawn_points)):
        ego = world.try_spawn_actor(ego_bp, sp)
        if ego:
            return ego

    raise RuntimeError("Could not spawn ego vehicle — no free spawn points.")


def _random_destination(
    world: carla.World, ego: carla.Vehicle, min_dist: float = 100.0
) -> carla.Location:
    """Pick a random spawn point at least min_dist metres from the ego."""
    ego_loc = ego.get_location()
    candidates = [
        sp.location
        for sp in world.get_map().get_spawn_points()
        if sp.location.distance(ego_loc) >= min_dist
    ]
    if not candidates:
        candidates = [sp.location for sp in world.get_map().get_spawn_points()]
    return random.choice(candidates)


def _attach_collision_sensor(
    world: carla.World, ego: carla.Vehicle
) -> Tuple["carla.Actor", List[str]]:
    """Attach a collision sensor and return it with the list it fills: the type of the actor
    hit, once per collision event."""
    bp = world.get_blueprint_library().find("sensor.other.collision")
    sensor = world.spawn_actor(bp, carla.Transform(), attach_to=ego)
    hits: List[str] = []
    sensor.listen(lambda event: hits.append(event.other_actor.type_id))
    return sensor, hits


# ---------------------------------------------------------------------------
# Single episode
# ---------------------------------------------------------------------------


def run_episode(
    client: carla.Client,
    checkpoint: str,
    town: str,
    n_vehicles: int,
    n_walkers: int,
    max_steps: int,
    device: str,
    seed: int,
    log: Path | None,
) -> dict:
    """Run one complete episode and return a metrics dict.

    All CARLA actors (ego, sensors, NPCs) are destroyed and world settings
    are restored in the finally block — even on Ctrl+C or exceptions.
    """
    world, original_settings, npc_actors = _setup_world(client, town, n_vehicles, n_walkers, seed)
    ego = _spawn_ego(world)
    for _ in range(SETTLE_TICKS):  # until the first tick, a new actor reads location (0, 0, 0)
        world.tick()
    collision_sensor, hits = _attach_collision_sensor(world, ego)
    agent = LearnedAgent(ego, checkpoint, device=device)

    destination = _random_destination(world, ego)
    agent.set_destination(destination)

    spectator = world.get_spectator()
    speeds = []
    rows: List[dict] = []
    hits_logged = 0
    step = 0
    # Capture final ego location before destroy for metrics
    final_ego_loc = ego.get_location()

    print(f"[episode] Starting. Destination: ({destination.x:.1f}, {destination.y:.1f})")

    try:
        for step in range(max_steps):
            frame = world.tick()

            # Spectator chase cam — follows ego from behind and above
            tf = ego.get_transform()
            spectator.set_transform(
                carla.Transform(
                    tf.location + carla.Location(z=10.0) - tf.get_forward_vector() * 10.0,
                    carla.Rotation(pitch=-25.0, yaw=tf.rotation.yaw),
                )
            )

            start = time.perf_counter()
            control = agent.run_step(frame)
            run_step_ms = (time.perf_counter() - start) * 1e3
            ego.apply_control(control)

            vel = ego.get_velocity()
            speed_kmh = (vel.x**2 + vel.y**2 + vel.z**2) ** 0.5 * 3.6
            speeds.append(speed_kmh)
            new_hits = hits[hits_logged:]
            hits_logged += len(new_hits)
            rows.append(
                {
                    "tick": step,
                    "frame": frame,
                    "model_stepped": int(agent.stepped),
                    "run_step_ms": round(run_step_ms, 3),
                    "camera_ms": round(agent.camera_ms, 3) if agent.stepped else "",
                    "model_ms": round(agent.model_ms, 3) if agent.stepped else "",
                    "speed_kmh": round(speed_kmh, 3),
                    "steer": round(control.steer, 4),
                    "throttle": round(control.throttle, 4),
                    "brake": round(control.brake, 4),
                    "nav_command": agent.last_nav_cmd,
                    "route_completion": round(agent.route_completion, 4),
                    "collisions": hits_logged,
                    "hit": ";".join(sorted(set(new_hits))),
                    # the light CARLA says governs the ego vehicle: logged, never a model input
                    "traffic_light": str(ego.get_traffic_light_state())
                    if ego.is_at_traffic_light()
                    else "",
                    **{
                        f"p_{cause.replace(' ', '_')}": round(p, 3)
                        for cause, p in zip(CAUSES, agent.situation)
                    },
                }
            )

            final_ego_loc = ego.get_location()
            dist_to_goal = final_ego_loc.distance(destination)
            if dist_to_goal < 5.0:
                print(f"[episode] Destination reached at step {step}!")
                break

            if step % 100 == 0:
                _nav_names = {
                    0: "LEFT",
                    1: "RIGHT",
                    2: "STRAIGHT",
                    3: "LANEFOLLOW",
                    4: "CHGLEFT",
                    5: "CHGRIGHT",
                }
                nav_label = _nav_names.get(agent.last_nav_cmd, str(agent.last_nav_cmd))
                print(
                    f"  step={step:>4}  speed={speed_kmh:.1f} km/h  "
                    f"dist_to_goal={dist_to_goal:.0f}m  "
                    f"collisions={len(hits)}  "
                    f"steer={agent.last_steer:+.3f}  accel={agent.last_accel:+.3f}  "
                    f"nav={nav_label}"
                )

    finally:
        route_completion = agent.route_completion
        print("[cleanup] Stopping agent and sensors...")
        agent.destroy()

        try:
            collision_sensor.stop()
        except Exception:
            pass
        _safe_destroy(collision_sensor)

        print("[cleanup] Destroying ego...")
        _safe_destroy(ego)

        print(f"[cleanup] Destroying {len(npc_actors)} NPC actors...")
        for actor in npc_actors:
            # Off the traffic manager first, whose thread would otherwise act on destroyed actors
            if isinstance(actor, carla.Vehicle):
                try:
                    actor.set_autopilot(False)
                except Exception:
                    pass
            _safe_destroy(actor)

        # Reset traffic manager to async mode
        try:
            client.get_trafficmanager().set_synchronous_mode(False)
        except Exception:
            pass

        # Restore world settings (async mode, original delta)
        print("[cleanup] Restoring world settings...")
        try:
            world.apply_settings(original_settings)
        except Exception:
            pass

        print("[cleanup] Done.")

    if log is not None and rows:
        log.parent.mkdir(parents=True, exist_ok=True)
        with log.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    stepped_ms = [r["run_step_ms"] for r in rows if r["model_stepped"]]
    return {
        "steps": step + 1,
        "collisions": len(hits),
        "hit": sorted(set(hits)),
        "avg_speed_kmh": float(np.mean(speeds)) if speeds else 0.0,
        "reached_goal": final_ego_loc.distance(destination) < 5.0,
        "route_completion_pct": 100 * route_completion,
        "model_step_ms_p50_p95_max": [
            round(float(v), 1) for v in np.percentile(stepped_ms, [50, 95, 100])
        ]
        if stepped_ms
        else [],
    }


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--checkpoint", required=True, help="Path to .pt checkpoint file")
    parser.add_argument(
        "--host",
        default="auto",
        help="CARLA host — 'auto' detects Windows host IP from WSL2 gateway (default: auto)",
    )
    parser.add_argument("--port", type=int, default=2000, help="CARLA server port (default: 2000)")
    parser.add_argument("--town", default="Town02", help="CARLA map to load (default: Town02)")
    parser.add_argument("--n-episodes", type=int, default=1, help="Number of episodes to run")
    parser.add_argument("--max-steps", type=int, default=3000, help="Max steps per episode")
    parser.add_argument("--n-vehicles", type=int, default=50, help="NPC vehicle count")
    parser.add_argument("--n-walkers", type=int, default=30, help="NPC walker count")
    parser.add_argument("--device", default="cuda", help="PyTorch device (default: cuda)")
    parser.add_argument("--seed", type=int, default=0, help="Seed for spawns and traffic")
    parser.add_argument("--log", type=Path, help="Per-tick CSV (episode number appended)")
    args = parser.parse_args()
    random.seed(args.seed)

    # Resolve host
    if args.host == "auto":
        host = get_windows_host_ip()
        if host is None:
            print("[main] Could not auto-detect Windows host IP — falling back to localhost.")
            host = "localhost"
        else:
            print(f"[main] Auto-detected Windows host IP: {host}")
    else:
        host = args.host

    client = carla.Client(host, args.port)
    client.set_timeout(30.0)
    print(f"[main] Connected to CARLA {client.get_server_version()} at {host}:{args.port}")

    all_results = []
    try:
        for ep in range(args.n_episodes):
            print(f"\n{'=' * 60}")
            print(f"Episode {ep + 1} / {args.n_episodes}")
            print(f"{'=' * 60}")
            result = run_episode(
                client,
                checkpoint=args.checkpoint,
                town=args.town,
                n_vehicles=args.n_vehicles,
                n_walkers=args.n_walkers,
                max_steps=args.max_steps,
                device=args.device,
                seed=args.seed,
                log=args.log.with_stem(f"{args.log.stem}_{ep + 1}") if args.log else None,
            )
            all_results.append(result)
            print(f"\n[episode {ep + 1}] Results: {result}")

    except KeyboardInterrupt:
        print("\n[main] Ctrl+C received — episode cleanup already handled above.")

    if all_results:
        print(f"\n{'=' * 60}")
        print("SUMMARY")
        print(f"{'=' * 60}")
        print(f"Episodes completed: {len(all_results)}")
        print(f"Avg speed:          {np.mean([r['avg_speed_kmh'] for r in all_results]):.1f} km/h")
        print(f"Total collisions:   {sum(r['collisions'] for r in all_results)}")
        reached = sum(r["reached_goal"] for r in all_results)
        print(f"Goal reached:       {reached} / {len(all_results)}")


if __name__ == "__main__":
    main()
