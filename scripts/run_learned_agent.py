"""Run the trained VisionRWKVDriver in a live CARLA simulation.

Requires a running CARLA server (0.9.16) on the specified host/port.

Usage (auto-detect Windows host, Town02):
    python scripts/run_learned_agent.py \
        --checkpoint runs_fast/fastsafe_20260424_093210/checkpoints/best.pt \
        --town Town02

Usage (explicit host, multiple episodes):
    python scripts/run_learned_agent.py \
        --checkpoint runs_fast/fastsafe_20260424_093210/checkpoints/best.pt \
        --host 172.20.0.1 \
        --town Town02 \
        --n-episodes 3 \
        --max-steps 2000

The script reports per-episode:
    - Route completion %
    - Collision count
    - Average speed (km/h)
    - Steps completed
"""

from __future__ import annotations

import argparse
import random
import subprocess
import sys
import time
from typing import List, Tuple

import numpy as np

try:
    import carla
except ImportError:
    print("ERROR: 'carla' package not found.")
    print("Install it or run: pip install carla==0.9.16")
    sys.exit(1)

from agents.learned.learned_agent import LearnedAgent


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def get_windows_host_ip() -> str | None:
    """Return the Windows host IP visible from WSL2 (default gateway)."""
    try:
        result = subprocess.run(
            ["ip", "route", "show", "default"],
            capture_output=True, text=True, check=True,
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
    settings.fixed_delta_seconds = 1.0 / 20.0  # 20 Hz, matching training FPS
    world.apply_settings(settings)

    traffic_manager = client.get_trafficmanager()
    traffic_manager.set_synchronous_mode(True)

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
    """Spawn the ego vehicle at a random spawn point."""
    bp_lib = world.get_blueprint_library()
    ego_bp = bp_lib.find("vehicle.lincoln.mkz_2017")
    if ego_bp is None:
        ego_bp = random.choice(bp_lib.filter("vehicle.*"))
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
) -> Tuple["carla.Actor", List[int]]:
    """Attach a collision sensor and return (sensor, collision_count_list)."""
    bp = world.get_blueprint_library().find("sensor.other.collision")
    sensor = world.spawn_actor(bp, carla.Transform(), attach_to=ego)
    count = [0]
    sensor.listen(lambda _: count.__setitem__(0, count[0] + 1))
    return sensor, count


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
) -> dict:
    """Run one complete episode and return a metrics dict.

    All CARLA actors (ego, sensors, NPCs) are destroyed and world settings
    are restored in the finally block — even on Ctrl+C or exceptions.
    """
    world, original_settings, npc_actors = _setup_world(client, town, n_vehicles, n_walkers)
    ego = _spawn_ego(world)
    collision_sensor, collision_count = _attach_collision_sensor(world, ego)
    agent = LearnedAgent(ego, checkpoint, device=device)

    destination = _random_destination(world, ego)
    agent.set_destination(destination)

    spectator = world.get_spectator()
    speeds = []
    step = 0
    # Capture final ego location before destroy for metrics
    final_ego_loc = ego.get_location()

    print(f"[episode] Starting. Destination: ({destination.x:.1f}, {destination.y:.1f})")

    try:
        for step in range(max_steps):
            world.tick()

            # Spectator chase cam — follows ego from behind and above
            tf = ego.get_transform()
            spectator.set_transform(carla.Transform(
                tf.location + carla.Location(z=10.0) - tf.get_forward_vector() * 10.0,
                carla.Rotation(pitch=-25.0, yaw=tf.rotation.yaw),
            ))

            control = agent.run_step()
            ego.apply_control(control)

            vel = ego.get_velocity()
            speed_kmh = (vel.x**2 + vel.y**2 + vel.z**2) ** 0.5 * 3.6
            speeds.append(speed_kmh)

            final_ego_loc = ego.get_location()
            dist_to_goal = final_ego_loc.distance(destination)
            if dist_to_goal < 5.0:
                print(f"[episode] Destination reached at step {step}!")
                break

            if step % 100 == 0:
                _nav_names = {0: "LEFT", 1: "RIGHT", 2: "STRAIGHT", 3: "LANEFOLLOW",
                              4: "CHGLEFT", 5: "CHGRIGHT"}
                nav_label = _nav_names.get(agent.last_nav_cmd, str(agent.last_nav_cmd))
                print(
                    f"  step={step:>4}  speed={speed_kmh:.1f} km/h  "
                    f"dist_to_goal={dist_to_goal:.0f}m  "
                    f"collisions={collision_count[0]}  "
                    f"steer={agent.last_steer:+.3f}  accel={agent.last_accel:+.3f}  "
                    f"nav={nav_label}"
                )

    finally:
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

    return {
        "steps": step + 1,
        "collisions": collision_count[0],
        "avg_speed_kmh": float(np.mean(speeds)) if speeds else 0.0,
        "reached_goal": final_ego_loc.distance(destination) < 5.0,
        "route_completion_pct": max(
            0.0,
            1.0 - final_ego_loc.distance(destination)
            / max(1.0, destination.distance(final_ego_loc))
        ) * 100,
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
        "--host", default="auto",
        help="CARLA host — 'auto' detects Windows host IP from WSL2 gateway (default: auto)",
    )
    parser.add_argument("--port", type=int, default=2000, help="CARLA server port (default: 2000)")
    parser.add_argument("--town", default="Town02", help="CARLA map to load (default: Town02)")
    parser.add_argument("--n-episodes", type=int, default=1, help="Number of episodes to run")
    parser.add_argument("--max-steps", type=int, default=3000, help="Max steps per episode")
    parser.add_argument("--n-vehicles", type=int, default=50, help="NPC vehicle count")
    parser.add_argument("--n-walkers", type=int, default=30, help="NPC walker count")
    parser.add_argument("--device", default="cuda", help="PyTorch device (default: cuda)")
    args = parser.parse_args()

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
            print(f"\n{'='*60}")
            print(f"Episode {ep + 1} / {args.n_episodes}")
            print(f"{'='*60}")
            result = run_episode(
                client,
                checkpoint=args.checkpoint,
                town=args.town,
                n_vehicles=args.n_vehicles,
                n_walkers=args.n_walkers,
                max_steps=args.max_steps,
                device=args.device,
            )
            all_results.append(result)
            print(f"\n[episode {ep + 1}] Results: {result}")

    except KeyboardInterrupt:
        print("\n[main] Ctrl+C received — episode cleanup already handled above.")

    if all_results:
        print(f"\n{'='*60}")
        print("SUMMARY")
        print(f"{'='*60}")
        print(f"Episodes completed: {len(all_results)}")
        print(f"Avg speed:          {np.mean([r['avg_speed_kmh'] for r in all_results]):.1f} km/h")
        print(f"Total collisions:   {sum(r['collisions'] for r in all_results)}")
        print(f"Goal reached:       {sum(r['reached_goal'] for r in all_results)} / {len(all_results)}")


if __name__ == "__main__":
    main()