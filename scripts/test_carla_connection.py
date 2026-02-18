"""Test CARLA server in synchronous mode with camera capture.

Validates the full pipeline: connect → sync mode → spawn vehicle → attach
camera → capture frames → save sample → cleanup.

Prerequisites:
    1. CARLA 0.9.16 server running on Windows host
    2. conda activate rwkv

Usage:
    python scripts/test_carla_connection.py [--frames 100] [--save-dir data/test_frames]
"""

import argparse
import queue
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

try:
    import carla
except ImportError:
    print("ERROR: carla package not installed. Run: pip install carla==0.9.16")
    sys.exit(1)

try:
    from PIL import Image
except ImportError:
    Image = None  # type: ignore[assignment, misc]


def get_windows_host_ip() -> str | None:
    """Get the Windows host IP from WSL2's default gateway."""
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


def main() -> None:
    parser = argparse.ArgumentParser(description="Test CARLA synchronous mode with camera")
    parser.add_argument("--frames", type=int, default=100, help="Number of ticks to run")
    parser.add_argument("--save-dir", type=str, default="data/test_frames", help="Save directory")
    parser.add_argument("--host", type=str, default="auto", help="CARLA host (auto=detect WSL2)")
    parser.add_argument("--port", type=int, default=2000, help="CARLA port")
    args = parser.parse_args()

    # Detect host
    if args.host == "auto":
        host = get_windows_host_ip()
        if host is None:
            print("Could not auto-detect Windows host IP. Trying localhost.")
            host = "localhost"
    else:
        host = args.host

    print(f"{'='*60}")
    print("CARLA Synchronous Mode Test")
    print(f"{'='*60}")
    print(f"Host:           {host}:{args.port}")
    print(f"Frames to run:  {args.frames}")
    print()

    # Connect
    client = carla.Client(host, args.port)
    client.set_timeout(30.0)
    print(f"Server version: {client.get_server_version()}")
    print(f"Client version: {client.get_client_version()}")

    world = client.get_world()
    original_settings = world.get_settings()

    actors_to_destroy = []
    image_queue: queue.Queue = queue.Queue()

    try:
        # === Enable synchronous mode ===
        settings = world.get_settings()
        settings.synchronous_mode = True
        settings.fixed_delta_seconds = 0.05  # 20 FPS
        world.apply_settings(settings)
        print("Synchronous mode: ENABLED (20 FPS)")

        # === Spawn ego vehicle ===
        blueprint_library = world.get_blueprint_library()
        vehicle_bp = blueprint_library.filter("vehicle.tesla.model3")[0]
        spawn_points = world.get_map().get_spawn_points()

        if not spawn_points:
            print("ERROR: No spawn points available on current map")
            sys.exit(1)

        vehicle = world.try_spawn_actor(vehicle_bp, spawn_points[0])
        if vehicle is None:
            print("ERROR: Failed to spawn vehicle")
            sys.exit(1)

        actors_to_destroy.append(vehicle)
        print(f"Vehicle spawned: {vehicle.type_id} (id={vehicle.id})")

        # === Attach RGB camera (224×224) ===
        camera_bp = blueprint_library.find("sensor.camera.rgb")
        camera_bp.set_attribute("image_size_x", "224")
        camera_bp.set_attribute("image_size_y", "224")
        camera_bp.set_attribute("fov", "100")

        camera_transform = carla.Transform(carla.Location(x=1.5, z=2.4))
        camera = world.spawn_actor(camera_bp, camera_transform, attach_to=vehicle)
        actors_to_destroy.append(camera)
        print("Camera attached: 224×224, FOV=100°")

        # Camera callback → queue
        camera.listen(lambda image: image_queue.put(image))

        # === Enable autopilot for driving ===
        vehicle.set_autopilot(True)
        print("Autopilot: ENABLED")
        print()

        # === Run simulation ===
        print(f"Running {args.frames} ticks...")
        saved_frame = None

        for tick_num in range(args.frames):
            world.tick()

            # Process camera image
            try:
                image = image_queue.get(timeout=2.0)
                array = np.frombuffer(image.raw_data, dtype=np.uint8)
                array = array.reshape(image.height, image.width, 4)[:, :, :3]  # BGRA → BGR → RGB
                array = array[:, :, ::-1].copy()  # BGR → RGB

                if tick_num == 0:
                    saved_frame = array

                if (tick_num + 1) % 25 == 0:
                    control = vehicle.get_control()
                    velocity = vehicle.get_velocity()
                    speed = (velocity.x**2 + velocity.y**2 + velocity.z**2) ** 0.5
                    print(
                        f"  Tick {tick_num + 1:4d}/{args.frames}: "
                        f"steer={control.steer:+.3f}, "
                        f"throttle={control.throttle:.3f}, "
                        f"speed={speed:.1f} m/s, "
                        f"frame={array.shape}"
                    )

            except queue.Empty:
                print(f"  WARNING: No camera image at tick {tick_num + 1}")

        # === Save a sample frame ===
        if saved_frame is not None:
            save_dir = Path(args.save_dir)
            save_dir.mkdir(parents=True, exist_ok=True)
            save_path = save_dir / "test_frame_000.png"

            if Image is not None:
                Image.fromarray(saved_frame).save(save_path)
                print(f"\nSample frame saved: {save_path} ({saved_frame.shape})")
            else:
                np.save(save_path.with_suffix(".npy"), saved_frame)
                print(
                    f"\nSample frame saved: {save_path.with_suffix('.npy')} (Pillow not available)"
                )

        print()
        print("CARLA synchronous mode test: PASSED ✓")

    except KeyboardInterrupt:
        print("\nInterrupted by user")

    except Exception as e:
        print(f"\nERROR: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)

    finally:
        # === Cleanup ===
        print("\nCleaning up...")
        camera.stop()
        for actor in reversed(actors_to_destroy):
            actor.destroy()

        # Restore original settings
        world.apply_settings(original_settings)
        print("Original settings restored")
        print("Done!")


if __name__ == "__main__":
    main()
