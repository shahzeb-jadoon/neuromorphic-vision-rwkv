#!/usr/bin/env python3
"""Raw bounding box diagnostic — prints exact BB values and manual corner computation.

This answers: WHY are the OBB corners displaced 333m from the vehicle?
Prints raw bb.location, bb.extent, transform, and step-by-step corner math.
"""

import math
import subprocess
import sys

try:
    import carla
except ImportError:
    print("ERROR: carla not installed")
    sys.exit(1)


def get_windows_host_ip() -> str | None:
    try:
        result = subprocess.run(
            ["ip", "route", "show", "default"],
            capture_output=True,
            text=True,
            check=True,
        )
        return result.stdout.strip().split()[2]
    except Exception:
        return None


def main() -> None:
    host = get_windows_host_ip() or "localhost"
    print(f"Connecting to {host}:2000 ...")
    client = carla.Client(host, 2000)
    client.set_timeout(30.0)
    world = client.get_world()
    carla_map = world.get_map()
    original_settings = world.get_settings()
    actors = []

    try:
        settings = world.get_settings()
        settings.synchronous_mode = True
        settings.fixed_delta_seconds = 0.05
        world.apply_settings(settings)

        bp_lib = world.get_blueprint_library()
        vehicle_bp = bp_lib.filter("vehicle.tesla.model3")[0]
        spawn_points = carla_map.get_spawn_points()
        sp = spawn_points[0]

        # Spawn single vehicle
        vehicle = world.spawn_actor(vehicle_bp, sp)
        actors.append(vehicle)
        print("Vehicle spawned at spawn point 0")

        # Settle physics — try different amounts
        for settle_ticks in [1, 5, 10, 20, 50]:
            for _ in range(
                max(
                    1,
                    settle_ticks
                    - (
                        0
                        if settle_ticks == 1
                        else (
                            [1, 5, 10, 20][[1, 5, 10, 20, 50].index(settle_ticks) - 1]
                            if settle_ticks > 1
                            else 0
                        )
                    ),
                )
            ):
                world.tick()

            t = vehicle.get_transform()
            bb = vehicle.bounding_box

            print(f"\n{'='*70}")
            print(f"AFTER {settle_ticks} TICKS")
            print(f"{'='*70}")

            print("  Transform:")
            print(f"    location: ({t.location.x:.4f}, {t.location.y:.4f}, {t.location.z:.4f})")
            print(
                f"    rotation: (pitch={t.rotation.pitch:.4f}, yaw={t.rotation.yaw:.4f}, "
                f"roll={t.rotation.roll:.4f})"
            )

            print("  BoundingBox:")
            print(
                f"    bb.location: ({bb.location.x:.4f}, {bb.location.y:.4f}, {bb.location.z:.4f})"
            )
            print(f"    bb.extent:   ({bb.extent.x:.4f}, {bb.extent.y:.4f}, {bb.extent.z:.4f})")
            print(
                f"    bb.rotation: (pitch={bb.rotation.pitch:.4f}, yaw={bb.rotation.yaw:.4f}, "
                f"roll={bb.rotation.roll:.4f})"
            )

            # BB center in world
            bb_center_world = t.transform(bb.location)
            print(
                f"  BB center (world): ({bb_center_world.x:.4f}, {bb_center_world.y:.4f}, "
                f"{bb_center_world.z:.4f})"
            )

            # Manual corner computation (what _get_actor_obb_2d does)
            cx, cy, cz = bb.location.x, bb.location.y, bb.location.z
            ex, ey = bb.extent.x, bb.extent.y
            local_corners_3d = [
                carla.Location(x=cx + ex, y=cy + ey, z=cz),
                carla.Location(x=cx + ex, y=cy - ey, z=cz),
                carla.Location(x=cx - ex, y=cy - ey, z=cz),
                carla.Location(x=cx - ex, y=cy + ey, z=cz),
            ]
            print("\n  Local corners (cx±ex, cy±ey, cz):")
            for i, lc in enumerate(local_corners_3d):
                print(f"    {i}: ({lc.x:.4f}, {lc.y:.4f}, {lc.z:.4f})")

            print("\n  World corners (via t.transform):")
            for i, lc in enumerate(local_corners_3d):
                wc = t.transform(lc)
                print(f"    {i}: ({wc.x:.4f}, {wc.y:.4f}, {wc.z:.4f})")

            # Manual rotation math for verification
            yaw_rad = math.radians(t.rotation.yaw)
            pitch_rad = math.radians(t.rotation.pitch)
            roll_rad = math.radians(t.rotation.roll)
            cos_y = math.cos(yaw_rad)
            sin_y = math.sin(yaw_rad)
            cos_p = math.cos(pitch_rad)
            sin_p = math.sin(pitch_rad)
            cos_r = math.cos(roll_rad)
            sin_r = math.sin(roll_rad)

            print(f"\n  Manual rotation check (yaw={t.rotation.yaw:.2f}°):")
            print(f"    cos(yaw)={cos_y:.4f} sin(yaw)={sin_y:.4f}")
            print(f"    cos(pitch)={cos_p:.4f} sin(pitch)={sin_p:.4f}")
            print(f"    cos(roll)={cos_r:.4f} sin(roll)={sin_r:.4f}")

            # UE4 rotation matrix (yaw-pitch-roll order, left-handed)
            # See: https://docs.unrealengine.com/4.27/en-US/API/Runtime/Core/Math/FRotator/
            # The UE4 rotation order is: Rz(yaw) * Ry(pitch) * Rx(roll)
            # For simplicity, compute just the yaw rotation (pitch and roll should be ~0)
            for i, lc in enumerate(local_corners_3d):
                # Yaw-only rotation (around Z axis)
                rx = lc.x * cos_y - lc.y * sin_y
                ry = lc.x * sin_y + lc.y * cos_y
                rz = lc.z
                wx = rx + t.location.x
                wy = ry + t.location.y
                wz = rz + t.location.z
                print(f"    Corner {i} manual: ({wx:.4f}, {wy:.4f}, {wz:.4f})")

            # Also try: what if CARLA uses the OPPOSITE yaw convention?
            print("\n  Alt rotation (negative yaw):")
            cos_y_neg = math.cos(-yaw_rad)
            sin_y_neg = math.sin(-yaw_rad)
            for i, lc in enumerate(local_corners_3d):
                rx = lc.x * cos_y_neg - lc.y * sin_y_neg
                ry = lc.x * sin_y_neg + lc.y * cos_y_neg
                wx = rx + t.location.x
                wy = ry + t.location.y
                print(f"    Corner {i} alt: ({wx:.4f}, {wy:.4f})")

        # Also check a traffic manager vehicle if any exist
        print(f"\n{'='*70}")
        print("CHECKING OTHER VEHICLES IN WORLD")
        print(f"{'='*70}")
        for v in world.get_actors().filter("*vehicle*"):
            if v.id == vehicle.id:
                continue
            vt = v.get_transform()
            vbb = v.bounding_box
            print(f"\n  Vehicle {v.id} ({v.type_id}):")
            print(f"    pos: ({vt.location.x:.2f}, {vt.location.y:.2f}) yaw={vt.rotation.yaw:.1f}°")
            print(f"    bb.loc: ({vbb.location.x:.4f}, {vbb.location.y:.4f}, {vbb.location.z:.4f})")
            print(f"    bb.ext: ({vbb.extent.x:.4f}, {vbb.extent.y:.4f}, {vbb.extent.z:.4f})")

            # Compute OBB corners
            cx, cy, cz = vbb.location.x, vbb.location.y, vbb.location.z
            ex, ey = vbb.extent.x, vbb.extent.y
            local_corners = [
                carla.Location(x=cx + ex, y=cy + ey, z=cz),
                carla.Location(x=cx + ex, y=cy - ey, z=cz),
                carla.Location(x=cx - ex, y=cy - ey, z=cz),
                carla.Location(x=cx - ex, y=cy + ey, z=cz),
            ]
            bb_center = vt.transform(vbb.location)
            print(f"    bb center world: ({bb_center.x:.2f}, {bb_center.y:.2f})")
            for i, lc in enumerate(local_corners):
                wc = vt.transform(lc)
                print(f"    corner {i}: ({wc.x:.2f}, {wc.y:.2f})")

    finally:
        for a in actors:
            try:
                a.destroy()
            except Exception:
                pass
        try:
            world.apply_settings(original_settings)
        except Exception:
            pass
        print("\nDone.")


if __name__ == "__main__":
    main()
