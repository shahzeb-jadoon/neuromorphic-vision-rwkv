#!/usr/bin/env python3
# mypy: ignore-errors
"""Query CARLA to inventory stop signs, yield signs, and roundabouts per town.

Usage:
    conda activate rwkv
    python scripts/carla_town_inventory.py

Requires CARLA server running. Reports:
  - Stop signs: pole-mounted (red sign) vs road-painted
  - Yield signs
  - Roundabouts (speed_limit actors or custom logic)
  - Traffic lights (for reference)
"""

import sys
import time
import subprocess


# Auto-detect CARLA host from WSL2 gateway
def get_carla_host():
    try:
        result = subprocess.run(["ip", "route", "show", "default"], capture_output=True, text=True)
        gateway = result.stdout.split()[2]
        return gateway
    except Exception:
        return "localhost"


def main():
    # Add CARLA egg to path
    sys.path.insert(0, "~/carla/PythonAPI/carla")
    try:
        import carla
    except ImportError:
        print("ERROR: Cannot import carla. Check CARLA PythonAPI path.")
        sys.exit(1)

    host = get_carla_host()
    print(f"Connecting to CARLA at {host}:2000...")
    client = carla.Client(host, 2000)
    client.set_timeout(30.0)

    towns = ["Town01", "Town03", "Town04", "Town05", "Town06", "Town07", "Town10HD"]

    print(f"\n{'=' * 80}")
    print("CARLA Town Sign & Feature Inventory")
    print(f"{'=' * 80}\n")

    summary = []

    for town in towns:
        print(f"--- Loading {town} ---")
        try:
            client.load_world(town)
        except Exception as e:
            print(f"  ERROR loading {town}: {e}")
            summary.append({"town": town, "error": str(e)})
            continue

        time.sleep(3)  # Let world settle
        world = client.get_world()

        # Get all actors by type
        actors = world.get_actors()

        # --- Stop Signs ---
        stop_signs = actors.filter("traffic.stop")
        stop_sign_list = list(stop_signs)

        pole_mounted = []
        road_painted = []
        for ss in stop_sign_list:
            loc = ss.get_location()

            # Heuristic: pole-mounted signs have their trigger volume
            # significantly offset from the sign (trigger on road surface)
            # Road-painted signs have trigger volume close to the sign location
            # Also: pole-mounted signs typically have z > 1.5m
            if loc.z > 1.0:
                pole_mounted.append(ss)
            else:
                road_painted.append(ss)

        # --- Yield Signs ---
        yield_signs = list(actors.filter("traffic.yield"))

        # --- Traffic Lights ---
        traffic_lights = list(actors.filter("traffic.traffic_light"))

        # --- Speed Limit Signs (may indicate roundabouts) ---
        speed_signs = list(actors.filter("traffic.speed_limit*"))

        # --- All traffic signs for reference ---
        all_traffic = list(actors.filter("traffic.*"))
        other_signs = [
            a
            for a in all_traffic
            if a.type_id not in {s.type_id for s in stop_sign_list}
            and a.type_id not in {s.type_id for s in yield_signs}
            and a.type_id not in {s.type_id for s in traffic_lights}
            and a.type_id not in {s.type_id for s in speed_signs}
        ]

        # Collect unique other sign types
        other_types = {}
        for a in other_signs:
            other_types[a.type_id] = other_types.get(a.type_id, 0) + 1

        info = {
            "town": town,
            "stop_total": len(stop_sign_list),
            "stop_pole": len(pole_mounted),
            "stop_road": len(road_painted),
            "yield": len(yield_signs),
            "traffic_lights": len(traffic_lights),
            "speed_signs": len(speed_signs),
            "other_types": other_types,
        }
        summary.append(info)

        print(f"  Stop signs: {len(stop_sign_list)} total")
        print(f"    Pole-mounted (z > 1m): {len(pole_mounted)}")
        print(f"    Road-level  (z <= 1m): {len(road_painted)}")
        for ss in stop_sign_list:
            loc = ss.get_location()
            tv = ss.trigger_volume
            print(
                f"      ID={ss.id} pos=({loc.x:.1f}, {loc.y:.1f}, {loc.z:.1f}) "
                f"TV_extent=({tv.extent.x:.1f}, {tv.extent.y:.1f}, {tv.extent.z:.1f})"
            )

        print(f"  Yield signs: {len(yield_signs)}")
        for ys in yield_signs:
            loc = ys.get_location()
            print(f"      ID={ys.id} pos=({loc.x:.1f}, {loc.y:.1f}, {loc.z:.1f})")

        print(f"  Traffic lights: {len(traffic_lights)}")
        print(f"  Speed limit signs: {len(speed_signs)}")
        if speed_signs:
            speed_types = {}
            for s in speed_signs:
                speed_types[s.type_id] = speed_types.get(s.type_id, 0) + 1
            for st, cnt in sorted(speed_types.items()):
                print(f"    {st}: {cnt}")

        if other_types:
            print("  Other traffic signs:")
            for tid, cnt in sorted(other_types.items()):
                print(f"    {tid}: {cnt}")

        print()

    # --- Summary Table ---
    print(f"\n{'=' * 80}")
    print("SUMMARY TABLE")
    print(f"{'=' * 80}")
    print(f"{'Town':<12} {'Stop(pole)':<12} {'Stop(road)':<12} {'Yield':<8} {'TL':<8} {'Speed':<8}")
    print(f"{'-' * 12} {'-' * 12} {'-' * 12} {'-' * 8} {'-' * 8} {'-' * 8}")
    for info in summary:
        if "error" in info:
            print(f"{info['town']:<12} ERROR: {info['error']}")
        else:
            print(
                f"{info['town']:<12} {info['stop_pole']:<12} {info['stop_road']:<12} "
                f"{info['yield']:<8} {info['traffic_lights']:<8} {info['speed_signs']:<8}"
            )

    print("\nNote: Roundabouts are identified by road topology, not actors.")
    print("Known roundabout towns: Town03 (large central), Town07 (multiple small)")
    print("\nPole-mounted = red octagonal stop sign on a post (z > 1m)")
    print("Road-level = stop text painted on road surface (z <= 1m)")


if __name__ == "__main__":
    main()
