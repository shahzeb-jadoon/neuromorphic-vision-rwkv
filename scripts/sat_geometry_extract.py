#!/usr/bin/env python3
"""Extract exact geometry from HDF5 data for SAT miss cases.

For each SAT miss at close range, reconstruct:
  - Ego position and heading
  - Corridor start point (ego_loc)
  - Vehicle position (from ego + offset)
  - Expected corridor coverage

This helps us understand EXACTLY why SAT fails geometrically.
"""

import sys
import json
import math

from typing import Any

import h5py
import numpy as np


def sat_overlap(poly_a: np.ndarray, poly_b: np.ndarray) -> bool:
    """Exact copy of the SAT function from privileged_expert.py."""

    def _project(vertices: np.ndarray, axis: np.ndarray) -> tuple[Any, Any]:
        dots = vertices @ axis
        return dots.min(), dots.max()

    for poly in (poly_a, poly_b):
        n = len(poly)
        for i in range(n):
            edge = poly[(i + 1) % n] - poly[i]
            axis = np.array([-edge[1], edge[0]])
            norm = np.linalg.norm(axis)
            if norm < 1e-8:
                continue
            axis /= norm
            min_a, max_a = _project(poly_a, axis)
            min_b, max_b = _project(poly_b, axis)
            if max_a < min_b or max_b < min_a:
                return False
    return True


def analyze_episode(path: str, max_cases: int = 5) -> int:
    """Analyze SAT misses in one episode, return count of cases printed."""
    cases_printed = 0
    with h5py.File(path, "r") as f:
        n = f["images"].shape[0]
        pos = f["position"][:]  # (N, 3) — x, y, z
        rot = f["rotation"][:]  # (N, 3) — pitch, yaw, roll
        segs_arr = f["n_corridor_segments"][:]
        buf_head = f["wp_buffer_head_dist"][:]

        ep_name = path.split("/")[-1].replace(".h5", "")

        for i in range(n):
            if cases_printed >= max_cases:
                break

            vlist = json.loads(f["nearby_vehicles"][i])

            # Find a close-range SAT miss
            for v in vlist:
                if not v.get("is_ahead", False):
                    continue
                if not v.get("same_direction", True):
                    continue
                if v.get("obb_in_lane", False):
                    continue  # SAT hit, skip
                if abs(v.get("lateral", 999)) >= 2.5:
                    continue  # not in lateral range
                if v.get("distance", 999) >= 15:
                    continue  # only close-range

                # Found a SAT miss case
                ego_x, ego_y, ego_z = pos[i]
                pitch, yaw, roll = rot[i]

                # Ego forward and right vectors
                yaw_rad = math.radians(yaw)
                fwd = np.array([math.cos(yaw_rad), math.sin(yaw_rad)])
                right = np.array([-fwd[1], fwd[0]])

                lat = v["lateral"]
                longi = v["longitudinal"]
                dist = v["distance"]
                n_segs = int(segs_arr[i])
                bh = float(buf_head[i])

                # Reconstruct approximate vehicle center position
                # (relative to ego_loc, not ego_bb_center)
                # lateral and longitudinal are measured from ego_bb_center
                # ego_bb_center ≈ ego_loc + fwd * bb_offset (typically ~1.5m)
                bb_offset = 1.5  # approximate
                ego_bb_x = ego_x + fwd[0] * bb_offset
                ego_bb_y = ego_y + fwd[1] * bb_offset

                v_center_x = ego_bb_x + fwd[0] * longi + right[0] * lat
                v_center_y = ego_bb_y + fwd[1] * longi + right[1] * lat

                # Print case details
                print(f"\n{'=' * 70}")
                print(f"SAT MISS CASE — {ep_name} frame {i}")
                print(f"{'=' * 70}")
                print(f"  Ego position:  ({ego_x:.1f}, {ego_y:.1f})")
                print(f"  Ego yaw:       {yaw:.1f}° → fwd=({fwd[0]:.3f}, {fwd[1]:.3f})")
                print(f"  Ego BB center: ({ego_bb_x:.1f}, {ego_bb_y:.1f})")
                print(f"  Vehicle center (approx): ({v_center_x:.1f}, {v_center_y:.1f})")
                print(f"  Longitudinal:  {longi:.1f}m (from ego BB center)")
                print(f"  Lateral:       {lat:.2f}m (+ = right)")
                print(f"  Bumper dist:   {dist:.1f}m")
                print(f"  Corridor segs: {n_segs}")
                print(f"  Buffer head:   {bh:.1f}m")

                # Now simulate what the corridor would look like
                # on a STRAIGHT road from ego_loc
                print("\n  --- Simulated straight-road corridor ---")
                print(f"  Corridor start: ({ego_x:.1f}, {ego_y:.1f})")
                half_lane = 2.5

                # If the road were straight, waypoints at 2m spacing
                # starting at buf_head distance
                wp_positions = []
                for wi in range(n_segs):
                    wp_dist = bh + wi * 2.0
                    wp_x = ego_x + fwd[0] * wp_dist
                    wp_y = ego_y + fwd[1] * wp_dist
                    wp_positions.append((wp_x, wp_y))

                # Build corridor segments (straight road case)
                prev = np.array([ego_x, ego_y])
                straight_segs = []
                for wx, wy in wp_positions:
                    curr = np.array([wx, wy])
                    seg_vec = curr - prev
                    seg_len = np.linalg.norm(seg_vec)
                    if seg_len < 1e-6:
                        continue
                    seg_dir = seg_vec / seg_len
                    seg_right = np.array([-seg_dir[1], seg_dir[0]])
                    pl = prev - seg_right * half_lane
                    pr = prev + seg_right * half_lane
                    cl = curr - seg_right * half_lane
                    cr = curr + seg_right * half_lane
                    straight_segs.append(np.array([pl, pr, cr, cl]))
                    prev = curr

                # Create a synthetic vehicle OBB
                # Typical CARLA car: ext_x=2.3, ext_y=1.0
                v_ext_x = 2.3
                v_ext_y = 1.0
                cos_y, sin_y = math.cos(yaw_rad), math.sin(yaw_rad)
                # Vehicle corners (assuming same heading as ego)
                v_corners = np.array(
                    [
                        [
                            v_center_x + cos_y * v_ext_x - sin_y * v_ext_y,
                            v_center_y + sin_y * v_ext_x + cos_y * v_ext_y,
                        ],
                        [
                            v_center_x + cos_y * v_ext_x + sin_y * v_ext_y,
                            v_center_y + sin_y * v_ext_x - cos_y * v_ext_y,
                        ],
                        [
                            v_center_x - cos_y * v_ext_x + sin_y * v_ext_y,
                            v_center_y - sin_y * v_ext_x - cos_y * v_ext_y,
                        ],
                        [
                            v_center_x - cos_y * v_ext_x - sin_y * v_ext_y,
                            v_center_y - sin_y * v_ext_x + cos_y * v_ext_y,
                        ],
                    ]
                )

                # Test SAT on straight-road corridor
                any_hit = False
                hit_segs = []
                for si, seg in enumerate(straight_segs):
                    if sat_overlap(seg, v_corners):
                        any_hit = True
                        hit_segs.append(si)

                print(f"  Straight-road SAT result: {'HIT' if any_hit else 'MISS'}")
                if any_hit:
                    print(f"  Overlapping segments: {hit_segs}")
                else:
                    # Find closest segment to vehicle
                    v_mid = np.array([v_center_x, v_center_y])
                    min_dist = float("in")
                    closest_seg = -1
                    for si, seg in enumerate(straight_segs):
                        seg_center = seg.mean(axis=0)
                        d = float(np.linalg.norm(seg_center - v_mid))
                        if d < min_dist:
                            min_dist = d
                            closest_seg = si
                    if closest_seg >= 0:
                        seg = straight_segs[closest_seg]
                        sc = seg.mean(0)
                        print(f"  Closest seg {closest_seg}: center=({sc[0]:.1f}, {sc[1]:.1f})")
                        print(f"    Seg corners: {seg.tolist()}")
                        print(f"    Vehicle corners: {v_corners.tolist()}")

                # KEY QUESTION: The straight-road test above should HIT.
                # If it does, the issue is that the ACTUAL corridor
                # (which we can't reconstruct offline) curves away.
                # If it DOESN'T, there's a fundamental bug.

                # Draw ASCII diagram
                print("\n  --- ASCII top-down view (X→, Y↓) ---")
                print("  (Note: CARLA coords, +Y = right when facing +X)")
                print("  Scale: 1 char ≈ 2m")
                print()

                # Simplified: just show relative positions
                print(f"  Vehicle at ({longi:.0f}m ahead, {lat:.1f}m lateral)")
                print(f"  First WP at {bh:.1f}m")
                print(f"  Corridor width: ±{half_lane}m = {half_lane * 2}m total")
                v_longi_from_ego = longi + bb_offset  # from ego_loc
                print(f"  Vehicle center at ~{v_longi_from_ego:.0f}m from ego_loc")

                wp_index = int((v_longi_from_ego - bh) / 2.0)
                print(f"  Expected in corridor segment ~{wp_index} (of {n_segs})")

                cases_printed += 1
                break  # one case per frame

    return cases_printed


def main() -> None:
    import glob

    data_dir = sys.argv[1] if len(sys.argv) > 1 else "/mnt/data/carla_data"
    max_episodes = int(sys.argv[2]) if len(sys.argv) > 2 else 10
    max_cases = int(sys.argv[3]) if len(sys.argv) > 3 else 3

    paths = sorted(glob.glob(f"{data_dir}/episode_*.h5"))

    # Find episodes with SAT misses
    total_cases = 0
    for path in paths[:max_episodes]:
        n = analyze_episode(path, max_cases=max_cases - total_cases)
        total_cases += n
        if total_cases >= max_cases:
            break

    if total_cases == 0:
        print("No close-range SAT misses found in sampled episodes.")


if __name__ == "__main__":
    main()
