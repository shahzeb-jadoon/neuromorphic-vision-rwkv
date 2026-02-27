"""Numerical test of SAT corridor for exact failing scenario."""

import numpy as np


def sat_overlap(poly_a: np.ndarray, poly_b: np.ndarray) -> bool:
    def _project(vertices: np.ndarray, axis: np.ndarray) -> tuple[float, float]:
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


ego_xy = np.array([207.0, 55.5])
yaw_rad = np.radians(180.0)
fwd = np.array([np.cos(yaw_rad), np.sin(yaw_rad)])
right_vec = np.array([-fwd[1], fwd[0]])
half_lane = 2.5
print(f"Ego: {ego_xy}, fwd={fwd}, right={right_vec}")

# Simulate waypoints on a straight road heading 180 deg
waypoints = []
for i in range(24):
    dist = 3.0 + i * 2.0
    center = ego_xy + fwd * dist
    waypoints.append((center, right_vec.copy()))

print(f"WPs: {len(waypoints)}, first={waypoints[0][0]}, last={waypoints[-1][0]}")

# Build corridor segments
prev_center = ego_xy.copy()
segments = []
for centre, _ in waypoints:
    seg_vec = centre - prev_center
    seg_len = np.linalg.norm(seg_vec)
    if seg_len < 1e-6:
        continue
    seg_dir = seg_vec / seg_len
    seg_right = np.array([-seg_dir[1], seg_dir[0]])
    prev_l = prev_center - seg_right * half_lane
    prev_r = prev_center + seg_right * half_lane
    cur_l = centre - seg_right * half_lane
    cur_r = centre + seg_right * half_lane
    segments.append(np.array([prev_l, prev_r, cur_r, cur_l]))
    prev_center = centre.copy()

print(f"Segments: {len(segments)}")

# Vehicle: long=28.6m from ego_bb_center, lat=0
ego_bb_center = ego_xy + fwd * 1.5
v_center = ego_bb_center + fwd * 28.6
v_origin = v_center - fwd * 1.5  # rear axle

cos_y, sin_y = np.cos(yaw_rad), np.sin(yaw_rad)
R = np.array([[cos_y, -sin_y], [sin_y, cos_y]])
cx, cy = 1.5, 0.0
ext_x, ext_y = 2.25, 1.0
local_corners = np.array(
    [
        [cx + ext_x, cy + ext_y],
        [cx + ext_x, cy - ext_y],
        [cx - ext_x, cy - ext_y],
        [cx - ext_x, cy + ext_y],
    ]
)
world_corners = (R @ local_corners.T).T + v_origin
print(f"Vehicle center: {v_center}")
print(f"Vehicle OBB: {world_corners}")
print(f"  x: [{world_corners[:,0].min():.2f}, {world_corners[:,0].max():.2f}]")
print(f"  y: [{world_corners[:,1].min():.2f}, {world_corners[:,1].max():.2f}]")

overlaps = []
for i, seg in enumerate(segments):
    if sat_overlap(seg, world_corners):
        overlaps.append(i)

if overlaps:
    print(f"\nSAT OK: {len(overlaps)} overlapping segments: {overlaps}")
else:
    print("\n!!! SAT FAILED — checking AABB overlaps...")
    for i, seg in enumerate(segments):
        sx = [seg[:, 0].min(), seg[:, 0].max()]
        sy = [seg[:, 1].min(), seg[:, 1].max()]
        vx = [world_corners[:, 0].min(), world_corners[:, 0].max()]
        vy = [world_corners[:, 1].min(), world_corners[:, 1].max()]
        if sx[1] >= vx[0] and vx[1] >= sx[0] and sy[1] >= vy[0] and vy[1] >= sy[0]:
            print(
                f"  Seg {i}: x=[{sx[0]:.1f},{sx[1]:.1f}] "
                f"y=[{sy[0]:.1f},{sy[1]:.1f}] "
                f"AABB=overlap SAT={sat_overlap(seg, world_corners)}"
            )
