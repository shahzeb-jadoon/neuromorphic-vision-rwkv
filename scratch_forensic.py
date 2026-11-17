import h5py
import json
import carla

episode_file = "data/test_collection/episode_88f3d598.h5"
route_file = "data/test_collection/episode_88f3d598.route.json"

# Load H5
frames = [3650, 3735, 3739, 3748, 3752, 3763, 3776, 3783, 3790, 3798, 3830, 3831, 3850]
with h5py.File(episode_file, "r") as h:
    town = h["metadata"].attrs.get("town", "Unknown")
    positions = h["position"][:]
    yaws = h["rotation"][:, 1]
    head_indices = h["wp_buffer_head_index"][:]

print(f"Town: {town}")

# Load Route
with open(route_file, "r") as f:
    route_data = json.load(f)["waypoints"]

# Connect to CARLA
client = carla.Client("localhost", 2000)
client.set_timeout(10.0)
world = client.get_world()
carla_map = world.get_map()

print(f"CARLA Map: {carla_map.name}")
if carla_map.name.split("/")[-1] != town:
    print("Warning: CARLA map does not match episode town!")

print("--- Forensic Data ---")
for frame in frames:
    ridx = head_indices[frame]
    if ridx < 0 or ridx >= len(route_data):
        continue

    pos = positions[frame]
    live_loc = carla.Location(x=float(pos[0]), y=float(pos[1]), z=float(pos[2]))
    live_wp = carla_map.get_waypoint(
        live_loc, project_to_road=True, lane_type=carla.LaneType.Driving
    )

    rwp = route_data[ridx]
    route_loc = carla.Location(x=float(rwp["x"]), y=float(rwp["y"]), z=float(rwp["z"]))
    route_wp = carla_map.get_waypoint(
        route_loc, project_to_road=True, lane_type=carla.LaneType.Driving
    )

    live_id = live_wp.lane_id if live_wp else None
    live_road = live_wp.road_id if live_wp else None
    route_id = route_wp.lane_id if route_wp else None
    route_road = route_wp.road_id if route_wp else None

    print(f"Frame {frame:4d}:")
    print(f"  Live Loc : x={pos[0]:.2f}, y={pos[1]:.2f}")
    live_is_junction = live_wp.is_junction if live_wp else False
    print(f"  Live WP  : road={live_road}, lane={live_id}, is_junc={live_is_junction}")
    print(f"  Route idx: {ridx}")
    print(f"  Route Loc: x={rwp['x']:.2f}, y={rwp['y']:.2f}")
    route_is_junction = route_wp.is_junction if route_wp else False
    stored_lane = rwp.get("lane_id")
    print(
        "  Route WP : "
        f"road={route_road}, lane={route_id}, stored_lane={stored_lane}, "
        f"is_junc={route_is_junction}"
    )
