"""Drive an agent along one route in CARLA and score it as the CARLA leaderboard does.

The harness owns the world: town, weather, seeded traffic, the car at the route's start and
the route planned through its keypoints. An agent only turns each tick into a control, so
experts and learned agents drive under the same conditions. Scoring is src.eval.driving_score.
"""

from __future__ import annotations

import csv
import json
import math
import queue
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Protocol

import carla
import numpy as np

from agents.navigation.global_route_planner import GlobalRoutePlanner
from agents.navigation.route import plan_route
from src.data.labels import ROUTE_RESOLUTION
from src.eval import driving_score as D

FPS = 20  # the rate the expert recorded at
SETTLE_TICKS = 10  # ticks a new car settles before the agent's first step, as the collector does
EGO_BLUEPRINT = "vehicle.tesla.model3"  # the car the expert recorded with
WALKER_SPEED = 1.4  # m/s
CLEAR_OF_EGO = 10.0  # m: no traffic spawns closer to the car's start
ROUTE_MISMATCH = 5.0  # m a planned route may differ in length from the one the suite recorded
# scenario_runner RunningStopTest
STOP_PROXIMITY = 4.0  # m from a stop sign to the car's lane point
STOP_LOOKAHEAD = 8  # lane points, 0.5 m apart, ahead of the car checked against its zone
STOP_ZONE_SCALE = 1.2
AGAINST_LANE = -0.17  # cosine below which the car reverses or drives against its lane


@dataclass(frozen=True)
class Route:
    id: str
    town: str
    keypoints: tuple[tuple[float, float, float], ...]  # x, y, z in CARLA's frame
    seed: int
    vehicles: int
    walkers: int
    weather: str  # a carla.WeatherParameters preset
    length_m: float = 0.0  # as planned when the suite was made; 0 if not recorded
    turns: int = 0
    lane_changes: int = 0
    lights: int = 0


def load_suite(path: Path) -> list[Route]:
    routes = json.loads(path.read_text())["routes"]
    return [Route(**{**r, "keypoints": tuple(tuple(p) for p in r["keypoints"])}) for r in routes]


class Agent(Protocol):
    def set_route(self, keypoints: list[Any]) -> None: ...

    def run_step(self, frame: int) -> Any: ...

    def destroy(self) -> None: ...


def _xy(location: Any) -> np.ndarray:
    return np.array([location.x, location.y])


def _rotate(x: float, y: float, yaw_deg: float) -> tuple[float, float]:
    c, s = math.cos(math.radians(yaw_deg)), math.sin(math.radians(yaw_deg))
    return c * x - s * y, s * x + c * y


def red_light_lines(world: Any) -> tuple[list[D.StopLine], dict[int, np.ndarray]]:
    """Each traffic light's stop lines and trigger-zone centre, built as scenario_runner's
    RunningRedLightTest does: lane points across the zone, advanced to the junction."""
    town = world.get_map()
    lines, centres = [], {}
    for light in world.get_actors().filter("traffic.traffic_light"):
        base = light.get_transform()
        centre = base.transform(light.trigger_volume.location)
        centres[light.id] = _xy(centre)
        extent = light.trigger_volume.extent
        starts: list[Any] = []
        for x in np.arange(-0.9 * extent.x, 0.9 * extent.x, 1.0):
            dx, dy = _rotate(float(x), 0.0, base.rotation.yaw)
            wp = town.get_waypoint(carla.Location(centre.x + dx, centre.y + dy, centre.z))
            if not starts or (starts[-1].road_id, starts[-1].lane_id) != (wp.road_id, wp.lane_id):
                starts.append(wp)
        for wp in starts:
            while not wp.is_intersection:
                ahead = wp.next(0.5)
                if not ahead or ahead[0].is_intersection:
                    break
                wp = ahead[0]
            yaw, half = wp.transform.rotation.yaw, 0.6 * wp.lane_width
            at = _xy(wp.transform.location)
            left, right = np.array(_rotate(half, 0, yaw + 90)), np.array(_rotate(half, 0, yaw - 90))
            forward = np.array(_rotate(1.0, 0.0, yaw))
            lines.append(
                D.StopLine(light.id, at + left, at + right, wp.road_id, wp.lane_id, forward)
            )
    return lines, centres


def _inside(point: Any, centre: Any, extent: Any, yaw_deg: float, scale: float) -> bool:
    x, y = _rotate(point.x - centre.x, point.y - centre.y, -yaw_deg)
    return bool(abs(x) <= extent.x * scale and abs(y) <= extent.y * scale)


def _stop_signs_ahead(town: Any, stops: list[Any], ego: Any) -> tuple[set[int], int | None]:
    """Stop signs whose zone holds the car's lane just ahead, and the one to watch if the car
    drives with its lane (scenario_runner RunningStopTest)."""
    transform = ego.get_transform()
    wp = town.get_waypoint(transform.location)
    path = [wp]
    for _ in range(STOP_LOOKAHEAD):
        ahead = path[-1].next(0.5)
        if not ahead:
            break
        path.append(ahead[0])
    affecting = set()
    for stop, centre, extent, yaw in stops:
        if stop.get_location().distance(wp.transform.location) > STOP_PROXIMITY:
            continue
        if any(_inside(p.transform.location, centre, extent, yaw, STOP_ZONE_SCALE) for p in path):
            affecting.add(stop.id)
    forward, velocity = transform.get_forward_vector(), ego.get_velocity()
    with_lane = (
        velocity.dot(forward) >= AGAINST_LANE
        and forward.dot(wp.transform.get_forward_vector()) >= AGAINST_LANE
    )
    return affecting, (min(affecting) if affecting and with_lane else None)


def _lane_place(town: Any, location: Any) -> D.LanePlace:
    """The driving lane under the car; distance and width of the nearer driving or parking lane."""
    driving = town.get_waypoint(location, project_to_road=True, lane_type=carla.LaneType.Driving)
    parking = town.get_waypoint(location, project_to_road=True, lane_type=carla.LaneType.Parking)
    near = driving
    if parking is not None and location.distance(parking.transform.location) < location.distance(
        driving.transform.location
    ):
        near = parking
    return D.LanePlace(
        driving.road_id,
        driving.lane_id,
        driving.is_junction,
        driving.transform.rotation.yaw,
        location.distance(near.transform.location),
        near.lane_width,
    )


class _World:
    """The route's town, weather and traffic, and the car at the route's start."""

    def __init__(self, client: Any, route: Route) -> None:
        self.client = client
        self.world = client.load_world(route.town)
        self.original = self.world.get_settings()
        settings = self.world.get_settings()
        settings.synchronous_mode, settings.fixed_delta_seconds = True, 1.0 / FPS
        self.world.apply_settings(settings)
        self.tm = client.get_trafficmanager()
        self.tm.set_synchronous_mode(True)
        self.tm.set_random_device_seed(route.seed)
        self.world.set_pedestrians_seed(route.seed)
        self.world.set_weather(getattr(carla.WeatherParameters, route.weather))
        self.actors: list[Any] = []
        self.controllers: list[Any] = []
        library = self.world.get_blueprint_library()

        start = self.world.get_map().get_waypoint(carla.Location(*route.keypoints[0])).transform
        start.location.z += 0.5
        ego_bp = library.find(EGO_BLUEPRINT)
        ego_bp.set_attribute("role_name", "hero")
        self.ego = self.world.spawn_actor(ego_bp, start)
        try:
            self._traffic(route, library, start)
        except BaseException:
            self.close()
            raise

    def _traffic(self, route: Route, library: Any, start: Any) -> None:
        rng = random.Random(route.seed)
        cars = sorted(
            (
                b
                for b in library.filter("vehicle.*")
                if int(b.get_attribute("number_of_wheels")) == 4
            ),
            key=lambda b: b.id,
        )
        spawns = [
            s
            for s in self.world.get_map().get_spawn_points()
            if s.location.distance(start.location) > CLEAR_OF_EGO
        ]
        rng.shuffle(spawns)
        for spawn in spawns[: route.vehicles]:
            bp = rng.choice(cars)
            bp.set_attribute("role_name", "background")
            car = self.world.try_spawn_actor(bp, spawn)
            if car is not None:
                car.set_autopilot(True, self.tm.get_port())
                self.actors.append(car)

        people = sorted(library.filter("walker.pedestrian.*"), key=lambda b: b.id)
        walkers = []
        for _ in range(route.walkers):
            spot = self.world.get_random_location_from_navigation()
            bp = rng.choice(people)
            if bp.has_attribute("is_invincible"):
                bp.set_attribute("is_invincible", "false")
            walker = self.world.try_spawn_actor(bp, carla.Transform(spot + carla.Location(z=1.0)))
            if walker is not None:
                walkers.append(walker)
        self.world.tick()
        brain = library.find("controller.ai.walker")
        for walker in walkers:
            self.controllers.append(self.world.spawn_actor(brain, carla.Transform(), walker))
        self.world.tick()
        for controller in self.controllers:
            controller.start()
            controller.go_to_location(self.world.get_random_location_from_navigation())
            controller.set_max_speed(WALKER_SPEED)
        self.actors += walkers
        for _ in range(SETTLE_TICKS):
            self.world.tick()

    def close(self) -> None:
        for controller in self.controllers:
            controller.stop()
        for actor in self.actors:
            if isinstance(actor, carla.Vehicle):
                actor.set_autopilot(False)  # the traffic manager would act on destroyed actors
        everyone = self.controllers + self.actors + [self.ego]
        self.client.apply_batch_sync([carla.command.DestroyActor(a) for a in everyone])
        self.tm.set_synchronous_mode(False)
        self.world.apply_settings(self.original)


def run_route(
    client: Any,
    route: Route,
    make_agent: Callable[[Any], Agent],
    out_dir: Path,
    telemetry: Callable[[Agent], dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Drive one route; write <id>.csv (a row a tick) and <id>.json (its score) to out_dir."""
    scene = _World(client, route)
    world, ego, town = scene.world, scene.ego, scene.world.get_map()
    sensor, agent = None, None
    try:
        keypoints = [carla.Location(*p) for p in route.keypoints]
        keypoints[0] = ego.get_location()
        planned = plan_route(GlobalRoutePlanner(town, ROUTE_RESOLUTION), keypoints)
        points = np.array([_xy(wp.transform.location) for wp, _ in planned])
        forwards = np.array([_rotate(1.0, 0.0, wp.transform.rotation.yaw) for wp, _ in planned])
        progress, deviation = D.RouteProgress(points, forwards), D.RouteDeviation(points)
        if route.length_m and abs(progress.length - route.length_m) > ROUTE_MISMATCH:
            raise ValueError(
                f"{route.id}: planned {progress.length:.1f} m, the suite says {route.length_m} m"
            )
        timeout, blocked, outside = D.RouteTimeout(), D.Blocked(), D.OutsideLanes()
        collisions, stop_signs = D.CollisionFilter(), D.StopSigns()
        lines, light_centres = red_light_lines(world)
        red_lights = D.RedLights(lines)
        lights = {light.id: light for light in world.get_actors().filter("traffic.traffic_light")}
        stops = [
            (
                s,
                s.get_transform().transform(s.trigger_volume.location),
                s.trigger_volume.extent,
                s.get_transform().rotation.yaw,
            )
            for s in world.get_actors().filter("traffic.stop")
        ]
        hits: queue.Queue[tuple[int, str]] = queue.Queue()
        sensor = world.spawn_actor(
            world.get_blueprint_library().find("sensor.other.collision"), carla.Transform(), ego
        )
        sensor.listen(lambda e: hits.put((e.other_actor.id, e.other_actor.type_id)))
        agent = make_agent(ego)
        agent.set_route(keypoints)
        half_length = ego.bounding_box.extent.x

        infractions: list[D.Infraction] = []
        rows: list[dict[str, Any]] = []
        step_ms: list[float] = []
        failure, tick, wall = None, 0, time.perf_counter()
        while failure is None and not progress.done:
            frame = world.tick()
            elapsed = tick / FPS
            transform = ego.get_transform()
            position, yaw = _xy(transform.location), transform.rotation.yaw
            v = ego.get_velocity()
            speed = math.hypot(v.x, v.y)
            gained = progress.update(position)
            bad_lane = outside.update(_lane_place(town, transform.location), yaw, gained)
            events = []

            collisions.update(elapsed, position)
            while not hits.empty():
                other_id, type_id = hits.get()
                kind = collisions.count(elapsed, position, speed, other_id, type_id)
                if kind:
                    events.append(
                        D.Infraction(kind, tick, float(position[0]), float(position[1]), type_id)
                    )
            red = {
                i: c
                for i, c in light_centres.items()
                if np.linalg.norm(c - position) <= D.RED_LIGHT_DISTANCE
                and lights[i].state == carla.TrafficLightState.Red
            }
            if red:
                tail = transform.location - transform.get_forward_vector() * (half_length + 1.0)
                tail_wp = town.get_waypoint(tail)
                run = red_lights.update(
                    position, yaw, half_length, (tail_wp.road_id, tail_wp.lane_id), red
                )
                if run is not None:
                    events.append(
                        D.Infraction(
                            "red_light", tick, float(position[0]), float(position[1]), str(run)
                        )
                    )
            affecting, ahead = _stop_signs_ahead(town, stops, ego) if stops else (set(), None)
            ran = stop_signs.update(affecting, ahead, speed)
            if ran is not None:
                events.append(
                    D.Infraction(
                        "stop_sign", tick, float(position[0]), float(position[1]), str(ran)
                    )
                )
            infractions += events

            if deviation.update(position):
                failure = "Agent deviated from the route"
            elif blocked.update(elapsed, speed):
                failure = "Agent got blocked"
            elif timeout.update(elapsed, gained, ego.get_speed_limit()):
                failure = "Agent timed out"

            start = time.perf_counter()
            control = agent.run_step(frame)
            step_ms.append((time.perf_counter() - start) * 1e3)
            ego.apply_control(control)
            rows.append(
                {
                    "tick": tick,
                    "frame": frame,
                    "x": round(float(position[0]), 2),
                    "y": round(float(position[1]), 2),
                    "yaw": round(yaw, 1),
                    "speed_kmh": round(speed * 3.6, 2),
                    "steer": round(control.steer, 4),
                    "throttle": round(control.throttle, 4),
                    "brake": round(control.brake, 4),
                    "completion": round(progress.completion, 2),
                    "outside_lanes": int(bad_lane),
                    "infraction": ";".join(e.kind for e in events),
                    **(telemetry(agent) if telemetry else {}),
                }
            )
            tick += 1
        wall = time.perf_counter() - wall
    finally:
        if sensor is not None:
            sensor.stop()
            sensor.destroy()
        if agent is not None:
            agent.destroy()
        scene.close()

    score = D.score_route(progress.completion, infractions, outside.percentage, failure)
    result = {
        "route": asdict(route),
        "length_m": round(progress.length, 1),
        "ticks": tick,
        "sim_s": round(tick / FPS, 2),
        "wall_s": round(wall, 1),
        "avg_speed_kmh": round(float(np.mean([r["speed_kmh"] for r in rows])) if rows else 0.0, 2),
        "run_step_ms_p50_p95": [round(float(x), 1) for x in np.percentile(step_ms, [50, 95])]
        if step_ms
        else [],
        "score": asdict(score),
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    with (out_dir / f"{route.id}.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]) if rows else ["tick"])
        writer.writeheader()
        writer.writerows(rows)
    (out_dir / f"{route.id}.json").write_text(json.dumps(result, indent=1))
    return result
