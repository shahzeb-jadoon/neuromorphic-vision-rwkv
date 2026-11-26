# Closed-loop evaluation

`scripts/run_route_suite.py` drives an agent over a frozen suite of routes in CARLA and scores
each route as the CARLA Leaderboard 2.0 does. The same harness scores the experts and the
learned agents, so they drive under the same conditions.

```bash
python scripts/run_route_suite.py --suite configs/routes/suite_v1.json --agent expert \
    --out runs/closed_loop/expert
python scripts/run_route_suite.py --suite configs/routes/suite_v1.json --agent learned \
    --checkpoint runs/short/short_rwkv7_rest/checkpoints/best.pt --out runs/closed_loop/rest
```

`--routes` drives a subset by id. Each route writes `<id>.csv`, a row a tick (position, speed,
control, route completion, whether the car is outside the lanes, infractions, and for a learned
agent its situation probabilities), and `<id>.json`, its score. With `--video` it also writes
`<id>.mp4`: a 640 × 360 camera 6 m behind and 3 m above the car, a frame a tick at 20 fps, so a
second of video is a second of simulation (about 7 MB for 3 minutes; it adds about 45% to the
wall time). `summary.json` gathers the routes driven. CARLA's host and port come from `.env`.

## The harness

`src/eval/closed_loop.py` owns the world. For each route it loads the town, sets the weather,
ticks synchronously at 20 Hz, spawns the car the expert recorded with (`vehicle.tesla.model3`)
at the route's start, then the traffic: vehicles on the traffic manager and walkers with an AI
controller, from seeded draws. The car settles for 10 ticks, as the collector lets it, and the
route is planned through the suite's keypoints at the leaderboard's 1 m spacing
(`agents/navigation/route.py`), the same plan the agent receives. An agent implements
`set_route(keypoints)`, `run_step(frame)` and `destroy()`.

## Scoring

`src/eval/driving_score.py` implements the leaderboard's rules on plain per-tick values, so they
are tested against hand-worked examples (`tests/test_driving_score.py`). Constants come from
`leaderboard/utils/statistics_manager.py` and scenario_runner's `atomic_criteria.py` and
`timer.py`, branch `leaderboard-2.0`.

- **Route completion**: the share of the route's length up to the farthest of the next two route
  points the car has passed; 100% once over 99% is passed within 10 m of the end.
- **Infraction penalty**: the product of 0.50 per collision with a pedestrian, 0.60 with a
  vehicle, 0.65 with anything static, 0.70 per red light and 0.80 per stop sign, times
  (1 − the share of the distance driven outside the lanes or against the lane's direction).
- **Driving score**: route completion × infraction penalty.
- A collision counts once: not again with the same actor within 5 s, nor within 5 m of the last
  counted one, nor while the car moves slower than 0.1 m/s.
- A red light counts when the stretch behind the car crosses the light's stop line on its lane,
  going the lane's way, with the light red and within 15 m. The leaderboard draws that line
  across the lane where the junction begins, which in Town01 lies 6.5 to 13 m beyond CARLA's
  stop waypoint.
- A stop sign counts when the car leaves its zone without having gone below 0.1 m/s.
- The route ends early when the car is 30 m from the route, or has passed 30% of it more than
  15 m away (deviation); after 180 s below 0.1 m/s (blocked); or when the time runs out: 300 s
  plus, for each metre of progress, 1 / (10% of the speed limit) seconds. An early end keeps the
  completion and penalty reached; only the status says why it ended.
- The summary averages the routes and gives each infraction per km driven (route length ×
  completion), and the share of routes that deviated, were blocked or timed out.

The leaderboard's minimum-speed criterion, which compares the car's speed with the background
traffic's, is not scored.

## Route suite v1

`configs/routes/suite_v1.json` holds 24 routes, 21.7 km in all: four in each of Town01, Town02,
Town03, Town04, Town05 and Town10HD. Town02 was never recorded, so its routes are held out.
`scripts/make_route_suite.py` picked them from 1,000 planned routes per town with seed 0: each
is 500 to 1,200 m long, turns at two junctions or more and starts and ends outside junctions;
a town's routes are picked one at a time, each covering the most road the earlier picks did not.
A route is stored as keypoints about every 50 m outside junctions, each on the lane the map
gives back for its location, and kept only if it plans again to the same length through them;
the harness refuses to drive a route whose plan differs from the recorded length by over 5 m.

| | Routes |
| --- | --- |
| Vehicles 20 / 50 / 100 (walkers 10 / 30 / 50) | 8 / 9 / 7 (Town02 at most 50) |
| Weather | the 10 presets v3 was recorded in, in turn |
| Junction turns | 2 to 10 a route |
| Lane changes the route asks for | none in Town01 and Town02; 1 to 3 in 14 of the 16 others |
| Traffic lights on the route | 1 to 9 a route |

Each route records its length, turns, lane changes and lights, so results can be split by them.
The model of Phase 2 has no lane-change command, so routes that need one test it beyond what it
was trained for. A committed suite is never edited; a change is a new version.
