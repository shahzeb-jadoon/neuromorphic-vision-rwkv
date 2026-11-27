# Watching simulator sessions

Driving scores miss part of how a car drives: a wobble, an odd line through a junction, late
braking, hesitation. Each simulator session that drives a candidate over a route suite is
watched in the simulator window and recorded, and what is seen goes into
[OBSERVATIONS.md](OBSERVATIONS.md).

## Before a session

A notice goes out before the session starts:

| Field | Example |
| --- | --- |
| Candidate | the expert at a commit, or a learned agent's checkpoint |
| Suite and routes | `configs/routes/suite_v1.json`, all 24 routes |
| Traffic | `cars` |
| Expected duration | about 4 h of wall time for the 24 routes with video and overlays |
| Start time | when the first route loads |
| Command | the exact command line |
| Output | the run's directory under `runs/` |

The expert's reference run took 1.9 h of simulation and 2.0 h of wall time without video or
overlays ([RUNS.md](../RUNS.md#expert-on-route-suite-v1)).

## During a session

```bash
python scripts/run_route_suite.py --suite configs/routes/suite_v1.json --agent expert \
    --out runs/bakeoff/<candidate> --video --follow --overlays
```

- `--follow` keeps the simulator window's view 6 m behind and 3 m above the car, where the chase
  camera is.
- `--overlays` draws the expert's view in the world, as the data collector draws it: the route
  ahead (green), the steering target (blue), the detection corridor (magenta), nearby vehicles
  and walkers by risk (white, yellow, red), stop targets (indigo) and the path driven (orange).
  Cameras see what is drawn in the world, so the chase video shows the overlays, and the runner
  refuses them for a learned agent, whose own cameras would see them.
- `--video` writes `<route id>.mp4`, a frame a tick at 20 fps: second *t* of the video is tick
  20 *t*, the row of `<route id>.csv` with the car's speed, controls and any infraction at that
  moment. Videos stay with the run, to watch again afterwards.

None of the three changes the drive: `town04_1` drove the same, tick for tick, with all three as
without. They cost wall time: `town04_1` took 185 s with all three and 97 s with none.

## Recording an observation

Live, note the route and what happened; afterwards, find the moment in the video and take the
simulation time from it. Each observation is a row of [OBSERVATIONS.md](OBSERVATIONS.md):
candidate, route, simulation time, severity, what was seen, and the video time or a screenshot
saved with the run.

| Severity | Meaning | For example |
| --- | --- | --- |
| 3 | an infraction, or nearly one | a collision, a red light run, stopping within a metre of a car, leaving the lane |
| 2 | driving a person would be faulted for | braking late and hard, a wrong line through a junction, stopping inside a junction, waiting with the way clear |
| 1 | unlike a person's driving, but harmless | a wobble, uneven speed, stopping well before the line |

An observation of severity 3 is checked against the route's scored infractions in
`<route id>.json`: one the score missed, or one it counted that did not happen, is a finding
about the scoring as much as about the driving.
