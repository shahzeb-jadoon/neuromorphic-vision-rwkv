# Training targets

How a sample's targets are computed from an episode's raw signals, in
`src/data/labels.py`, `src/data/schema_adapter.py` and `src/data/driving_dataset.py`.

## Per frame

| Target | Definition |
| --- | --- |
| `controls` | Steering in [−1, 1] and acceleration = throttle − brake in [−1, 1], as the expert applied them |
| `nav_command` | RoadOption 1–6 minus one: LEFT 0, RIGHT 1, STRAIGHT 2, LANEFOLLOW 3, CHANGELANELEFT 4, CHANGELANERIGHT 5. Any other value is an error, never clamped |
| `future_waypoints` | Positions 5, 10, 15 and 20 frames ahead, in the vehicle's frame at that frame: x forward, y to the right, metres. Offsets count frames, so they reach 0.25–1 s ahead in 20 Hz recordings and 0.5–2 s in the 10 Hz testbed cache |
| `target_speed` | The expert's target speed divided by 120 km/h |
| `driving_mode` | The expert's mode 1–8 minus one; the v6 YieldStop is folded into StopSignStop |
| `loss_weight` | 0.1 for a stationary frame, otherwise 1 (below) |

## Frames used

Clips are taken only from an episode's usable frames:

- **Collision**: frames from 3 s before the collision on are dropped, since the expert was
  already failing. Some collisions are other vehicles hitting the stopped ego car, where
  this drops good frames.
- **Timeout**: the final stop is dropped. The collector ends an episode once the vehicle
  has stood below 0.1 m/s for 75–120 s of wall-clock time, so these frames show it stuck
  rather than driving. In v3 the stop lasts 26–43 s (10th to 90th percentile).
- **Completed** episodes are used whole, including those that end stopped at a light when
  the frame budget ran out.
- A **golden-subset excerpt** is used whole: its status describes its source episode.

On the v3 dataset these rules drop 5.3% of frames: 183,966 stuck frames and 9,021 around
collisions.

## Stationary weighting

A frame whose horizontal speed is below 0.14 m/s (0.5 km/h) has loss weight 0.1. Standing
frames are a third (33.0%) of the usable v3 testbed frames, and their targets barely vary:
no steering, braking, waypoints at the vehicle. At 0.1 they make up 4.7% of the total
weight, enough to learn to stay stopped without letting the easiest frames dominate the
loss and the validation metric.

## Mirroring

During training half of the clips are mirrored left to right (`mirror` in
`driving_dataset.py`): the panorama is flipped, which also swaps the side cameras;
steering and waypoint y change sign; LEFT and RIGHT swap, as do the two lane changes. A
mirrored scene shows left-hand traffic.
