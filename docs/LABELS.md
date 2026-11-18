# Training targets

How a sample's targets are computed from an episode's raw signals, and which clips
training draws, in `src/data/labels.py`, `src/data/schema_adapter.py` and
`src/data/driving_dataset.py`.

Every episode is read at 10 frames per second (`dataset.fps`): frame k is recorded row
k × (recording rate / 10), and its targets come from the full-rate signals at that row.
A 20 Hz recording and its 10 Hz training cache therefore give identical targets.

## Per frame

| Target | Definition |
| --- | --- |
| `controls` | Steering in [−1, 1] and acceleration = throttle − brake in [−1, 1], as the expert applied them |
| `nav_command` | RoadOption minus one, derived from the path driven ([below](#navigation-commands)): LEFT 0, RIGHT 1, LANEFOLLOW 3; STRAIGHT (2) and the lane changes (4, 5) do not occur. A recorded command outside 1–6 is still an error, never clamped |
| `future_waypoints` | Positions 5, 10, 15 and 20 frames (0.5–2 s) ahead, in the vehicle's frame at that frame: x forward, y to the right, metres |
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

## Navigation commands

The recorded commands of v3 often end before the turn they announce
([DATA_V3.md](DATA_V3.md#known-limitations)), so training derives the command from the
path the vehicle drove (`path_commands` in `labels.py`). Along the distance travelled, a
turn is a stretch curving one way by at least 1°/m that peaks above 3°/m (a radius under
19 m) and changes the heading by 45° or more: v3's junction turns peak near 6°/m, its road
curves below 2°/m. The command is LEFT or RIGHT from 30 m before a turn until the turn
ends, and LANEFOLLOW elsewhere; turns the same way whose commands would come within 10 m
share one. Measured along the distance travelled, a command holds however long the
vehicle waits before its turn.

Going straight through a junction cannot be told from following the lane, so STRAIGHT
becomes LANEFOLLOW. At driving time the agent has to derive its command the same way,
from the planned route: `path_commands` takes any path with headings.

On every recorded row of v3, the derived commands start where the recorded ones do (median
1 m later: the expert also announced turns about 30 m ahead) but last until the turn
ends. While moving, LANEFOLLOW covers 6.7% of the frames that steer by 0.5 or more instead
of 32.7%, and 97.4% of the turning frames under a derived turn command steer its way.

## Clip sampling

Each training epoch draws as many clips as the dataset's fixed clips (one every `stride`
frames), with replacement, and a drawn clip may start at any usable frame (`get_sampler`
in `driving_dataset.py`). Validation and evaluation use the fixed clips. A clip's chance
is 1 + 5 × its share of turning frames (|steering| > 0.1), so a clip that turns
throughout is six times as likely as one that never does. Weighting by navigation
command instead, as the April run did, barely helps: LEFT and RIGHT also cover the
approach to a junction, and turning frames rose only from 12% to 17% of the frames drawn.

On the v3 testbed training split, with 32-frame clips and the April run's stride of 512
(`scripts/sampling_report.py --stride 512`):

| Clips | Reachable | Drawn in 50 epochs | Turning share of drawn frames |
| --- | ---: | ---: | ---: |
| Fixed, every 512 frames | 6.70% | 6.70% | 14.3% |
| Random starts | 100.00% | 95.75% | 11.8% |
| Random starts, turn-weighted | 100.00% | 90.77% | 32.5% |
| Episode-ordered, batch 4 | — | 99.99% | 11.9% |

The 63 frames no clip can reach (0.005%) are in four episodes shorter than a clip. With the base config's
stride of 16 (85,184 clips per epoch) the weighted sampler draws 99.98% of frames.

For training that carries the model's state from one clip to the next,
`EpisodeOrderedSampler` gives each batch row whole episodes as consecutive clips; each
epoch shuffles the episodes and shifts the clip boundaries by a random offset, so every
frame can be drawn. It covers every episode once per epoch, in order, so it cannot
favour turns.

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
