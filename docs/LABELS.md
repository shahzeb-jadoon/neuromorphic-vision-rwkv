# Training targets

How a sample's targets and route inputs are computed from an episode's raw signals, and
which clips training draws, in `src/data/labels.py`, `src/data/schema_adapter.py` and
`src/data/driving_dataset.py`.

Every episode is read at 10 frames per second (`dataset.fps`): frame k is recorded row
k × (recording rate / 10), and its targets come from the full-rate signals at that row.
A 20 Hz recording and its 10 Hz training cache therefore give identical targets.

## Per frame

| Target | Definition |
| --- | --- |
| `controls` | Steering in [−1, 1] and acceleration = throttle − brake in [−1, 1], as the expert applied them |
| `nav_command` | RoadOption minus one, what the route does at the next junction ([below](#navigation-commands)): LEFT 0, RIGHT 1, STRAIGHT 2, LANEFOLLOW 3 (no junction maneuver). The lane changes (4, 5) do not occur in v3. A recorded command outside 1–6 is still an error, never clamped |
| `target_point` | The path 30 m ahead, in the vehicle's frame: x forward, y to the right, metres ([below](#target-point)); an input, not a target |
| `future_waypoints` | Positions 5, 10, 15 and 20 frames (0.5–2 s) ahead, in the vehicle's frame at that frame: x forward, y to the right, metres |
| `waypoint_mask` | False where a waypoint lies past the end of the recording, so is clamped to its last position: such waypoints are left out of the waypoint loss, while the frame's other targets stay. In v3 these are the last 2 s of the 684 completed episodes, which stop at the frame budget mid-drive: 13,680 frames (0.8%) |
| `speed` | The vehicle's horizontal speed divided by 120 km/h; an input, not a target. Training hides it from half of the clips, which see a learned "unknown speed" instead |
| `target_speed` | The expert's target speed divided by 120 km/h |
| `driving_mode` | The expert's mode 1–8 minus one; the v6 YieldStop is folded into StopSignStop |
| `situation` | Which of eleven situations hold, several at once: junctions, turning, red lights, stop signs, and vehicles and pedestrians in the path ([below](#situations)) |
| `loss_weight` | 0 for a frame kept out of the loss ([below](#frames-kept-out-of-the-loss)), 0.1 for a stationary frame, otherwise 1 |

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

## Frames kept out of the loss

The expert sometimes stops for a red light in the wrong place (`misplaced_stops` in
`labels.py`). A stop of 2 s or more that begins at a red or yellow light within 30 m, with
no vehicle in the path, is measured from the stop line of its lane, taken from CARLA's maps.
A correct stop leaves the vehicle's centre within 3 m of the line, as 85.7% of these stops
in v3 do. A stop more than 4 m past the line stands over it, mostly inside the junction: the
expert noticed the light late, a median 0.7 s before standing against 5.0 s normally, and
braked hard. Town01's stops sit closest to the line (median 1.5 m past), so its stops past
4 m are the least clear-cut.
More than 6 m before the line, with nothing in the path, the expert misjudged where the
line is, at a few places in Town04, Town06 and Town10HD.

Such a stop and the 5 s before it keep their frames, so clips still show them and a
stateful model still sees them, but their loss weight is 0, so no target is learned from
them. The v3 testbed has 75 stops past the line and 47 short of it: 30,583 frames (1.78%).

## Steering spikes

The v3 expert's steering sometimes jumps for a single frame and returns. On the testbed,
2.12% of moving frames lie more than 0.1 from the 5-frame running median of their episode's
steering (per episode: median 2.09%, 95th percentile 4.86%), and 360 frames jump by more than
1.0. Nearly identical images then carry different targets, so no model can fit them. Training
keeps them for now. Their cause is not yet known: the expert's injected steering noise
(standard deviation 0.001) is far too small to explain them.

## Navigation commands

The recorded commands of v3 often end before the turn they announce
([DATA_V3.md](DATA_V3.md#known-limitations)), so training derives the command from the
path the vehicle drove and the junctions of its town (`route` in `labels.py`), with CARLA's
meaning: LEFT, RIGHT or STRAIGHT is what the route does at the next junction, from 30 m
before it until the vehicle leaves it, and LANEFOLLOW means no junction maneuver. A
junction crossing's heading change, measured from 5 m before entering to 10 m after
leaving, decides it: under 35° is STRAIGHT, as in CARLA's route planner. Visits under 3 m
only clip an outline and are ignored, and a junction's command starts as soon as the
previous junction is left when its approach would otherwise begin less than 10 m later.
Measured along the distance travelled, a command holds however long the vehicle waits.

Lane following is not a command that competes with these: the vehicle follows its lane
through a junction too. Lane changes would be a second, separate input; v3 records none,
and the v7 schema adds them. Bends between junctions are LANEFOLLOW: the target point
shows them.

Junction outlines come from CARLA's maps (`scripts/export_junctions.py`, 123 junctions in
the six v3 towns) and are stored as `junctions.json` beside the episodes, with the stop lines
of the towns' traffic lights (357, from each light's stop waypoints); reading labels without
it is an error. At driving time the agent derives its command the same way from the
planned route, whose junction waypoints CARLA marks.

On the v3 testbed the derived commands agree with the recorded LEFT and RIGHT on 88% of
their frames, and 92% of the turning frames under a turn command steer its way. The
recorded STRAIGHT lies inside a junction on only 26% of its rows, ending early as the
recorded turns do.

## Target point

`target_point` is where the path is 30 m further on, in the vehicle's frame at that frame,
as TransFuser-style models condition on the route. In v3 it comes from the driven path,
continued straight along the final heading past the end of a recording; in v7 it comes
from the recorded route, and at driving time from the planned one. Mirroring flips its y.
The model embeds it after the temporal core, in units of 30 m.

The v3 point shows the exact line the expert went on to drive, which a planned route does not,
so it gives away upcoming steering. Judge v3 offline scores against a baseline that sees only
the target point, not only against the per-command baseline.

## Situations

Several can hold at once (`SITUATIONS` in `labels.py`). Junction and turning situations come
from the path, red lights and stop signs from the recorded telemetry, and vehicles and
pedestrians from the road users the expert recorded around it each frame. Each cause comes as
a pair, moving and standing, and a "stopped" situation holds only while standing.

| Situation | True when | v3 frames |
| --- | --- | ---: |
| approaching junction | within 30 m before a junction (as its command) | 44.7% |
| in junction | inside a junction outline | 21.3% |
| turning | the path curves by 3°/m or more (a radius under 19 m) | 7.2% |
| approaching red light | moving, with a red or yellow light within 30 m | 6.1% |
| stopped at red light | standing (under 0.5 km/h) with a red or yellow light within 30 m | 27.8% |
| approaching stop sign | moving, with a stop sign within 30 m | 1.6% |
| stopped at stop sign | standing with a stop sign within 30 m | 0.9% |
| following vehicle | moving, with a vehicle in the path within 25 m, bumper to bumper | 8.2% |
| stopped for vehicle | standing with a vehicle in the path within 25 m | 12.1% |
| yielding to pedestrian | moving, with a pedestrian in the path within 15 m | 0.1% |
| stopped for pedestrian | standing with a pedestrian in the path within 15 m | 0.3% |

A vehicle is in the path when its box overlaps the expert's corridor along the route, or its
centre lies within 1.75 m of the line straight ahead, whichever way it is heading: in 18% of the
expert's standing emergency stops the car in its path is oncoming or crossing. A
pedestrian is in the path within 2.5 m of that line, now or where its velocity takes it in
1 s. The distances are the expert's own: it starts following at 25 m and slows for a
pedestrian at 15 m. They separate its modes cleanly: a vehicle is in the path within 25 m in
99% of its following and emergency-stop frames, and in 0.3% of its traffic-light stops and
0.2% of its free driving. The expert's modes are not used: its PedestrianStop creeps at
8 km/h, and its EmergencyStop is mostly a calm wait close behind a queued car.

The 30 m reach includes waiting in a queue: 94.5% of standing frames at a red light are
within 30 m of it, and nearly all beyond 15 m are queued behind another vehicle.

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
steering, waypoint y and target point y change sign; LEFT and RIGHT swap, as do the two
lane changes; no situation has a side. A mirrored scene shows left-hand traffic.
