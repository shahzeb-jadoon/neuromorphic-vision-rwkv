# Data card: v3 driving dataset

The first dataset recorded for this project: the privileged expert driving in CARLA,
saved as schema v3 episodes. The full recording has been retired; what remains is listed
under [Preserved](#preserved). Figures come from `scripts/scan_dataset.py` and the
collection manifest.

| | |
| --- | --- |
| Episodes | 1,099, all readable, no non-finite values |
| Frames | 3,624,673 at 20 Hz (50.3 hours); 3,624,525 before collision cut-offs |
| Size | 3.65 TB |
| Collected | 10–15 March 2026 (manifest timestamps, UTC) |
| Simulator | CARLA 0.9.16, synchronous mode, 20 Hz |
| Expert | Privileged rule-based expert at tag `expert/v3-collection`, target speed 30 km/h |
| Schema | v3, `src/data/hdf5_schema.py` |

## Episodes

| Town | Episodes |
| --- | ---: |
| Town01 | 195 |
| Town03 | 191 |
| Town04 | 195 |
| Town05 | 195 |
| Town06 | 180 |
| Town10HD | 143 |

Town02 was not recorded. Town07 episodes were removed in April 2026; three timed-out
Town07 episodes were kept separately.

- Weather: 10 presets (clear, cloudy, wet, soft, mid and hard rain at noon or sunset),
  95–118 episodes each.
- Traffic: 20, 50 or 100 vehicles (362, 368 and 369 episodes), always 10 pedestrians.
- Outcome: 684 completed, 267 timed out, 148 ended in a collision. 681 of the completed
  episodes stopped at the 4,000-frame budget rather than at the end of a route.
- Length: 54 to 4,000 frames.

## Each frame

- Images: three cameras (60° field of view, yaw −60°, 0°, +60°), 640×320 each, stitched
  into one 1920×320 RGB panorama; gzip level 1, one frame per chunk.
- Controls: steering in [−1, 1], throttle and brake in [0, 1], as the expert applied them.
- Navigation command (RoadOption): LANEFOLLOW 54.0%, RIGHT 25.4%, LEFT 18.6%,
  STRAIGHT 1.9% of frames; no lane-change commands.
- Expert driving mode: FollowLane 34.6%, TrafficLightStop 25.2%, EmergencyStop 11.9%,
  VehicleFollowing 9.9%, Turning 9.0%, ApproachingTurn 6.6%, StopSignStop 2.2%,
  PedestrianStop 0.4%.
- Turning frames (|steering| > 0.1): 12.6%.
- Ego state: world position and rotation, world-frame velocity (m/s), target speed (km/h).
- Expert telemetry: traffic-light state and distance, stop-sign distance, a stationary
  flag, corridor diagnostics (SAT and lateral detections, corridor segments, waypoint
  buffer size and head distance), and the nearby vehicles and pedestrians as JSON.

## Known limitations

- The planned route and goal are not stored, so waypoint targets have to be derived from
  the driven trajectory.
- LEFT and RIGHT cover 44% of frames, approaches included, while only 12.6% of frames
  steer by more than 0.1.
- Commands can end before the turn they announce. 28% of the frames that steer by 0.5 or
  more are labelled LANEFOLLOW (in 799 episodes), 12% of them in the first 3 s after
  spawning and most of the rest in junction turns; 180 episodes have a LEFT or RIGHT run
  in which the car never steers that way. The recorded STRAIGHT lies inside a junction on
  only 26% of its rows. Training derives the commands from the path driven and the
  junction outlines instead ([LABELS.md](LABELS.md#navigation-commands)).
- `traffic_light_distance` keeps 4,000 rows in 418 episodes that ended before the frame
  budget; rows past the episode's end are padding.
- The expert stops for some red lights in the wrong place. 75 stops stand more than 4 m
  past the stop line, mostly inside the junction, after it noticed the light late; 47 stand more
  than 6 m before it with nothing in the path, at a few places in Town04, Town06 and
  Town10HD. Training keeps them out of the loss
  ([LABELS.md](LABELS.md#frames-kept-out-of-the-loss)).
- The expert's driving modes name its rules, not what is in front of it: its
  PedestrianStop creeps at 8 km/h, and its EmergencyStop is mostly a calm wait close behind
  a queued car. The situation labels come from the recorded road users instead.
- Steering labels include the expert's Gaussian steering noise (σ = 0.001), added before
  the control was applied and recorded.
- Most completed episodes end at the frame budget, mid-route.
- Collision episodes end on the frame after the collision. Timed-out episodes end with the
  vehicle standing still for 26–43 s (10th to 90th percentile) before the collector gave
  up. Training uses neither that stop nor the 3 s before a collision
  ([LABELS.md](LABELS.md)).

## Checks at the Phase 1 gate (2026-11-18)

On the testbed cache, with the Phase 1 labels ([LABELS.md](LABELS.md)):

| Check | Result |
| --- | --- |
| Every usable frame of both splits read through the dataset (`scripts/dataset_pass.py`) | 0 problems in 53,079 clips covering 99.0% of 1,715,947 frames; the rest are episode tails shorter than a clip |
| Training frames a clip can reach (`scripts/sampling_report.py --stride 512`) | 100.00% |
| Loader memory over 30 minutes (`scripts/loader_soak.py`, 4 workers, batches of 8 × 32 frames) | Flat: main process 583 MiB throughout, each worker 505–551 MiB between batches and at most 1.43 GiB while holding one |
| Label audit (`scripts/label_audit.py`), rendered on both machines | No unreadable episode; identical numbers |

Reviewing the audit's sample frames found red-light stops in the wrong place, and situations
copied from the expert's modes: "stopped for pedestrian" also covered creeping at 8 km/h.
Both were fixed ([LABELS.md](LABELS.md#frames-kept-out-of-the-loss)), and the dataset pass
and the audit were run again with the new labels: 0 problems in the same 53,079 clips,
every "stopped" situation standing on 100% of its frames and every moving one on none,
1.78% of frames kept out of the loss. The sampler and loader did not change, so the
reachability and memory results stand.

## Preserved

- Golden subset: 12 windows of 800 frames, two per town (6 completed, 3 timed out,
  3 collisions), with the original image bytes; selected by `scripts/extract_golden.py`.
- Testbed cache: every episode with images resized to 960×160 the way training resizes
  them, stored as JPEG (quality 90) at 10 Hz, and every signal at 20 Hz unchanged; built
  by `scripts/build_testbed_cache.py` and read through `src/data/schema_adapter.py`.
- Junction outlines and traffic-light stop lines of the six towns from CARLA's maps
  (`scripts/export_junctions.py`), as `junctions.json` beside the testbed cache and both
  golden copies.
- The episode inventory from `scripts/scan_dataset.py`, the collection manifests and log.
- Offline baselines of the April model on this data: [RUNS.md](RUNS.md).
