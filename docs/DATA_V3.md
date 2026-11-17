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
- Steering labels include the expert's Gaussian steering noise (σ = 0.001), added before
  the control was applied and recorded.
- Most completed episodes end at the frame budget, mid-route.
- Collision episodes continue after the collision; the loader drops those frames.

## Preserved

- Golden subset: 12 windows of 800 frames, two per town (6 completed, 3 timed out,
  3 collisions), with the original image bytes; selected by `scripts/extract_golden.py`.
- Testbed cache: every episode with images resized to 960×160 the way training resizes
  them, stored as JPEG (quality 90) at 10 Hz, and every signal at 20 Hz unchanged; built
  by `scripts/build_testbed_cache.py` and read through `src/data/schema_adapter.py`.
- The episode inventory from `scripts/scan_dataset.py`, the collection manifests and log.
- Offline baselines of the April model on this data: [RUNS.md](RUNS.md).
