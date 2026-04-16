# Option H: Segment Planner and Route-Link Diagnostics

## Objective

Option H replaces waypoint-dense global search with a lane-segment graph that is easier to cache, faster to inspect, and better aligned with junction semantics.

This change also adds sidecar diagnostics so route failures can be diagnosed from route metadata alone, without opening HDF5 frames first.

## Core Planner Model

`LaneGraphPlanner` now models the map as directed lane-segment nodes derived from CARLA topology.

Each segment node includes:

- entry and exit waypoint IDs
- entry and exit transforms
- road/section/lane identifiers
- junction membership

Edges are typed with explicit transition kinds:

- `forward`
- `lane_change_left`
- `lane_change_right`
- `junction_connector`

A* runs on segment IDs, then expands each segment back into waypoint-level output for the expert route interface.

## Junction Connector Semantics (Important)

A key CARLA topology detail is now explicit in the pipeline:

- Roads inside junctions frequently act as connector roads.
- Those connector roads link roads outside the junction to other roads outside the junction.
- Correctly identifying these links is required to avoid mislabeling valid transitions as route corruption.

The planner and sidecar diagnostics now treat connector links as first-class transitions instead of generic road changes.

## Cache Behavior

Per-town planner caches are written to disk and reused when valid.

Cache payload includes:

- segment nodes
- typed edges
- lane memberships
- precomputed distance-to-junction map
- connector link metadata

This keeps planning startup costs stable across episodes in the same town.

## Sidecar Additions

Episode route sidecars (`*.route.json`) now include:

- `route_planner_backend`
- `planner_diagnostics`
: planner graph summary and last-plan diagnostics when available
- `route_links`
: transition-level link metadata between adjacent route waypoints

`route_links` includes:

- `links`: per-transition properties (`from_index`, `to_index`, road/lane IDs, junction flags, kind, optional `connector_road_id`)
- `link_kind_counts`: aggregate counts by transition type
- `junction_connectors`: inferred connector roads and which external roads they connect
- `junction_connector_transfer_count`: number of detected connector transfers

## Validator Coverage

`validate_route_sidecars.py` now validates `route_links` integrity:

- transition count consistency against waypoint count
- per-link field agreement with waypoint properties
- connector road ID correctness on junction transitions
- consistency of reported `junction_connectors` against inferred waypoint triples

This enables immediate detection of property-assignment or link-labeling regressions.

## Triage Workflow

1. Open the route sidecar and inspect `route_links.link_kind_counts`.
2. Check `junction_connectors` for expected connector roads in dense intersections.
3. If anomalies exist, inspect `route_links.issues` from validator output.
4. Only then move to HDF5-level frame debugging if needed.

This workflow reduces time-to-root-cause for route failures and zero-frame terminations.
