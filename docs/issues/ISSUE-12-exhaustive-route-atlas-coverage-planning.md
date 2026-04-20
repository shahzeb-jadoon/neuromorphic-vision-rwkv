# ISSUE-12 — Exhaustive Route Atlas Coverage Planning

> **Status:** Open (planning started)
> **Priority:** High (parallelizable with active collection)
> **Scope:** Offline/background precompute script + atlas artifact format

## Problem

Current collection generates routes online per episode. This can over-sample
some regions and leave weak road-lane/connector coverage in long campaigns.

## Goal

Create a separate offline script that precomputes a route atlas with explicit
coverage targets so collection can consume curated route plans instead of only
online sampling.

## Initial Town Scope

Use an initial bounded scope to reduce OOM risk during atlas generation:

- Town01
- Town03
- Town04
- Town05
- Town06
- Town07
- Town10HD

Keep Town12/Town15 out of the first atlas pass until large-map memory behavior
is validated.

## MVP Script

Planned script path:

- `scripts/build_route_atlas.py`

Planned output:

- `data/atlas/<town>/atlas_manifest.json`
- `data/atlas/<town>/routes/*.json`

## Coverage Metrics (MVP)

Track at least:

- Road-lane segment coverage ratio.
- Junction approach-to-exit connector coverage ratio.
- Turn-label distribution coverage (LEFT/RIGHT/STRAIGHT).
- Route reuse/repetition penalty.

## OOM / Storage Guardrails

- Run one town per process.
- Write artifacts incrementally (checkpoint every N routes).
- Resume from checkpoints on restart.
- Hard cap routes per town and max waypoints per route.
- Keep atlas artifacts compressed JSON (or JSONL+gzip) to limit disk footprint.

## Suggested Command Shape

Single town:

```bash
python scripts/build_route_atlas.py \
  --town Town03 \
  --output data/atlas \
  --max-routes 150 \
  --target-coverage 0.95
```

Multi-town sequential sweep:

```bash
for t in Town01 Town03 Town04 Town05 Town06 Town07 Town10HD; do
  python scripts/build_route_atlas.py --town "$t" --output data/atlas --max-routes 150 --target-coverage 0.95
done
```

## Acceptance Criteria (MVP)

- Script runs end-to-end for at least one town without OOM.
- Atlas manifest records achieved coverage metrics.
- Atlas artifacts can be loaded by collection code in a dry-run parser.
- Generation can be resumed after interruption.

## Follow-up Integration Tasks

- Add atlas-consumer mode to `scripts/collect_data.py`.
- Add validation utility for atlas coverage and route integrity.
- Add tests for manifest schema and resume behavior.
