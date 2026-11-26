"""Drive a route suite in CARLA with the expert or a learned agent, scored as the CARLA
leaderboard scores routes (src/eval/driving_score.py).

    python scripts/run_route_suite.py --suite configs/routes/suite_v1.json --agent expert \
        --out runs/closed_loop/expert
    python scripts/run_route_suite.py --suite configs/routes/suite_v1.json --agent learned \
        --checkpoint runs/short/short_rwkv7_rest/checkpoints/best.pt --out runs/closed_loop/rest

Writes <route id>.csv (a row a tick) and <route id>.json (its score) for each route, and
summary.json over the routes driven. CARLA_HOST and CARLA_PORT come from .env.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import carla

from agents.navigation.global_route_planner import GlobalRoutePlanner
from agents.navigation.route import plan_route
from src.data.labels import CAUSES, ROUTE_RESOLUTION
from src.eval import closed_loop as C
from src.eval import driving_score as D
from src.utils.settings import setting


class ExpertAgent:
    """The repository's privileged expert, driving the harness's route."""

    def __init__(self, vehicle: Any) -> None:
        from agents.expert.privileged_expert import ExpertConfig, PrivilegedExpert

        world = vehicle.get_world()
        self.planner = GlobalRoutePlanner(world.get_map(), ROUTE_RESOLUTION)
        self.expert = PrivilegedExpert(vehicle, world, self.planner, ExpertConfig())

    def set_route(self, keypoints: list[Any]) -> None:
        self.expert.set_route(plan_route(self.planner, keypoints))

    def run_step(self, frame: int) -> Any:
        return self.expert.run_step()[0]

    def destroy(self) -> None:
        pass


def learned_telemetry(agent: Any) -> dict[str, Any]:
    return {
        "model_stepped": int(agent.stepped),
        "nav_command": agent.last_nav_cmd,
        **{f"p_{c.replace(' ', '_')}": round(p, 3) for c, p in zip(CAUSES, agent.situation)},
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--suite", type=Path, required=True)
    parser.add_argument("--agent", choices=["expert", "learned"], required=True)
    parser.add_argument("--checkpoint", help="the learned agent's checkpoint")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--routes", nargs="*", help="route ids to drive (default: all)")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    routes = [r for r in C.load_suite(args.suite) if not args.routes or r.id in args.routes]
    if args.agent == "learned":
        if not args.checkpoint:
            parser.error("--agent learned needs --checkpoint")
        from agents.learned.learned_agent import LearnedAgent

        def make(vehicle: Any) -> Any:
            return LearnedAgent(vehicle, args.checkpoint, device=args.device)

        telemetry: Any = learned_telemetry
    else:
        make, telemetry = ExpertAgent, None

    client = carla.Client(setting("CARLA_HOST"), int(setting("CARLA_PORT", "2000")))
    client.set_timeout(60.0)
    results = []
    for route in routes:
        result = C.run_route(client, route, make, args.out, telemetry)
        score = result["score"]
        print(
            f"{route.id:>12} {route.town:>9}  DS {score['driving_score']:6.1f}  "
            f"RC {score['route_completion']:6.1f}  penalty {score['infraction_penalty']:.3f}  "
            f"{len(score['infractions'])} infractions  {result['sim_s']:.0f} s sim, "
            f"{result['wall_s']:.0f} s wall  {score['status']}",
            flush=True,
        )
        results.append(result)

    scores = [D.RouteScore.from_dict(r["score"]) for r in results]
    summary = D.summarize(scores, [r["length_m"] for r in results])
    summary["routes"] = len(results)
    (args.out / "summary.json").write_text(json.dumps(summary, indent=1))
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
