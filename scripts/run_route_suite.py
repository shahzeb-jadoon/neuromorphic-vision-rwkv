"""Drive a route suite in CARLA with the expert or a learned agent, scored as the CARLA
leaderboard scores routes (src/eval/driving_score.py).

    python scripts/run_route_suite.py --suite configs/routes/suite_v1.json --agent expert \
        --out runs/closed_loop/expert
    python scripts/run_route_suite.py --suite configs/routes/suite_v1.json --agent learned \
        --checkpoint runs/short/short_rwkv7_rest/checkpoints/best.pt --out runs/closed_loop/rest

Writes <route id>.csv (a row a tick), <route id>.json (its score) and, with --video,
<route id>.mp4 (a 640x360 chase camera at 20 fps) for each route, and summary.json over the
routes driven. Each route is driven in a process of its own, so its result does not depend on
the routes driven before it. CARLA_HOST and CARLA_PORT come from .env.
"""

from __future__ import annotations

import argparse
import json
import multiprocessing
from pathlib import Path
from typing import Any

import carla

from agents.navigation.global_route_planner import GlobalRoutePlanner
from agents.navigation.route import plan_route
from src.data.labels import CAUSES, ROUTE_RESOLUTION
from src.eval import closed_loop as C
from src.eval import driving_score as D
from src.utils.settings import setting

# A crashed route is driven once more, as the leaderboard repeats a route the simulator crashed
# on. Routes repeat exactly, so a retry gives the drive an uncrashed run would have given; an
# agent that crashes crashes again and the route is reported as crashed.
RETRIES = 1


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


def drive(route: C.Route, args: argparse.Namespace) -> None:
    """Drive one route; run in a fresh process, whose traffic manager no other route has used."""
    if args.agent == "learned":
        from agents.learned.learned_agent import LearnedAgent

        def make(vehicle: Any) -> Any:
            return LearnedAgent(vehicle, args.checkpoint, device=args.device)

        telemetry: Any = learned_telemetry
    else:
        make, telemetry = ExpertAgent, None

    client = carla.Client(setting("CARLA_HOST"), int(setting("CARLA_PORT", "2000")))
    client.set_timeout(60.0)
    C.run_route(client, route, make, args.out, telemetry, video=args.video, traffic=args.traffic)


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
    parser.add_argument("--video", action="store_true", help="record a chase camera per route")
    parser.add_argument(
        "--traffic",
        choices=C.TRAFFIC,
        default="cars",
        help="cars: leaderboard 2.0's background vehicles (default); all: every 4-wheeled "
        "vehicle on the traffic manager's defaults",
    )
    args = parser.parse_args()
    if args.agent == "learned" and not args.checkpoint:
        parser.error("--agent learned needs --checkpoint")

    routes = [r for r in C.load_suite(args.suite) if not args.routes or r.id in args.routes]
    spawn = multiprocessing.get_context("spawn")
    results, crashed = [], []
    for route in routes:
        for _ in range(1 + RETRIES):
            child = spawn.Process(target=drive, args=(route, args))
            child.start()
            child.join()
            if not child.exitcode:
                break
            print(f"{route.id:>12} {route.town:>9}  crashed ({child.exitcode})", flush=True)
        else:
            crashed.append(route.id)
            continue
        result = json.loads((args.out / f"{route.id}.json").read_text())
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
    summary: dict[str, Any] = D.summarize(scores, [r["length_m"] for r in results])
    summary |= {"routes": len(results), "crashed": crashed, "traffic": args.traffic}
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "summary.json").write_text(json.dumps(summary, indent=1))
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
