#!/usr/bin/env python3
"""Replay CARLA recorder files with optional TL overlays and interactive controls.

Controls in terminal:
  p -> pause/resume replay (pause also disables follow camera for free-fly)
  f -> toggle ego follow camera
  q -> quit replay
  h -> print control help

When paused, use the CARLA viewport controls for free-fly inspection:
click viewport, hold right mouse button, then WASD + Q/E.
"""

from __future__ import annotations

import argparse
import logging
import select
import sys
import termios
import time
import tty
from typing import Any

try:
    import carla
except ImportError as exc:  # pragma: no cover
    raise SystemExit("CARLA Python API is required.") from exc

from scripts.visualize_cf183fc1_tl_stops import (
    _bound_tl_list,
    _collect_live_highlights,
    _draw_live_highlights,
    _draw_overlay,
    _extract_episode_stops,
    _resolve_episode_path,
    _resolve_profile_defaults,
    _set_initial_spectator_view,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


class _TerminalKeyReader:
    """Non-blocking single-char reader for interactive replay controls."""

    def __init__(self) -> None:
        self._enabled = False
        self._fd: int | None = None
        self._old_attrs: Any = None

    def __enter__(self) -> "_TerminalKeyReader":
        if not sys.stdin.isatty():
            return self
        self._fd = sys.stdin.fileno()
        self._old_attrs = termios.tcgetattr(self._fd)
        tty.setcbreak(self._fd)
        self._enabled = True
        return self

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        if self._enabled and self._fd is not None and self._old_attrs is not None:
            termios.tcsetattr(self._fd, termios.TCSADRAIN, self._old_attrs)

    def poll(self) -> str | None:
        if not self._enabled:
            return None
        assert self._fd is not None
        ready, _, _ = select.select([self._fd], [], [], 0.0)
        if not ready:
            return None
        ch = sys.stdin.read(1)
        return ch if ch else None


def _detect_wsl2_host() -> str:
    import subprocess

    try:
        result = subprocess.run(
            ["ip", "route", "show", "default"],
            capture_output=True,
            text=True,
            check=True,
        )
        return result.stdout.strip().split()[2]
    except Exception:
        return "localhost"


def _set_replayer_speed(client: Any, speed: float) -> bool:
    setter = getattr(client, "set_replayer_time_factor", None)
    if setter is None:
        logger.warning("This CARLA build does not expose set_replayer_time_factor().")
        return False
    setter(speed)
    return True


def _find_hero_vehicle(world: Any) -> Any | None:
    vehicles = list(world.get_actors().filter("vehicle.*"))
    if not vehicles:
        return None
    for veh in vehicles:
        role_name = veh.attributes.get("role_name", "")
        if role_name.lower() == "hero":
            return veh
    return vehicles[0]


def _update_spectator_follow(
    world: Any,
    vehicle: Any,
    offset: tuple[float, float, float],
) -> None:
    vt = vehicle.get_transform()
    fwd = vt.get_forward_vector()
    right = vt.get_right_vector()
    up = carla.Location(x=0.0, y=0.0, z=1.0)

    cam_loc = vt.location + carla.Location(
        x=fwd.x * offset[0] + right.x * offset[1] + up.x * offset[2],
        y=fwd.y * offset[0] + right.y * offset[1] + up.y * offset[2],
        z=fwd.z * offset[0] + right.z * offset[1] + up.z * offset[2],
    )
    cam_rot = carla.Rotation(pitch=-15.0, yaw=vt.rotation.yaw, roll=0.0)
    world.get_spectator().set_transform(carla.Transform(cam_loc, cam_rot))


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Replay CARLA recording with optional TL overlays and interactive pause",
    )
    parser.add_argument("recording", help="CARLA recorder file (episode_xxx.log)")
    parser.add_argument("--episode", default=None, help="Episode H5 for TL overlays")
    parser.add_argument("--town", default="Town10HD", help="Expected town/map")
    parser.add_argument("--tl-ids", type=int, nargs="+", default=None)
    parser.add_argument("--host", default="auto")
    parser.add_argument("--port", type=int, default=2000)
    parser.add_argument("--speed", type=float, default=1.0)
    parser.add_argument("--start", type=float, default=0.0)
    parser.add_argument("--duration", type=float, default=0.0)
    parser.add_argument(
        "--camera-id",
        type=int,
        default=0,
        help="Actor id for replay camera binding (0 keeps spectator free)",
    )
    parser.add_argument(
        "--auto-load-town",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--follow-ego",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Auto-follow hero vehicle when not paused",
    )
    parser.add_argument(
        "--follow-offset",
        type=float,
        nargs=3,
        default=[-8.0, 0.0, 5.0],
        metavar=("X", "Y", "Z"),
    )
    parser.add_argument("--life-time", type=float, default=0.65)
    parser.add_argument("--refresh-sec", type=float, default=0.20)
    parser.add_argument("--stop-speed-kmh", type=float, default=0.8)
    parser.add_argument(
        "--allow-initial-hold-stop",
        action="store_true",
        help="Use earliest near-stop even with no prior motion",
    )
    parser.add_argument("--min-prior-speed-kmh", type=float, default=2.0)
    parser.add_argument(
        "--show-no-stop-tls",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument("--width-overlays", type=float, nargs="*", default=[])
    parser.add_argument("--highlight-live-ids", type=int, nargs="*", default=[])
    parser.add_argument(
        "--set-initial-view",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    return parser.parse_args()


def _print_controls() -> None:
    print("\nControls: p=pause/resume, f=toggle follow, q=quit, h=help")
    print("When paused, use CARLA viewport free-fly: RMB + WASD + Q/E.\n")


def main() -> None:
    args = _parse_args()
    host = _detect_wsl2_host() if args.host == "auto" else args.host

    client = carla.Client(host, args.port)
    client.set_timeout(60.0)
    world = client.get_world()

    map_name = world.get_map().name
    if args.auto_load_town and args.town not in map_name:
        logger.info("Current world is '%s', loading '%s'", map_name, args.town)
        client.set_timeout(120.0)
        client.load_world(args.town)
        world = client.get_world()
        client.set_timeout(60.0)
        logger.info("Loaded world: %s", world.get_map().name)
    elif args.town not in map_name:
        logger.warning("Current world is '%s', expected '%s'", map_name, args.town)

    bound_tls: list[Any] = []
    live_highlights: list[Any] = []
    if args.episode is not None:
        episode_path = _resolve_episode_path(args.episode)
        tl_ids = list(args.tl_ids) if args.tl_ids is not None else None
        if tl_ids is None or len(tl_ids) == 0:
            _prof_ep, prof_tl_ids = _resolve_profile_defaults(args.town)
            tl_ids = list(prof_tl_ids) if prof_tl_ids is not None else None
        if tl_ids is None or len(tl_ids) == 0:
            raise SystemExit("No --tl-ids provided and no town profile defaults found for overlay.")

        stops = _extract_episode_stops(
            episode_path=episode_path,
            tl_ids=tl_ids,
            stop_speed_kmh=args.stop_speed_kmh,
            allow_initial_hold_stop=args.allow_initial_hold_stop,
            min_prior_speed_kmh=args.min_prior_speed_kmh,
        )

        stops_for_overlay = stops
        if not args.show_no_stop_tls:
            hidden = [s for s in stops if s.first_stop_frame is None]
            if hidden:
                hidden_txt = ", ".join(f"TL{s.requested_tl_id}" for s in hidden)
                logger.info(
                    "Hiding no-stop TLs: %s (use --show-no-stop-tls to include)",
                    hidden_txt,
                )
            stops_for_overlay = [s for s in stops if s.first_stop_frame is not None]
            if not stops_for_overlay:
                logger.warning("No near-stop frames found; drawing all requested TLs.")
                stops_for_overlay = stops

        bound_tls = _bound_tl_list(world, stops_for_overlay)
        live_highlights = _collect_live_highlights(world, args.highlight_live_ids)

        logger.info(
            "Overlay loaded from %s with %d TL zones",
            episode_path.name,
            len(bound_tls),
        )
        if args.set_initial_view and bound_tls:
            _set_initial_spectator_view(world, bound_tls)

    width_overlays = [w for w in args.width_overlays if 0.0 < w < 1.0]

    logger.info(
        "Replaying %s speed=%.2fx start=%.1fs duration=%.1fs camera_id=%d",
        args.recording,
        args.speed,
        args.start,
        args.duration,
        args.camera_id,
    )
    result = client.replay_file(
        args.recording,
        args.start,
        args.duration,
        args.camera_id,
        False,
    )
    logger.info("Replay started: %s", result)
    speed_control_ok = _set_replayer_speed(client, args.speed)

    paused = False
    follow_enabled = args.follow_ego
    raw_follow_offset = args.follow_offset
    follow_offset: tuple[float, float, float] = (
        float(raw_follow_offset[0]),
        float(raw_follow_offset[1]),
        float(raw_follow_offset[2]),
    )

    _print_controls()

    try:
        with _TerminalKeyReader() as keys:
            while True:
                if bound_tls:
                    _draw_overlay(world, bound_tls, args.life_time, width_overlays)
                if live_highlights:
                    _draw_live_highlights(world, live_highlights, args.life_time)

                if follow_enabled and not paused:
                    hero = _find_hero_vehicle(world)
                    if hero is not None:
                        _update_spectator_follow(world, hero, follow_offset)

                key = keys.poll()
                if key is not None:
                    k = key.lower()
                    if k == "q":
                        logger.info("Quit requested by user.")
                        break
                    if k == "h":
                        _print_controls()
                    elif k == "f":
                        follow_enabled = not follow_enabled
                        logger.info("Follow ego: %s", follow_enabled)
                    elif k == "p":
                        if not speed_control_ok:
                            logger.warning("Pause/resume unsupported in this CARLA build.")
                        else:
                            paused = not paused
                            if paused:
                                _set_replayer_speed(client, 0.0)
                                follow_enabled = False
                                logger.info("Replay paused. Follow disabled; free-fly enabled.")
                            else:
                                _set_replayer_speed(client, args.speed)
                                logger.info("Replay resumed at %.2fx", args.speed)

                time.sleep(max(0.03, args.refresh_sec))
    except KeyboardInterrupt:
        logger.info("Interrupted by user.")
    finally:
        try:
            client.stop_replayer(keep_actors=False)
            logger.info("Replay stopped.")
        except Exception:
            pass


if __name__ == "__main__":
    main()
