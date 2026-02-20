"""Replay a CARLA recording from the built-in recorder.

Usage:
    python scripts/replay_recording.py episode_1234567890.log
    python scripts/replay_recording.py episode_1234567890.log --speed 2.0
    python scripts/replay_recording.py --list          # show available recordings
    python scripts/replay_recording.py recording.log --info  # show recording info

The recording file is created by ``collect_data.py --record`` and stored
inside the CARLA server's working directory (usually wherever
CarlaUE4.exe was launched from).

Note:
    Replay happens *server-side* — open the CARLA viewport window on
    the Windows host to watch it.  The spectator camera will follow the
    ego vehicle (actor id 0 by default).
"""

from __future__ import annotations

import argparse
import logging
import sys
import time

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

try:
    import carla

    CARLA_AVAILABLE = True
except ImportError:
    CARLA_AVAILABLE = False


def _detect_wsl2_host() -> str:
    """Detect Windows host IP from WSL2 default gateway."""
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


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Replay a CARLA recording",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("recording", nargs="?", help="Recording filename (e.g. episode_xxx.log)")
    parser.add_argument("--host", type=str, default="auto", help="CARLA host")
    parser.add_argument("--port", type=int, default=2000, help="CARLA port")
    parser.add_argument("--speed", type=float, default=1.0, help="Playback speed multiplier")
    parser.add_argument("--start", type=float, default=0.0, help="Start time (seconds)")
    parser.add_argument("--duration", type=float, default=0.0, help="Duration (0 = full)")
    parser.add_argument("--camera-id", type=int, default=0, help="Actor ID for spectator to follow")
    parser.add_argument("--info", action="store_true", help="Show recording info and exit")
    parser.add_argument("--list", action="store_true", help="List available recordings")
    args = parser.parse_args()

    if not CARLA_AVAILABLE:
        logger.error("carla package not installed. Run: pip install carla==0.9.16")
        sys.exit(1)

    host = _detect_wsl2_host() if args.host == "auto" else args.host
    logger.info("Connecting to CARLA at %s:%d", host, args.port)
    client = carla.Client(host, args.port)
    client.set_timeout(60.0)
    logger.info("Connected — server %s", client.get_server_version())

    if args.list:
        # CARLA stores recordings in <server_working_dir>
        logger.info(
            "Recordings are stored in the CARLA server's working directory.\n"
            "Check the folder where CarlaUE4.exe was launched."
        )
        return

    if not args.recording:
        parser.error("Please specify a recording filename (or use --list)")

    if args.info:
        info = client.show_recorder_file_info(args.recording, show_all=True)
        print(info)
        return

    # Start replay
    logger.info(
        "Replaying %s at %.1fx speed (start=%.1fs, duration=%.1fs, camera on actor %d)",
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
        False,  # spawn_sensors — False to avoid re-spawning cameras
    )
    logger.info("Replay started: %s", result)
    logger.info("Watch the CARLA viewport window. Press Ctrl+C to stop.")

    try:
        while True:
            time.sleep(1.0)
    except KeyboardInterrupt:
        logger.info("Stopping replay...")
        client.stop_replayer(keep_actors=False)
        logger.info("Replay stopped.")


if __name__ == "__main__":
    main()
