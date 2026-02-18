"""Verify CARLA server connectivity from WSL2.

Prerequisites:
    1. CARLA 0.9.16 server running on Windows host.
    2. conda activate rwkv
    3. pip install carla==0.9.16

Note:
    WSL2 runs in a lightweight VM — 'localhost' refers to WSL2 itself,
    not the Windows host. The Windows host IP is the default gateway
    of the WSL2 virtual network, obtained via `ip route`.
    This IP can change on every WSL restart.
"""

import subprocess
import sys

import carla


def get_windows_host_ip():
    """Get the Windows host IP from WSL2's default gateway."""
    try:
        result = subprocess.run(
            ["ip", "route", "show", "default"],
            capture_output=True,
            text=True,
            check=True,
        )
        # Output: "default via 172.20.0.1 dev eth0"
        return result.stdout.strip().split()[2]
    except (subprocess.CalledProcessError, IndexError):
        return None


def main():
    host = get_windows_host_ip()
    if host is None:
        print("ERROR: Could not detect Windows host IP from WSL2 gateway.")
        print("       Falling back to localhost (may not work in WSL2).")
        host = "localhost"

    port = 2000
    timeout = 10.0

    print(f"Windows host:   {host}")
    print(f"Connecting to CARLA server at {host}:{port} ...")

    try:
        client = carla.Client(host, port)
        client.set_timeout(timeout)

        server_version = client.get_server_version()
        client_version = client.get_client_version()
        print(f"Server version: {server_version}")
        print(f"Client version: {client_version}")

        if server_version != client_version:
            print(f"WARNING: Version mismatch — server={server_version}, client={client_version}")

        world = client.get_world()
        map_name = world.get_map().name
        print(f"Map:            {map_name}")
        print("CARLA test:     PASSED")

    except Exception as e:
        print(f"CARLA test:     FAILED — {e}")
        print()
        print("Troubleshooting:")
        print("  1. Is CARLA server running on Windows? (CarlaUE4.exe)")
        print("  2. Is it listening on port 2000?")
        print(f"  3. Can WSL2 reach Windows host? (try: curl {host}:2000)")
        print(f"  4. Detected host IP: {host}")
        print("  5. If IP is wrong, check: ip route show default")
        sys.exit(1)


if __name__ == "__main__":
    main()
