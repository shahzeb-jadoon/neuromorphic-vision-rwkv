"""Verify CARLA server connectivity from WSL2.

Prerequisites:
    1. CARLA 0.9.16 server running on Windows host.
    2. conda activate rwkv
    3. pip install carla==0.9.16
"""
import sys
import carla


def main():
    host = "localhost"
    port = 2000
    timeout = 10.0

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
        print("  3. Can WSL2 reach Windows localhost? (try: curl localhost:2000)")
        sys.exit(1)


if __name__ == "__main__":
    main()
