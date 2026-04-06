import subprocess
import carla

try:
    result = subprocess.run(["ip", "route", "show", "default"], capture_output=True, text=True)
    host = result.stdout.strip().split()[2]
except Exception:
    host = "localhost"

try:
    client = carla.Client(host, 2000)
    client.set_timeout(10.0)
    current_map = client.get_world().get_map().name
    print(f"\n✅ Currently loaded CARLA map: {current_map}\n")
except Exception as e:
    print(f"Failed to connect: {e}")
