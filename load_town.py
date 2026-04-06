import subprocess
import carla

try:
    result = subprocess.run(["ip", "route", "show", "default"], capture_output=True, text=True)
    host = result.stdout.strip().split()[2]
except Exception:
    host = "localhost"

print(f"Connecting to CARLA on {host}:2000... (Timeout set to 120s)")
try:
    client = carla.Client(host, 2000)
    client.set_timeout(120.0)
    print("Loading Town10HD... this may take up to a minute.")
    client.load_world("Town10HD")
    print("Town10HD loaded successfully! You can now switch to the CARLA window to look around.")
except Exception as e:
    print(f"Failed to connect or load map: {e}")
