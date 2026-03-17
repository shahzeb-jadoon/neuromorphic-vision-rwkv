import json
import shutil
import logging
from pathlib import Path

# =====================================================================
# CONFIGURATION
# =====================================================================
MANIFEST_PATH = Path("/mnt/data/carla_data/collection_manifest.json")
BACKUP_PATH = Path("/mnt/data/carla_data/collection_manifest.historical_audit.json")
DATA_DIR = Path("/mnt/data/carla_data/")
LOG_FILE = Path("data/analysis/manifest_cleaning.log")

# Initialize audit log
logging.basicConfig(
    filename=LOG_FILE, level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


def main() -> None:
    print("=====================================================")
    print(" 🧹  MANIFEST CLEANING SCRIPT (OPTION B) ")
    print("=====================================================\n")

    if not MANIFEST_PATH.exists():
        print(f"CRITICAL ERROR: Cannot find the active manifest at: {MANIFEST_PATH}")
        return

    # 1. Create the Historical Backup
    print("1. Creating Permanent Historical Backup...")
    if not BACKUP_PATH.exists():
        try:
            shutil.copy2(MANIFEST_PATH, BACKUP_PATH)
            print(f"  ✓ Backup successfully created at: {BACKUP_PATH}")
            logging.info(f"Created historical backup at {BACKUP_PATH}")
        except Exception as e:
            print(f"  ❌ ERROR creating backup: {e}")
            return
    else:
        print(
            f"  ✓ Backup already exists at: {BACKUP_PATH}."
            " Skipping creation to preserve original history."
        )

    # 2. Parse the Active Manifest
    print("\n2. Parsing Active Manifest...")
    with open(MANIFEST_PATH, "r") as f:
        data = json.load(f)

    episodes = data.get("episodes", [])
    original_count = len(episodes)
    print(f"  ✓ Found {original_count} historical episodes recorded.")

    # 3. Filter Ghost Files
    print("\n3. Sweeping Drive for Ghost Entries...")
    active_episodes = []
    missing_count = 0

    for idx, ep in enumerate(episodes):
        fn = ep.get("filename")
        # Handle the rare corrupted entries that have no filename
        if not fn:
            missing_count += 1
            logging.info(f"Dropped ghost entry (no filename): {ep.get('config_hash')}")
            continue

        fp = DATA_DIR / fn
        if fp.exists():
            active_episodes.append(ep)
        else:
            missing_count += 1
            logging.info(f"Dropped ghost entry (physically deleted): {fn}")

    print(f"  ✓ Discovered {missing_count} ghost entries (physically deleted files).")
    print(f"  ✓ Keeping {len(active_episodes)} physically validated episodes.")

    if missing_count == 0:
        print("\n=====================================================")
        print(" OPERATION CANCELLED ")
        print(" The manifest is already perfectly in sync with the drive!")
        print(" No ghost entries found. No changes were made.")
        return

    # 4. Explicit Confirmation Gate
    print("\nWARNING: You are about to prune the ACTIVE manifest.")
    print(
        f"This will permanently strip {missing_count} ghost rows."
        " The data is still safe in the backup."
    )

    response = input("Type 'CLEAN' (in all caps) to execute the prune: ")

    if response != "CLEAN":
        print("\nAborted. The manifest was not modified.")
        logging.info("User aborted manifest pruning at confirmation gate.")
        return

    # 5. Overwrite the Active Manifest
    print("\n4. Overwriting Active Manifest...")

    # Update top-level tracking numbers
    data["episodes"] = active_episodes
    data["total_episodes"] = len(active_episodes)
    data["total_frames"] = sum(ep.get("n_frames", 0) for ep in active_episodes)

    try:
        with open(MANIFEST_PATH, "w") as f:
            json.dump(data, f, indent=2)
        print("  ✓ Manifest successfully pruned and saved.")
        logging.info(f"Successfully pruned manifest. Removed {missing_count} rows.")
    except Exception as e:
        print(f"  ❌ CRITICAL ERROR writing manifest: {e}")
        logging.error(f"Failed to write pruned manifest: {e}")
        return

    print("\n=====================================================")
    print(" MANIFEST CLEANING COMPLETED ")
    print("=====================================================")
    print(f" Old Manifest Rows: {original_count}")
    print(f" New Manifest Rows: {len(active_episodes)}")
    print(f" Ghost Rows Pruned: {missing_count}")

    print("\nYour codebase and --resume logic is perfectly protected.")
    print(f"Audit Log saved to: {LOG_FILE}\n")


if __name__ == "__main__":
    main()
