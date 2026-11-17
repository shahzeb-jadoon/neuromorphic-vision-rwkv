#!/usr/bin/env python3
# mypy: ignore-errors
"""Remove Town07 episodes from collected data and update manifest.

Creates a backup of the manifest before modification.
Moves Town07 HDF5 files to a backup directory (not deleted).

Usage:
    python scripts/remove_town07_data.py /mnt/data/carla_data
    python scripts/remove_town07_data.py /mnt/data/carla_data --delete
        # Actually delete instead of backup
"""

import argparse
import json
import shutil
import sys
from datetime import datetime
from pathlib import Path

import h5py


def find_town07_episodes(data_dir: Path) -> list[Path]:
    """Find all HDF5 episodes that were collected in Town07."""
    town07_files = []
    for hdf5_path in sorted(data_dir.glob("episode_*.h5")):
        try:
            with h5py.File(hdf5_path, "r") as f:
                town = f["metadata"].attrs.get("town", "")
                if isinstance(town, bytes):
                    town = town.decode()
                if town == "Town07":
                    town07_files.append(hdf5_path)
        except Exception as e:
            print(f"  WARNING: Could not read {hdf5_path.name}: {e}")
    return town07_files


def backup_manifest(manifest_path: Path) -> Path:
    """Create timestamped backup of the manifest."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_path = manifest_path.with_name(f"{manifest_path.stem}_backup_{timestamp}.json")
    shutil.copy2(manifest_path, backup_path)
    return backup_path


def update_manifest(manifest_path: Path, removed_files: list[Path]) -> int:
    """Remove Town07 entries from the manifest. Returns count of removed entries."""
    if not manifest_path.exists():
        print(f"  No manifest found at {manifest_path} — skipping manifest update")
        return 0

    with open(manifest_path) as f:
        manifest = json.load(f)

    removed_names = {p.name for p in removed_files}
    original_count = 0
    removed_count = 0

    # Handle different manifest formats
    if isinstance(manifest, list):
        original_count = len(manifest)
        manifest = [entry for entry in manifest if not _entry_matches_town07(entry, removed_names)]
        removed_count = original_count - len(manifest)
    elif isinstance(manifest, dict):
        # Could be keyed by episode name or have an "episodes" list
        if "episodes" in manifest:
            original_count = len(manifest["episodes"])
            manifest["episodes"] = [
                entry
                for entry in manifest["episodes"]
                if not _entry_matches_town07(entry, removed_names)
            ]
            removed_count = original_count - len(manifest["episodes"])
        elif "completed" in manifest:
            original_count = len(manifest["completed"])
            manifest["completed"] = [
                entry
                for entry in manifest["completed"]
                if not _entry_matches_town07(entry, removed_names)
            ]
            removed_count = original_count - len(manifest["completed"])
        else:
            # Keyed by filename or ID
            keys_to_remove = []
            for key, val in manifest.items():
                if isinstance(val, dict) and val.get("town") == "Town07":
                    keys_to_remove.append(key)
                elif key in removed_names:
                    keys_to_remove.append(key)
            original_count = len(manifest)
            for key in keys_to_remove:
                del manifest[key]
            removed_count = len(keys_to_remove)

    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)

    return removed_count


def _entry_matches_town07(entry, removed_names: set[str]) -> bool:
    """Check if a manifest entry is a Town07 episode."""
    if isinstance(entry, str):
        return Path(entry).name in removed_names or "Town07" in entry
    if isinstance(entry, dict):
        if entry.get("town") == "Town07":
            return True
        filename = entry.get("filename", entry.get("path", entry.get("episode", "")))
        if isinstance(filename, str) and Path(filename).name in removed_names:
            return True
    return False


def main():
    parser = argparse.ArgumentParser(description="Remove Town07 data from collection")
    parser.add_argument("data_dir", type=Path, help="Path to collected data directory")
    parser.add_argument(
        "--delete",
        action="store_true",
        help="Actually delete files instead of moving to backup dir",
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="Show what would be removed without doing it"
    )
    args = parser.parse_args()

    data_dir = args.data_dir
    if not data_dir.exists():
        print(f"ERROR: {data_dir} does not exist")
        sys.exit(1)

    print(f"Scanning {data_dir} for Town07 episodes...")
    town07_files = find_town07_episodes(data_dir)

    if not town07_files:
        print("No Town07 episodes found. Nothing to do.")
        sys.exit(0)

    # Calculate total size
    total_bytes = sum(f.stat().st_size for f in town07_files)
    total_gb = total_bytes / (1024**3)

    print(f"\nFound {len(town07_files)} Town07 episodes ({total_gb:.1f} GB):")
    for f in town07_files[:10]:
        print(f"  {f.name} ({f.stat().st_size / (1024**2):.0f} MB)")
    if len(town07_files) > 10:
        print(f"  ... and {len(town07_files) - 10} more")

    if args.dry_run:
        print("\n[DRY RUN] No files modified.")
        sys.exit(0)

    # Confirm
    print(
        f"\nThis will {'DELETE' if args.delete else 'MOVE'} "
        f"{len(town07_files)} files ({total_gb:.1f} GB)"
    )
    response = input("Continue? [y/N] ").strip().lower()
    if response != "y":
        print("Aborted.")
        sys.exit(0)

    # Backup manifest
    manifest_path = data_dir / "collection_manifest.json"
    if manifest_path.exists():
        backup_path = backup_manifest(manifest_path)
        print(f"\nManifest backed up to: {backup_path}")

    # Move or delete Town07 files
    if args.delete:
        print(f"\nDeleting {len(town07_files)} Town07 episode files...")
        for f in town07_files:
            f.unlink()
            print(f"  Deleted: {f.name}")
    else:
        backup_dir = data_dir / "town07_backup"
        backup_dir.mkdir(exist_ok=True)
        print(f"\nMoving {len(town07_files)} Town07 episodes to {backup_dir}/...")
        for f in town07_files:
            dest = backup_dir / f.name
            shutil.move(str(f), str(dest))
            print(f"  Moved: {f.name}")
        print(f"\nBackup location: {backup_dir}")
        print(f"To permanently delete: rm -rf {backup_dir}")

    # Update manifest
    removed_count = update_manifest(manifest_path, town07_files)
    print(f"\nManifest: removed {removed_count} Town07 entries")

    # Summary
    remaining = list(data_dir.glob("episode_*.h5"))
    print(f"\nDone! Remaining episodes: {len(remaining)}")


if __name__ == "__main__":
    main()
