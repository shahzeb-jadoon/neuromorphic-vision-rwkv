import csv
from pathlib import Path

# Paths
data_dir = Path("/mnt/data/carla_data/")
in_csv = Path("data/analysis/option3_delete_candidates.csv")


def main() -> None:
    print("=====================================================")
    print(" OPTION 3 DELETION EXECUTION SCRIPT ")
    print("=====================================================\n")

    if not in_csv.exists():
        print(f"ERROR: Cannot find the candidate CSV at: {in_csv}")
        print("Please ensure you are running this from the project root directory.")
        return

    # Read the candidate list
    with in_csv.open("r") as f:
        rows = list(csv.DictReader(f))

    # Verification pass (Dry Run)
    total_bytes = 0
    missing_files = []

    print(f"Scanning {len(rows)} candidate files on disk...")
    for r in rows:
        fn = r["filename"]
        fp = data_dir / fn

        if not fp.exists():
            missing_files.append(fn)
        else:
            total_bytes += fp.stat().st_size

    if missing_files:
        print(
            f"\nWARNING: {len(missing_files)} files from the list"
            " are physically missing from the drive."
        )
        print("This means the drive state has changed since the CSV was generated.")
        print("Aborting to be safe. Please regenerate the CSV.")
        return

    total_gb = total_bytes / 1e9
    print(f"Found all {len(rows)} files exactly. They total {total_gb:.2f} GB of space.")
    print("All files belong to verified Option 3 (Tiers A, B, C, and D) candidates.\n")

    # The Final Confirmation Prompt
    print(f"WARNING: This script will PERMANENTLY DELETE {len(rows)} files ({total_gb:.2f} GB).")
    print(
        "These files are the obsolete/shorter paired retries"
        " evaluated in the final storage decision."
    )

    response = input("\nType 'YES' (in all caps) to permanently delete these files: ")

    if response != "YES":
        print("\nAborted. No files were deleted.")
        return

    # Execution Pass
    print("\nExecuting deletions...")
    deleted_count = 0
    freed_bytes = 0

    for r in rows:
        fn = r["filename"]
        fp = data_dir / fn

        try:
            sz = fp.stat().st_size
            fp.unlink()  # This is the actual deletion command
            deleted_count += 1
            freed_bytes += sz
            # Print an update every 50 files so you aren't staring at a blank screen
            if deleted_count % 50 == 0:
                print(f"  ... deleted {deleted_count} / {len(rows)}")

        except Exception as e:
            print(f"  ERROR deleting {fn}: {e}")

    final_gb = freed_bytes / 1e9
    print("\n=====================================================")
    print(" DELETION COMPLETE ")
    print(f" Successfully removed {deleted_count} files.")
    print(f" Reclaimed {final_gb:.2f} GB of SSD space.")
    print("=====================================================")
    print("Your drive now has plenty of room for Town 07. Good luck!")


if __name__ == "__main__":
    main()
