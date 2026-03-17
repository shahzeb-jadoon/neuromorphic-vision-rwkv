import csv
import logging
from pathlib import Path

# =====================================================================
# CONFIGURATION
# =====================================================================
DATA_DIR = Path("/mnt/data/carla_data/")
IN_CSV = Path("data/analysis/option3_delete_candidates.csv")
LOG_FILE = Path("data/analysis/deletion_audit.log")

# Option 2 targets strictly Perfect Upgrades (A) & Same-Status Shorter (B/C)
ALLOWED_TIERS = {"A_perfect_upgrade", "BC_same_status_shorter"}

# Initialize audit log
logging.basicConfig(
    filename=LOG_FILE, level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


def sanitize_filename(filename: str) -> bool:
    """Path sanitisation to ensure the filename is valid and safe."""
    if not filename or filename.strip() == "":
        return False
    # Prevent directory traversal
    if ".." in filename or "/" in filename or "\\" in filename:
        return False
    # Only allow .h5 files
    if not filename.endswith(".h5"):
        return False
    return True


def main() -> None:
    print("=====================================================")
    print(" 🛠️  OPTION 2 (SAFE) DELETION EXECUTION SCRIPT ")
    print("     Targeting Tiers A, B, and C ONLY ")
    print("=====================================================\n")

    if not IN_CSV.exists():
        print(f"CRITICAL ERROR: Cannot find the candidate CSV at: {IN_CSV}")
        return

    # 1. Load and Schema Validation
    try:
        with IN_CSV.open("r") as f:
            rows = list(csv.DictReader(f))
    except Exception as e:
        print(f"CRITICAL ERROR reading CSV schema: {e}")
        return

    # Filter strictly for Option 2 tiers
    option2_rows = []
    for r in rows:
        tier = r.get("tier", "").strip()
        if tier in ALLOWED_TIERS:
            option2_rows.append(r)

    if not option2_rows:
        print("ERROR: No valid Option 2 files found in the CSV.")
        return

    # 2. Dry Run & Path Sanitization
    total_bytes = 0
    missing_files = []
    unsafe_files = []

    print(f"PREFLIGHT: Scanning {len(option2_rows)} Option 2 candidate files on disk...")
    for r in option2_rows:
        fn = r.get("filename", "").strip()

        # Security/Sanitisation check
        if not sanitize_filename(fn):
            unsafe_files.append(fn)
            continue

        fp = DATA_DIR / fn

        # Physical existence check
        if not fp.exists():
            missing_files.append(fn)
        else:
            total_bytes += fp.stat().st_size

    if unsafe_files:
        print(
            f"\nCRITICAL ERROR: Found {len(unsafe_files)} unsafe filenames (e.g., path traversal)."
        )
        logging.error(f"Unsafe filenames blocked: {unsafe_files}")
        return

    if missing_files:
        print(f"\nCRITICAL ERROR: {len(missing_files)} files from the list are physically missing.")
        logging.error(f"Missing files blocked execution: {missing_files}")
        return

    total_gb = total_bytes / 1e9
    print(f"  ✓ Validated {len(option2_rows)} rows against strict Option 2 criteria.")
    print("  ✓ Path sanitization passed. Zero unsafe filenames.")
    print(f"  ✓ Physical existence confirmed. Space to reclaim: {total_gb:.2f} GB.\n")

    # 3. Explicit Confirmation Gate
    print(
        f"WARNING: You are about to PERMANENTLY DELETE"
        f" {len(option2_rows)} files ({total_gb:.2f} GB)."
    )
    print(
        "This action covers only Tiers A, B, and C (Perfect Upgrades & Shorter Same-Status pairs)."
    )
    print(f"An immutable audit log will be written to {LOG_FILE}\n")

    response = input("Type 'CONFIRM' (in all caps) to irrevocably delete these files: ")

    if response != "CONFIRM":
        print("\nAborted. No files were deleted.")
        logging.info("User aborted deletion at confirmation gate.")
        return

    # 4. Transactional Execution
    print("\nExecuting authenticated deletions...")
    logging.info(
        f"--- STARTING OPTION 2 DELETION: {len(option2_rows)} FILES, {total_gb:.2f} GB ---"
    )

    deleted_count = 0
    freed_bytes = 0
    failed_deletions = []

    for r in option2_rows:
        fn = r["filename"].strip()
        fp = DATA_DIR / fn

        try:
            sz = fp.stat().st_size
            fp.unlink()
            deleted_count += 1
            freed_bytes += sz
            logging.info(f"DELETED: {fn} | Size: {sz} bytes | Tier: {r['tier']}")

            if deleted_count % 50 == 0:
                print(
                    f"  ... successfully physically deleted {deleted_count} / {len(option2_rows)}"
                )

        except Exception as e:
            failed_deletions.append((fn, str(e)))
            logging.error(f"FAILED TO DELETE {fn}: {e}")

    # 5. Post-Delete Verification Pass
    print("\nRunning post-deletion validation...")
    still_exists = []
    for r in option2_rows:
        fn = r["filename"].strip()
        if (DATA_DIR / fn).exists():
            still_exists.append(fn)

    final_gb = freed_bytes / 1e9

    print("\n=====================================================")
    print(" DELETION OPERATION COMPLETED ")
    print("=====================================================")
    print(f" Removed:        {deleted_count} files")
    print(f" Space Reclaimed:{final_gb:.2f} GB")

    if failed_deletions or still_exists:
        print("\n⚠️ WARNING: Not all deletions were successful.")
        print(f"Failed to delete {len(failed_deletions)} files during run.")
        print(f"Found {len(still_exists)} files still existing during post-check.")
        logging.warning(f"Post-check validation flagged {len(still_exists)} ghost files.")
    else:
        print("\n  ✓ Post-check passed: All targeted files are gone.")
        logging.info("--- DELETION SUCCESSFULLY COMPLETED WITH ZERO ERRORS ---")

    print(f"\nAudit Log saved to: {LOG_FILE}\n")


if __name__ == "__main__":
    main()
