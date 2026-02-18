"""Validate HDF5 driving dataset for integrity and anomalies.

Checks each episode file for:
  - HDF5 file integrity (can be opened and read)
  - All required dataset fields present
  - Schema version metadata
  - NaN/Inf values in control signals
  - All-black images (camera failure)
  - Image shape consistency
  - Control value ranges (steering [-1,1], throttle [0,1], brake [0,1])
  - Nav command range [0, 3]
  - Frame count consistency (n_frames attr vs actual array length)

Usage:
    python scripts/validate_dataset.py data/collected/
    python scripts/validate_dataset.py data/toy/ --strict
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass, field
from pathlib import Path

import h5py
import numpy as np

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data import hdf5_schema as S  # noqa: E402

REQUIRED_FIELDS = [S.IMAGES, S.STEERING, S.THROTTLE, S.VELOCITY, S.NAV_COMMAND]
OPTIONAL_FIELDS = [S.BRAKE]


@dataclass
class ValidationResult:
    """Result of validating a single episode file."""

    filepath: Path
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    n_frames: int = 0

    @property
    def ok(self) -> bool:
        return len(self.errors) == 0


def validate_episode(filepath: Path, strict: bool = False) -> ValidationResult:
    """Validate a single episode HDF5 file.

    Args:
        filepath: Path to the episode .h5 file.
        strict: If True, treat warnings as errors.

    Returns:
        ValidationResult with errors and warnings.
    """
    result = ValidationResult(filepath=filepath)

    # --- 1. Can we open the file? ---
    try:
        f = h5py.File(filepath, "r")
    except Exception as e:
        result.errors.append(f"Cannot open file: {e}")
        return result

    try:
        # --- 2. Required fields present ---
        for field_name in REQUIRED_FIELDS:
            if field_name not in f:
                result.errors.append(f"Missing required field: {field_name}")

        for field_name in OPTIONAL_FIELDS:
            if field_name not in f:
                result.warnings.append(f"Missing optional field: {field_name}")

        if result.errors:
            # Can't do further checks without required fields
            return result

        # --- 3. Schema version ---
        if S.METADATA_GROUP in f:
            meta = f[S.METADATA_GROUP]
            if S.META_SCHEMA_VERSION not in meta.attrs:
                result.warnings.append("Missing schema_version in metadata")
            else:
                version = int(meta.attrs[S.META_SCHEMA_VERSION])
                if version != S.SCHEMA_VERSION:
                    result.warnings.append(
                        f"Schema version mismatch: file={version}, expected={S.SCHEMA_VERSION}"
                    )
        else:
            result.warnings.append(f"Missing {S.METADATA_GROUP} group")

        # --- 4. Get frame counts and check consistency ---
        images = f[S.IMAGES]
        n_frames = images.shape[0]
        result.n_frames = n_frames

        # Check all arrays have same frame count
        for field_name in [S.STEERING, S.THROTTLE, S.NAV_COMMAND]:
            arr_len = f[field_name].shape[0]
            if arr_len != n_frames:
                result.errors.append(
                    f"Frame count mismatch: {S.IMAGES}={n_frames}, {field_name}={arr_len}"
                )

        if S.VELOCITY in f:
            vel_len = f[S.VELOCITY].shape[0]
            if vel_len != n_frames:
                result.errors.append(
                    f"Frame count mismatch: {S.IMAGES}={n_frames}, {S.VELOCITY}={vel_len}"
                )

        if S.BRAKE in f:
            brake_len = f[S.BRAKE].shape[0]
            if brake_len != n_frames:
                result.errors.append(
                    f"Frame count mismatch: {S.IMAGES}={n_frames}, {S.BRAKE}={brake_len}"
                )

        # Check metadata n_frames matches actual
        if S.METADATA_GROUP in f and S.META_N_FRAMES in f[S.METADATA_GROUP].attrs:
            meta_n_frames = int(f[S.METADATA_GROUP].attrs[S.META_N_FRAMES])
            if meta_n_frames != n_frames:
                result.errors.append(
                    f"Metadata n_frames={meta_n_frames} != actual array length={n_frames}"
                )

        # --- 5. Image shape consistency ---
        img_shape = images.shape
        if len(img_shape) != 4:
            result.errors.append(f"Images should be 4D (N,H,W,C), got shape {img_shape}")
        elif img_shape[3] != 3:
            result.errors.append(f"Images should have 3 channels, got {img_shape[3]}")

        # --- 6. Check for all-black images (sample first, middle, last) ---
        check_indices = [0, n_frames // 2, n_frames - 1]
        black_count = 0
        for idx in check_indices:
            if idx < n_frames:
                img = images[idx]
                if np.all(img == 0):
                    black_count += 1

        if black_count == len(check_indices):
            result.errors.append(
                f"All {len(check_indices)} sampled images are all-black (camera failure?)"
            )
        elif black_count > 0:
            result.warnings.append(
                f"{black_count}/{len(check_indices)} sampled images are all-black"
            )

        # --- 7. NaN/Inf in controls ---
        steering = f[S.STEERING][:]
        throttle = f[S.THROTTLE][:]

        if np.any(np.isnan(steering)) or np.any(np.isinf(steering)):
            result.errors.append("NaN or Inf values in steering")
        if np.any(np.isnan(throttle)) or np.any(np.isinf(throttle)):
            result.errors.append("NaN or Inf values in throttle")

        if S.VELOCITY in f:
            velocity = f[S.VELOCITY][:]
            if np.any(np.isnan(velocity)) or np.any(np.isinf(velocity)):
                result.errors.append("NaN or Inf values in velocity")

        if S.BRAKE in f:
            brake = f[S.BRAKE][:]
            if np.any(np.isnan(brake)) or np.any(np.isinf(brake)):
                result.errors.append("NaN or Inf values in brake")

        # --- 8. Control value ranges ---
        if np.any(steering < -1.0) or np.any(steering > 1.0):
            result.warnings.append(
                f"Steering out of [-1, 1]: min={steering.min():.4f}, max={steering.max():.4f}"
            )

        if np.any(throttle < 0.0) or np.any(throttle > 1.0):
            result.warnings.append(
                f"Throttle out of [0, 1]: min={throttle.min():.4f}, max={throttle.max():.4f}"
            )

        if S.BRAKE in f:
            brake = f[S.BRAKE][:]
            if np.any(brake < 0.0) or np.any(brake > 1.0):
                result.warnings.append(
                    f"Brake out of [0, 1]: min={brake.min():.4f}, max={brake.max():.4f}"
                )

        # --- 9. Nav command range ---
        nav_cmd = f[S.NAV_COMMAND][:]
        valid_cmds = {0, 1, 2, 3}
        unique_cmds = set(np.unique(nav_cmd).tolist())
        invalid_cmds = unique_cmds - valid_cmds
        if invalid_cmds:
            result.errors.append(f"Invalid nav command IDs: {invalid_cmds}")

    finally:
        f.close()

    # In strict mode, warnings become errors
    if strict:
        result.errors.extend(result.warnings)
        result.warnings = []

    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate HDF5 driving dataset")
    parser.add_argument("data_dir", type=str, help="Directory containing episode_*.h5 files")
    parser.add_argument("--strict", action="store_true", help="Treat warnings as errors")
    parser.add_argument("--verbose", "-v", action="store_true", help="Show per-file details")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    if not data_dir.exists():
        print(f"ERROR: Directory not found: {data_dir}")
        sys.exit(1)

    episodes = sorted(data_dir.glob("episode_*.h5"))
    if not episodes:
        print(f"ERROR: No episode_*.h5 files found in {data_dir}")
        sys.exit(1)

    print(f"Validating {len(episodes)} episodes in {data_dir}")
    print()

    total_frames = 0
    error_count = 0
    warning_count = 0
    failed_files: list[str] = []

    for ep_path in episodes:
        result = validate_episode(ep_path, strict=args.strict)
        total_frames += result.n_frames

        if args.verbose or not result.ok:
            status = "PASS" if result.ok else "FAIL"
            print(f"  [{status}] {ep_path.name} ({result.n_frames} frames)")

            for err in result.errors:
                print(f"    ERROR: {err}")
                error_count += 1
            for warn in result.warnings:
                print(f"    WARN:  {warn}")
                warning_count += 1
        elif result.ok and not args.verbose:
            # Just count, don't print each passing file
            pass

        if not result.ok:
            failed_files.append(ep_path.name)

    # Summary
    print()
    print("=" * 60)
    print(f"Episodes:  {len(episodes)}")
    print(f"Frames:    {total_frames:,}")
    print(f"Errors:    {error_count}")
    print(f"Warnings:  {warning_count}")
    print(f"Failed:    {len(failed_files)}")

    if failed_files:
        print()
        print("Failed files:")
        for name in failed_files:
            print(f"  - {name}")
        print()
        print("VALIDATION FAILED")
        sys.exit(1)
    else:
        print()
        print("ALL EPISODES VALID")


if __name__ == "__main__":
    main()
