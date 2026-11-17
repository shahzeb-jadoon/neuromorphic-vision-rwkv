"""Verify everything preserved from a dataset against its source, before the source is deleted.

Checks, each against the original files:
- golden subset copies: signals equal to the source window, image chunks byte-identical,
  file checksums equal to the index;
- testbed cache: one file per episode, the expected image count, every signal identical,
  and sampled images close to the resized source frame (PSNR);
- copied files (manifests, logs, backups, run folders): sha256 equal to the source.

Prints counts and checksums and writes them to OUT.json; exits non-zero on any failure.

usage: python scripts/verify_preserved.py --scan SCAN.json --source DIR --cache DIR
           --golden DIR [DIR ...] --copy SRC DST [--copy SRC DST ...] --out OUT.json
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from src.data import hdf5_schema as S

MIN_PSNR_DB = 30.0


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(1 << 22), b""):
            h.update(block)
    return h.hexdigest()


def signals_equal(a: h5py.Dataset, b: h5py.Dataset, rows: slice | None = None) -> bool:
    x = a[rows] if rows is not None else a[()]
    y = b[()]
    if a.dtype.kind == "O":
        return list(x) == list(y)
    return a.dtype == b.dtype and np.array_equal(x, y)


def check_golden(directory: Path, source: Path) -> dict[str, Any]:
    index = json.loads((directory / "index.json").read_text())["windows"]
    problems = []
    for w in index:
        path = directory / w["file"]
        if sha256(path) != w["sha256"]:
            problems.append(f"{w['file']}: checksum differs from index")
            continue
        rows = slice(w["start"], w["start"] + w["frames"])
        with h5py.File(source / w["source"], "r") as f, h5py.File(path, "r") as g:
            for name, ds in f.items():
                if not isinstance(ds, h5py.Dataset):
                    continue
                if name == S.IMAGES and ds.chunks == (1, *ds.shape[1:]):
                    same = all(
                        ds.id.read_direct_chunk((w["start"] + i, 0, 0, 0))[1]
                        == g[name].id.read_direct_chunk((i, 0, 0, 0))[1]
                        for i in range(w["frames"])
                    )
                elif name == S.IMAGES:
                    same = np.array_equal(ds[rows], g[name][()])
                else:
                    same = signals_equal(ds, g[name], rows)
                if not same:
                    problems.append(f"{w['file']}: {name} differs from source")
    return {"windows": len(index), "frames": sum(w["frames"] for w in index), "problems": problems}


def psnr(a: np.ndarray, b: np.ndarray) -> float:
    mse = float(np.mean((a.astype(np.float64) - b.astype(np.float64)) ** 2))
    return 99.0 if mse == 0 else 10 * np.log10(255.0**2 / mse)


def check_cache_episode(src: Path, cache: Path) -> dict[str, Any]:
    if not cache.exists():
        return {"file": src.name, "problems": ["missing"]}
    problems: list[str] = []
    with h5py.File(src, "r") as f, h5py.File(cache, "r") as g:
        meta = g[S.METADATA_GROUP].attrs
        stride = int(meta[S.META_IMAGE_STRIDE])
        n = int(f[S.IMAGES].shape[0])
        jpegs = g[S.IMAGES_JPEG]
        if len(jpegs) != -(-n // stride):
            problems.append(f"{len(jpegs)} images for {n} source frames")
        for name, ds in f.items():
            if isinstance(ds, h5py.Dataset) and name != S.IMAGES and not signals_equal(ds, g[name]):
                problems.append(f"{name} differs from source")
        h, w = np.asarray(Image.open(io.BytesIO(jpegs[0].tobytes()))).shape[:2]
        scores = []
        for k in sorted({0, len(jpegs) // 2, len(jpegs) - 1}):
            frame = torch.from_numpy(f[S.IMAGES][k * stride]).permute(2, 0, 1)[None].float()
            small = F.interpolate(frame, size=(h, w), mode="bilinear", align_corners=False)
            expected = small.round().clamp(0, 255)[0].permute(1, 2, 0).numpy()
            decoded = np.asarray(Image.open(io.BytesIO(jpegs[k].tobytes())).convert("RGB"))
            scores.append(psnr(decoded, expected))
        if min(scores) < MIN_PSNR_DB:
            problems.append(f"sampled image PSNR {min(scores):.1f} dB")
    return {
        "file": src.name,
        "images": int(len(jpegs)),
        "min_psnr_db": round(min(scores), 1),
        "sha256": sha256(cache),
        "problems": problems,
    }


def check_cache(scan: dict[str, Any], source: Path, cache: Path, workers: int) -> dict[str, Any]:
    names = [e["file"] for e in scan["episodes"]]
    with ProcessPoolExecutor(workers) as pool:
        results = list(
            pool.map(check_cache_episode, [source / n for n in names], [cache / n for n in names])
        )
    return {
        "episodes": len(results),
        "images": sum(r.get("images", 0) for r in results),
        "bytes": sum(
            (cache / r["file"]).stat().st_size for r in results if (cache / r["file"]).exists()
        ),
        "min_psnr_db": min((r["min_psnr_db"] for r in results if "min_psnr_db" in r), default=None),
        "problems": [f"{r['file']}: {p}" for r in results for p in r["problems"]],
        "sha256": {r["file"]: r["sha256"] for r in results if "sha256" in r},
    }


def check_copy(src: Path, dst: Path) -> dict[str, Any]:
    files = (
        sorted(p for p in src.rglob("*") if p.is_file() and not p.name.startswith("."))
        if src.is_dir()
        else [src]
    )
    checksums, problems = {}, []
    for p in files:
        rel = p.relative_to(src) if src.is_dir() else Path(p.name)
        target = dst / rel if src.is_dir() else dst
        digest = sha256(p)
        checksums[str(rel)] = digest
        if not target.exists() or sha256(target) != digest:
            problems.append(f"{rel}: copy missing or different")
    return {"source": src.name, "files": len(files), "sha256": checksums, "problems": problems}


def check_registry(registry: Path, copies: list[dict[str, Any]]) -> dict[str, Any]:
    """Compare copied files with the sha256 lines (`<hex>  <path>`) recorded in a registry."""
    recorded = {}
    for line in registry.read_text().splitlines():
        parts = line.split()
        if (
            len(parts) == 2
            and len(parts[0]) == 64
            and all(c in "0123456789abcdef" for c in parts[0])
        ):
            recorded[parts[1]] = parts[0]
    found = {rel: digest for c in copies for rel, digest in c["sha256"].items() if rel in recorded}
    problems = [
        f"{rel}: differs from {registry.name}" for rel, d in found.items() if recorded[rel] != d
    ]
    missing = sorted(set(recorded) - set(found))
    return {
        "registry": registry.name,
        "matched": len(found),
        "missing": missing,
        "problems": problems,
    }


def main() -> None:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--scan", type=Path, required=True)
    p.add_argument("--source", type=Path, required=True)
    p.add_argument("--cache", type=Path, required=True)
    p.add_argument("--golden", type=Path, nargs="+", required=True)
    p.add_argument(
        "--copy", type=Path, nargs=2, action="append", default=[], metavar=("SRC", "DST")
    )
    p.add_argument(
        "--registry", type=Path, help="Checksum registry for copied files, e.g. docs/RUNS.md"
    )
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--workers", type=int, default=8)
    args = p.parse_args()

    scan = json.loads(args.scan.read_text())
    report: dict[str, Any] = {
        "golden": {str(d): check_golden(d, args.source) for d in args.golden},
        "cache": check_cache(scan, args.source, args.cache, args.workers),
        "copies": [check_copy(src, dst) for src, dst in args.copy],
    }
    if args.registry:
        report["registry"] = check_registry(args.registry, report["copies"])
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=1))

    problems = [x for g in report["golden"].values() for x in g["problems"]]
    problems += report["cache"]["problems"] + [x for c in report["copies"] for x in c["problems"]]
    if "registry" in report:
        problems += report["registry"]["problems"] + [
            f"{m}: not checked" for m in report["registry"]["missing"]
        ]
    for d, g in report["golden"].items():
        print(f"golden {d}: {g['windows']} windows, {g['frames']:,} frames")
    c = report["cache"]
    print(
        f"cache: {c['episodes']} episodes, {c['images']:,} images, {c['bytes'] / 1e9:.1f} GB, "
        f"lowest sampled PSNR {c['min_psnr_db']} dB"
    )
    for c in report["copies"]:
        print(f"copy {c['source']}: {c['files']} files, sha256 checked")
    if "registry" in report:
        r = report["registry"]
        print(f"registry {r['registry']}: {r['matched']} recorded checksums matched")
    print(
        f"\n{'PASS' if not problems else 'FAIL'}: {len(problems)} problems; checksums in {args.out}"
    )
    for x in problems[:50]:
        print("  ", x)
    raise SystemExit(1 if problems else 0)


if __name__ == "__main__":
    main()
