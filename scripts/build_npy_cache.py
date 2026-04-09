"""
build_npy_cache.py

Pre-build the .npy memory-mapped cache and copy/create file_index.npy from
HDF5 datasets into a separate output directory tree, without copying the
.h5 files themselves.

The output mirrors the input directory structure exactly:

  <dst>/<dataset>/<split>/
      file_index.npy
      npy_cache/
          <stem>_<key>.npy
          ...

Usage
-----
# Full mirror of a source tree into scratch space:
python build_npy_cache.py \\
    --src  /global/cfs/projectdirs/m2616/TauCPML/DataTesting/processed_h5 \\
    --dst  /pscratch/sd/m/milescb/processed_h5

# Process only a single split directory (--flat skips tree discovery):
python build_npy_cache.py \\
    --src  /global/.../processed_h5/tau/train \\
    --dst  /pscratch/.../processed_h5/tau/train \\
    --flat

# Override which keys to cache:
python build_npy_cache.py \\
    --src /global/.../processed_h5 --dst /pscratch/.../processed_h5 \\
    --keys data pid tracks cells_per_cluster decay_mode data_pid

# Parallel split-level processing:
python build_npy_cache.py \\
    --src /global/.../processed_h5 --dst /pscratch/.../processed_h5 \\
    --workers 4 --chunk-size 20000
"""

import argparse
import concurrent.futures
import sys
from pathlib import Path

import h5py
import numpy as np

DEFAULT_KEYS = [
    "data",
    "pid",
    "tracks",
    "cells_per_cluster",
    "decay_mode",
    "tau_targets",
    "charged_pion_targets",
    "neutral_pion_targets"
]


# ---------------------------------------------------------------------------
# file_index
# ---------------------------------------------------------------------------

def build_file_index(h5_files: list[Path]) -> np.ndarray:
    """Build (N, 2) int32 array of (file_idx, sample_idx) pairs."""
    entries: list[tuple[int, int]] = []
    for file_idx, h5_path in enumerate(h5_files):
        with h5py.File(h5_path, "r") as f:
            n = len(f["data"])
        entries.extend((file_idx, i) for i in range(n))
        print(f"    Indexed {h5_path.name}: {n:,} samples")
    return np.array(entries, dtype=np.int32)


def ensure_file_index(
    src_split_dir: Path,
    dst_split_dir: Path,
    h5_files: list[Path],
    overwrite: bool,
) -> None:
    dst_index = dst_split_dir / "file_index.npy"
    if dst_index.exists() and not overwrite:
        print("  file_index.npy: already exists, skipping")
        return

    src_index = src_split_dir / "file_index.npy"
    if src_index.exists():
        print("  file_index.npy: copying from source")
        dst_index.write_bytes(src_index.read_bytes())
    else:
        print("  file_index.npy: building from scratch")
        index = build_file_index(h5_files)
        np.save(str(dst_index), index)
        print(f"  file_index.npy: saved ({len(index):,} entries)")


# ---------------------------------------------------------------------------
# npy cache
# ---------------------------------------------------------------------------

def cache_h5_file(
    h5_path: Path,
    cache_dir: Path,
    keys: list[str],
    chunk_size: int,
    overwrite: bool,
) -> None:
    """Convert one HDF5 file into per-key .npy mmap arrays under cache_dir."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    stem = h5_path.stem

    with h5py.File(h5_path, "r") as f:
        available = set(f.keys())
        to_cache = [k for k in keys if k in available]
        skipped  = [k for k in keys if k not in available]

        if skipped:
            print(f"  [{h5_path.name}] keys not found (skipped): {skipped}")

        for key in to_cache:
            npy_path = cache_dir / f"{stem}_{key}.npy"
            if npy_path.exists() and not overwrite:
                print(f"  [{h5_path.name}] '{key}': already cached, skipping")
                continue

            ds = f[key]
            shape, dtype = ds.shape, ds.dtype
            print(f"  [{h5_path.name}] caching '{key}' {shape} {dtype}")

            out = np.lib.format.open_memmap(
                str(npy_path), mode="w+", dtype=dtype, shape=shape
            )
            for start in range(0, shape[0], chunk_size):
                end = min(start + chunk_size, shape[0])
                out[start:end] = ds[start:end]
            del out  # flush to disk

    print(f"  [{h5_path.name}] done")


# ---------------------------------------------------------------------------
# split-level entry point
# ---------------------------------------------------------------------------

def process_split(
    src_split_dir: Path,
    dst_split_dir: Path,
    keys: list[str],
    chunk_size: int,
    overwrite: bool,
) -> None:
    h5_files = sorted(src_split_dir.glob("*.h5")) + sorted(src_split_dir.glob("*.hdf5"))
    if not h5_files:
        print(f"  WARNING: no .h5/.hdf5 files in {src_split_dir}, skipping")
        return

    dst_split_dir.mkdir(parents=True, exist_ok=True)

    print(f"  {len(h5_files)} HDF5 file(s)")

    # 1. file_index.npy
    ensure_file_index(src_split_dir, dst_split_dir, h5_files, overwrite)

    # 2. npy_cache/<stem>_<key>.npy  (one set per h5 file)
    cache_dir = dst_split_dir / "npy_cache"
    for h5_path in h5_files:
        cache_h5_file(h5_path, cache_dir, keys, chunk_size, overwrite)


# ---------------------------------------------------------------------------
# tree discovery
# ---------------------------------------------------------------------------

def find_split_dirs(root: Path) -> list[Path]:
    """Return every directory under root that directly contains .h5/.hdf5 files."""
    if any(root.glob("*.h5")) or any(root.glob("*.hdf5")):
        return [root]
    return sorted(
        p for p in root.rglob("*")
        if p.is_dir() and (any(p.glob("*.h5")) or any(p.glob("*.hdf5")))
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Build .npy mmap cache + file_index.npy from HEP HDF5 datasets "
            "into a separate output directory (no .h5 files copied)."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--src", required=True, metavar="DIR",
                   help="Root of the source HDF5 tree.")
    p.add_argument("--dst", required=True, metavar="DIR",
                   help="Root of the output tree. Mirrors --src structure.")
    p.add_argument("--keys", nargs="+", default=DEFAULT_KEYS, metavar="KEY",
                   help="HDF5 dataset keys to convert to .npy.")
    p.add_argument("--chunk-size", type=int, default=10_000, metavar="N",
                   help="Rows read per HDF5 chunk.")
    p.add_argument("--overwrite", action="store_true",
                   help="Overwrite existing .npy / file_index.npy files.")
    p.add_argument("--workers", type=int, default=1, metavar="N",
                   help="Parallel threads (one per split directory).")
    p.add_argument("--flat", action="store_true",
                   help="Treat --src/--dst as a single leaf split dir, skip tree discovery.")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    src_root = Path(args.src).resolve()
    dst_root = Path(args.dst).resolve()

    if not src_root.exists():
        print(f"ERROR: --src '{src_root}' does not exist.", file=sys.stderr)
        sys.exit(1)

    if args.flat:
        split_pairs = [(src_root, dst_root)]
    else:
        src_splits = find_split_dirs(src_root)
        if not src_splits:
            print(f"ERROR: No .h5 files found under {src_root}", file=sys.stderr)
            sys.exit(1)
        split_pairs = [
            (s, dst_root / s.relative_to(src_root)) for s in src_splits
        ]

    print(f"Source root : {src_root}")
    print(f"Dest root   : {dst_root}")
    print(f"Keys        : {args.keys}")
    print(f"Chunk size  : {args.chunk_size:,}")
    print(f"Overwrite   : {args.overwrite}")
    print(f"Workers     : {args.workers}")
    print(f"Splits found: {len(split_pairs)}")
    for src_s, dst_s in split_pairs:
        rel = src_s.relative_to(src_root) if not args.flat else Path(".")
        print(f"  {rel}  ->  {dst_s}")
    print()

    def _task(pair: tuple[Path, Path]) -> None:
        src_s, dst_s = pair
        label = src_s.relative_to(src_root) if not args.flat else src_s.name
        print(f"── {label} ──")
        process_split(src_s, dst_s, args.keys, args.chunk_size, args.overwrite)
        print()

    if args.workers <= 1:
        for pair in split_pairs:
            _task(pair)
    else:
        with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
            futures = {pool.submit(_task, p): p for p in split_pairs}
            for fut in concurrent.futures.as_completed(futures):
                exc = fut.exception()
                if exc:
                    src_s, _ = futures[fut]
                    print(f"ERROR in {src_s}: {exc}", file=sys.stderr)

    print("Done.")


if __name__ == "__main__":
    main()