"""
merge_h5.py — Merge per-job HDF5 files into train/val/test splits.

Each per-job file contains datasets:
    data        [N, MAX_CLUSTERS, NUM_CLUSTER_FEATURES]
    tracks      [N, MAX_TRACKS,   NUM_TRACK_FEATURES]
    cells_per_cluster [N, MAX_CLUSTERS, MAX_CELLS_PER_CLUSTER, NUM_CELL_FEATURES]
    pid         [N]
    decay_mode  [N]

Usage:
    # After rucio download, point at the directory containing all *.h5 files:
    python scripts/merge_h5.py --raw_dir /path/to/downloaded/h5s \\
                                --output_dir /path/to/training_data

    # Optionally adjust split fractions:
    python scripts/merge_h5.py --raw_dir ... --output_dir ... \\
        --train_frac 0.6 --val_frac 0.2 --seed 42

    # If files are spread across multiple subdirectories (one per rucio dataset):
    python scripts/merge_h5.py \\
        --raw_dir /path/jz2_download /path/tau_download /path/ele_download \\
        --output_dir /path/to/training_data
"""

import os
import glob
import argparse
import numpy as np
import h5py
from tqdm import tqdm

DATASETS = ["data", "tracks", "cells_per_cluster", "pid", "decay_mode"]


def find_h5_files(directories):
    files = []
    for d in directories:
        found = sorted(glob.glob(os.path.join(d, "**", "*.h5"), recursive=True))
        files.extend(found)
    return files


def load_h5(filepath):
    arrays = {}
    with h5py.File(filepath, "r") as f:
        for key in DATASETS:
            arrays[key] = f[key][:]
    return arrays


def main():
    parser = argparse.ArgumentParser(description="Merge per-job HDF5 files into train/val/test splits")
    parser.add_argument("--raw_dir", type=str, nargs="+",
                        default=["/pscratch/sd/m/milescb/OmniTau/OmniLearnedData/training_data/raw"],
                        help="Directory (or directories) containing per-job .h5 files")
    parser.add_argument("--output_dir", type=str,
                        default="/pscratch/sd/m/milescb/OmniTau/OmniLearnedData/training_data",
                        help="Directory to write train/val/test splits")
    parser.add_argument("--train_frac", type=float, default=0.6)
    parser.add_argument("--val_frac", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    h5_files = find_h5_files(args.raw_dir)
    if not h5_files:
        raise FileNotFoundError(f"No .h5 files found under: {args.raw_dir}")

    print(f"Found {len(h5_files)} file(s) to merge:")
    for p in h5_files:
        print(f"  {p}")

    all_arrays = {k: [] for k in DATASETS}
    for filepath in tqdm(h5_files, desc="Loading"):
        arrays = load_h5(filepath)
        for k in DATASETS:
            all_arrays[k].append(arrays[k])

    merged = {k: np.concatenate(all_arrays[k], axis=0) for k in DATASETS}
    n_total = len(merged["pid"])

    print(f"\nTotal samples: {n_total}")
    for label, name in [(0, "QCD"), (1, "tau"), (2, "electron")]:
        print(f"  {name} ({label}): {np.sum(merged['pid'] == label)}")
    dm = merged["decay_mode"]
    print(f"  Decay modes 0-4: {[int(np.sum(dm == i)) for i in range(5)]}")
    print(f"  Decay mode N/A:  {int(np.sum((dm < 0) | (dm > 4)))}")

    rng = np.random.default_rng(args.seed)
    idx = rng.permutation(n_total)
    merged = {k: v[idx] for k, v in merged.items()}

    n_train = int(n_total * args.train_frac)
    n_val = int(n_total * args.val_frac)
    slices = {
        "train": slice(0, n_train),
        "val":   slice(n_train, n_train + n_val),
        "test":  slice(n_train + n_val, None),
    }

    print(f"\nSplit sizes:")
    for name, sl in slices.items():
        print(f"  {name}: {len(merged['pid'][sl])}")

    for split_name, sl in slices.items():
        split_dir = os.path.join(args.output_dir, split_name)
        os.makedirs(split_dir, exist_ok=True)
        out_path = os.path.join(split_dir, "data.h5")
        print(f"Writing {out_path} ...")
        with h5py.File(out_path, "w") as f:
            for k in DATASETS:
                arr = merged[k][sl]
                kwargs = {"compression": "gzip"} if arr.ndim > 1 else {}
                f.create_dataset(k, data=arr, **kwargs)

        n_samples = len(merged["pid"][sl])
        file_indices = np.array([(0, i) for i in range(n_samples)], dtype=np.int32)
        np.save(os.path.join(split_dir, "file_index.npy"), file_indices)

    print("\nDone.")


if __name__ == "__main__":
    main()
