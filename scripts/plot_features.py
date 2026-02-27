"""
Plot cluster and track feature distributions from HDF5 data.

Usage:
    python scripts/plot_features.py --input datasets/tau/train/data.h5 --output plots/
"""

import numpy as np
import h5py
import matplotlib.pyplot as plt
import argparse
import os

# Feature names matching prepare_data.py
CLUSTER_FEATURE_NAMES = [
    "cls_E (log)",
    "cls_ET (log)",
    "cls_Eta",
    "cls_Phi",
    "cls_FIRST_ENG_DENS",
]

TRACK_FEATURE_NAMES = [
    "trk_E (log)",
    "trk_pT (log)",
    "trk_Eta",
    "trk_Phi",
    "trk_charge",
    "trk_d0",
    "trk_z0",
    "trk_z0sintheta",
    "trk_nTRTHits",
    "trk_nTRTHighThresholdHits",
    "trk_nSCTHits",
    "trk_nPixelHits",
    "trk_nBLayerHits",
]

CELL_FEATURE_NAMES = [
    "cell_E (log)",
    "cell_ET (log)",
    "cell_Eta",
    "cell_Phi",
    "cell_Eta_dup",
    "cell_Phi_dup",
    "cell_sintheta",
    "cell_costheta",
    "cell_sinphi",
    "cell_cosphi",
    "cell_layer",
    "cell_x",
    "cell_y",
    "cell_z",
]

REGRESSION_TARGET_NAMES = [
    "truth_loge",
]

CLASS_NAMES = {0: "QCD", 1: "Tau", 2: "Electron"}
CLASS_COLORS = {0: "#e41a1c", 1: "#377eb8", 2: "#4daf4a"}
DECAY_MODE_NAMES = {0: "1p0n", 1: "1p1n", 2: "1pXn", 3: "3p0n", 4: "3pXn"}
DECAY_MODE_COLORS = {0: "#e41a1c", 1: "#377eb8", 2: "#4daf4a", 3: "#984ea3", 4: "#ff7f00"}

MAX_TRACKS = 20
MAX_CELLS = 500
NUM_TRACK_FEATURES = len(TRACK_FEATURE_NAMES)
NUM_CELL_FEATURES = len(CELL_FEATURE_NAMES)


def load_data(filepath):
    """Load HDF5 data."""
    print(f"Loading data from {filepath}...")
    with h5py.File(filepath, "r") as f:
        data = {
            "clusters": f["data"][:],
            "tracks": f["tracks"][:],
            "cells": f["cells"][:],
            "pid": f["pid"][:],
            "decay_mode": f["decay_mode"][:],
        }
        # Load regression targets if present
        if "truth_targets" in f:
            data["targets"] = f["truth_targets"][:]
    print(f"  Loaded {len(data['pid'])} jets")
    print(f"  Clusters shape: {data['clusters'].shape}")
    print(f"  Tracks shape: {data['tracks'].shape}")
    print(f"  Cells shape: {data['cells'].shape}")
    print(f"  Regression targets shape: {data['targets'].shape if 'targets' in data else 'N/A'}")
    
    return data


def get_valid_cluster_values(clusters, feature_idx):
    """Extract non-zero (valid) cluster values for a feature."""
    # Clusters are zero-padded; use feature 0 (cls_E) to find valid entries
    valid_mask = clusters[:, :, 0] != 0
    values = clusters[:, :, feature_idx][valid_mask]
    return values

def get_valid_cell_values(cells, feature_idx):
    """Extract non-zero (valid) cell values for a feature."""
    # Cells are zero-padded; use feature 0 (cell_E) to find valid entries
    valid_mask = cells[:, :, 0] != 0
    values = cells[:, :, feature_idx][valid_mask]
    return values

def get_valid_target_values(targets, target_idx):
    """Extract non-zero (valid) target values for a regression target."""
    # target values are zero-padded; use feature 0 (log E) to find valid entries
    valid_mask = targets[:, :, 0] != 0
    values = targets[:, :, target_idx][valid_mask]
    return values

def get_track_features(tracks, track_idx, feature_idx):
    """Extract a specific track feature from flattened global array."""
    # tracks is [N_jets, MAX_TRACKS, NUM_TRACK_FEATURES]
    n_jets = tracks.shape[0]
    reshaped = tracks.reshape(n_jets, MAX_TRACKS, NUM_TRACK_FEATURES)
    return reshaped[:, track_idx, feature_idx]


def get_all_track_values(tracks, feature_idx):
    """Get all non-zero track values for a feature across all tracks."""
    n_jets = tracks.shape[0]
    reshaped = tracks.reshape(n_jets, MAX_TRACKS, NUM_TRACK_FEATURES)
    # Use feature 1 (log trackPt) to identify valid tracks
    valid_mask = reshaped[:, :, 1] != 0
    values = reshaped[:, :, feature_idx][valid_mask]
    return values


def plot_cluster_features(data, output_dir):
    """Plot individual cluster feature distributions."""
    print("\nPlotting cluster features...")
    
    clusters = data["clusters"]
    pid = data["pid"]
    
    n_features = clusters.shape[2]
    
    for i in range(n_features):
        feature_name = CLUSTER_FEATURE_NAMES[i] if i < len(CLUSTER_FEATURE_NAMES) else f"Feature {i}"
        
        fig, ax = plt.subplots(figsize=(8, 6))
        
        for class_id in [0, 1, 2]:
            mask = pid == class_id
            if mask.sum() == 0:
                continue
            values = get_valid_cluster_values(clusters[mask], i)
            if len(values) > 0:
                # Remove extreme outliers for better visualization
                p1, p99 = np.percentile(values, [1, 99])
                values_clipped = values[(values >= p1) & (values <= p99)]
                ax.hist(values_clipped, bins=50, alpha=0.5, density=True,
                       label=CLASS_NAMES[class_id], color=CLASS_COLORS[class_id])
        
        ax.set_xlabel(feature_name, fontsize=12)
        ax.set_ylabel("Density", fontsize=12)
        ax.legend(fontsize=10)
        ax.set_title(f"Cluster: {feature_name}", fontsize=13)
        
        plt.tight_layout()
        safe_name = feature_name.replace(" ", "_").replace("(", "").replace(")", "").lower()
        plt.savefig(os.path.join(output_dir, f"cluster_{safe_name}.png"), dpi=150, bbox_inches="tight")
        plt.close()
    
    print(f"  Saved {n_features} cluster feature plots")


def plot_track_features(data, output_dir):
    """Plot individual track feature distributions."""
    print("\nPlotting track features...")
    
    tracks = data["tracks"]
    pid = data["pid"]
    
    n_features = NUM_TRACK_FEATURES
    
    for i in range(n_features):
        feature_name = TRACK_FEATURE_NAMES[i] if i < len(TRACK_FEATURE_NAMES) else f"Feature {i}"
        
        fig, ax = plt.subplots(figsize=(8, 6))
        
        for class_id in [0, 1, 2]:
            mask = pid == class_id
            if mask.sum() == 0:
                continue
            values = get_all_track_values(tracks[mask], i)
            if len(values) > 0:
                # Remove extreme outliers
                p1, p99 = np.percentile(values, [1, 99])
                values_clipped = values[(values >= p1) & (values <= p99)]
                if len(values_clipped) > 0:
                    ax.hist(values_clipped, bins=50, alpha=0.5, density=True,
                           label=CLASS_NAMES[class_id], color=CLASS_COLORS[class_id])
        
        ax.set_xlabel(feature_name, fontsize=12)
        ax.set_ylabel("Density", fontsize=12)
        ax.legend(fontsize=10)
        ax.set_title(f"Track: {feature_name}", fontsize=13)
        
        plt.tight_layout()
        safe_name = feature_name.replace(" ", "_").replace("(", "").replace(")", "").lower()
        plt.savefig(os.path.join(output_dir, f"track_{safe_name}.png"), dpi=150, bbox_inches="tight")
        plt.close()
    
    print(f"  Saved {n_features} track feature plots")


def plot_cell_features(data, output_dir):
    """Plot individual cell feature distributions."""
    print("\nPlotting cell features...")
    
    cells = data["cells"]
    pid = data["pid"]
    
    n_features = NUM_CELL_FEATURES
    
    for i in range(n_features):
        feature_name = CELL_FEATURE_NAMES[i] if i < len(CELL_FEATURE_NAMES) else f"Feature {i}"
        
        fig, ax = plt.subplots(figsize=(8, 6))
        
        for class_id in [0, 1, 2]:
            mask = pid == class_id
            if mask.sum() == 0:
                continue
            values = get_valid_cell_values(cells[mask], i)
            if len(values) > 0:
                p1, p99 = np.percentile(values, [1, 99])
                values_clipped = values[(values >= p1) & (values <= p99)]
                if len(values_clipped) > 0:
                    ax.hist(values_clipped, bins=50, alpha=0.5, density=True,
                           label=CLASS_NAMES[class_id], color=CLASS_COLORS[class_id])
        
        ax.set_xlabel(feature_name, fontsize=12)
        ax.set_ylabel("Density", fontsize=12)
        ax.legend(fontsize=10)
        ax.set_title(f"Cell: {feature_name}", fontsize=13)
        
        plt.tight_layout()
        safe_name = feature_name.replace(" ", "_").replace("(", "").replace(")", "").lower()
        plt.savefig(os.path.join(output_dir, f"cell_{safe_name}.png"), dpi=150, bbox_inches="tight")
        plt.close()
    
    print(f"  Saved {n_features} cell feature plots")


def plot_decay_mode_features(data, output_dir):
    """Plot individual feature distributions by detailed decay mode (for tau jets only)."""
    print("\nPlotting features by detailed decay mode (tau jets only)...")
    
    clusters = data["clusters"]
    tracks = data["tracks"]
    pid = data["pid"]
    decay_mode = data["decay_mode"]
    
    # Select tau jets with valid decay mode
    tau_mask = (pid == 1) & (decay_mode >= 0)
    
    if tau_mask.sum() == 0:
        print("  No tau jets with valid decay mode found, skipping...")
        return
    
    tau_clusters = clusters[tau_mask]
    tau_tracks = tracks[tau_mask]
    tau_decay = decay_mode[tau_mask]
    
    # Plot cluster features by detailed decay mode
    n_features = clusters.shape[2]
    
    for i in range(n_features):
        feature_name = CLUSTER_FEATURE_NAMES[i] if i < len(CLUSTER_FEATURE_NAMES) else f"Feature {i}"
        
        fig, ax = plt.subplots(figsize=(8, 6))
        
        for dm in range(5):  # 0-4
            mask = tau_decay == dm
            if mask.sum() == 0:
                continue
            values = get_valid_cluster_values(tau_clusters[mask], i)
            if len(values) > 0:
                p1, p99 = np.percentile(values, [1, 99])
                values_clipped = values[(values >= p1) & (values <= p99)]
                ax.hist(values_clipped, bins=50, alpha=0.5, density=True,
                       label=DECAY_MODE_NAMES[dm], color=DECAY_MODE_COLORS[dm])
        
        ax.set_xlabel(feature_name, fontsize=12)
        ax.set_ylabel("Density", fontsize=12)
        ax.legend(fontsize=10)
        ax.set_title(f"Tau Cluster: {feature_name} (by Decay Mode)", fontsize=13)
        
        plt.tight_layout()
        safe_name = feature_name.replace(" ", "_").replace("(", "").replace(")", "").lower()
        plt.savefig(os.path.join(output_dir, f"cluster_by_decay_mode_{safe_name}.png"), dpi=150, bbox_inches="tight")
        plt.close()
    
    print(f"  Saved {n_features} cluster feature decay mode plots")
    
    # Plot track features by detailed decay mode
    n_features = NUM_TRACK_FEATURES
    
    for i in range(n_features):
        feature_name = TRACK_FEATURE_NAMES[i] if i < len(TRACK_FEATURE_NAMES) else f"Feature {i}"
        
        fig, ax = plt.subplots(figsize=(8, 6))
        
        for dm in range(5):  # 0-4
            mask = tau_decay == dm
            if mask.sum() == 0:
                continue
            values = get_all_track_values(tau_tracks[mask], i)
            if len(values) > 0:
                p1, p99 = np.percentile(values, [1, 99])
                values_clipped = values[(values >= p1) & (values <= p99)]
                if len(values_clipped) > 0:
                    ax.hist(values_clipped, bins=50, alpha=0.5, density=True,
                           label=DECAY_MODE_NAMES[dm], color=DECAY_MODE_COLORS[dm])
        
        ax.set_xlabel(feature_name, fontsize=12)
        ax.set_ylabel("Density", fontsize=12)
        ax.legend(fontsize=10)
        ax.set_title(f"Tau Track: {feature_name} (by Decay Mode)", fontsize=13)
        
        plt.tight_layout()
        safe_name = feature_name.replace(" ", "_").replace("(", "").replace(")", "").lower()
        plt.savefig(os.path.join(output_dir, f"track_by_decay_mode_{safe_name}.png"), dpi=150, bbox_inches="tight")
        plt.close()
    
    print(f"  Saved {n_features} track feature decay mode plots")


def plot_cell_features_by_decay_mode(data, output_dir):
    """Plot individual cell feature distributions by decay mode (for tau jets only)."""
    print("\nPlotting cell features by detailed decay mode (tau jets only)...")
    
    cells = data["cells"]
    pid = data["pid"]
    decay_mode = data["decay_mode"]
    
    # Select tau jets with valid decay mode
    tau_mask = (pid == 1) & (decay_mode >= 0)
    
    if tau_mask.sum() == 0:
        print("  No tau jets with valid decay mode found, skipping...")
        return
    
    tau_cells = cells[tau_mask]
    tau_decay = decay_mode[tau_mask]
    
    n_features = NUM_CELL_FEATURES
    
    for i in range(n_features):
        feature_name = CELL_FEATURE_NAMES[i] if i < len(CELL_FEATURE_NAMES) else f"Feature {i}"
        
        fig, ax = plt.subplots(figsize=(8, 6))
        
        for dm in range(5):  # 0-4
            mask = tau_decay == dm
            if mask.sum() == 0:
                continue
            values = get_valid_cell_values(tau_cells[mask], i)
            if len(values) > 0:
                p1, p99 = np.percentile(values, [1, 99])
                values_clipped = values[(values >= p1) & (values <= p99)]
                if len(values_clipped) > 0:
                    ax.hist(values_clipped, bins=50, alpha=0.5, density=True,
                           label=DECAY_MODE_NAMES[dm], color=DECAY_MODE_COLORS[dm])
        
        ax.set_xlabel(feature_name, fontsize=12)
        ax.set_ylabel("Density", fontsize=12)
        ax.legend(fontsize=10)
        ax.set_title(f"Tau Cell: {feature_name} (by Decay Mode)", fontsize=13)
        
        plt.tight_layout()
        safe_name = feature_name.replace(" ", "_").replace("(", "").replace(")", "").lower()
        plt.savefig(os.path.join(output_dir, f"cell_by_decay_mode_{safe_name}.png"), dpi=150, bbox_inches="tight")
        plt.close()
    
    print(f"  Saved {n_features} cell feature decay mode plots")


def plot_summary_stats(data, output_dir):
    """Plot summary statistics."""
    print("\nPlotting summary statistics...")
    
    clusters = data["clusters"]
    tracks = data["tracks"]
    cells = data["cells"]
    pid = data["pid"]
    decay_mode = data["decay_mode"]
    
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    
    # Class distribution
    ax = axes[0, 0]
    class_counts = [np.sum(pid == i) for i in range(3)]
    bars = ax.bar(list(CLASS_NAMES.values()), class_counts, color=list(CLASS_COLORS.values()))
    ax.set_ylabel("Number of Jets")
    ax.set_title("Jet Type Distribution")
    for bar, count in zip(bars, class_counts):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height(), f'{count}',
                ha='center', va='bottom', fontsize=10)
    
    # Decay mode distribution (tau only)
    ax = axes[0, 1]
    tau_mask = pid == 1
    dm_tau = decay_mode[tau_mask]
    dm_counts = [np.sum(dm_tau == i) for i in range(5)]
    na_count = np.sum(dm_tau == -1)
    bars = ax.bar(list(DECAY_MODE_NAMES.values()) + ["N/A"], dm_counts + [na_count],
                  color=list(DECAY_MODE_COLORS.values()) + ["gray"])
    ax.set_ylabel("Number of Tau Jets")
    ax.set_title("Tau Decay Mode Distribution")
    for bar, count in zip(bars, dm_counts + [na_count]):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height(), f'{count}',
                ha='center', va='bottom', fontsize=10)

    # Truth label distribution (all jets)
    ax = axes[0, 2]
    truth_counts = [np.sum(pid == i) for i in range(3)]
    bars = ax.bar(list(CLASS_NAMES.values()), truth_counts, color=list(CLASS_COLORS.values()))
    ax.set_ylabel("Number of Jets")
    ax.set_title("Truth Label Distribution")
    for bar, count in zip(bars, truth_counts):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height(), f'{count}',
                ha='center', va='bottom', fontsize=10)
    
    # Number of valid clusters per jet
    ax = axes[1, 0]
    n_clusters = (clusters[:, :, 0] != 0).sum(axis=1)
    for class_id in [0, 1, 2]:
        mask = pid == class_id
        ax.hist(n_clusters[mask], bins=range(0, 22), alpha=0.5, density=True,
               label=CLASS_NAMES[class_id], color=CLASS_COLORS[class_id])
    ax.set_xlabel("Number of Clusters")
    ax.set_ylabel("Density")
    ax.set_title("Clusters per Jet")
    ax.legend()
    
    # Number of valid tracks per jet
    ax = axes[1, 1]
    reshaped = tracks.reshape(-1, MAX_TRACKS, NUM_TRACK_FEATURES)
    n_tracks = (reshaped[:, :, 1] != 0).sum(axis=1)
    for class_id in [0, 1, 2]:
        mask = pid == class_id
        ax.hist(n_tracks[mask], bins=range(0, 22), alpha=0.5, density=True,
               label=CLASS_NAMES[class_id], color=CLASS_COLORS[class_id])
    ax.set_xlabel("Number of Tracks")
    ax.set_ylabel("Density")
    ax.set_title("Tracks per Jet")
    ax.legend()

    # Number of valid cells per jet
    ax = axes[1, 2]
    n_cells = (cells[:, :, 0] != 0).sum(axis=1)
    for class_id in [0, 1, 2]:
        mask = pid == class_id
        ax.hist(n_cells[mask], bins=range(0, 52), alpha=0.5, density=True,
               label=CLASS_NAMES[class_id], color=CLASS_COLORS[class_id])
    ax.set_xlabel("Number of Cells")
    ax.set_ylabel("Density")
    ax.set_title("Cells per Jet")
    ax.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "summary_stats.png"), dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved summary_stats.png")


def plot_regression_targets(data, output_dir):
    """Plot all regression target distributions by class."""
    if "targets" not in data or len(data["targets"]) == 0:
        print("\nNo regression targets found, skipping...")
        return
    
    print("\nPlotting regression targets...")
    
    targets = data["targets"]
    pid = data["pid"]
    n_targets = len(REGRESSION_TARGET_NAMES)
    n_cols = 3
    n_rows = (n_targets + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 4 * n_rows))
    axes = axes.flatten()
    
    for i in range(n_targets):
        
        ax = axes[i]
        target_name = REGRESSION_TARGET_NAMES[i] if i < len(REGRESSION_TARGET_NAMES) else f"Target {i}"
        
        for class_id in [0, 1, 2]:
            mask = pid == class_id
            if mask.sum() == 0:
                continue
            values = get_valid_target_values(targets[mask], i)
            if len(values) > 0:
                # Remove extreme outliers for better visualization
                if len(values) > 10:
                    p1, p99 = np.percentile(values, [1, 99])
                    values_clipped = values[(values >= p1) & (values <= p99)]
                    if len(values_clipped) > 0:
                        ax.hist(values_clipped, bins=50, alpha=0.5, density=True,
                            label=CLASS_NAMES[class_id], color=CLASS_COLORS[class_id])
        
        ax.set_xlabel(target_name)
        ax.set_ylabel("Density")
        ax.legend()
        ax.set_title(f"Regression Target: {target_name}")
    
    # Hide unused subplots
    for i in range(n_targets, len(axes)):
        axes[i].set_visible(False)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "regression_targets.png"), dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved regression_targets.png")


def plot_2d_correlations(data, output_dir):
    """Plot 2D correlations for key features."""
    print("\nPlotting 2D correlations...")
    
    clusters = data["clusters"]
    cells = data["cells"]
    pid = data["pid"]
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 9))
    
    # Cluster Eta vs Phi for each class
    for idx, class_id in enumerate([0, 1, 2]):
        ax = axes[0, idx]
        mask = pid == class_id
        eta = get_valid_cluster_values(clusters[mask], 2)
        phi = get_valid_cluster_values(clusters[mask], 3)
        
        # Sample if too many points
        if len(eta) > 50000:
            sample_idx = np.random.choice(len(eta), 50000, replace=False)
            eta = eta[sample_idx]
            phi = phi[sample_idx]
        
        ax.hexbin(eta, phi, gridsize=50, cmap='Blues', mincnt=1)
        ax.set_xlabel("Eta")
        ax.set_ylabel("Phi")
        ax.set_title(f"{CLASS_NAMES[class_id]} Cluster Positions")
        ax.set_xlim(-0.5, 0.5)
        ax.set_ylim(-0.5, 0.5)

    # Cell Eta vs Phi for each class
    for idx, class_id in enumerate([0, 1, 2]):
        ax = axes[1, idx]
        mask = pid == class_id
        eta = get_valid_cell_values(cells[mask], 2)
        phi = get_valid_cell_values(cells[mask], 3)

        if len(eta) > 50000:
            sample_idx = np.random.choice(len(eta), 50000, replace=False)
            eta = eta[sample_idx]
            phi = phi[sample_idx]

        ax.hexbin(eta, phi, gridsize=50, cmap='Greens', mincnt=1)
        ax.set_xlabel("Eta")
        ax.set_ylabel("Phi")
        ax.set_title(f"{CLASS_NAMES[class_id]} Cell Positions")
        ax.set_xlim(-0.5, 0.5)
        ax.set_ylim(-0.5, 0.5)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "cluster_cell_2d_positions.png"), dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved cluster_cell_2d_positions.png")


def main():
    parser = argparse.ArgumentParser(description="Plot feature distributions from HDF5 data")
    parser.add_argument("--input", type=str, default="datasets/tau/train/data.h5",
                       help="Path to input HDF5 file")
    parser.add_argument("--output", type=str, default="plots/",
                       help="Output directory for plots")
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output, exist_ok=True)
    
    # Load data
    data = load_data(args.input)
    
    # Generate all plots
    plot_summary_stats(data, args.output)
    plot_cluster_features(data, args.output)
    plot_track_features(data, args.output)
    plot_cell_features(data, args.output)
    plot_regression_targets(data, args.output)
    plot_decay_mode_features(data, args.output)
    plot_cell_features_by_decay_mode(data, args.output)
    plot_2d_correlations(data, args.output)
    
    print(f"\nAll plots saved to {args.output}")
    print("\nGenerated files:")
    for f in os.listdir(args.output):
        if f.endswith(".png"):
            print(f"  - {f}")


if __name__ == "__main__":
    main()
