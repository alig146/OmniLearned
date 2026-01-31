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
from matplotlib.backends.backend_pdf import PdfPages

# Feature names matching prepare_data.py
CLUSTER_FEATURE_NAMES = [
    "cls_dEta (Δη)",
    "cls_dPhi (Δφ)",
    "cls_et (log pT)",
    "cls_e (log E)",
    "cls_SECOND_R",
    "cls_SECOND_LAMBDA",
    "cls_FIRST_ENG_DENS",
    "cls_EM_PROBABILITY",
    "cls_CENTER_MAG",
    "cls_CENTER_LAMBDA",
]

TRACK_FEATURE_NAMES = [
    "dEta",
    "dPhi",
    "trackPt (log)",
    "theta",
    "numberOfInnermostPixelLayerHits",
    "numberOfPixelHits",
    "numberOfPixelSharedHits",
    "numberOfPixelDeadSensors",
    "numberOfSCTHits",
    "numberOfSCTSharedHits",
    "numberOfSCTDeadSensors",
    "numberOfTRTHighThresholdHits",
    "numberOfTRTHits",
    "nPixHits",
    "nSCTHits",
    "nSiHits",
    "nIBLHitsAndExp",
    "expectInnermostPixelLayerHit",
    "expectNextToInnermostPixelLayerHit",
    "numberOfContribPixelLayers",
    "numberOfPixelHoles",
    "numberOfSCTHoles",
    "d0_old",
    "qOverP",
]

REGRESSION_TARGET_NAMES = [
    "truth_loge",
]

CLASS_NAMES = {0: "QCD", 1: "Tau", 2: "Electron"}
CLASS_COLORS = {0: "#e41a1c", 1: "#377eb8", 2: "#4daf4a"}
DECAY_MODE_NAMES = {0: "1p0n", 1: "1p1n", 2: "1pXn", 3: "3p0n", 4: "3pXn"}
DECAY_MODE_COLORS = {0: "#e41a1c", 1: "#377eb8", 2: "#4daf4a", 3: "#984ea3", 4: "#ff7f00"}

MAX_TRACKS = 20
NUM_TRACK_FEATURES = len(TRACK_FEATURE_NAMES)


def load_data(filepath):
    """Load HDF5 data."""
    print(f"Loading data from {filepath}...")
    with h5py.File(filepath, "r") as f:
        data = {
            "clusters": f["data"][:],
            "tracks": f["tracks"][:],
            "pid": f["pid"][:],
            "decay_mode": f["decay_mode"][:],
        }
        # Load regression targets if present
        if "truth_targets" in f:
            data["targets"] = f["truth_targets"][:]
    print(f"  Loaded {len(data['pid'])} jets")
    print(f"  Clusters shape: {data['clusters'].shape}")
    print(f"  Tracks shape: {data['tracks'].shape}")
    print(f"  Regression targets shape: {data['targets'].shape if 'targets' in data else 'N/A'}")
    
    return data


def get_valid_cluster_values(clusters, feature_idx):
    """Extract non-zero (valid) cluster values for a feature."""
    # Clusters are zero-padded; use feature 2 (log pT) to find valid entries
    valid_mask = clusters[:, :, 2] != 0
    values = clusters[:, :, feature_idx][valid_mask]
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
    # Use feature 2 (log trackPt) to identify valid tracks
    valid_mask = reshaped[:, :, 2] != 0
    values = reshaped[:, :, feature_idx][valid_mask]
    return values


def plot_cluster_features(data, output_dir):
    """Plot all cluster feature distributions."""
    print("\nPlotting cluster features...")
    
    clusters = data["clusters"]
    pid = data["pid"]
    
    n_features = clusters.shape[2]
    n_cols = 3
    n_rows = (n_features + n_cols - 1) // n_cols
    
    # Plot by jet type
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 4 * n_rows))
    axes = axes.flatten()
    
    for i in range(n_features):
        ax = axes[i]
        feature_name = CLUSTER_FEATURE_NAMES[i] if i < len(CLUSTER_FEATURE_NAMES) else f"Feature {i}"
        
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
        
        ax.set_xlabel(feature_name)
        ax.set_ylabel("Density")
        ax.legend()
        ax.set_title(f"Cluster: {feature_name}")
    
    # Hide unused subplots
    for i in range(n_features, len(axes)):
        axes[i].set_visible(False)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "cluster_features_by_class.png"), dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved cluster_features_by_class.png")


def plot_track_features(data, output_dir):
    """Plot all track feature distributions."""
    print("\nPlotting track features...")
    
    tracks = data["tracks"]
    pid = data["pid"]
    
    n_features = NUM_TRACK_FEATURES
    n_cols = 4
    n_rows = (n_features + n_cols - 1) // n_cols
    
    # Plot by jet type
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(16, 4 * n_rows))
    axes = axes.flatten()
    
    for i in range(n_features):
        ax = axes[i]
        feature_name = TRACK_FEATURE_NAMES[i] if i < len(TRACK_FEATURE_NAMES) else f"Feature {i}"
        
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
        
        ax.set_xlabel(feature_name)
        ax.set_ylabel("Density")
        ax.legend(fontsize=8)
        ax.set_title(f"Track: {feature_name}", fontsize=10)
    
    # Hide unused subplots
    for i in range(n_features, len(axes)):
        axes[i].set_visible(False)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "track_features_by_class.png"), dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved track_features_by_class.png")


def plot_decay_mode_features(data, output_dir):
    """Plot feature distributions by detailed decay mode (for tau jets only)."""
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
    n_cols = 3
    n_rows = (n_features + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 4 * n_rows))
    axes = axes.flatten()
    
    for i in range(n_features):
        ax = axes[i]
        feature_name = CLUSTER_FEATURE_NAMES[i] if i < len(CLUSTER_FEATURE_NAMES) else f"Feature {i}"
        
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
        
        ax.set_xlabel(feature_name)
        ax.set_ylabel("Density")
        ax.legend(fontsize=8)
        ax.set_title(f"Cluster: {feature_name}")
    
    for i in range(n_features, len(axes)):
        axes[i].set_visible(False)
    
    plt.suptitle("Tau Cluster Features by Detailed Decay Mode", fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "cluster_features_by_decay_mode.png"), dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved cluster_features_by_decay_mode.png")
    
    # Plot track features by detailed decay mode
    n_features = NUM_TRACK_FEATURES
    n_cols = 4
    n_rows = (n_features + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(16, 4 * n_rows))
    axes = axes.flatten()
    
    for i in range(n_features):
        ax = axes[i]
        feature_name = TRACK_FEATURE_NAMES[i] if i < len(TRACK_FEATURE_NAMES) else f"Feature {i}"
        
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
        
        ax.set_xlabel(feature_name)
        ax.set_ylabel("Density")
        ax.legend(fontsize=6)
        ax.set_title(f"Track: {feature_name}", fontsize=10)
    
    for i in range(n_features, len(axes)):
        axes[i].set_visible(False)
    
    plt.suptitle("Tau Track Features by Detailed Decay Mode", fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "track_features_by_decay_mode.png"), dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved track_features_by_decay_mode.png")


def plot_summary_stats(data, output_dir):
    """Plot summary statistics."""
    print("\nPlotting summary statistics...")
    
    clusters = data["clusters"]
    tracks = data["tracks"]
    pid = data["pid"]
    decay_mode = data["decay_mode"]
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
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
    
    # Number of valid clusters per jet
    ax = axes[1, 0]
    n_clusters = (clusters[:, :, 2] != 0).sum(axis=1)
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
    n_tracks = (reshaped[:, :, 2] != 0).sum(axis=1)
    for class_id in [0, 1, 2]:
        mask = pid == class_id
        ax.hist(n_tracks[mask], bins=range(0, 22), alpha=0.5, density=True,
               label=CLASS_NAMES[class_id], color=CLASS_COLORS[class_id])
    ax.set_xlabel("Number of Tracks")
    ax.set_ylabel("Density")
    ax.set_title("Tracks per Jet")
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
    pid = data["pid"]
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # dEta vs dPhi for each class
    for idx, class_id in enumerate([0, 1, 2]):
        ax = axes[idx]
        mask = pid == class_id
        deta = get_valid_cluster_values(clusters[mask], 0)
        dphi = get_valid_cluster_values(clusters[mask], 1)
        
        # Sample if too many points
        if len(deta) > 50000:
            sample_idx = np.random.choice(len(deta), 50000, replace=False)
            deta = deta[sample_idx]
            dphi = dphi[sample_idx]
        
        ax.hexbin(deta, dphi, gridsize=50, cmap='Blues', mincnt=1)
        ax.set_xlabel("Δη")
        ax.set_ylabel("Δφ")
        ax.set_title(f"{CLASS_NAMES[class_id]} Cluster Positions")
        ax.set_xlim(-0.5, 0.5)
        ax.set_ylim(-0.5, 0.5)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "cluster_2d_positions.png"), dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved cluster_2d_positions.png")


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
    plot_regression_targets(data, args.output)
    plot_decay_mode_features(data, args.output)
    plot_2d_correlations(data, args.output)
    
    print(f"\nAll plots saved to {args.output}")
    print("\nGenerated files:")
    for f in os.listdir(args.output):
        if f.endswith(".png"):
            print(f"  - {f}")


if __name__ == "__main__":
    main()
