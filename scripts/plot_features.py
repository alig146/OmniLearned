"""
Plot cluster and track feature distributions from HDF5 data.

Usage:
    python scripts/plot_features.py --input datasets/tau/train/data.h5 --output plots/
"""

from plotting.features import (
    # constants
    CLUSTER_FEATURE_NAMES, TRACK_FEATURE_NAMES, CELL_FEATURE_NAMES,
    TAU_TARGET_NAMES, PION_FEATURE_NAMES,
    MAX_CLUSTERS, MAX_TRACKS, MAX_CELLS_PER_CLUSTER,
    NUM_TRACK_FEATURES, NUM_CELL_FEATURES,
    CLASS_NAMES, CLASS_COLORS, DECAY_MODE_NAMES, DECAY_MODE_COLORS,
    PLOT_STYLE,
    # data loading
    load_data,
    # feature extractors
    get_valid_cluster_values, get_valid_track_values, get_valid_cell_values,
    get_valid_target_values,
    get_valid_cluster_values_at_index, get_valid_track_values_at_index,
    flatten_cells,
    # plot primitives
    _safe_feature_name, _clip_and_hist, _apply_axis_style, _save_and_close, _make_fig,
    # grouped histogram helpers
    _plot_feature_grid, _plot_featurebyindex_grid,
    # group array builders
    _build_class_arrays, _build_decay_mode_arrays,
)

import argparse
import os

import numpy as np
import matplotlib.pyplot as plt
import mplhep as hep
hep.style.use(hep.style.ATLAS)


# ---------------------------------------------------------------------------
# Per-feature plots
# ---------------------------------------------------------------------------

def plot_cluster_features(data, output_dir):
    """Plot individual cluster feature distributions by class."""
    print("\nPlotting cluster features...")
    clusters = data["clusters"]
    arrays   = _build_class_arrays(clusters, data["pid"])
    _plot_feature_grid(
        arrays, CLUSTER_FEATURE_NAMES, get_valid_cluster_values,
        CLASS_NAMES, CLASS_COLORS,
        "Cluster", "cluster", output_dir,
    )


def plot_track_features(data, output_dir):
    """Plot individual track feature distributions by class."""
    print("\nPlotting track features...")
    tracks = data["tracks"]
    arrays = _build_class_arrays(tracks, data["pid"])
    _plot_feature_grid(
        arrays, TRACK_FEATURE_NAMES, get_valid_track_values,
        CLASS_NAMES, CLASS_COLORS,
        "Track", "track", output_dir,
    )


def plot_cell_features(data, output_dir):
    """Plot individual cell feature distributions by class."""
    print("\nPlotting cell features...")
    cells_flat = flatten_cells(data["cells_per_cluster"])
    arrays     = _build_class_arrays(cells_flat, data["pid"])
    _plot_feature_grid(
        arrays, CELL_FEATURE_NAMES, get_valid_cell_values,
        CLASS_NAMES, CLASS_COLORS,
        "Cell", "cell", output_dir,
    )


# ---------------------------------------------------------------------------
# Per-feature (per-index) plots
# ---------------------------------------------------------------------------

def plot_percluster_features(data, output_dir):
    """Plot individual per-cluster feature distributions by class."""
    print("\nPlotting per-cluster features...")
    clusters = data["clusters"]
    arrays   = _build_class_arrays(clusters, data["pid"])
    print(f"  Found classes: {list(arrays.keys())}")
    _plot_featurebyindex_grid(
        arrays, CLUSTER_FEATURE_NAMES, get_valid_cluster_values_at_index,
        CLASS_NAMES, CLASS_COLORS,
        "Cluster", "cluster", output_dir, MAX_CLUSTERS, "cluster",
    )


def plot_pertrack_features(data, output_dir, max_track_index=15):
    """Plot individual per-track feature distributions by class."""
    print("\nPlotting per-track features...")
    tracks = data["tracks"]
    arrays = _build_class_arrays(tracks, data["pid"])
    print(f"  Found classes: {list(arrays.keys())}")
    n_slots = min(MAX_TRACKS, max_track_index + 1)
    _plot_featurebyindex_grid(
        arrays, TRACK_FEATURE_NAMES, get_valid_track_values_at_index,
        CLASS_NAMES, CLASS_COLORS,
        "Track", "track", output_dir, n_slots, "track",
    )


# ---------------------------------------------------------------------------
# Decay-mode breakdowns
# ---------------------------------------------------------------------------

def _tau_arrays_by_decay_mode(base_array, pid, decay_mode):
    """Return {dm: base_array[tau & decay_mode==dm]}."""
    tau_mask = (pid == 1) & (decay_mode >= 0)
    return _build_decay_mode_arrays(base_array[tau_mask], decay_mode[tau_mask])


def plot_decay_mode_features(data, output_dir):
    """Plot cluster and track features by decay mode (tau jets only)."""
    print("\nPlotting features by detailed decay mode (tau jets only)...")
    pid, decay_mode = data["pid"], data["decay_mode"]

    if ((pid == 1) & (decay_mode >= 0)).sum() == 0:
        print("  No tau jets with valid decay mode found, skipping...")
        return

    cluster_arrays = _tau_arrays_by_decay_mode(data["clusters"], pid, decay_mode)
    _plot_feature_grid(
        cluster_arrays, CLUSTER_FEATURE_NAMES, get_valid_cluster_values,
        DECAY_MODE_NAMES, DECAY_MODE_COLORS,
        "Tau Cluster", "cluster_by_decay_mode", output_dir,
    )

    track_arrays = _tau_arrays_by_decay_mode(data["tracks"], pid, decay_mode)
    _plot_feature_grid(
        track_arrays, TRACK_FEATURE_NAMES, get_valid_track_values,
        DECAY_MODE_NAMES, DECAY_MODE_COLORS,
        "Tau Track", "track_by_decay_mode", output_dir,
    )


def plot_cell_features_by_decay_mode(data, output_dir):
    """Plot cell feature distributions by decay mode (tau jets only)."""
    print("\nPlotting cell features by detailed decay mode (tau jets only)...")
    pid, decay_mode = data["pid"], data["decay_mode"]

    if ((pid == 1) & (decay_mode >= 0)).sum() == 0:
        print("  No tau jets with valid decay mode found, skipping...")
        return

    cells_flat  = flatten_cells(data["cells_per_cluster"])
    cell_arrays = _tau_arrays_by_decay_mode(cells_flat, pid, decay_mode)
    _plot_feature_grid(
        cell_arrays, CELL_FEATURE_NAMES, get_valid_cell_values,
        DECAY_MODE_NAMES, DECAY_MODE_COLORS,
        "Tau Cell", "cell_by_decay_mode", output_dir,
    )


def plot_perindex_decay_mode_features(data, output_dir, max_track_index=15):
    """Plot cluster and track features by decay mode (tau jets only)."""
    print("\nPlotting features by detailed decay mode (tau jets only)...")
    pid, decay_mode = data["pid"], data["decay_mode"]

    if ((pid == 1) & (decay_mode >= 0)).sum() == 0:
        print("  No tau jets with valid decay mode found, skipping...")
        return

    cluster_arrays = _tau_arrays_by_decay_mode(data["clusters"], pid, decay_mode)
    _plot_featurebyindex_grid(
        cluster_arrays, CLUSTER_FEATURE_NAMES, get_valid_cluster_values_at_index,
        DECAY_MODE_NAMES, DECAY_MODE_COLORS,
        "Tau Cluster", "cluster_by_decay_mode", output_dir, MAX_CLUSTERS, "cluster",
    )

    n_slots = min(MAX_TRACKS, max_track_index + 1)
    track_arrays = _tau_arrays_by_decay_mode(data["tracks"], pid, decay_mode)
    _plot_featurebyindex_grid(
        track_arrays, TRACK_FEATURE_NAMES, get_valid_track_values_at_index,
        DECAY_MODE_NAMES, DECAY_MODE_COLORS,
        "Tau Track", "track_by_decay_mode", output_dir, n_slots, "track",
    )


# ---------------------------------------------------------------------------
# Truth tau and pion targets
# ---------------------------------------------------------------------------

def _plot_tau_targets(tau_targets, output_dir):
    """One file per tau kinematic target, overlaying a single series."""
    for feat_idx, feat_name in enumerate(TAU_TARGET_NAMES):
        vals = tau_targets[:, feat_idx]
        if feat_name == "truth_pt":
            vals = vals[vals != 0]
        fig, ax = _make_fig(figsize=(6, 6))
        _clip_and_hist(ax, vals, label="Tau", color="steelblue")
        _apply_axis_style(ax, feat_name, "Density", f"Truth tau {feat_name}")
        ax.legend(fontsize=PLOT_STYLE["fontsize_legend"])
        safe = _safe_feature_name(feat_name)
        _save_and_close(fig, os.path.join(output_dir, f"truth_tau_{safe}.png"))
    print(f"  Saved {len(TAU_TARGET_NAMES)} truth tau target plots")


def plot_truth_targets(data, output_dir):
    """Plot truth tau and pion kinematic target distributions.

    For each pion type (charged / neutral), truth and reco are overlaid on the
    same axes when both are present, or plotted alone otherwise.
    """
    pid      = data["pid"]
    tau_mask = pid == 1

    if "tau_targets" in data:
        print("\nPlotting truth tau targets...")
        _plot_tau_targets(data["tau_targets"][tau_mask], output_dir)

    pion_configs = [
        ("charged",  "charged_pion_targets", "blue", "reco_charged_pions", "red"),
        ("neutral",  "neutral_pion_targets",  "blue", "reco_neutral_pions", "red"),
    ]

    for kind_lc, truth_key, truth_color, reco_key, reco_color in pion_configs:
        has_truth = truth_key in data
        has_reco  = reco_key  in data
        if not has_truth and not has_reco:
            continue

        print(f"\nPlotting {kind_lc} pion targets (truth={has_truth}, reco={has_reco})...")

        truth_arr = data[truth_key][tau_mask] if has_truth else None
        reco_arr  = data[reco_key][tau_mask]  if has_reco  else None

        for feat_idx, feat_name in enumerate(PION_FEATURE_NAMES):
            fig, ax = _make_fig(figsize=(6, 6))

            if truth_arr is not None:
                valid = truth_arr[:, 0] != -999.0
                _clip_and_hist(ax, truth_arr[valid, feat_idx],
                               label="Truth", color=truth_color)

            if reco_arr is not None:
                valid = reco_arr[:, 0] != -999.0
                _clip_and_hist(ax, reco_arr[valid, feat_idx],
                               label="Reco", color=reco_color)

            title = f"{kind_lc} pion sum {feat_name}"
            _apply_axis_style(ax, f"{kind_lc} pion sum {feat_name}", "Density", title)
            ax.legend(fontsize=PLOT_STYLE["fontsize_legend"])
            safe = _safe_feature_name(feat_name)
            _save_and_close(fig, os.path.join(output_dir, f"pion_{kind_lc}_{safe}.png"))

        print(f"  Saved {len(PION_FEATURE_NAMES)} {kind_lc} pion plots")


# ---------------------------------------------------------------------------
# Summary and 2-D correlation plots
# ---------------------------------------------------------------------------

def _annotate_bars(ax, bars, counts):
    """Add count labels above each bar."""
    for bar, count in zip(bars, counts):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), str(count),
                ha="center", va="bottom", fontsize=10)


def plot_summary_stats(data, output_dir):
    """Plot summary statistics."""
    print("\nPlotting summary statistics...")
    clusters   = data["clusters"]
    tracks     = data["tracks"]
    pid        = data["pid"]
    decay_mode = data["decay_mode"]
    cells_pc   = data.get("cells_per_cluster", None)

    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    
    # scale from validation to training size
    train_scale = 10

    # Class distribution
    ax = axes[0, 0]
    class_counts = [np.sum(pid == i) * train_scale for i in range(3)]
    bars = ax.bar(list(CLASS_NAMES.values()), class_counts, color=list(CLASS_COLORS.values()))
    _apply_axis_style(ax, "", "Number of Jets", "Jet Type Distribution")
    _annotate_bars(ax, bars, class_counts)

    # Decay mode distribution (tau only)
    ax = axes[0, 1]
    tau_mask  = pid == 1
    dm_tau    = decay_mode[tau_mask]
    dm_counts = [np.sum(dm_tau == i) * train_scale for i in range(5)]
    na_count  = np.sum(dm_tau == -1)
    bars = ax.bar(list(DECAY_MODE_NAMES.values()) + ["N/A"],
                  dm_counts + [na_count],
                  color=list(DECAY_MODE_COLORS.values()) + ["gray"])
    _apply_axis_style(ax, "", "Number of Tau Jets", "Tau Decay Mode Distribution")
    _annotate_bars(ax, bars, dm_counts + [na_count])

    # # Truth label distribution (redundant but kept for layout symmetry)
    # ax = axes[0, 2]
    # bars = ax.bar(list(CLASS_NAMES.values()), class_counts, color=list(CLASS_COLORS.values()))
    # _apply_axis_style(ax, "", "Number of Jets", "Truth Label Distribution")
    # _annotate_bars(ax, bars, class_counts)

    # Number of valid clusters per jet (use cls_E != 0)
    ax = axes[1, 0]
    n_clusters = (clusters[:, :, 3] != 0).sum(axis=1)
    for class_id in CLASS_NAMES:
        mask = pid == class_id
        ax.hist(n_clusters[mask], bins=range(0, MAX_CLUSTERS+1), density=True,
                label=CLASS_NAMES[class_id], color=CLASS_COLORS[class_id], histtype='step')
    _apply_axis_style(ax, "Number of Clusters", "Density", "Clusters per Jet")
    ax.legend(fontsize=PLOT_STYLE["fontsize_legend"])

    # Number of valid tracks per jet (use trk_E != 0, feature 3)
    ax = axes[1, 1]
    reshaped = tracks.reshape(-1, MAX_TRACKS, NUM_TRACK_FEATURES)
    n_tracks  = (reshaped[:, :, 3] != 0).sum(axis=1)
    for class_id in CLASS_NAMES:
        mask = pid == class_id
        ax.hist(n_tracks[mask], bins=range(0, MAX_TRACKS+1), density=True,
                label=CLASS_NAMES[class_id], color=CLASS_COLORS[class_id], histtype='step')
    _apply_axis_style(ax, "Number of Tracks", "Density", "Tracks per Jet")
    ax.legend(fontsize=PLOT_STYLE["fontsize_legend"])

    # # Cells per cluster distribution (by class)
    # ax = axes[1, 2]
    # if cells_pc is not None:
    #     reshaped = cells_pc.reshape(-1, MAX_CLUSTERS, MAX_CELLS_PER_CLUSTER, NUM_CELL_FEATURES)
    #     n_cells = (reshaped[:, :, :, 3] != 0).sum(axis=1)
    #     for class_id in CLASS_NAMES:
    #         mask = pid == class_id
    #         ax.hist(n_cells[mask].flatten(), bins=range(0, MAX_CELLS_PER_CLUSTER+1),
    #                 density=True, label=CLASS_NAMES[class_id],
    #                 color=CLASS_COLORS[class_id], histtype='step')
    #     _apply_axis_style(ax, "Number of Cells per Cluster", "Density",
    #                       "Cells per Cluster Distribution")
    #     ax.legend(fontsize=PLOT_STYLE["fontsize_legend"])
    # else:
    #     ax.set_visible(False)

    _save_and_close(fig, os.path.join(output_dir, "summary_stats.png"))
    print("  Saved summary_stats.png")


def plot_regression_targets(data, output_dir):
    """Plot all regression target distributions by class."""
    if "targets" not in data or len(data["targets"]) == 0:
        print("\nNo regression targets found, skipping...")
        return

    print("\nPlotting regression targets...")

    targets = data["targets"]
    pid = data["pid"]
    n_targets = len(TAU_TARGET_NAMES)
    n_cols = 3
    n_rows = (n_targets + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 4 * n_rows))
    axes = axes.flatten()

    for i in range(n_targets):
        ax = axes[i]
        target_name = TAU_TARGET_NAMES[i] if i < len(TAU_TARGET_NAMES) else f"Target {i}"

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
                        ax.hist(values_clipped, bins=50, density=True,
                            label=CLASS_NAMES[class_id], color=CLASS_COLORS[class_id],
                            histtype='step')

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
    print("  Saved regression_targets.png")


def plot_2d_correlations(data, output_dir):
    """Plot regular 2D DeltaEta-DeltaPhi heatmaps per PID class and decay mode."""
    print("\nPlotting 2D correlations...")
    clusters   = data["clusters"]
    cells_flat = flatten_cells(data["cells_per_cluster"]) if "cells_per_cluster" in data else None
    pid        = data["pid"]
    decay_mode = data["decay_mode"]

    # Include explicit N/A mode for non-tau or missing decay-mode labels.
    decay_mode_names = dict(DECAY_MODE_NAMES)
    decay_mode_names[-1] = "N/A"

    n_saved = 0
    for class_id, class_name in CLASS_NAMES.items():
        class_mask = pid == class_id
        if np.sum(class_mask) == 0 or class_id != 1:
            continue

        for dm_id, dm_name in decay_mode_names.items():
            mask = class_mask & (decay_mode == dm_id)
            if np.sum(mask) == 0:
                continue

            n_cols = 2 if cells_flat is not None else 1
            fig, axes = plt.subplots(1, n_cols, figsize=(6 * n_cols, 5))
            if n_cols == 1:
                axes = [axes]

            eta, phi = (get_valid_cluster_values(clusters[mask], feat) for feat in (0, 1))
            if len(eta) > 50000:
                sample_idx = np.random.choice(len(eta), 50000, replace=False)
                eta, phi = eta[sample_idx], phi[sample_idx]
            axes[0].hist2d(
                eta,
                phi,
                bins=20,
                range=[[-0.5, 0.5], [-0.5, 0.5]],
                cmap="viridis",
                cmin=1,
            )
            _apply_axis_style(axes[0], "Delta Eta", "Delta Phi",
                              f"{class_name} {dm_name} Cluster Positions")
            axes[0].set_xlim(-0.5, 0.5)
            axes[0].set_ylim(-0.5, 0.5)

            if cells_flat is not None:
                ceta, cphi = (get_valid_cell_values(cells_flat[mask], feat) for feat in (0, 1))
                if len(ceta) > 50000:
                    sample_idx = np.random.choice(len(ceta), 50000, replace=False)
                    ceta, cphi = ceta[sample_idx], cphi[sample_idx]
                axes[1].hist2d(
                    ceta,
                    cphi,
                    bins=20,
                    range=[[-0.5, 0.5], [-0.5, 0.5]],
                    cmap="magma",
                    cmin=1,
                )
                _apply_axis_style(axes[1], "Delta Eta", "Delta Phi",
                                  f"{class_name} {dm_name} Cell Positions")
                axes[1].set_xlim(-0.5, 0.5)
                axes[1].set_ylim(-0.5, 0.5)

            safe_class = _safe_feature_name(class_name)
            safe_dm = _safe_feature_name(dm_name)
            out_name = f"cluster_cell_2d_positions_{safe_class}_decay_mode_{safe_dm}.png"
            _save_and_close(fig, os.path.join(output_dir, out_name))
            n_saved += 1

    print(f"  Saved {n_saved} PID x decay-mode 2D correlation plots")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Plot feature distributions from HDF5 data")
    parser.add_argument("--input",  type=str,
                        default="/pscratch/sd/m/milescb/processed_h5/tau/val/data.h5",
                        help="Path to input HDF5 file")
    parser.add_argument("--output", type=str, default="plots/",
                        help="Output directory for plots")
    parser.add_argument("--max-events", type=int, default=None,
                        help="Maximum number of events to load (for faster testing)")
    parser.add_argument("--cells-per-cluster", action="store_true", default=False,
                        help="Load and plot cells_per_cluster data")
    parser.add_argument("--split-tracks-and-cells", action="store_true", default=False)
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)
    data = load_data(args.input, max_events=args.max_events, load_cells=args.cells_per_cluster)

    plot_summary_stats(data, args.output)
    plot_cluster_features(data, args.output)
    plot_track_features(data, args.output)
    if args.cells_per_cluster:
        plot_cell_features(data, args.output)
    plot_decay_mode_features(data, args.output)
    if args.cells_per_cluster:
        plot_cell_features_by_decay_mode(data, args.output)
    plot_truth_targets(data, args.output)
    plot_2d_correlations(data, args.output)

    if args.cells_per_cluster:
        plot_percluster_features(data, args.output)
        plot_pertrack_features(data, args.output)
        plot_perindex_decay_mode_features(data, args.output)

    print(f"\nAll plots saved to {args.output}")
    print("\nGenerated files:")
    for f in sorted(os.listdir(args.output)):
        if f.endswith(".png"):
            print(f"  - {f}")


if __name__ == "__main__":
    main()
