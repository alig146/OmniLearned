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

import mplhep as hep
hep.style.use(hep.style.ATLAS)

# Feature names matching prepare_data.py
CLUSTER_FEATURE_NAMES = [
    "cls_dEta",
    "cls_dPhi",
    "cls_ET (log)",
    "cls_E (log)",
    "cls_Eta",
    "cls_Phi",
    "cls_FIRST_ENG_DENS",
    "cls_SECOND_R",
    "cls_EM_PROBABILITY",
    "cls_SECOND_LAMBDA",
    "cls_CENTER_LAMBDA",
    "cls_CENTER_MAG",
]

TRACK_FEATURE_NAMES = [
    "trk_dEta",
    "trk_dPhi",
    "trk_pT (log)",
    "trk_E (log)",
    "trk_Eta",
    "trk_Phi",
    "trk_charge",
    "trk_qOverP",
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
    "cell_dEta",
    "cell_dPhi",
    "cell_ET (log)",
    "cell_E (log)",
    "cell_Eta",
    "cell_Phi",
    "cell_sintheta",
    "cell_costheta",
    "cell_sinphi",
    "cell_cosphi",
    "cell_layer",
    "cell_x",
    "cell_y",
    "cell_z",
]

TAU_TARGET_NAMES   = ["truth_pt", "truth_eta", "truth_phi"]
PION_FEATURE_NAMES = ["pt", "eta", "phi"]

# Must match prepare_data.py
MAX_PION_REGRESSION_TARGETS = 4
MAX_CLUSTERS = 20
MAX_TRACKS = 20
MAX_CELLS_PER_CLUSTER = 20

NUM_CLUSTER_FEATURES = len(CLUSTER_FEATURE_NAMES)
NUM_TRACK_FEATURES = len(TRACK_FEATURE_NAMES)
NUM_CELL_FEATURES  = len(CELL_FEATURE_NAMES)

CLASS_NAMES   = {0: "QCD", 1: "Tau", 2: "Electron"}
CLASS_COLORS  = {0: "#e41a1c", 1: "#377eb8", 2: "#4daf4a"}
DECAY_MODE_NAMES  = {0: "1p0n", 1: "1p1n", 2: "1pXn", 3: "3p0n", 4: "3pXn"}
DECAY_MODE_COLORS = {0: "#e41a1c", 1: "#377eb8", 2: "#4daf4a", 3: "#984ea3", 4: "#ff7f00"}

# Shared plot style defaults
PLOT_STYLE = dict(fontsize_label=12, fontsize_legend=10, fontsize_title=13, dpi=150)


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


def load_data(filepath, max_events=None, load_cells=False):
    """Load HDF5 data."""
    print(f"Loading data from {filepath}...")
    if max_events is not None:
        print(f"  Only loading up to {max_events} events...")
    with h5py.File(filepath, "r") as f:
        n_total = f["pid"].shape[0]
        if max_events is None or max_events >= n_total:
            slc = slice(None)
        else:
            # Random subset without replacement; sorted indices for stable HDF5 fancy indexing.
            slc = np.sort(np.random.choice(n_total, size=max_events, replace=False))
        data = {
            "clusters":  f["data"][slc],
            "tracks":    f["tracks"][slc],
            "pid":       f["pid"][slc],
            "decay_mode": f["decay_mode"][slc],
        }
        if load_cells and "cells_per_cluster" in f:
            data["cells_per_cluster"] = f["cells_per_cluster"][slc]

        for key in ("tau_targets", "charged_pion_targets", "neutral_pion_targets"):
            if key in f:
                data[key] = f[key][slc]

    print(f"  Loaded {len(data['pid'])} jets")
    print(f"  Clusters shape:          {data['clusters'].shape}")
    print(f"  Tracks shape:            {data['tracks'].shape}")
    if "cells_per_cluster" in data:
        print(f"  Cells per cluster shape: {data['cells_per_cluster'].shape}")
    for key in ("tau_targets", "charged_pion_targets", "neutral_pion_targets"):
        if key in data:
            print(f"  {key} shape: {data[key].shape}")
    return data


# ---------------------------------------------------------------------------
# Feature extractors
# ---------------------------------------------------------------------------

def get_valid_cluster_values(clusters, feature_idx):
    """Extract non-zero (valid) cluster values. Uses cls_E (feature 3) as presence mask."""
    valid_mask = clusters[:, :, 3] != 0
    return clusters[:, :, feature_idx][valid_mask]

def get_valid_target_values(targets, target_idx):
    """Extract non-zero (valid) target values for a regression target."""
    # target values are zero-padded; use feature 0 (log E) to find valid entries
    valid_mask = targets[:, :, 0] != 0
    values = targets[:, :, target_idx][valid_mask]
    return values

def get_valid_cell_values(cells, feature_idx):
    """Extract non-zero (valid) cell values. Uses cell_E (feature 3) as presence mask."""
    valid_mask = cells[:, :, 3] != 0
    return cells[:, :, feature_idx][valid_mask]


def get_valid_track_values(tracks, feature_idx):
    """Extract non-zero (valid) track values. Uses trk_pT (feature 3) as presence mask."""
    reshaped = tracks.reshape(-1, MAX_TRACKS, NUM_TRACK_FEATURES)
    valid_mask = reshaped[:, :, 3] != 0
    return reshaped[:, :, feature_idx][valid_mask]


def get_valid_cluster_values_at_index(clusters, feature_idx, cluster_idx):
    """Extract valid cluster values for one cluster slot across all jets."""
    values = clusters[:, cluster_idx, feature_idx]
    valid_mask = clusters[:, cluster_idx, 3] != 0
    return values[valid_mask]


def get_valid_track_values_at_index(tracks, feature_idx, track_idx):
    """Extract valid track values for one track slot across all jets."""
    reshaped = tracks.reshape(-1, MAX_TRACKS, NUM_TRACK_FEATURES)
    values = reshaped[:, track_idx, feature_idx]
    valid_mask = reshaped[:, track_idx, 3] != 0
    return values[valid_mask]


def flatten_cells(cells_per_cluster):
    """Flatten (N, C, P, F) → (N, C*P, F) for per-jet cell access."""
    N, C, P, F = cells_per_cluster.shape
    return cells_per_cluster.reshape(N, C * P, F)


# ---------------------------------------------------------------------------
# Plot primitives
# ---------------------------------------------------------------------------

def _safe_feature_name(feature_name):
    """Convert a feature name to a filesystem-safe string."""
    return (
        feature_name.replace(" ", "_")
        .replace("(", "")
        .replace(")", "")
        .replace("/", "_")
        .replace("\\", "_")
        .lower()
    )


def _clip_and_hist(ax, values, label, color, bins=50):
    """Clip to [p1, p99] and overlay a normalised histogram."""
    if len(values) == 0:
        return
    p1, p99 = np.percentile(values, [1, 99])
    clipped = values[(values >= p1) & (values <= p99)]
    if len(clipped) > 0:
        ax.hist(clipped, bins=bins, alpha=0.5, density=True, label=label, color=color, histtype='step')


def _apply_axis_style(ax, xlabel, ylabel, title):
    """Apply consistent axis labels and title."""
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)


def _save_and_close(fig, path):
    """Save figure and close it."""
    plt.tight_layout()
    fig.savefig(path, dpi=PLOT_STYLE["dpi"], bbox_inches="tight")
    plt.close(fig)


def _make_fig(title=None, **subplot_kwargs):
    """Create a figure/axes pair with an optional suptitle."""
    fig, axes = plt.subplots(**subplot_kwargs)
    if title:
        plt.suptitle(title, fontsize=PLOT_STYLE["fontsize_title"], y=1.02)
    return fig, axes


# ---------------------------------------------------------------------------
# Grouped histogram helpers
# ---------------------------------------------------------------------------


def _plot_feature_by_group(ax, arrays_by_group, feature_idx, extractor,
                           group_names, group_colors):
    """
    Overlay one histogram per group on *ax*.

    Parameters
    ----------
    arrays_by_group : dict  {group_id: array}   – pre-filtered data arrays
    feature_idx     : int
    extractor       : callable(array, feature_idx) → 1-D values
    group_names     : dict  {group_id: str}
    group_colors    : dict  {group_id: str}
    """
    for gid, arr in arrays_by_group.items():
        if len(arr) == 0:
            continue
        _clip_and_hist(ax, extractor(arr, feature_idx),
                       group_names[gid], group_colors[gid])
    ax.legend(fontsize=PLOT_STYLE["fontsize_legend"])


def _plot_feature_grid(data_arrays, feature_names, extractor,
                       group_names, group_colors,
                       title_prefix, filename_prefix, output_dir):
    """
    For every feature, create one figure that overlays all groups.

    Parameters
    ----------
    data_arrays  : dict {group_id: array}
    feature_names: list[str]
    extractor    : callable(array, feature_idx) → 1-D values
    group_names  : dict {group_id: str}
    group_colors : dict {group_id: str}
    title_prefix : str   e.g. "Cluster"
    filename_prefix : str e.g. "cluster"
    output_dir   : str
    """
    for i, feature_name in enumerate(feature_names):
        fig, ax = _make_fig(figsize=(6, 6))
        _plot_feature_by_group(ax, data_arrays, i, extractor, group_names, group_colors)
        _apply_axis_style(ax, feature_name, "Density", f"{title_prefix}: {feature_name}")
        safe = _safe_feature_name(feature_name)
        _save_and_close(fig, os.path.join(output_dir, f"{filename_prefix}_{safe}.png"))
    print(f"  Saved {len(feature_names)} {filename_prefix} feature plots")


# ---------------------------------------------------------------------------
# Per-feature plots
# ---------------------------------------------------------------------------

def _build_class_arrays(base_array, pid):
    """Return {class_id: base_array[pid==class_id]} for classes 0,1,2."""
    return {cid: base_array[pid == cid] for cid in CLASS_NAMES if (pid == cid).sum() > 0}


def _build_decay_mode_arrays(base_array, decay_modes):
    """Return {dm: base_array[decay_modes==dm]} for decay modes 0-4."""
    return {dm: base_array[decay_modes == dm] for dm in DECAY_MODE_NAMES
            if (decay_modes == dm).sum() > 0}


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


def _plot_featurebyindex_grid(data_arrays, feature_names, extractor_by_index,
                              group_names, group_colors,
                              title_prefix, filename_prefix, output_dir,
                              max_index, index_label):
    """
    For every feature and index slot, create one figure that overlays all groups.

    Parameters
    ----------
    data_arrays       : dict {group_id: array}
    feature_names     : list[str]
    extractor_by_index: callable(array, feature_idx, slot_idx) -> 1-D values
    group_names       : dict {group_id: str}
    group_colors      : dict {group_id: str}
    title_prefix      : str
    filename_prefix   : str
    output_dir        : str
    max_index         : int
    index_label       : str
    """
    n_saved = 0
    for feature_idx, feature_name in enumerate(feature_names):
        safe_feature = _safe_feature_name(feature_name)
        for slot_idx in range(max_index):
            fig, ax = _make_fig(figsize=(6, 6))
            for gid, arr in data_arrays.items():
                if len(arr) == 0:
                    continue
                vals = extractor_by_index(arr, feature_idx, slot_idx)
                _clip_and_hist(ax, vals, group_names[gid], group_colors[gid])

            _apply_axis_style(
                ax,
                feature_name,
                "Density",
                f"{title_prefix} {slot_idx}: {feature_name}",
            )
            ax.legend(fontsize=PLOT_STYLE["fontsize_legend"])
            out_name = f"{filename_prefix}_{index_label}_{slot_idx}_{safe_feature}.png"
            _save_and_close(fig, os.path.join(output_dir, out_name))
            n_saved += 1

    print(f"  Saved {n_saved} {filename_prefix} per-{index_label} feature plots")


def _plot_feature_grid(data_arrays, feature_names, extractor,
                       group_names, group_colors,
                       title_prefix, filename_prefix, output_dir):
    """
    For every feature, create one figure that overlays all groups.

    Parameters
    ----------
    data_arrays  : dict {group_id: array}
    feature_names: list[str]
    extractor    : callable(array, feature_idx) → 1-D values
    group_names  : dict {group_id: str}
    group_colors : dict {group_id: str}
    title_prefix : str   e.g. "Cluster"
    filename_prefix : str e.g. "cluster"
    output_dir   : str
    """
    for i, feature_name in enumerate(feature_names):
        fig, ax = _make_fig(figsize=(6, 6))
        _plot_feature_by_group(ax, data_arrays, i, extractor, group_names, group_colors)
        _apply_axis_style(ax, feature_name, "Density", f"{title_prefix}: {feature_name}")
        safe = _safe_feature_name(feature_name)
        _save_and_close(fig, os.path.join(output_dir, f"{filename_prefix}_{safe}.png"))
    print(f"  Saved {len(feature_names)} {filename_prefix} feature plots")



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

    cells_flat   = flatten_cells(data["cells_per_cluster"])
    cell_arrays  = _tau_arrays_by_decay_mode(cells_flat, pid, decay_mode)
    _plot_feature_grid(
        cell_arrays, CELL_FEATURE_NAMES, get_valid_cell_values,
        DECAY_MODE_NAMES, DECAY_MODE_COLORS,
        "Tau Cell", "cell_by_decay_mode", output_dir,
    )

# ---------------------------------------------------------------------------
# Per-feature (per-index) plots (for Decay Mode Breakdown)
# ---------------------------------------------------------------------------

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


def _plot_pion_targets(pion_targets, pion_kind, color, output_dir):
    """
    One file per pion feature (pt, eta, phi) of the summed pion 4-vector.
    pion_targets has shape [N, 3] — (pt_sum, eta_sum, phi_sum) per event,
    as produced by prepare_data_v2.py's _sum_pion_4vectors.
    """
    kind_lc = pion_kind.lower()
    # Events with no pions are sentinel-filled with -999; exclude them
    valid_mask = pion_targets[:, 0] != -999.0

    for feat_idx, feat_name in enumerate(PION_FEATURE_NAMES):
        vals = pion_targets[valid_mask, feat_idx]
        fig, ax = _make_fig(figsize=(6, 6))
        _clip_and_hist(ax, vals, label=f"{pion_kind} sum", color=color)
        _apply_axis_style(ax, f"{kind_lc} pion sum {feat_name}", "Density",
                          f"{pion_kind} pion sum {feat_name}")
        ax.legend(fontsize=PLOT_STYLE["fontsize_legend"])
        safe = _safe_feature_name(feat_name)
        _save_and_close(fig, os.path.join(output_dir, f"truth_{kind_lc}_pion_{safe}.png"))

    print(f"  Saved {len(PION_FEATURE_NAMES)} {kind_lc} pion target plots")


def plot_truth_targets(data, output_dir):
    """Plot truth tau and pion kinematic target distributions."""
    pid      = data["pid"]
    tau_mask = pid == 1

    if "tau_targets" in data:
        print("\nPlotting truth tau targets...")
        _plot_tau_targets(data["tau_targets"][tau_mask], output_dir)

    for pion_kind, ds_key, color in [
        ("Charged", "charged_pion_targets", "#e41a1c"),
        ("Neutral", "neutral_pion_targets",  "#377eb8"),
    ]:
        if ds_key not in data:
            continue
        print(f"\nPlotting truth {pion_kind.lower()} pion targets...")
        _plot_pion_targets(data[ds_key][tau_mask], pion_kind, color, output_dir)


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

    fig, axes = plt.subplots(2, 3, figsize=(16, 10))

    # Class distribution
    ax = axes[0, 0]
    class_counts = [np.sum(pid == i) for i in range(3)]
    bars = ax.bar(list(CLASS_NAMES.values()), class_counts, color=list(CLASS_COLORS.values()))
    _apply_axis_style(ax, "", "Number of Jets", "Jet Type Distribution")
    _annotate_bars(ax, bars, class_counts)

    # Decay mode distribution (tau only)
    ax = axes[0, 1]
    tau_mask  = pid == 1
    dm_tau    = decay_mode[tau_mask]
    dm_counts = [np.sum(dm_tau == i) for i in range(5)]
    na_count  = np.sum(dm_tau == -1)
    bars = ax.bar(list(DECAY_MODE_NAMES.values()) + ["N/A"],
                  dm_counts + [na_count],
                  color=list(DECAY_MODE_COLORS.values()) + ["gray"])
    _apply_axis_style(ax, "", "Number of Tau Jets", "Tau Decay Mode Distribution")
    _annotate_bars(ax, bars, dm_counts + [na_count])

    # Truth label distribution (redundant but kept for layout symmetry)
    ax = axes[0, 2]
    bars = ax.bar(list(CLASS_NAMES.values()), class_counts, color=list(CLASS_COLORS.values()))
    _apply_axis_style(ax, "", "Number of Jets", "Truth Label Distribution")
    _annotate_bars(ax, bars, class_counts)

    # Number of valid clusters per jet (use cls_E != 0)
    ax = axes[1, 0]
    n_clusters = (clusters[:, :, 3] != 0).sum(axis=1)
    for class_id in CLASS_NAMES:
        mask = pid == class_id
        ax.hist(n_clusters[mask], bins=range(0, MAX_CLUSTERS+1), alpha=0.5, density=True,
                label=CLASS_NAMES[class_id], color=CLASS_COLORS[class_id], histtype='step')
    _apply_axis_style(ax, "Number of Clusters", "Density", "Clusters per Jet")
    ax.legend(fontsize=PLOT_STYLE["fontsize_legend"])

    # Number of valid tracks per jet (use trk_E != 0, feature 3)
    ax = axes[1, 1]
    reshaped = tracks.reshape(-1, MAX_TRACKS, NUM_TRACK_FEATURES)
    n_tracks  = (reshaped[:, :, 3] != 0).sum(axis=1)
    for class_id in CLASS_NAMES:
        mask = pid == class_id
        ax.hist(n_tracks[mask], bins=range(0, MAX_TRACKS+1), alpha=0.5, density=True,
                label=CLASS_NAMES[class_id], color=CLASS_COLORS[class_id], histtype='step')
    _apply_axis_style(ax, "Number of Tracks", "Density", "Tracks per Jet")
    ax.legend(fontsize=PLOT_STYLE["fontsize_legend"])

    # Cells per cluster distribution (by class)
    ax = axes[1, 2]
    if cells_pc is not None:
        reshaped = cells_pc.reshape(-1, MAX_CLUSTERS, MAX_CELLS_PER_CLUSTER, NUM_CELL_FEATURES)
        n_cells = (reshaped[:, :, :, 3] != 0).sum(axis=1)
        for class_id in CLASS_NAMES:
            mask = pid == class_id
            ax.hist(n_cells[mask].flatten(), bins=range(0, MAX_CELLS_PER_CLUSTER+1),
                    alpha=0.5, density=True, label=CLASS_NAMES[class_id], color=CLASS_COLORS[class_id], histtype='step')
        _apply_axis_style(ax, "Number of Cells per Cluster", "Density",
                          "Cells per Cluster Distribution")
        ax.legend(fontsize=PLOT_STYLE["fontsize_legend"])
    else:
        ax.set_visible(False)

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
                        ax.hist(values_clipped, bins=50, alpha=0.5, density=True,
                            label=CLASS_NAMES[class_id], color=CLASS_COLORS[class_id], histtype='step')
        
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
    parser.add_argument("--input",  type=str, default="/pscratch/sd/m/milescb/processed_h5_new/tau/val/data.h5",
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
