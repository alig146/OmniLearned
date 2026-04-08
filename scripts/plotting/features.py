"""
Constants, data loading, feature extractors, and plot primitives
shared across feature-plotting scripts.
"""

import os
import numpy as np
import h5py
import matplotlib.pyplot as plt


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

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
NUM_TRACK_FEATURES   = len(TRACK_FEATURE_NAMES)
NUM_CELL_FEATURES    = len(CELL_FEATURE_NAMES)

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

        for key in ("tau_targets", "charged_pion_targets", "neutral_pion_targets",
                    "reco_tau_4mom", "reco_charged_pions", "reco_neutral_pions"):
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
        ax.hist(clipped, bins=bins, density=True, label=label, color=color, histtype='step')

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
    data_arrays     : dict {group_id: array}
    feature_names   : list[str]
    extractor       : callable(array, feature_idx) → 1-D values
    group_names     : dict {group_id: str}
    group_colors    : dict {group_id: str}
    title_prefix    : str   e.g. "Cluster"
    filename_prefix : str   e.g. "cluster"
    output_dir      : str
    """
    for i, feature_name in enumerate(feature_names):
        fig, ax = _make_fig(figsize=(6, 6))
        _plot_feature_by_group(ax, data_arrays, i, extractor, group_names, group_colors)
        _apply_axis_style(ax, feature_name, "Density", f"{title_prefix}: {feature_name}")
        safe = _safe_feature_name(feature_name)
        _save_and_close(fig, os.path.join(output_dir, f"{filename_prefix}_{safe}.png"))
    print(f"  Saved {len(feature_names)} {filename_prefix} feature plots")


def _plot_featurebyindex_grid(data_arrays, feature_names, extractor_by_index,
                              group_names, group_colors,
                              title_prefix, filename_prefix, output_dir,
                              max_index, index_label):
    """
    For every feature and index slot, create one figure that overlays all groups.

    Parameters
    ----------
    data_arrays        : dict {group_id: array}
    feature_names      : list[str]
    extractor_by_index : callable(array, feature_idx, slot_idx) -> 1-D values
    group_names        : dict {group_id: str}
    group_colors       : dict {group_id: str}
    title_prefix       : str
    filename_prefix    : str
    output_dir         : str
    max_index          : int
    index_label        : str
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


# ---------------------------------------------------------------------------
# Group array builders
# ---------------------------------------------------------------------------

def _build_class_arrays(base_array, pid):
    """Return {class_id: base_array[pid==class_id]} for classes 0,1,2."""
    return {cid: base_array[pid == cid] for cid in CLASS_NAMES if (pid == cid).sum() > 0}

def _build_decay_mode_arrays(base_array, decay_modes):
    """Return {dm: base_array[decay_modes==dm]} for decay modes 0-4."""
    return {dm: base_array[decay_modes == dm] for dm in DECAY_MODE_NAMES
            if (decay_modes == dm).sum() > 0}
