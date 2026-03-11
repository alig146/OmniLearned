# -*- coding: utf-8 -*-
"""
Prepare HDF5 datasets for OmniLearned tau identification.

Optimized for large datasets (100GB+ ROOT):
- Chunked ROOT reading (--chunk_size) limits memory per file
- Parallel file processing (--workers) uses multiple CPU cores
- Staging + merge avoids holding all processed data in RAM
"""
import sys
if sys.version_info[0] < 3:
    sys.exit("This script requires Python 3.")
"""
Converts ROOT files to HDF5 format with:
- data: Cluster point cloud features [N, MAX_CLUSTERS, NUM_CLUSTER_FEATURES]
- tracks: Track point cloud features [N, MAX_TRACKS, NUM_TRACK_FEATURES]  
- pid: Jet type label (0=QCD, 1=tau, 2=electron)
- decay_mode: Tau decay mode (0=1p0n, 1=1p1n, 2=1pXn, 3=3p0n, 4=3pXn, 5=Other, 6=NotSet, -1=N/A or Error)

Both clusters and tracks are treated as point clouds with first features being:
- Clusters: [dEta, dPhi, log(et), log(e), ...]
- Tracks: [dEta, dPhi, log(pt), theta, ...]

Usage:
    python prepare_data.py --input_dir /path/to/root/files --output_dir /path/to/output
"""

import os
os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"

import uproot
import numpy as np
import awkward as ak
import h5py
import argparse
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed
from concurrent.futures.process import BrokenProcessPool
import shutil


# Configuration
MAX_CLUSTERS = 15
MAX_TRACKS = 15
MAX_CELLS = 500

# Cluster branches (particles in point cloud)
# First 4 MUST be: [dEta, dPhi, log(pT), log(E)]
CLUSTER_BRANCHES = [
    ("cls_dEta", "cls_dEta", False),           # Feature 0: η
    ("cls_dPhi", "cls_dPhi", False),           # Feature 1: φ
    ("cls_ET", "cls_ET", True),                  # Feature 3: E_{T}
    ("cls_E", "cls_E", True),                # Feature 2: E
    ("cls_Eta", "cls_Eta", False),           # Feature 0: η
    ("cls_Phi", "cls_Phi", False),           # Feature 1: φ
    ("cls_FIRST_ENG_DENS", "cls_FIRST_ENG_DENS", False),
    ("cls_SECOND_R", "cls_SECOND_R", False),
    ("cls_EM_PROBABILITY", "cls_EM_PROBABILITY", False),
    ("cls_SECOND_LAMBDA", "cls_SECOND_LAMBDA", False),
    ("cls_CENTER_LAMBDA", "cls_CENTER_LAMBDA", False),
    ("cls_CENTER_MAG", "cls_CENTER_MAG", False),
#### Commenting these out for now. Making a to-do list of everything else to add in the future, but want to get a basic version working first with just the 4 main features.
    #("cls_SECOND_R", "TauClusters.cls_SECOND_R", False),
    #("cls_SECOND_LAMBDA", "TauClusters.cls_SECOND_LAMBDA", False),
    #("cls_EM_PROBABILITY", "TauClusters.cls_EM_PROBABILITY", False),
    #("cls_CENTER_MAG", "TauClusters.cls_CENTER_MAG", False),
    #("cls_CENTER_LAMBDA", "TauClusters.cls_CENTER_LAMBDA", False),
]

# Track branches (all features, will be flattened into global)
TRACK_BRANCHES = [
    ("trk_dEta", "trk_dEta", False),
    ("trk_dPhi", "trk_dPhi", False),    #("trk_eProbNN", "trk_eProbNN", False), # We should not use this anyway, but eventually if we have samples we should
    ("trk_pT", "trk_pT", True),
    ("trk_E", "trk_E", True),
    ("trk_Eta", "trk_Eta", False),
    ("trk_Phi", "trk_Phi", False),
    ("trk_charge", "trk_charge", False),
    ("trk_d0", "trk_d0", False),
    ("trk_z0", "trk_z0", False),
    ("trk_z0sintheta", "trk_z0sintheta", False),
    ("trk_nTRTHits", "trk_nTRTHits", False),
    ("trk_nTRTHighThresholdHits", "trk_nTRTHighThresholdHits", False),
    ("trk_nSCTHits", "trk_nSCTHits", False),
    ("trk_nPixelHits", "trk_nPixelHits", False),
    ("trk_nBLayerHits", "trk_nBLayerHits", False),
]
#### Commenting these out for now. Making a to-do list of everything else to add in the future, but want to get a basic version working first with just the 4 main features.
#    ("dEta", "TauTracks.dEta", False),
#    ("dPhi", "TauTracks.dPhi", False),
#    ("trackPt", "TauTracks.trackPt", True),                # log(pT)
#    ("theta", "TauTracks.theta", False),
#    ("numberOfInnermostPixelLayerHits", "TauTracks.numberOfInnermostPixelLayerHits", False),
#    ("numberOfPixelHits", "TauTracks.numberOfPixelHits", False),
#    ("numberOfPixelSharedHits", "TauTracks.numberOfPixelSharedHits", False),
#    ("numberOfPixelDeadSensors", "TauTracks.numberOfPixelDeadSensors", False),
#    ("numberOfSCTHits", "TauTracks.numberOfSCTHits", False),
#    ("numberOfSCTSharedHits", "TauTracks.numberOfSCTSharedHits", False),
#    ("numberOfSCTDeadSensors", "TauTracks.numberOfSCTDeadSensors", False),
#    ("numberOfTRTHighThresholdHits", "TauTracks.numberOfTRTHighThresholdHits", False),
#    ("numberOfTRTHits", "TauTracks.numberOfTRTHits", False),
#    ("nPixHits", "TauTracks.nPixHits", False),
#    ("nSCTHits", "TauTracks.nSCTHits", False),
#    ("nSiHits", "TauTracks.nSiHits", False),
#    ("nIBLHitsAndExp", "TauTracks.nIBLHitsAndExp", False),
#    ("expectInnermostPixelLayerHit", "TauTracks.expectInnermostPixelLayerHit", False),
#    ("expectNextToInnermostPixelLayerHit", "TauTracks.expectNextToInnermostPixelLayerHit", False),
#    ("numberOfContribPixelLayers", "TauTracks.numberOfContribPixelLayers", False),
#    ("numberOfPixelHoles", "TauTracks.numberOfPixelHoles", False),
#    ("numberOfSCTHoles", "TauTracks.numberOfSCTHoles", False),
#    ("d0_old", "TauTracks.d0_old", False),
#    ("qOverP", "TauTracks.qOverP", False),


CELL_BRANCHES = [
    ("cell_dEta", "cell_dEta", False),
    ("cell_dPhi", "cell_dPhi", False),
    ("cell_ET", "cell_ET", True),
    ("cell_E", "cell_E", True),
    ("cell_Eta", "cell_Eta", False),
    ("cell_Phi", "cell_Phi", False),
    ("cell_sintheta", "cell_sintheta", False),
    ("cell_costheta", "cell_costheta", False),
    ("cell_sinphi", "cell_sinphi", False),
    ("cell_cosphi", "cell_cosphi", False),
    ("cell_layer", "cell_layer", False),
    ("cell_x", "cell_x", False),
    ("cell_y", "cell_y", False),
    ("cell_z", "cell_z", False),
]

NUM_CLUSTER_FEATURES = len(CLUSTER_BRANCHES)
NUM_TRACK_FEATURES = len(TRACK_BRANCHES)
NUM_CELL_FEATURES = len(CELL_BRANCHES)

# TODO: these guys need to actually be saved
OTHER_BRANCHES = [
    "truth_label",
    "truth_pt",
    "truth_decayMode", # 0=1p0n, 1=1p1n, 2=1pXn, 3=3p0n, 4=3pXn, 5=Other, 6=NotSet, 7=Error
    "truth_pdgId", 
]


def get_all_branches():
    """Get list of all branches to read."""
    branches = []
    branches += [b for _, b, _ in CELL_BRANCHES]
    branches += [b for _, b, _ in CLUSTER_BRANCHES]
    branches += [b for _, b, _ in TRACK_BRANCHES]
    branches += OTHER_BRANCHES
    return list(set(branches))


def safe_log(arr, epsilon=1e-8):
    """Apply log transformation safely."""
    arr = np.array(arr, dtype=np.float32)
    return np.log(np.maximum(arr, epsilon))


def _is_sequence_like(values):
    """Return True if values behaves like a sequence (list/array/awkward list)."""
    try:
        len(values)
        return True
    except TypeError:
        return False


def _get_jet_values(event_values, jet_idx, n_jets):
    """Get per-jet values from branches that may be per-event or per-jet nested."""
    if not _is_sequence_like(event_values):
        return [event_values] if jet_idx == 0 else []

    # Common in these ntuples: one jet per event, and branch already stores
    # the full object list for that event (e.g. clusters/cells/tracks).
    if n_jets == 1 and len(event_values) > 0:
        first_item = event_values[0]
        if not _is_sequence_like(first_item):
            return event_values

    if len(event_values) <= jet_idx:
        return []

    jet_values = event_values[jet_idx]
    if _is_sequence_like(jet_values):
        return jet_values

    return [jet_values]


def _fill_point_cloud_features(
    output_for_jet,
    feature_specs,
    event_feature_arrays,
    jet_idx,
    n_jets,
    max_items,
    reorder_indices=None,
):
    """Fill a single jet point-cloud tensor from feature specs and event data."""
    for feat_idx, (feat_name, _, apply_log) in enumerate(feature_specs):
        jet_values = _get_jet_values(event_feature_arrays[feat_name], jet_idx, n_jets)
        n_values = len(jet_values)
        if n_values == 0:
            continue

        values = np.asarray(ak.to_numpy(jet_values), dtype=np.float32)
        if reorder_indices is not None and len(reorder_indices) > 0:
            valid_idx = reorder_indices[reorder_indices < len(values)]
            if len(valid_idx) > 0:
                values = values[valid_idx]

        n = min(len(values), max_items)
        values = values[:n]
        if apply_log:
            values = safe_log(values)
        output_for_jet[:n, feat_idx] = values


def _get_jet_cell_values(event_values, jet_idx, n_jets, cluster_sort_indices):
    """Get flattened per-jet cell values from branches stored as cells-per-cluster."""
    if not _is_sequence_like(event_values):
        return [event_values] if jet_idx == 0 else []

    # In these ntuples, cells are often stored for one jet/event as
    # [cluster0_cells, cluster1_cells, ...]. Keep full event content for n_jets==1.
    if n_jets == 1:
        jet_cells_by_cluster = event_values
    else:
        if len(event_values) <= jet_idx:
            return []
        jet_cells_by_cluster = event_values[jet_idx]

    if not _is_sequence_like(jet_cells_by_cluster) or len(jet_cells_by_cluster) == 0:
        return []

    first_item = jet_cells_by_cluster[0]
    if not _is_sequence_like(first_item):
        # Already flat cells for this jet.
        return jet_cells_by_cluster

    clusters_as_lists = [list(cluster_cells) for cluster_cells in jet_cells_by_cluster]

    if cluster_sort_indices is not None and len(cluster_sort_indices) > 0:
        ordered_clusters = [
            clusters_as_lists[idx]
            for idx in cluster_sort_indices
            if idx < len(clusters_as_lists)
        ]
    else:
        ordered_clusters = clusters_as_lists

    # Flatten cluster->cells into one jet-level cell sequence.
    flattened = []
    for cluster_cells in ordered_clusters:
        flattened.extend(cluster_cells)
    return flattened


def _fill_cell_features(
    output_for_jet,
    feature_specs,
    event_feature_arrays,
    jet_idx,
    n_jets,
    max_items,
    cluster_sort_indices,
):
    """Fill cell features for a jet, flattening cells-per-cluster after cluster reordering."""
    for feat_idx, (feat_name, _, apply_log) in enumerate(feature_specs):
        jet_cells = _get_jet_cell_values(
            event_feature_arrays[feat_name],
            jet_idx,
            n_jets,
            cluster_sort_indices,
        )
        if len(jet_cells) == 0:
            continue

        values = np.asarray(ak.to_numpy(jet_cells), dtype=np.float32)
        n = min(len(values), max_items)
        values = values[:n]
        if apply_log:
            values = safe_log(values)
        output_for_jet[:n, feat_idx] = values


def _pad_and_convert(arr, max_items):
    """Pad ragged awkward array to fixed width and convert to numpy float32."""
    padded = ak.fill_none(ak.pad_none(arr, target=max_items, clip=True, axis=-1), 0.0)
    return np.asarray(ak.to_numpy(padded), dtype=np.float32)


def _vectorized_point_cloud(events, feature_specs, max_items):
    """Build (N, max_items, N_features) array from ragged branches — no Python loop over events."""
    n = len(events)
    n_feat = len(feature_specs)
    out = np.zeros((n, max_items, n_feat), dtype=np.float32)
    for feat_idx, (_, branch_name, apply_log) in enumerate(feature_specs):
        arr = events[branch_name]
        np_arr = _pad_and_convert(arr, max_items)
        if apply_log:
            nonzero = np_arr != 0
            np_arr[nonzero] = np.log(np.maximum(np_arr[nonzero], 1e-8))
        out[:, :, feat_idx] = np_arr
    return out


def _vectorized_cells(events, cell_specs, max_cells):
    """Build (N, max_cells, N_features) from doubly-ragged cell branches.

    Cell branches may be events x clusters x cells (doubly ragged) or events x cells (flat).
    For doubly-ragged, cells are flattened across clusters (without reordering) then padded.
    Cluster reordering is skipped in the vectorized path because the cluster counts in
    cls_E and the cell branches can differ; the cell order matters less than for clusters.
    """
    n = len(events)
    n_feat = len(cell_specs)
    out = np.zeros((n, max_cells, n_feat), dtype=np.float32)
    for feat_idx, (_, branch_name, apply_log) in enumerate(cell_specs):
        arr = events[branch_name]
        ndim = arr.ndim if hasattr(arr, 'ndim') else ak.Array(arr).ndim
        if ndim >= 3:
            flat = ak.flatten(arr, axis=-1)
        else:
            flat = arr
        np_arr = _pad_and_convert(flat, max_cells)
        if apply_log:
            nonzero = np_arr != 0
            np_arr[nonzero] = np.log(np.maximum(np_arr[nonzero], 1e-8))
        out[:, :, feat_idx] = np_arr
    return out


def _process_chunk(events, label, n_events_in_chunk):
    """Vectorized chunk processing — operates on all events at once via awkward/numpy."""
    n_jets_per_event = ak.num(events["truth_label"])
    total_jets = int(ak.sum(n_jets_per_event))
    if total_jets == 0:
        return None

    all_single = int(ak.max(n_jets_per_event)) == 1

    if all_single:
        # --- Fast vectorized path (1 jet per event, most common) ---
        # Clusters
        chunk_data = _vectorized_point_cloud(events, CLUSTER_BRANCHES, MAX_CLUSTERS)

        # Sort clusters by energy descending
        cls_e_idx = next(i for i, (n, _, _) in enumerate(CLUSTER_BRANCHES) if n == "cls_E")
        csort = np.argsort(-chunk_data[:, :, cls_e_idx], axis=1)
        chunk_data = np.take_along_axis(chunk_data, csort[:, :, np.newaxis], axis=1)

        # Tracks
        chunk_tracks = _vectorized_point_cloud(events, TRACK_BRANCHES, MAX_TRACKS)
        trk_pt_idx = next(i for i, (n, _, _) in enumerate(TRACK_BRANCHES) if n == "trk_pT")
        tsort = np.argsort(-chunk_tracks[:, :, trk_pt_idx], axis=1)
        chunk_tracks = np.take_along_axis(chunk_tracks, tsort[:, :, np.newaxis], axis=1)

        # Cells — flatten across clusters then pad
        chunk_cells = _vectorized_cells(events, CELL_BRANCHES, MAX_CELLS)

        # Labels
        chunk_pid = np.full(total_jets, label, dtype=np.int64)
        dm = events["truth_decayMode"]
        dm_flat = ak.to_numpy(ak.flatten(dm)).astype(np.int64)
        if label == 1:
            dm_flat[(dm_flat < 0) | (dm_flat > 6)] = -1
            chunk_decay_mode = dm_flat
        else:
            chunk_decay_mode = np.full(total_jets, -1, dtype=np.int64)

        return chunk_data, chunk_tracks, chunk_cells, chunk_pid, chunk_decay_mode

    # --- Fallback: multi-jet events (rare) — loop per event/jet ---
    chunk_data = np.zeros((total_jets, MAX_CLUSTERS, NUM_CLUSTER_FEATURES), dtype=np.float32)
    chunk_tracks = np.zeros((total_jets, MAX_TRACKS, NUM_TRACK_FEATURES), dtype=np.float32)
    chunk_cells = np.zeros((total_jets, MAX_CELLS, NUM_CELL_FEATURES), dtype=np.float32)
    chunk_pid = np.full(total_jets, label, dtype=np.int64)
    chunk_decay_mode = np.full(total_jets, -1, dtype=np.int64)

    jet_counter = 0
    for evt_idx in range(n_events_in_chunk):
        n_jets = int(n_jets_per_event[evt_idx])
        cluster_arrays = {fn: events[bn][evt_idx] for fn, bn, _ in CLUSTER_BRANCHES}
        track_arrays = {fn: events[bn][evt_idx] for fn, bn, _ in TRACK_BRANCHES}
        cell_arrays = {fn: events[bn][evt_idx] for fn, bn, _ in CELL_BRANCHES}
        decay_mode = events["truth_decayMode"][evt_idx]

        for jet_idx in range(n_jets):
            jet_cluster_energy = _get_jet_values(cluster_arrays["cls_E"], jet_idx, n_jets)
            if len(jet_cluster_energy) > 0:
                cluster_energy = np.asarray(ak.to_numpy(jet_cluster_energy), dtype=np.float32)
                cluster_sort_indices = np.argsort(-cluster_energy)
            else:
                cluster_sort_indices = np.array([], dtype=np.int64)

            _fill_point_cloud_features(
                chunk_data[jet_counter], CLUSTER_BRANCHES, cluster_arrays,
                jet_idx, n_jets, MAX_CLUSTERS, reorder_indices=cluster_sort_indices,
            )
            _fill_point_cloud_features(
                chunk_tracks[jet_counter], TRACK_BRANCHES, track_arrays,
                jet_idx, n_jets, MAX_TRACKS,
            )
            _fill_cell_features(
                chunk_cells[jet_counter], CELL_BRANCHES, cell_arrays,
                jet_idx, n_jets, MAX_CELLS, cluster_sort_indices,
            )
            dm_val = int(decay_mode[jet_idx]) if label == 1 else -1
            chunk_decay_mode[jet_counter] = dm_val if 0 <= dm_val <= 6 else -1
            jet_counter += 1

    trk_pt_idx = next(i for i, (n, _, _) in enumerate(TRACK_BRANCHES) if n == "trk_pT")
    sort_indices = np.argsort(-chunk_tracks[:, :, trk_pt_idx], axis=1)
    chunk_tracks = np.take_along_axis(chunk_tracks, sort_indices[:, :, np.newaxis], axis=1)

    return chunk_data, chunk_tracks, chunk_cells, chunk_pid, chunk_decay_mode


def process_file(filepath, label, chunk_size="500 MB"):
    """
    Process a single ROOT file with chunked reading (memory-efficient for large files).

    Returns:
        data: Cluster features [N_jets, MAX_CLUSTERS, NUM_CLUSTER_FEATURES]
        tracks: Track features [N_jets, MAX_TRACKS, NUM_TRACK_FEATURES]
        cells: Cell features [N_jets, MAX_CELLS, NUM_CELL_FEATURES]
        pid: Jet labels [N_jets]
        decay_mode: Decay mode labels [N_jets]
    """
    print(f"Processing {filepath} with label {label} (chunk_size={chunk_size})")

    try:
        with uproot.open(filepath) as f:
            n_entries = f["CollectionTree"].num_entries
    except Exception as e:
        print(f"Error opening {filepath}: {e}")
        return None, None, None, None, None

    if n_entries == 0:
        print(f"File {filepath} is empty, skipping...")
        return None, None, None, None, None

    branches = get_all_branches()
    chunk_results = []

    for chunk_batch in tqdm(
        uproot.iterate(f"{filepath}:CollectionTree", branches, step_size=chunk_size, library="ak", report=False),
        desc="  Chunks",
        leave=False,
    ):
        n_in_chunk = len(chunk_batch)
        result = _process_chunk(chunk_batch, label, n_in_chunk)
        if result is not None:
            chunk_results.append(result)
        del chunk_batch

    if len(chunk_results) == 0:
        print(f"  No jets in {filepath}")
        return None, None, None, None, None

    all_data = np.concatenate([r[0] for r in chunk_results], axis=0)
    all_tracks = np.concatenate([r[1] for r in chunk_results], axis=0)
    all_cells = np.concatenate([r[2] for r in chunk_results], axis=0)
    all_pid = np.concatenate([r[3] for r in chunk_results], axis=0)
    all_decay_mode = np.concatenate([r[4] for r in chunk_results], axis=0)
    del chunk_results

    print(f"  Output: data={all_data.shape}, tracks={all_tracks.shape}, cells={all_cells.shape}")
    return all_data, all_tracks, all_cells, all_pid, all_decay_mode


def _process_file_to_staging(args):
    """Worker: process one file and write to staging. Returns (staging_path, n_jets) or (None, 0)."""
    filepath, label, chunk_size, staging_dir, file_idx = args
    try:
        result = process_file(filepath, label, chunk_size=chunk_size)
    except Exception as e:
        raise RuntimeError(f"Error processing {filepath}: {e}") from e
    if result is None or result[0] is None:
        return None, 0
    data, tracks, cells, pid, decay_mode = result
    n_jets = len(data)
    staging_path = os.path.join(staging_dir, f"staging_{file_idx:05d}.h5")
    with h5py.File(staging_path, "w") as hf:
        hf.create_dataset("data", data=data, compression="gzip")
        hf.create_dataset("tracks", data=tracks, compression="gzip")
        hf.create_dataset("cells", data=cells, compression="gzip")
        hf.create_dataset("pid", data=pid)
        hf.create_dataset("decay_mode", data=decay_mode)
    return staging_path, n_jets


def main():
    parser = argparse.ArgumentParser(description="Prepare HDF5 datasets for OmniLearned")
    parser.add_argument("--input_dir", type=str,
                        default="/pscratch/sd/m/milescb/OmniTau/OmniLearnedData/ntuples/",
                        help="Directory containing ROOT files")
    parser.add_argument("--output_dir", type=str,
                        default="/pscratch/sd/m/milescb/OmniTau/OmniLearnedData/training_data",
                        help="Directory to save HDF5 files")
    parser.add_argument("--train_frac", type=float, default=0.6)
    parser.add_argument("--val_frac", type=float, default=0.2)
    parser.add_argument(
        "--chunk_size",
        type=str,
        default="500 MB",
        help="ROOT read chunk size: int (entries) or str (e.g. '500 MB', '2 GB'). Lower = less memory, more I/O.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Number of parallel workers for processing files. 1 = sequential. Use ≤ CPU cores; each worker uses ~chunk_size memory.",
    )
    args = parser.parse_args()

    # Parse chunk_size: int (entries) or str e.g. "500 MB" for uproot
    try:
        chunk_size = int(args.chunk_size)
    except ValueError:
        chunk_size = args.chunk_size  # e.g. "500 MB", "2 GB"

    os.makedirs(args.output_dir, exist_ok=True)
    
    # JZ2 files (label 0)
    jz2_files = [
        "user.nkyriaco.48954090.EXT0._000005.ntuple.root",
        #"user.nkyriaco.48954090.EXT0._000247.ntuple.root",
        #"user.nkyriaco.48954090.EXT0._000256.ntuple.root",
        #"user.nkyriaco.48954090.EXT0._000542.ntuple.root",
        #"user.nkyriaco.48954090.EXT0._000806.ntuple.root",
        #"user.nkyriaco.48954090.EXT0._000932.ntuple.root",
        #"user.nkyriaco.48954090.EXT0._001252.ntuple.root",
    ]
    
    # Gammatautau files (label 1)
    gammatautau_files = [
        "user.nkyriaco.48954085.EXT0._000050.ntuple.root",
        #"user.nkyriaco.48954085.EXT0._000116.ntuple.root",
        #"user.nkyriaco.48954085.EXT0._000307.ntuple.root",
        #"user.nkyriaco.48954085.EXT0._000406.ntuple.root",
        #"user.nkyriaco.48954085.EXT0._000433.ntuple.root",
        #"user.nkyriaco.48954085.EXT0._000595.ntuple.root",
        #"user.nkyriaco.48954085.EXT0._000690.ntuple.root",
        #"user.nkyriaco.48954085.EXT0._000751.ntuple.root",
        #"user.nkyriaco.48954085.EXT0._000906.ntuple.root",
        #"user.nkyriaco.48954085.EXT0._000923.ntuple.root",
    ]
    
    # Gammaee files (label 2)
    gammaee_files = [
        "user.nkyriaco.48954086.EXT0._000004.ntuple.root",
        #"user.nkyriaco.48954086.EXT0._000022.ntuple.root",
        #"user.nkyriaco.48954086.EXT0._000050.ntuple.root",
        #"user.nkyriaco.48954086.EXT0._000053.ntuple.root",
        #"user.nkyriaco.48954086.EXT0._000058.ntuple.root",
        #"user.nkyriaco.48954086.EXT0._000062.ntuple.root",
        #"user.nkyriaco.48954086.EXT0._000078.ntuple.root",
        #"user.nkyriaco.48954086.EXT0._000100.ntuple.root",
        #"user.nkyriaco.48954086.EXT0._000101.ntuple.root",
        #"user.nkyriaco.48954086.EXT0._000167.ntuple.root",
    ]
    
    files_and_labels = []

    # Add JZ2 files with label 0
    for fname in jz2_files:
        files_and_labels.append((os.path.join(args.input_dir,
                                              "user.nkyriaco.JZ2.Ntuple_03_03_26_Prod1_EXT0", fname), 0))

    # Add Gammatautau files with label 1
    for fname in gammatautau_files:
        files_and_labels.append((os.path.join(args.input_dir,
                                              "user.nkyriaco.Gammatautau.Ntuple_03_03_26_Prod1_EXT0", fname), 1))

    # Add Gammaee files with label 2
    for fname in gammaee_files:
        files_and_labels.append((os.path.join(args.input_dir,
                                              "user.nkyriaco.Gammaee.Ntuple_03_03_26_Prod1_EXT0", fname), 2))

    # Filter to existing files
    valid_files = [(fp, lb) for fp, lb in files_and_labels if os.path.exists(fp)]
    for fp, _ in files_and_labels:
        if not os.path.exists(fp):
            print(f"Warning: {fp} not found, skipping...")

    if len(valid_files) == 0:
        print("No data processed!")
        return

    # Staging dir for per-file output (enables memory-efficient merge)
    staging_dir = os.path.join(args.output_dir, "_staging")
    os.makedirs(staging_dir, exist_ok=True)

    worker_args = [
        (fp, lb, chunk_size, staging_dir, idx)
        for idx, (fp, lb) in enumerate(valid_files)
    ]

    # Rough memory check: each worker holds ~chunk_size + processed output
    if args.workers > 1 and isinstance(chunk_size, str):
        ck = chunk_size.upper().replace(" ", "")
        if "GB" in ck:
            try:
                gb = float(ck.replace("GB", ""))
                est_gb = args.workers * gb * 2.5  # chunk + awkward + processed
                if est_gb > 32:
                    print(f"Warning: {args.workers} workers x {chunk_size} ≈ {est_gb:.0f} GB peak memory.")
                    print("  If workers are killed (BrokenProcessPool), try: --workers 2 --chunk_size '1 GB'")
            except ValueError:
                pass

    if args.workers <= 1:
        staging_paths = []
        for warg in tqdm(worker_args, desc="Processing files"):
            path, n = _process_file_to_staging(warg)
            if path is not None:
                staging_paths.append(path)
    else:
        staging_paths = []
        try:
            with ProcessPoolExecutor(max_workers=args.workers) as ex:
                futures = {ex.submit(_process_file_to_staging, warg): warg for warg in worker_args}
                for fut in tqdm(as_completed(futures), total=len(futures), desc="Processing files"):
                    try:
                        path, n = fut.result()
                        if path is not None:
                            staging_paths.append(path)
                    except BrokenProcessPool:
                        print("\n\nWorker process was killed (likely out-of-memory).")
                        print("Try reducing memory usage:")
                        print("  python prepare_data.py ... --workers 2 --chunk_size '1 GB'")
                        print("  or:  --workers 1 --chunk_size '500 MB'  (sequential, lowest memory)")
                        raise
        except BrokenProcessPool:
            if len(staging_paths) > 0:
                print(f"\nSome files ({len(staging_paths)}) completed before failure. Consider re-running with --workers 1")
                print("to process remaining files, or reduce --chunk_size and --workers.")
            raise

    if len(staging_paths) == 0:
        print("No data processed!")
        shutil.rmtree(staging_dir, ignore_errors=True)
        return

    # Merge: read each staging file, shuffle, assign to train/val/test, append to output
    print(f"\nMerging {len(staging_paths)} staging files into train/val/test...")
    np.random.seed(42)
    train_frac, val_frac = args.train_frac, args.val_frac
    split_paths = {"train": os.path.join(args.output_dir, "train"),
                  "val": os.path.join(args.output_dir, "val"),
                  "test": os.path.join(args.output_dir, "test")}
    for p in split_paths.values():
        os.makedirs(p, exist_ok=True)

    split_offsets = {k: 0 for k in split_paths}
    split_files = {k: os.path.join(split_paths[k], "data.h5") for k in split_paths}
    h5_handles = {}
    sample_shapes = None

    try:
        for staging_path in tqdm(staging_paths, desc="  Merge"):
            with h5py.File(staging_path, "r") as sf:
                data = sf["data"][:]
                tracks = sf["tracks"][:]
                cells = sf["cells"][:]
                pid = sf["pid"][:]
                decay_mode = sf["decay_mode"][:]

            if sample_shapes is None:
                sample_shapes = (data.shape[1:], tracks.shape[1:], cells.shape[1:])

            n = len(data)
            r = np.random.random(n)
            train_mask = r < train_frac
            val_mask = (r >= train_frac) & (r < train_frac + val_frac)
            test_mask = r >= train_frac + val_frac

            for split_name, mask in [("train", train_mask), ("val", val_mask), ("test", test_mask)]:
                if not mask.any():
                    continue
                d, t, c, p, dm = data[mask], tracks[mask], cells[mask], pid[mask], decay_mode[mask]
                fp = split_files[split_name]
                if split_name not in h5_handles:
                    h5_handles[split_name] = h5py.File(fp, "w")
                    h5_handles[split_name].create_dataset("data", data=d, compression="gzip", maxshape=(None,) + d.shape[1:])
                    h5_handles[split_name].create_dataset("tracks", data=t, compression="gzip", maxshape=(None,) + t.shape[1:])
                    h5_handles[split_name].create_dataset("cells", data=c, compression="gzip", maxshape=(None,) + c.shape[1:])
                    h5_handles[split_name].create_dataset("pid", data=p, maxshape=(None,))
                    h5_handles[split_name].create_dataset("decay_mode", data=dm, maxshape=(None,))
                    split_offsets[split_name] = len(d)
                else:
                    hf = h5_handles[split_name]
                    for ds_name, arr in [("data", d), ("tracks", t), ("cells", c), ("pid", p), ("decay_mode", dm)]:
                        hf[ds_name].resize(split_offsets[split_name] + len(arr), axis=0)
                        hf[ds_name][split_offsets[split_name]:] = arr
                    split_offsets[split_name] += len(d)
    finally:
        for h in h5_handles.values():
            h.close()

    # Create empty HDF5 for splits that received no data (dataloader expects all splits to exist)
    if sample_shapes is not None:
        cluster_shp, track_shp, cell_shp = sample_shapes
        for split_name in split_paths:
            if split_name not in h5_handles:
                with h5py.File(split_files[split_name], "w") as hf:
                    hf.create_dataset("data", shape=(0,) + cluster_shp, maxshape=(None,) + cluster_shp, compression="gzip")
                    hf.create_dataset("tracks", shape=(0,) + track_shp, maxshape=(None,) + track_shp, compression="gzip")
                    hf.create_dataset("cells", shape=(0,) + cell_shp, maxshape=(None,) + cell_shp, compression="gzip")
                    hf.create_dataset("pid", shape=(0,), maxshape=(None,), dtype=np.int64)
                    hf.create_dataset("decay_mode", shape=(0,), maxshape=(None,), dtype=np.int64)

    # Cleanup staging
    shutil.rmtree(staging_dir, ignore_errors=True)

    # Load final arrays for stats and file_index (only metadata, not full data)
    data = None
    with h5py.File(split_files["train"], "r") as f:
        n_train = f["data"].shape[0]
    with h5py.File(split_files["val"], "r") as f:
        n_val = f["data"].shape[0]
    with h5py.File(split_files["test"], "r") as f:
        n_test = f["data"].shape[0]
    n_total = n_train + n_val + n_test

    # Write file_index.npy for each split
    for split_name in split_paths:
        n = n_train if split_name == "train" else (n_val if split_name == "val" else n_test)
        file_indices = np.array([(0, i) for i in range(n)], dtype=np.int32)
        np.save(os.path.join(split_paths[split_name], "file_index.npy"), file_indices)
    
    print(f"\nTotal samples: {n_total}")
    print(f"  train: {n_train}, val: {n_val}, test: {n_test}")

    # Aggregate stats from splits
    pid_all, decay_all = [], []
    for split_name in split_paths:
        with h5py.File(split_files[split_name], "r") as f:
            pid_all.append(f["pid"][:])
            decay_all.append(f["decay_mode"][:])
    pid = np.concatenate(pid_all)
    decay_mode = np.concatenate(decay_all)
    print(f"  QCD (0): {np.sum(pid == 0)}")
    print(f"  Tau (1): {np.sum(pid == 1)}")
    print(f"  Electron (2): {np.sum(pid == 2)}")
    print(f"  Decay mode 1p0n (0): {np.sum(decay_mode == 0)}")
    print(f"  Decay mode 1p1n (1): {np.sum(decay_mode == 1)}")
    print(f"  Decay mode 1pXn (2): {np.sum(decay_mode == 2)}")
    print(f"  Decay mode 3p0n (3): {np.sum(decay_mode == 3)}")
    print(f"  Decay mode 3pXn (4): {np.sum(decay_mode == 4)}")
    print(f"  Decay mode Other (5): {np.sum(decay_mode == 5)}")
    print(f"  Decay mode NotSet (6): {np.sum(decay_mode == 6)}")
    print(f"  Decay mode N/A (-1): {np.sum(decay_mode == -1)}")
    
    print("\nDone!")
    print(f"\n" + "="*60)
    print("DATA STRUCTURE (Option B: Tracks as Separate Tokens)")
    print("="*60)
    with h5py.File(split_files["train"], "r") as f:
        dshape, tshape, cshape = f["data"].shape, f["tracks"].shape, f["cells"].shape
    print(f"\n  data (clusters): (N, {dshape[1]}, {dshape[2]})")
    print(f"    - {MAX_CLUSTERS} clusters x {NUM_CLUSTER_FEATURES} features")
    print(f"    - First 4: dEta, dPhi, log(et), log(e)")
    print(f"\n  tracks: (N, {tshape[1]}, {tshape[2]})")
    print(f"    - {MAX_TRACKS} tracks x {NUM_TRACK_FEATURES} features")
    print(f"    - First 4: dEta, dPhi, log(pt), theta")
    print(f"\nCluster features ({NUM_CLUSTER_FEATURES}):")
    for i, (name, _, _) in enumerate(CLUSTER_BRANCHES):
        print(f"    {i}: {name}")
    print(f"\nTrack features ({NUM_TRACK_FEATURES}):")
    for i, (name, _, _) in enumerate(TRACK_BRANCHES):
        print(f"    {i}: {name}")
    print(f"\nCell features ({NUM_CELL_FEATURES}):")
    for i, (name, _, _) in enumerate(CELL_BRANCHES):
        print(f"    {i}: {name}")
    print(f"\n" + "="*60)
    print("TRAINING COMMAND")
    print("="*60)
    print(f"\n  omnilearned train --dataset tau --path datasets \\")
    print(f"    --num-feat {NUM_CLUSTER_FEATURES} --num-classes 3 \\")
    print(f"    --use-tracks --track-dim {NUM_TRACK_FEATURES} \\")
    print(f"    --aux-tasks-str 'decay_mode:7,electron_vs_qcd:2'")


if __name__ == "__main__":
    main()
