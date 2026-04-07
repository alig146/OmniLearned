"""
Converts ROOT files to HDF5 format with:
- data: Cluster point cloud features [N, MAX_CLUSTERS, NUM_CLUSTER_FEATURES]
- tracks: Track point cloud features [N, MAX_TRACKS, NUM_TRACK_FEATURES] 
- cells: Cell point cloud per cluster [N, MAX_CLUSTERS, MAX_CELLS_PER_CLUSTER, NUM_CELL_FEATURES] 
- pid: Jet type label (0=QCD, 1=tau, 2=electron)
- decay_mode: Tau decay mode (0=1p0n, 1=1p1n, 2=1pXn, 3=3p0n, 4=3pXn, 5=Other, 6=NotSet, -1=N/A or Error)
- tau_targets: Tau regression targets [N, NUM_TAU_REGRESSION_TARGETS] (truth_pt, truth_eta, truth_phi)
- charged_pion_targets: Charged pion sum [N, 3] — (pt, eta, phi) of the 4-vector sum of all charged pions per event
- neutral_pion_targets: Neutral pion sum [N, 3] — (pt, eta, phi) of the 4-vector sum of all neutral pions per event
- tau

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
import vector
import h5py
import argparse
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed
from concurrent.futures.process import BrokenProcessPool
import shutil

vector.register_awkward()


# Configuration
MAX_CLUSTERS = 20
MAX_TRACKS = 20
MAX_CELLS_PER_CLUSTER = 10

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
    ("trk_qOverP", "trk_qOverP", False),
    ("trk_d0", "trk_d0", False),
    ("trk_z0", "trk_z0", False),
    ("trk_z0sintheta", "trk_z0sintheta", False),
    ("trk_nTRTHits", "trk_nTRTHits", False),
    ("trk_nTRTHighThresholdHits", "trk_nTRTHighThresholdHits", False),
    ("trk_nSCTHits", "trk_nSCTHits", False),
    ("trk_nPixelHits", "trk_nPixelHits", False),
    ("trk_nBLayerHits", "trk_nBLayerHits", False),
]


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

TAU_REGRESSION_TARGETS = [
    ("truth_tau_Vispt", "truth_tau_Vispt", False), 
    ("truth_tau_Viseta", "truth_tau_Viseta", False),
    ("truth_tau_Visphi", "truth_tau_Visphi", False),
]

CHARGED_PION_BRANCHES = [
    ("truth_chargedPion_Vispt",  "truth_chargedPion_Vispt",  False),
    ("truth_chargedPion_Viseta", "truth_chargedPion_Viseta", False),
    ("truth_chargedPion_Visphi", "truth_chargedPion_Visphi", False),
]
NEUTRAL_PION_BRANCHES = [
    ("truth_neutralPion_Vispt",  "truth_neutralPion_Vispt",  False),
    ("truth_neutralPion_Viseta", "truth_neutralPion_Viseta", False),
    ("truth_neutralPion_Visphi", "truth_neutralPion_Visphi", False),
]
PION_REGRESSION_TARGETS = CHARGED_PION_BRANCHES + NEUTRAL_PION_BRANCHES

TAUTRACK_CLASSIFICATION_BRANCHES = [
    ("trk_truthType", "trk_truthType", False), # Could be trk_originClass (need to double-check) # 0,8=Undefined, 1=TTT, 2=CT, 3,4=IT, 5,6,7=FT
]


NUM_CLUSTER_FEATURES = len(CLUSTER_BRANCHES)
NUM_TRACK_FEATURES = len(TRACK_BRANCHES)
NUM_CELL_FEATURES = len(CELL_BRANCHES)
NUM_TAU_REGRESSION_TARGETS = len(TAU_REGRESSION_TARGETS)
NUM_PION_FEATURES = 3  # pt, eta, phi
NUM_TAU_TRACK_CLASSIFICATION_TARGETS = len(TAUTRACK_CLASSIFICATION_BRANCHES)


DECAY_MODE = [
    # 0=1p0n, 1=1p1n, 2=1pXn, 3=3p0n, 4=3pXn, 5=Other, 6=NotSet, 7=Error
    "truth_decayMode",
    "reco_TauNNDecayMode",
    "reco_TauPanTauBDTDecayMode",
]

RECO_ID = [
    "reco_TauRNNEleScore_Raw",
    "reco_TauRNNEleScore_SigTrans",
    "reco_TauRNNJetScore_Raw",
    "reco_TauRNNJetScore_SigTrans",
    "reco_TauGNNJetScore_Raw",
    "reco_TauGNNJetScore_SigTrans",
]

# Currently, these are flattened vectors
RECO_CHARGED_PION_4MOM = [
    "reco_chargedPion_pt",
    "reco_chargedPion_eta",
    "reco_chargedPion_phi",
    "tau_nChargedTracks" # use to split
]
RECO_NEUTRAL_PION_4MOM = [
    "reco_PanTauPi0_pt",
    "reco_PanTauPi0_eta",
    "reco_PanTauPi0_phi",
    "reco_PanTauPi0_n" # use to split
]

RECO_TAU_4MOM = [
    "reco_TauPanTauCellBased_pt",
    "reco_TauPanTauCellBased_eta",
    "reco_TauPanTauCellBased_phi",
    "reco_TauFinalCalib_pt",
    "reco_TauFinalCalib_eta",
    "reco_TauFinalCalib_phi"
]

def get_all_branches(label=None, use_cells=True):
    """Get list of all branches to read. Only includes cell branches for tau (label=1)."""
    branches = []

    if label == 1:
        if use_cells:
            branches += [b for _, b, _ in CELL_BRANCHES] 
        branches += [b for _, b, _ in TAUTRACK_CLASSIFICATION_BRANCHES]

    branches += [b for _, b, _ in CLUSTER_BRANCHES]
    branches += [b for _, b, _ in TRACK_BRANCHES]
    branches += [b for _, b, _ in TAU_REGRESSION_TARGETS]
    branches += [b for _, b, _ in PION_REGRESSION_TARGETS]
    branches += DECAY_MODE
    branches += RECO_ID
    branches += RECO_CHARGED_PION_4MOM
    branches += RECO_NEUTRAL_PION_4MOM
    branches += RECO_TAU_4MOM
    return list(set(branches))


def _list_root_files(dir_path, num_files=None):
    """List only ROOT files to avoid sidecar files such as .asetup.save."""
    if num_files is not None:
        return [f for f in os.listdir(dir_path) if f.endswith(".root")][:num_files]
    return [f for f in os.listdir(dir_path) if f.endswith(".root")]


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


def _vectorized_scalar_targets(events, feature_specs):
    """Build (N, N_features) array for per-event scalar regression targets."""
    n = len(events)
    n_feat = len(feature_specs)
    out = np.zeros((n, n_feat), dtype=np.float32)
    for feat_idx, (_, branch_name, apply_log) in enumerate(feature_specs):
        arr = ak.to_numpy(ak.flatten(events[branch_name])).astype(np.float32)
        if apply_log:
            nz = arr != 0
            arr[nz] = np.log(np.maximum(arr[nz], 1e-8))
        out[:, feat_idx] = arr
    return out

def _vectorized_scalar_targets_no_decorator(events, features):
    """Build (N, N_features) array for per-event scalar regression targets."""
    n = len(events)
    n_feat = len(features)
    out = np.zeros((n, n_feat), dtype=np.float32)
    for feat_idx, branch_name in enumerate(features):
        arr = ak.to_numpy(ak.flatten(events[branch_name])).astype(np.float32)
        out[:, feat_idx] = arr
    return out

CHARGED_PION_MASS = 139.57018  # MeV
NEUTRAL_PION_MASS = 134.97700  # MeV

def _temp_obtain_reco_pions_sum(events, pion_4mom: list[str], pion_mass=CHARGED_PION_MASS):
    """Sum reco pion 4-vectors per event. pion_4mom is [pt_branch, eta_branch, phi_branch, ...].
    Returns (N, 3) array of (pt_sum, eta_sum, phi_sum); events with no pions yield (-999, -999, -999)."""
    pt_branch, eta_branch, phi_branch = pion_4mom[:3]
    pt_arr   = events[pt_branch]
    eta_arr  = events[eta_branch]
    phi_arr  = events[phi_branch]
    mass_arr = ak.full_like(pt_arr, pion_mass)

    pions = ak.zip(
        {"pt": pt_arr, "eta": eta_arr, "phi": phi_arr, "mass": mass_arr},
        with_name="Momentum4D",
        behavior=vector.backends.awkward.behavior,
    )
    no_pions = ak.to_numpy(ak.num(pions, axis=1) == 0)
    sum_vec  = ak.sum(pions, axis=1)

    pt_sum  = ak.to_numpy(sum_vec.pt).astype(np.float32)
    eta_sum = ak.to_numpy(sum_vec.eta).astype(np.float32)
    phi_sum = ak.to_numpy(sum_vec.phi).astype(np.float32)

    pt_sum[no_pions]  = -999.0
    eta_sum[no_pions] = -999.0
    phi_sum[no_pions] = -999.0

    return np.stack([pt_sum, eta_sum, phi_sum], axis=1)


def _vectorized_cells_per_cluster(label, events, cell_specs, max_clusters, max_cells_pc, cluster_sort=None, use_cells=True):
    """Build tau cell tensor (N, max_clusters, max_cells_pc, N_features); return None for non-tau labels to avoid large zero allocations.
    Cell branches are doubly ragged: events x clusters x cells.
    This function keeps the cluster structure intact instead of flattening."""
    if label != 1 or not use_cells:
        return None

    n = len(events)
    n_feat = len(cell_specs)
    out = np.zeros((n, max_clusters, max_cells_pc, n_feat), dtype=np.float32)

    for feat_idx, (_, branch_name, apply_log) in enumerate(cell_specs):
        arr = events[branch_name]
        ndim = arr.ndim if hasattr(arr, 'ndim') else ak.Array(arr).ndim
        if ndim < 3:
            raise ValueError("Incorrect useage of cell vectorization func.")

        for evt in range(n):
            evt_clusters = arr[evt]
            n_cls = min(len(evt_clusters), max_clusters)
            for ci in range(n_cls):
                cells = evt_clusters[ci]
                nc = min(len(cells), max_cells_pc)
                if nc > 0:
                    vals = ak.to_numpy(cells[:nc]).astype(np.float32)
                    if apply_log:
                        nz = vals != 0
                        vals[nz] = np.log(np.maximum(vals[nz], 1e-8))
                    out[evt, ci, :nc, feat_idx] = vals

    # TODO: if sorting, should be done BEFORE filling above
    # if cluster_sort is not None:
    #     idx = cluster_sort[:, :, np.newaxis, np.newaxis]
    #     idx = np.broadcast_to(idx, out.shape)
    #     out = np.take_along_axis(out, idx, axis=1)

    return out


def _process_chunk(events, label, n_events_in_chunk, use_cells=True):
    """Vectorized chunk processing — operates on all events at once via awkward/numpy."""
    total_jets = len(events)
    
    # Clusters
    chunk_data = _vectorized_point_cloud(events, CLUSTER_BRANCHES, MAX_CLUSTERS)

    # TODO: fix the sorting: first by tracks (and everything else sorts on this)
    # TODO: Then, for the clusters associated with the tracks, sort as below
    # # Sort clusters by energy descending
    # cls_e_idx = next(i for i, (n, _, _) in enumerate(CLUSTER_BRANCHES) if n == "cls_E")
    # csort = np.argsort(-chunk_data[:, :, cls_e_idx], axis=1)
    # chunk_data = np.take_along_axis(chunk_data, csort[:, :, np.newaxis], axis=1)

    # Tracks
    chunk_tracks = _vectorized_point_cloud(events, TRACK_BRANCHES, MAX_TRACKS)
    # trk_pt_idx = next(i for i, (n, _, _) in enumerate(TRACK_BRANCHES) if n == "trk_pT")
    # tsort = np.argsort(-chunk_tracks[:, :, trk_pt_idx], axis=1)
    # chunk_tracks = np.take_along_axis(chunk_tracks, tsort[:, :, np.newaxis], axis=1)

    # Cells per cluster — preserves cluster association (uses pre-sort csort)
    chunk_cells_pc = _vectorized_cells_per_cluster(
        label, events, CELL_BRANCHES, MAX_CLUSTERS, MAX_CELLS_PER_CLUSTER, use_cells=use_cells)

    chunk_pid = np.full(total_jets, label, dtype=np.int32)
    
    dm = events["truth_decayMode"]
    dm_flat = ak.to_numpy(ak.flatten(dm)).astype(np.int32)
    if label == 1:
        dm_flat[(dm_flat < 0) | (dm_flat > 6)] = -1
        chunk_decay_mode = dm_flat
    else:
        chunk_decay_mode = np.full(total_jets, -1, dtype=np.int32)

    # Truth regression targets
    chunk_tau_targets = _vectorized_scalar_targets(events, TAU_REGRESSION_TARGETS)
    chunk_charged_pion_targets = _vectorized_scalar_targets(events, CHARGED_PION_BRANCHES)
    chunk_neutral_pion_targets = _vectorized_scalar_targets(events, NEUTRAL_PION_BRANCHES)
    
    # Reco comparisons
    chunk_reco_id = _vectorized_scalar_targets_no_decorator(events, RECO_ID)
    chunk_reco_decay_mode = _vectorized_scalar_targets_no_decorator(events, DECAY_MODE[1:])
    chunk_reco_tau_4mom = _vectorized_scalar_targets_no_decorator(events, RECO_TAU_4MOM)
    chuck_reco_charged_pions = _temp_obtain_reco_pions_sum(events, RECO_CHARGED_PION_4MOM, 
                                                           CHARGED_PION_MASS)
    chuck_reco_neutral_pions = _temp_obtain_reco_pions_sum(events, RECO_NEUTRAL_PION_4MOM, 
                                                        NEUTRAL_PION_MASS)


    # Tau Track targets - only for tau jets, else None to avoid large zero arrays
    # TODO: ongoing edits
    chunk_tau_track_targets = _vectorized_point_cloud(events, TAUTRACK_CLASSIFICATION_BRANCHES, MAX_TRACKS) if label == 1 else None

    return (chunk_data, chunk_tracks, chunk_cells_pc, chunk_pid, chunk_decay_mode,
            chunk_tau_targets, chunk_charged_pion_targets, chunk_neutral_pion_targets,
            chunk_tau_track_targets,
            chunk_reco_id, chunk_reco_decay_mode, chunk_reco_tau_4mom,
            chuck_reco_charged_pions, chuck_reco_neutral_pions)


def process_file(filepath, label, chunk_size="250 MB", use_cells=True):
    """
    Process a single ROOT file with chunked reading (memory-efficient for large files).

    Returns:
        data: Cluster features [N_jets, MAX_CLUSTERS, NUM_CLUSTER_FEATURES]
        tracks: Track features [N_jets, MAX_TRACKS, NUM_TRACK_FEATURES]
        cells_per_cluster: Cell features [N_jets, MAX_CLUSTERS, MAX_CELLS_PER_CLUSTER, NUM_CELL_FEATURES]
        pid: Jet labels [N_jets]
        decay_mode: Decay mode labels [N_jets]
        tau_targets: Tau regression targets [N_jets, NUM_TAU_REGRESSION_TARGETS]
        charged_pion_targets: Charged pion sum [N_jets, 3] — (pt, eta, phi) of 4-vector sum per event
        neutral_pion_targets: Neutral pion sum [N_jets, 3] — (pt, eta, phi) of 4-vector sum per event
    """

    try:
        with uproot.open(filepath) as f:
            n_entries = f["CollectionTree"].num_entries
    except Exception as e:
        print(f"Error opening {filepath}: {e}")
        return None, None, None, None, None, None, None, None, None, None, None, None, None, None

    if n_entries == 0:
        print(f"File {filepath} is empty, skipping...")
        return None, None, None, None, None, None, None, None, None, None, None, None, None, None

    branches = get_all_branches(label=label, use_cells=use_cells)
    chunk_results = []

    for chunk_batch in uproot.iterate(f"{filepath}:CollectionTree", branches, step_size=chunk_size, library="ak", report=False):
        n_in_chunk = len(chunk_batch)
        result = _process_chunk(chunk_batch, label, n_in_chunk, use_cells=use_cells)
        if result is not None:
            chunk_results.append(result)
        del chunk_batch

    if len(chunk_results) == 0:
        print(f"  No jets in {filepath}")
        return None, None, None, None, None, None, None, None, None, None, None, None, None, None

    all_data = np.concatenate([r[0] for r in chunk_results], axis=0)
    all_tracks = np.concatenate([r[1] for r in chunk_results], axis=0)
    all_cells_pc = np.concatenate([r[2] for r in chunk_results], axis=0) if (label == 1 and use_cells) else None
    all_pid = np.concatenate([r[3] for r in chunk_results], axis=0)
    all_decay_mode = np.concatenate([r[4] for r in chunk_results], axis=0)
    all_tau_targets = np.concatenate([r[5] for r in chunk_results], axis=0)
    all_charged_pion_targets = np.concatenate([r[6] for r in chunk_results], axis=0)
    all_neutral_pion_targets = np.concatenate([r[7] for r in chunk_results], axis=0)
    all_tau_track_targets = np.concatenate([r[8] for r in chunk_results], axis=0) if label == 1 else None
    all_reco_id = np.concatenate([r[9]  for r in chunk_results], axis=0)
    all_reco_decay_mode = np.concatenate([r[10] for r in chunk_results], axis=0)
    all_reco_tau_4mom = np.concatenate([r[11] for r in chunk_results], axis=0)
    all_reco_charged_pions = np.concatenate([r[12] for r in chunk_results], axis=0)
    all_reco_neutral_pions = np.concatenate([r[13] for r in chunk_results], axis=0)
    del chunk_results

    return (all_data, all_tracks, all_cells_pc, all_pid, all_decay_mode,
            all_tau_targets, all_charged_pion_targets, all_neutral_pion_targets,
            all_tau_track_targets,
            all_reco_id, all_reco_decay_mode, all_reco_tau_4mom,
            all_reco_charged_pions, all_reco_neutral_pions)


def _process_file_to_staging(args):
    """Worker: process one file and write to staging. Returns (staging_path, n_jets) or (None, 0)."""
    filepath, label, chunk_size, staging_dir, file_idx, use_cells = args
    try:
        result = process_file(filepath, label, chunk_size=chunk_size, use_cells=use_cells)
    except Exception as e:
        raise RuntimeError(f"Error processing {filepath}: {e}") from e
    if result is None or result[0] is None:
        return None, 0
    (data, tracks, cells_pc, pid, decay_mode, tau_targets, charged_pion_targets, neutral_pion_targets,
     tau_track_targets, reco_id, reco_decay_mode, reco_tau_4mom, reco_charged_pions, reco_neutral_pions) = result
    n_jets = len(data)
    staging_path = os.path.join(staging_dir, f"staging_{file_idx:05d}.h5")
    with h5py.File(staging_path, "w") as hf:
        hf.create_dataset("data", data=data, compression="gzip")
        hf.create_dataset("tracks", data=tracks, compression="gzip")
        if use_cells:
            if cells_pc is not None:
                hf.create_dataset("cells_per_cluster", data=cells_pc, compression="gzip")
            else:
                hf.create_dataset(
                    "cells_per_cluster",
                    shape=(n_jets, MAX_CLUSTERS, MAX_CELLS_PER_CLUSTER, NUM_CELL_FEATURES),
                    dtype=np.float32,
                    compression="gzip",
                    fillvalue=0.0,
                )
        hf.create_dataset("pid", data=pid)
        hf.create_dataset("decay_mode", data=decay_mode)
        hf.create_dataset("tau_targets", data=tau_targets, compression="gzip")
        hf.create_dataset("charged_pion_targets", data=charged_pion_targets, compression="gzip")
        hf.create_dataset("neutral_pion_targets", data=neutral_pion_targets, compression="gzip")
        if tau_track_targets is not None:
            hf.create_dataset("tau_track_targets", data=tau_track_targets, compression="gzip")
        else:
            hf.create_dataset(
                "tau_track_targets",
                shape=(n_jets, MAX_TRACKS, NUM_TAU_TRACK_CLASSIFICATION_TARGETS),
                dtype=np.float32,
                compression="gzip",
                fillvalue=0.0,
            )
        hf.create_dataset("reco_id",            data=reco_id,            compression="gzip")
        hf.create_dataset("reco_decay_mode",    data=reco_decay_mode,    compression="gzip")
        hf.create_dataset("reco_tau_4mom",      data=reco_tau_4mom,      compression="gzip")
        hf.create_dataset("reco_charged_pions", data=reco_charged_pions, compression="gzip")
        hf.create_dataset("reco_neutral_pions", data=reco_neutral_pions, compression="gzip")
    return staging_path, n_jets


def main():
    parser = argparse.ArgumentParser(description="Prepare HDF5 datasets for OmniLearned")
    parser.add_argument("--label", type=int, default=None,
                        help="Label for --input_file: 0=QCD, 1=tau, 2=electron")
    parser.add_argument("--input_dir", type=str,
                        default="/global/cfs/projectdirs/m2616/TauCPML/DataTesting/ntuples/",
                        help="Directory containing ROOT files")
    parser.add_argument("--output_dir", type=str,
                        default="/global/cfs/projectdirs/m2616/TauCPML/DataTesting/processed_h5/tau",
                        help="Directory to save HDF5 files")
    parser.add_argument("--train_frac", type=float, default=0.8)
    parser.add_argument("--val_frac", type=float, default=0.1)
    parser.add_argument("--chunk_size", type=str, default="500 MB",
        help="ROOT read chunk size: int (entries) or str (e.g. '500 MB', '2 GB'). Lower = less memory, more I/O.",
    )
    parser.add_argument("--workers", type=int, default=9,
        help="Number of parallel workers for processing files. 1 = sequential. Use ≤ CPU cores; each worker uses ~chunk_size memory.",
    )
    parser.add_argument(
        "--no_cells", action="store_true", default=True,
        help="Skip reading and saving cell features (cells_per_cluster). Reduces memory and disk usage.",
    )
    parser.add_argument("--num-files", type=int, default=None)
    args = parser.parse_args()
    use_cells = not args.no_cells

    # Parse chunk_size: int (entries) or str e.g. "500 MB" for uproot
    try:
        chunk_size = int(args.chunk_size)
    except ValueError:
        chunk_size = args.chunk_size  # e.g. "500 MB", "2 GB"

    os.makedirs(args.output_dir, exist_ok=True)

    jz0_rucio_name = "user.nkyriaco.JZ0.Ntuple_04_06_26_Prod1_EXT0"
    jz1_rucio_name = "user.nkyriaco.JZ1.Ntuple_04_06_26_Prod1_EXT0"
    jz2_rucio_name = "user.nkyriaco.JZ2.Ntuple_04_06_26_Prod1_EXT0"
    jz3_rucio_name = "user.nkyriaco.JZ3.Ntuple_04_06_26_Prod1_EXT0"
    jz4_rucio_name = "user.nkyriaco.JZ4.Ntuple_04_06_26_Prod1_EXT0"

    tautau_rucio_name = "user.nkyriaco.Gammatautau.Ntuple_04_06_26_Prod1_EXT0"
    ee_rucio_name = "user.nkyriaco.Gammaee.Ntuple_04_06_26_Prod1_EXT0"

    num_files = args.num_files
    jz0_files = _list_root_files(os.path.join(args.input_dir, jz0_rucio_name), num_files=num_files)
    jz1_files = _list_root_files(os.path.join(args.input_dir, jz1_rucio_name), num_files=num_files)
    jz2_files = _list_root_files(os.path.join(args.input_dir, jz2_rucio_name), num_files=num_files)
    jz3_files = _list_root_files(os.path.join(args.input_dir, jz3_rucio_name), num_files=num_files)
    jz4_files = _list_root_files(os.path.join(args.input_dir, jz4_rucio_name), num_files=num_files)

    # Gammatautau files (label 1)
    gammatautau_files = _list_root_files(os.path.join(args.input_dir, tautau_rucio_name), 
                                         num_files=num_files)
    
    # Gammaee files (label 2)
    gammaee_files = _list_root_files(os.path.join(args.input_dir, ee_rucio_name), 
                                     num_files=num_files)
    
    files_and_labels = []

    # Add JZ0 files with label 0
    for fname in jz0_files:
        files_and_labels.append((os.path.join(args.input_dir, jz0_rucio_name, fname), 0))

    # Add JZ1 files with label 0
    for fname in jz1_files:
        files_and_labels.append((os.path.join(args.input_dir, jz1_rucio_name, fname), 0))

    # Add JZ2 files with label 0
    for fname in jz2_files:
        files_and_labels.append((os.path.join(args.input_dir, jz2_rucio_name, fname), 0))

    # Add JZ3 files with label 0
    for fname in jz3_files:
        files_and_labels.append((os.path.join(args.input_dir, jz3_rucio_name, fname), 0))

    # Add JZ4 files with label 0
    for fname in jz4_files:
        files_and_labels.append((os.path.join(args.input_dir, jz4_rucio_name, fname), 0))

    # Add Gammatautau files with label 1
    for fname in gammatautau_files:
        files_and_labels.append((os.path.join(args.input_dir, tautau_rucio_name, fname), 1))

    # Add Gammaee files with label 2
    for fname in gammaee_files:
        files_and_labels.append((os.path.join(args.input_dir, ee_rucio_name, fname), 2))

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
        (fp, lb, chunk_size, staging_dir, idx, use_cells)
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
                cells_pc = sf["cells_per_cluster"][:] if use_cells else None
                pid = sf["pid"][:]
                decay_mode = sf["decay_mode"][:]
                tau_targets = sf["tau_targets"][:]
                charged_pion_targets = sf["charged_pion_targets"][:]
                neutral_pion_targets = sf["neutral_pion_targets"][:]
                tau_track_targets = sf["tau_track_targets"][:]

            if sample_shapes is None:
                sample_shapes = (
                    data.shape[1:], tracks.shape[1:], cells_pc.shape[1:] if use_cells else None,
                    tau_targets.shape[1:], charged_pion_targets.shape[1:],
                    neutral_pion_targets.shape[1:], tau_track_targets.shape[1:]
                )

            n = len(data)
            r = np.random.random(n)
            train_mask = r < train_frac
            val_mask = (r >= train_frac) & (r < train_frac + val_frac)
            test_mask = r >= train_frac + val_frac

            for split_name, mask in [("train", train_mask), ("val", val_mask), ("test", test_mask)]:
                if not mask.any():
                    continue
                d = data[mask]
                t = tracks[mask]
                cpc = cells_pc[mask] if use_cells else None
                p = pid[mask]
                dm = decay_mode[mask]
                tt = tau_targets[mask]
                cpt = charged_pion_targets[mask]
                npt = neutral_pion_targets[mask]
                ttt = tau_track_targets[mask]
                fp = split_files[split_name]
                if split_name not in h5_handles:
                    h5_handles[split_name] = h5py.File(fp, "w")
                    h5_handles[split_name].create_dataset("data", data=d, compression="gzip", 
                                                          maxshape=(None,) + d.shape[1:])
                    h5_handles[split_name].create_dataset("tracks", data=t, compression="gzip", 
                                                          maxshape=(None,) + t.shape[1:])
                    if use_cells:
                        h5_handles[split_name].create_dataset("cells_per_cluster", data=cpc, 
                                    compression="gzip", maxshape=(None,) + cpc.shape[1:])
                    h5_handles[split_name].create_dataset("pid", data=p, maxshape=(None,))
                    h5_handles[split_name].create_dataset("decay_mode", data=dm, maxshape=(None,))
                    h5_handles[split_name].create_dataset("tau_targets", data=tt, 
                                    compression="gzip", maxshape=(None,) + tt.shape[1:])
                    h5_handles[split_name].create_dataset("charged_pion_targets", 
                                    data=cpt, compression="gzip", maxshape=(None,) + cpt.shape[1:])
                    h5_handles[split_name].create_dataset("neutral_pion_targets", 
                                    data=npt, compression="gzip", maxshape=(None,) + npt.shape[1:])
                    h5_handles[split_name].create_dataset("tau_track_targets", 
                                    data=ttt, compression="gzip", maxshape=(None,) + ttt.shape[1:])
                    split_offsets[split_name] = len(d)
                else:
                    hf = h5_handles[split_name]
                    cells_items = [("cells_per_cluster", cpc)] if use_cells else []
                    for ds_name, arr in [
                        ("data", d), ("tracks", t), *cells_items,
                        ("pid", p), ("decay_mode", dm),
                        ("tau_targets", tt),
                        ("charged_pion_targets", cpt), ("neutral_pion_targets", npt),
                        ("tau_track_targets", ttt)
                    ]:
                        hf[ds_name].resize(split_offsets[split_name] + len(arr), axis=0)
                        hf[ds_name][split_offsets[split_name]:] = arr
                    split_offsets[split_name] += len(d)
    finally:
        for h in h5_handles.values():
            h.close()

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
        dshape, tshape = f["data"].shape, f["tracks"].shape
        if use_cells:
            cpc_shape = f["cells_per_cluster"].shape
        tau_shape = f["tau_targets"].shape
        cpt_shape = f["charged_pion_targets"].shape
        npt_shape = f["neutral_pion_targets"].shape
        ttt_shape = f["tau_track_targets"].shape
    print(f"\n  data (clusters): (N, {dshape[1]}, {dshape[2]})")
    print(f"    - {MAX_CLUSTERS} clusters x {NUM_CLUSTER_FEATURES} features")
    print(f"    - First 4: dEta, dPhi, log(et), log(e)")
    print(f"\n  tracks: (N, {tshape[1]}, {tshape[2]})")
    print(f"    - {MAX_TRACKS} tracks x {NUM_TRACK_FEATURES} features")
    print(f"    - First 4: dEta, dPhi, log(pt), theta")
    if use_cells:
        print(f"\n  cells_per_cluster: (N, {cpc_shape[1]}, {cpc_shape[2]}, {cpc_shape[3]})")
        print(f"    - {MAX_CLUSTERS} clusters x {MAX_CELLS_PER_CLUSTER} cells x {NUM_CELL_FEATURES} features")
    print(f"\n  tau_targets: (N, {tau_shape[1]})")
    print(f"    - {NUM_TAU_REGRESSION_TARGETS} scalar targets per jet: truth_pt, truth_eta, truth_phi")
    if use_cells:
        print(f"\n  charged_pion_targets: (N, {cpt_shape[1]})")
        print("    - (pt, eta, phi) of 4-vector sum of charged pions per event")
    print(f"\n  neutral_pion_targets: (N, {npt_shape[1]})")
    print("    - (pt, eta, phi) of 4-vector sum of neutral pions per event")
    print(f"\n  tau_track_targets: (N, {ttt_shape[1]}, {ttt_shape[2]})")
    print(f"    - up to {MAX_TRACKS} tracks x {NUM_TAU_TRACK_CLASSIFICATION_TARGETS} features (tau_track_class), 0-padded")
    print(f"\nCluster features ({NUM_CLUSTER_FEATURES}):")
    for i, (name, _, _) in enumerate(CLUSTER_BRANCHES):
        print(f"    {i}: {name}")
    print(f"\nTrack features ({NUM_TRACK_FEATURES}):")
    for i, (name, _, _) in enumerate(TRACK_BRANCHES):
        print(f"    {i}: {name}")
    if use_cells:
        print(f"\nCell features ({NUM_CELL_FEATURES}):")
        for i, (name, _, _) in enumerate(CELL_BRANCHES):
            print(f"    {i}: {name}")
    print(f"\n" + "="*60)
    print("TRAINING COMMAND")
    print("="*60)
    print(f"\n  omnilearned train --dataset tau --path datasets \\")
    print(f"    --num-feat {NUM_CLUSTER_FEATURES} --num-classes 3 \\")
    print(f"    --use-tracks --track-dim {NUM_TRACK_FEATURES} \\")
    print(f"    --use-cells --cell-dim {NUM_CELL_FEATURES} \\")
    print(f"    --aux-tasks-str 'decay_mode:5'")


if __name__ == "__main__":
    main()