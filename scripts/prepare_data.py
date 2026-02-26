"""
Prepare HDF5 datasets for OmniLearned tau identification.

Converts ROOT files to HDF5 format with:
- data: Cluster point cloud features [N, MAX_CLUSTERS, NUM_CLUSTER_FEATURES]
- tracks: Track point cloud features [N, MAX_TRACKS, NUM_TRACK_FEATURES]  
- pid: Jet type label (0=QCD, 1=tau, 2=electron)
- decay_mode: Tau decay mode (0=1-prong, 1=3-prong, -1=N/A)

Both clusters and tracks are treated as point clouds with first features being:
- Clusters: [dEta, dPhi, log(et), log(e), ...]
- Tracks: [dEta, dPhi, log(pt), theta, ...]

Usage:
    python prepare_data.py --input_dir /path/to/root/files --output_dir /path/to/output
"""

import uproot
import numpy as np
import awkward as ak
import h5py
import os
import argparse
from tqdm import tqdm


# Configuration
MAX_CLUSTERS = 15
MAX_TRACKS = 10
MAX_CELLS = 20

# Cluster branches (particles in point cloud)
# First 4 MUST be: [dEta, dPhi, log(pT), log(E)]
CLUSTER_BRANCHES = [
    ("cls_E", "cls_E", True),                # Feature 2: E
    ("cls_ET", "cls_ET", True),                  # Feature 3: E_{T}
    ("cls_Eta", "cls_Eta", False),           # Feature 0: η
    ("cls_Phi", "cls_Phi", False),           # Feature 1: φ
    ("cls_FIRST_ENG_DENS", "cls_FIRST_ENG_DENS", False),

#### Commenting these out for now. Making a to-do list of everything else to add in the future, but want to get a basic version working first with just the 4 main features.
    #("cls_SECOND_R", "TauClusters.cls_SECOND_R", False),
    #("cls_SECOND_LAMBDA", "TauClusters.cls_SECOND_LAMBDA", False),
    #("cls_EM_PROBABILITY", "TauClusters.cls_EM_PROBABILITY", False),
    #("cls_CENTER_MAG", "TauClusters.cls_CENTER_MAG", False),
    #("cls_CENTER_LAMBDA", "TauClusters.cls_CENTER_LAMBDA", False),
]

# Track branches (all features, will be flattened into global)
TRACK_BRANCHES = [

    ("trk_E", "trk_E", True),
    ("trk_pT", "trk_pT", True),
    ("trk_Eta", "trk_Eta", False),
    ("trk_Phi", "trk_Phi", False),
    #("trk_eProbNN", "trk_eProbNN", False), # We should not use this anyway, but eventually if we have samples we should
    ("trk_charge", "trk_charge", False),
    ("trk_d0", "trk_d0", False),
    ("trk_z0", "trk_z0", False),
    ("trk_z0sintheta", "trk_z0sintheta", False),
    ("trk_nTRTHits", "trk_nTRTHits", False),
    ("trk_nTRTHighThresholdHits", "trk_nTRTHighThresholdHits", False),
    ("trk_nSCTHits", "trk_nSCTHits", False),
    ("trk_nPixelHits", "trk_nPixelHHits", False),
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
    ("cell_E", "cell_E", True),
    ("cell_ET", "cell_ET", True),
    ("cell_Eta", "cell_Eta", False),
    ("cell_Phi", "cell_Phi", False),
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


def process_file(filepath, label):
    """
    Process a single ROOT file.
    
    Returns:
        data: Cluster features [N_jets, MAX_CLUSTERS, NUM_CLUSTER_FEATURES]
        tracks: Track features [N_jets, MAX_TRACKS, NUM_TRACK_FEATURES]
        cells: Cell features [N_jets, MAX_CELLS, NUM_CELL_FEATURES]
        pid: Jet labels [N_jets]
        decay_mode: Decay mode labels [N_jets]
    """
    print(f"Processing {filepath} with label {label}")
    
    try:
        f = uproot.open(f"{filepath}:CollectionTree")
    except Exception as e:
        print(f"Error opening {filepath}: {e}")
        return None, None, None, None, None
    
    if f.num_entries == 0:
        print(f"File {filepath} is empty, skipping...")
        return None, None, None, None, None
    
    branches = get_all_branches()
    events = f.arrays(branches, library='ak')
    n_events = len(events)
    print(f"  Found {n_events} events")
    
    n_jets_per_event = ak.num(events["TauJets.pt"])
    total_jets = int(ak.sum(n_jets_per_event))
    print(f"  Total jets: {total_jets}")
    print(f"  Cluster features: {NUM_CLUSTER_FEATURES}")
    print(f"  Track features: {NUM_TRACK_FEATURES}")
    print(f"  Cell features: {NUM_CELL_FEATURES}")
    
    # Clusters as point cloud
    all_data = np.zeros((total_jets, MAX_CLUSTERS, NUM_CLUSTER_FEATURES), dtype=np.float32)
    # Tracks as point cloud (NOT flattened)
    all_tracks = np.zeros((total_jets, MAX_TRACKS, NUM_TRACK_FEATURES), dtype=np.float32)
    # Cells as point cloud
    all_cells = np.zeros((total_jets, MAX_CELLS, NUM_CELL_FEATURES), dtype=np.float32)

    all_pid = np.full(total_jets, label, dtype=np.int64)
    all_decay_mode = np.full(total_jets, -1, dtype=np.int64)
    
    jet_counter = 0
    
    for evt_idx in tqdm(range(n_events), desc="  Events", leave=False):
        n_jets = int(n_jets_per_event[evt_idx])
        
        # Get cluster data
        cluster_arrays = {}
        for feat_name, branch_name, _ in CLUSTER_BRANCHES:
            cluster_arrays[feat_name] = events[branch_name][evt_idx]
        
        # Get track data
        track_arrays = {}
        for feat_name, branch_name, _ in TRACK_BRANCHES:
            track_arrays[feat_name] = [events[branch_name][evt_idx]]
        
        # Get cell data
        cell_arrays = {}
        for feat_name, branch_name, _ in CELL_BRANCHES:
            cell_arrays[feat_name] = events[branch_name][evt_idx]
        
        decay_mode = events["truth_decayMode"][evt_idx]
        
        for jet_idx in range(n_jets):
            # =========================================
            # CLUSTERS (point cloud 1)
            # =========================================
            for feat_idx, (feat_name, _, apply_log) in enumerate(CLUSTER_BRANCHES):
                event_clusters = cluster_arrays[feat_name]
                if len(event_clusters) > jet_idx:
                    jet_clusters = event_clusters[jet_idx]
                    if len(jet_clusters) > 0:
                        n = min(len(jet_clusters), MAX_CLUSTERS)
                        values = ak.to_numpy(jet_clusters[:n])
                        if apply_log:
                            values = safe_log(values)
                        all_data[jet_counter, :n, feat_idx] = values

            # =========================================
            # TRACKS (point cloud 2 - NOT flattened)
            # =========================================
            for feat_idx, (feat_name, _, apply_log) in enumerate(TRACK_BRANCHES):
                event_tracks = track_arrays[feat_name]
                if len(event_tracks) > jet_idx:
                    jet_tracks = event_tracks[jet_idx]
                    if len(jet_tracks) > 0:
                        n = min(len(jet_tracks), MAX_TRACKS)
                        values = jet_tracks[:n]
                        if apply_log:
                            values = safe_log(values)
                        all_tracks[jet_counter, :n, feat_idx] = values

            # =========================================
            # CELLS (point cloud 3)
            # =========================================
            for feat_idx, (feat_name, _, apply_log) in enumerate(CELL_BRANCHES):
                event_cells = cell_arrays[feat_name]
                if len(event_cells) > jet_idx:
                    jet_cells = event_cells[jet_idx]
                    if len(jet_cells) > 0:
                        n = min(len(jet_cells), MAX_CELLS)
                        values = ak.to_numpy(jet_cells[:n])
                        if apply_log:
                            values = safe_log(values)
                        all_cells[jet_counter, :n, feat_idx] = values

            # Decay mode (only for tau)
            #if label == 1:
            #    dm = int(decay_mode[jet_idx])
            #    if 0 <= dm <= 4:
            #        all_decay_mode[jet_counter] = dm
            #    else:
            #        all_decay_mode[jet_counter] = -1 
            all_decay_mode[jet_counter] =int(decay_mode[jet_idx]) if label == 1 else -1             
            
            jet_counter += 1
    
    print(f"  Output: data={all_data.shape}, tracks={all_tracks.shape}, cells={all_cells.shape}")
    return all_data, all_tracks, all_cells, all_pid, all_decay_mode


def main():
    parser = argparse.ArgumentParser(description="Prepare HDF5 datasets for OmniLearned")
    parser.add_argument("--input_dir", type=str, default="/global/homes/a/agarabag/pscratch/OmniLearned/samples",
                        help="Directory containing ROOT files")
    parser.add_argument("--output_dir", type=str, default="/global/homes/a/agarabag/pscratch/OmniLearned/datasets/tau",
                        help="Directory to save HDF5 files")
    parser.add_argument("--train_frac", type=float, default=0.6)
    parser.add_argument("--val_frac", type=float, default=0.2)
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    files_and_labels = [
        (os.path.join(args.input_dir, "JZ.root"), 0),
        #(os.path.join(args.input_dir, "Gammatautau.root"), 1),
        #(os.path.join(args.input_dir, "Gammaee.root"), 2),
    ]
    
    all_data = []
    all_tracks = []
    all_cells = []
    all_pid = []
    all_decay_mode = []
    
    for filepath, label in files_and_labels:
        if not os.path.exists(filepath):
            print(f"Warning: {filepath} not found, skipping...")
            continue
        
        data, tracks, cells, pid, decay_mode = process_file(filepath, label)
        
        if data is not None:
            all_data.append(data)
            all_tracks.append(tracks)
            all_cells.append(cells)
            all_pid.append(pid)
            all_decay_mode.append(decay_mode)
    
    if len(all_data) == 0:
        print("No data processed!")
        return
    
    data = np.concatenate(all_data, axis=0)
    tracks = np.concatenate(all_tracks, axis=0)
    cells = np.concatenate(all_cells, axis=0)
    pid = np.concatenate(all_pid, axis=0)
    decay_mode = np.concatenate(all_decay_mode, axis=0)
    
    print(f"\nTotal samples: {len(data)}")
    print(f"  QCD (0): {np.sum(pid == 0)}")
    print(f"  Tau (1): {np.sum(pid == 1)}")
    print(f"  Electron (2): {np.sum(pid == 2)}")
    print(f"  Decay mode 1p0n: {np.sum(decay_mode == 0)}")
    print(f"  Decay mode 1p1n: {np.sum(decay_mode == 1)}")
    print(f"  Decay mode 1pXn: {np.sum(decay_mode == 2)}")
    print(f"  Decay mode 3p0n: {np.sum(decay_mode == 3)}")
    print(f"  Decay mode 3pXn: {np.sum(decay_mode == 4)}")
    print(f"  Decay mode N/A: {np.sum((decay_mode < 0) | (decay_mode > 4))}")
    
    np.random.seed(42)
    indices = np.random.permutation(len(data))
    data = data[indices]
    tracks = tracks[indices]
    cells = cells[indices]
    pid = pid[indices]
    decay_mode = decay_mode[indices]
    
    n_total = len(data)
    n_train = int(n_total * args.train_frac)
    n_val = int(n_total * args.val_frac)
    
    splits = {
        "train": (data[:n_train], tracks[:n_train], cells[:n_train], pid[:n_train], decay_mode[:n_train]),
        "val": (data[n_train:n_train+n_val], tracks[n_train:n_train+n_val], cells[n_train:n_train+n_val], pid[n_train:n_train+n_val], decay_mode[n_train:n_train+n_val]),
        "test": (data[n_train+n_val:], tracks[n_train+n_val:], cells[n_train+n_val:], pid[n_train+n_val:], decay_mode[n_train+n_val:]),
    }
    
    print(f"\nSplit sizes:")
    for name, (d, *_) in splits.items():
        print(f"  {name}: {len(d)}")
    
    for split_name, (split_data, split_tracks, split_cells, split_pid, split_dm) in splits.items():
        output_path = os.path.join(args.output_dir, split_name)
        os.makedirs(output_path, exist_ok=True)
        
        filepath = os.path.join(output_path, "data.h5")
        print(f"Saving {filepath}...")
        
        with h5py.File(filepath, 'w') as f:
            f.create_dataset('data', data=split_data, compression='gzip')
            f.create_dataset('tracks', data=split_tracks, compression='gzip')
            f.create_dataset('cells', data=split_cells, compression='gzip')
            f.create_dataset('pid', data=split_pid)
            f.create_dataset('decay_mode', data=split_dm)
        
        n_samples = len(split_data)
        file_indices = np.array([(0, i) for i in range(n_samples)], dtype=np.int32)
        np.save(os.path.join(output_path, "file_index.npy"), file_indices)
    
    print("\nDone!")
    print(f"\n" + "="*60)
    print("DATA STRUCTURE (Option B: Tracks as Separate Tokens)")
    print("="*60)
    print(f"\n  data (clusters): {data.shape}")
    print(f"    - {MAX_CLUSTERS} clusters x {NUM_CLUSTER_FEATURES} features")
    print(f"    - First 4: dEta, dPhi, log(et), log(e)")
    print(f"\n  tracks: {tracks.shape}")
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
    print(f"    --aux-tasks-str 'decay_mode:2,electron_vs_qcd:2'")


if __name__ == "__main__":
    main()
