import torch
import h5py
from argparse import ArgumentParser
from torch.utils.data import Dataset, DataLoader
import requests
import re
import os
from urllib.parse import urljoin
import numpy as np
from pathlib import Path


class FileGroupedBatchSampler(torch.utils.data.Sampler):
    """
    Yields batch indices grouped by file for mmap cache locality.
    Shuffles file order and within-file order each epoch via set_epoch().
    """

    def __init__(self, file_indices, batch_size, drop_last=False):
        self._batch_size = batch_size
        self._drop_last = drop_last
        self._epoch = 0

        self._by_file = {}
        for pos, (file_idx, _) in enumerate(file_indices):
            self._by_file.setdefault(file_idx, []).append(pos)
        self._total = len(file_indices)

    def set_epoch(self, epoch):
        self._epoch = epoch

    def __iter__(self):
        rng = np.random.default_rng(self._epoch)
        file_keys = list(self._by_file.keys())
        rng.shuffle(file_keys)

        all_indices = []
        for fk in file_keys:
            positions = list(self._by_file[fk])
            rng.shuffle(positions)
            all_indices.extend(positions)

        for start in range(0, len(all_indices), self._batch_size):
            batch = all_indices[start : start + self._batch_size]
            if self._drop_last and len(batch) < self._batch_size:
                continue
            yield batch

    def __len__(self):
        if self._drop_last:
            return self._total // self._batch_size
        return (self._total + self._batch_size - 1) // self._batch_size


def collate_point_cloud(batch, max_part=5000):
    """
    Collate function for point clouds and labels with truncation performed per batch.

    Args:
        batch (list of dicts): Each element is a dictionary with keys:
            - "X" (Tensor): Point cloud of shape (N, F)
            - "y" (Tensor): Label tensor
            - "cond" (optional, Tensor): Conditional info
            - "pid" (optional, Tensor): Particle IDs
            - "add_info" (optional, Tensor): Extra features

    Returns:
        Dict[str, torch.Tensor]: Dictionary containing collated tensors:
            - "X": (B, M, F) Truncated point clouds
            - "y": (B, num_classes)
            - "cond", "pid", "add_info" (optional, shape (B, M, ...))
    """
    batch_X = [item["X"] for item in batch]
    batch_y = [item["y"] for item in batch]

    point_clouds = torch.stack(batch_X)  # (B, N, F)
    labels = torch.stack(batch_y)  # (B, num_classes)

    # Truncate to actual max valid particles in this batch (feature index 2 == pT)
    valid_mask = point_clouds[:, :, 2] != 0
    max_particles = min(valid_mask.sum(dim=1).max().item(), max_part)

    truncated_X = point_clouds[:, :max_particles, :].contiguous()  # (B, M, F)
    result = {"X": truncated_X, "y": labels}

    sequence_fields = ["pid", "add_info", "data_pid", "vertex_pid"]
    jet_level_fields = ["cond", "decay_mode", "tau_targets", "charged_pion_targets",
                        "neutral_pion_targets", "reco_tau_4mom", "reco_charged_pions",
                        "reco_neutral_pions", "reco_id", "reco_decay_mode",
                        "tau_vertex_targets", "vertex_slot_mask"]
    # Tracks are appended as separate tokens, so no cluster-dim truncation needed
    point_cloud_fields = ["tracks"]
    tau_track_target_fields = ["tau_track_targets"]
    # cells_per_cluster shares the cluster dimension with X and must be truncated
    cluster_aligned_fields = ["cells_per_cluster"]

    for field in sequence_fields:
        if all(field in item for item in batch):
            stacked = torch.stack([item[field] for item in batch])
            if stacked.dim() >= 2 and stacked.shape[1] >= max_particles:
                stacked = stacked[:, :max_particles].contiguous()
            result[field] = stacked
        else:
            result[field] = None

    for field in jet_level_fields:
        if all(field in item for item in batch):
            result[field] = torch.stack([item[field] for item in batch])
        else:
            result[field] = None

    for field in point_cloud_fields:
        if all(field in item for item in batch):
            stacked = torch.stack([item[field] for item in batch])
            result[field] = stacked
        else:
            result[field] = None

    for field in tau_track_target_fields:
        if all(field in item for item in batch):
            stacked = torch.stack([item[field] for item in batch])
            result[field] = stacked
        else:
            result[field] = None

    for field in cluster_aligned_fields:
        if all(field in item for item in batch):
            stacked = torch.stack([item[field] for item in batch])
            stacked = stacked[:, :max_particles].contiguous()
            result[field] = stacked
        else:
            result[field] = None

    return result


def get_url(
    dataset_name,
    dataset_type,
    base_url="https://portal.nersc.gov/cfs/dasrepo/omnilearned/",
):
    url = f"{base_url}/{dataset_name}/{dataset_type}/"
    try:
        requests.head(url, allow_redirects=True, timeout=5)
        return url
    except requests.RequestException:
        print(
            "ERROR: Request timed out, visit https://www.nersc.gov/users/status for status on  portal.nersc.gov"
        )
        return None


def download_h5_files(base_url, destination_folder):
    """
    Downloads all .h5 files from the specified directory URL.

    Args:
        base_url (str): The base URL of the directory containing the .h5 files.
        destination_folder (str): The local folder to save the downloaded files.
    """

    response = requests.get(base_url)
    if response.status_code != 200:
        print(f"Failed to access {base_url}")
        return

    file_links = re.findall(r'href="([^"]+\.h5)"', response.text)

    for file_name in file_links:
        file_url = urljoin(base_url, file_name)
        file_path = os.path.join(destination_folder, file_name)

        print(f"Downloading {file_url} to {file_path}")
        with requests.get(file_url, stream=True) as r:
            r.raise_for_status()
            with open(file_path, "wb") as f:
                for chunk in r.iter_content(chunk_size=8192):
                    f.write(chunk)
        print(f"Downloaded {file_name}")


class HEPDataset(Dataset):
    def __init__(
        self,
        file_paths,
        file_indices=None,
        use_cond=False,
        use_pid=False,
        pid_idx=4,
        use_add=False,
        num_add=4,
        label_shift=0,
        clip_inputs=False,
        mode="",
        nevts=-1,
        use_tracks=False,
        use_cells=False,
        do_regression_aux_tasks=False,
        do_vertex_classification=False
    ):
        self.use_cond = use_cond
        self.use_pid = use_pid
        self.use_add = use_add
        self.use_tracks = use_tracks
        self.use_cells = use_cells
        self.pid_idx = pid_idx
        self.num_add = num_add
        self.label_shift = label_shift
        self.do_regression_aux_tasks = do_regression_aux_tasks
        self.do_vertex_classification = do_vertex_classification

        self.file_paths = file_paths
        self._file_cache = {}
        self.file_indices = file_indices
        self.clip_inputs = clip_inputs
        self.mode = mode
        self.nevts = int(nevts)
        if self.nevts < 0:
            self.nevts = len(self.file_indices)

        self._npy_cache_paths = {}
        self._prepare_npy_cache()

    def __len__(self):
        return min(self.nevts, len(self.file_indices))

    def _required_keys(self):
        keys = ["data", "pid"]
        if self.use_cond:
            keys.append("global")
        if self.use_tracks:
            keys.append("tracks")
        if self.use_cells:
            keys.append("cells_per_cluster")
        if self.do_regression_aux_tasks:
            keys.extend(["tau_targets", "charged_pion_targets", "neutral_pion_targets"])
        if self.do_vertex_classification:
            keys.append("tau_vertex_targets")
            keys.append("vertex_slot_mask")
        if self.use_tracks:
            keys.extend(["tau_track_targets"])
        keys.extend(["decay_mode", "data_pid",
                     "reco_id", "reco_decay_mode", "reco_tau_4mom", "reco_charged_pions", "reco_neutral_pions"])
        return keys

    def _prepare_npy_cache(self):
        """One-time conversion of HDF5 datasets to .npy for mmap-based access."""
        required = self._required_keys()
        for file_idx, file_path in enumerate(self.file_paths):
            cache_dir = Path(file_path).parent / "npy_cache"
            stem = Path(file_path).stem

            with h5py.File(file_path, "r") as f:
                available = set(f.keys())
                keys_to_cache = [k for k in required if k in available]

                missing = [
                    k for k in keys_to_cache
                    if not (cache_dir / f"{stem}_{k}.npy").exists()
                ]

                if missing:
                    cache_dir.mkdir(parents=True, exist_ok=True)
                    print(f"Building numpy cache for {Path(file_path).name} (one-time)...")
                    for key in missing:
                        npy_path = cache_dir / f"{stem}_{key}.npy"
                        ds = f[key]
                        out = np.lib.format.open_memmap(
                            str(npy_path), mode="w+",
                            dtype=ds.dtype, shape=ds.shape,
                        )
                        chunk_size = 10000
                        for start in range(0, ds.shape[0], chunk_size):
                            end = min(start + chunk_size, ds.shape[0])
                            out[start:end] = ds[start:end]
                        del out
                        print(f"  Cached {key} {ds.shape}")

            paths = {}
            for key in keys_to_cache:
                npy_path = cache_dir / f"{stem}_{key}.npy"
                if npy_path.exists():
                    paths[key] = str(npy_path)
            self._npy_cache_paths[file_idx] = paths

    def _get_file(self, file_idx):
        if file_idx not in self._file_cache:
            if self._npy_cache_paths.get(file_idx):
                data = {}
                for key, npy_path in self._npy_cache_paths[file_idx].items():
                    data[key] = np.load(npy_path, mmap_mode="r")
                self._file_cache[file_idx] = data
            else:
                file_path = self.file_paths[file_idx]
                self._file_cache[file_idx] = h5py.File(file_path, "r")
        return self._file_cache[file_idx]

    def _build_sample(
        self,
        x,
        label,
        cond=None,
        data_pid=None,
        decay_mode=None,
        tracks=None,
        tau_track_targets=None,
        cells_per_cluster=None,
        tau_targets=None,
        charged_pion_targets=None,
        neutral_pion_targets=None,
        reco_id=None,
        reco_decay_mode=None,
        reco_tau_4mom=None,
        reco_charged_pions=None,
        reco_neutral_pions=None,
        tau_vertex_targets=None,
        vertex_slot_mask=None,
    ):
        sample = {}

        sample["X"] = torch.tensor(x, dtype=torch.float32)
        if self.clip_inputs:
            mask_part = (torch.hypot(sample["X"][:, 0], sample["X"][:, 1]) < 0.8) & (
                sample["X"][:, 2] > 0.0
            )
            sample["X"][:, 3] = np.clip(
                sample["X"][:, 3], a_min=sample["X"][:, 2], a_max=None
            )
            sample["X"] = sample["X"] * mask_part.unsqueeze(-1).float()

        pid_dtype = torch.float32 if self.mode == "regression" else torch.int64
        sample["y"] = torch.tensor(label - self.label_shift, dtype=pid_dtype)

        if cond is not None and self.use_cond:
            sample["cond"] = torch.tensor(cond, dtype=torch.float32)

        if self.use_pid:
            sample["pid"] = sample["X"][:, self.pid_idx].int()
            sample["X"] = torch.cat(
                (sample["X"][:, : self.pid_idx], sample["X"][:, self.pid_idx + 1 :]),
                dim=1,
            )

        if self.use_add:
            sample["add_info"] = sample["X"][:, -self.num_add :]
            sample["X"] = sample["X"][:, : -self.num_add]

        if data_pid is not None and self.mode in ["segmentation", "ftag"]:
            data_dtype = torch.float32 if self.mode == "segmentation" else torch.int64
            sample["data_pid"] = torch.tensor(data_pid, dtype=data_dtype)

        if decay_mode is not None:
            sample["decay_mode"] = torch.tensor(decay_mode, dtype=torch.int64)

        if tracks is not None and self.use_tracks:
            sample["tracks"] = torch.tensor(tracks, dtype=torch.float32)

        if tracks is not None and self.use_tracks and tau_track_targets is not None:
            sample["tau_track_targets"] = torch.tensor(tau_track_targets, dtype=torch.int64)

        if cells_per_cluster is not None and self.use_cells:
            sample["cells_per_cluster"] = torch.tensor(cells_per_cluster, dtype=torch.float32)
            
        # Regression truth targets (e.g., TES correction)
        if tau_targets is not None and self.do_regression_aux_tasks:
            sample["tau_targets"] = torch.tensor(tau_targets, dtype=torch.float32)
        if charged_pion_targets is not None and self.do_regression_aux_tasks:
            sample["charged_pion_targets"] = torch.tensor(charged_pion_targets, dtype=torch.float32)
        if neutral_pion_targets is not None and self.do_regression_aux_tasks:
            sample["neutral_pion_targets"] = torch.tensor(neutral_pion_targets, dtype=torch.float32)

        if reco_id is not None:
            sample["reco_id"] = torch.tensor(reco_id, dtype=torch.float32)

        if reco_decay_mode is not None:
            sample["reco_decay_mode"] = torch.tensor(reco_decay_mode, dtype=torch.int64)

        if reco_tau_4mom is not None:
            sample["reco_tau_4mom"] = torch.tensor(reco_tau_4mom, dtype=torch.float32)
        if reco_charged_pions is not None:
            sample["reco_charged_pions"] = torch.tensor(reco_charged_pions, dtype=torch.float32)
        if reco_neutral_pions is not None:
            sample["reco_neutral_pions"] = torch.tensor(reco_neutral_pions, dtype=torch.float32)

        if tau_vertex_targets is not None and self.do_vertex_classification:
            sample["tau_vertex_targets"] = torch.tensor(tau_vertex_targets, dtype=torch.int32)

        if vertex_slot_mask is not None and self.do_vertex_classification:
            sample["vertex_slot_mask"] = torch.tensor(vertex_slot_mask, dtype=torch.bool)

        return sample

    def __getitem__(self, idx):
        file_idx, sample_idx = self.file_indices[idx]
        f = self._get_file(file_idx)

        cond = f["global"][sample_idx] if "global" in f and self.use_cond else None
        data_pid = f["data_pid"][sample_idx] if self.mode in ["segmentation", "ftag"] else None
        decay_mode = f["decay_mode"][sample_idx] if "decay_mode" in f else None
        tracks = f["tracks"][sample_idx] if self.use_tracks and "tracks" in f else None
        tau_track_targets = f["tau_track_targets"][sample_idx] if self.use_tracks and "tau_track_targets" in f else None
        cells = f["cells_per_cluster"][sample_idx] if self.use_cells and "cells_per_cluster" in f else None
        tau_targets = f["tau_targets"][sample_idx] if self.do_regression_aux_tasks and "tau_targets" in f else None
        charged_pion_targets = f["charged_pion_targets"][sample_idx] if self.do_regression_aux_tasks and "charged_pion_targets" in f else None
        neutral_pion_targets = f["neutral_pion_targets"][sample_idx] if self.do_regression_aux_tasks and "neutral_pion_targets" in f else None
        tau_vertex_targets = f["tau_vertex_targets"][sample_idx] if self.do_vertex_classification and "tau_vertex_targets" in f else None
        vertex_slot_mask = f["vertex_slot_mask"][sample_idx] if self.do_vertex_classification and "vertex_slot_mask" in f else None
        reco_id = f["reco_id"][sample_idx] if "reco_id" in f else None
        reco_decay_mode = f["reco_decay_mode"][sample_idx] if "reco_decay_mode" in f else None
        reco_tau_4mom = f["reco_tau_4mom"][sample_idx] if "reco_tau_4mom" in f else None
        reco_charged_pions = f["reco_charged_pions"][sample_idx] if "reco_charged_pions" in f else None
        reco_neutral_pions = f["reco_neutral_pions"][sample_idx] if "reco_neutral_pions" in f else None

        return self._build_sample(
            f["data"][sample_idx],
            f["pid"][sample_idx],
            cond=cond,
            data_pid=data_pid,
            decay_mode=decay_mode,
            tracks=tracks,
            tau_track_targets=tau_track_targets,
            cells_per_cluster=cells,
            tau_targets=tau_targets,
            charged_pion_targets=charged_pion_targets,
            neutral_pion_targets=neutral_pion_targets,
            reco_id=reco_id,
            reco_decay_mode=reco_decay_mode,
            reco_tau_4mom=reco_tau_4mom,
            reco_charged_pions=reco_charged_pions,
            reco_neutral_pions=reco_neutral_pions,
            tau_vertex_targets=tau_vertex_targets,
            vertex_slot_mask=vertex_slot_mask,
        )

    def __getitems__(self, indices):
        """Batched loading: group by file, sort indices for sequential HDF5 access."""
        if len(indices) == 0:
            return []

        samples = [None] * len(indices)
        by_file = {}
        for out_pos, dataset_idx in enumerate(indices):
            file_idx, sample_idx = self.file_indices[dataset_idx]
            by_file.setdefault(file_idx, []).append((out_pos, sample_idx))

        for file_idx, items in by_file.items():
            f = self._get_file(file_idx)
            output_positions = np.array([p for p, _ in items], dtype=np.int64)
            sample_indices = np.array([s for _, s in items], dtype=np.int64)
            order = np.argsort(sample_indices)
            sorted_positions = output_positions[order]
            sorted_indices = sample_indices[order]

            batch_x = f["data"][sorted_indices]
            batch_y = f["pid"][sorted_indices]
            batch_cond = f["global"][sorted_indices] if "global" in f and self.use_cond else None
            batch_data_pid = f["data_pid"][sorted_indices] if self.mode in ["segmentation", "ftag"] else None
            batch_decay = f["decay_mode"][sorted_indices] if "decay_mode" in f else None
            batch_tracks = f["tracks"][sorted_indices] if self.use_tracks and "tracks" in f else None
            batch_tau_track_targets = f["tau_track_targets"][sorted_indices] if self.use_tracks and "tau_track_targets" in f else None
            batch_cells = f["cells_per_cluster"][sorted_indices] if self.use_cells and "cells_per_cluster" in f else None
            batch_tau_targets = f["tau_targets"][sorted_indices] if "tau_targets" in f else None
            batch_charged_pion = f["charged_pion_targets"][sorted_indices] if "charged_pion_targets" in f else None
            batch_neutral_pion = f["neutral_pion_targets"][sorted_indices] if "neutral_pion_targets" in f else None
            batch_tau_vertex = f["tau_vertex_targets"][sorted_indices] if self.do_vertex_classification and "tau_vertex_targets" in f else None
            batch_vertex_slot_mask = f["vertex_slot_mask"][sorted_indices] if self.do_vertex_classification and "vertex_slot_mask" in f else None
            batch_reco_id = f["reco_id"][sorted_indices] if "reco_id" in f else None
            batch_reco_decay_mode = f["reco_decay_mode"][sorted_indices] if "reco_decay_mode" in f else None
            batch_reco_tau_4mom = f["reco_tau_4mom"][sorted_indices] if "reco_tau_4mom" in f else None
            batch_reco_charged = f["reco_charged_pions"][sorted_indices] if "reco_charged_pions" in f else None
            batch_reco_neutral = f["reco_neutral_pions"][sorted_indices] if "reco_neutral_pions" in f else None

            for i, out_pos in enumerate(sorted_positions):
                samples[out_pos] = self._build_sample(
                    batch_x[i],
                    batch_y[i],
                    cond=batch_cond[i] if batch_cond is not None else None,
                    data_pid=batch_data_pid[i] if batch_data_pid is not None else None,
                    decay_mode=batch_decay[i] if batch_decay is not None else None,
                    tracks=batch_tracks[i] if batch_tracks is not None else None,
                    tau_track_targets=batch_tau_track_targets[i] if batch_tau_track_targets is not None else None,
                    cells_per_cluster=batch_cells[i] if batch_cells is not None else None,
                    tau_targets=batch_tau_targets[i] if batch_tau_targets is not None else None,
                    charged_pion_targets=batch_charged_pion[i] if batch_charged_pion is not None else None,
                    neutral_pion_targets=batch_neutral_pion[i] if batch_neutral_pion is not None else None,
                    reco_id=batch_reco_id[i] if batch_reco_id is not None else None,
                    reco_decay_mode=batch_reco_decay_mode[i] if batch_reco_decay_mode is not None else None,
                    reco_tau_4mom=batch_reco_tau_4mom[i] if batch_reco_tau_4mom is not None else None,
                    reco_charged_pions=batch_reco_charged[i] if batch_reco_charged is not None else None,
                    reco_neutral_pions=batch_reco_neutral[i] if batch_reco_neutral is not None else None,
                    tau_vertex_targets=batch_tau_vertex[i] if batch_tau_vertex is not None else None,
                    vertex_slot_mask=batch_vertex_slot_mask[i] if batch_vertex_slot_mask is not None else None,

                )

        return samples

    def __del__(self):
        for f in self._file_cache.values():
            if hasattr(f, "close"):
                try:
                    f.close()
                except Exception:
                    pass


def load_data(
    dataset_name,
    path,
    batch=100,
    dataset_type="train",
    distributed=True,
    use_cond=False,
    use_pid=False,
    pid_idx=4,
    use_add=False,
    num_add=4,
    num_workers=16,
    rank=0,
    size=1,
    clip_inputs=False,
    mode="",
    shuffle=True,
    nevts=-1,
    use_tracks=False,
    use_cells=False,
    do_regression_aux_tasks=False,
    do_vertex_classification=False,
):
    supported_datasets = [
        "top",
        "qg",
        "pretrain",
        "atlas",
        "aspen",
        "jetclass",
        "jetclass2",
        "h1",
        "toy",
        "cms_qcd",
        "cms_bsm",
        "cms_top",
        "aspen_bsm",
        "aspen_bsm_ad_sb",
        "aspen_bsm_ad_sr",
        "aspen_top_ad_sb",
        "aspen_top_ad_sr",
        "aspen_top_ad_sr_hl",
        "qcd_dijet",
        "jetnet150",
        "jetnet30",
        "dctr",
        "atlas_flav",
        "custom",
        "camels",
        "quijote",
        "microboone",
        "aspen_bsm_ad_sb",
        "aspen_bsm_ad_sr",
        "aspen_bsm_ad_sr_hl",
        "tau",
    ]
    if dataset_name not in supported_datasets:
        raise ValueError(
            f"Dataset '{dataset_name}' not supported. Choose from {supported_datasets}."
        )

    if dataset_name == "pretrain":
        names = ["atlas", "aspen", "jetclass", "jetclass2", "h1", "cms_qcd", "cms_bsm"]
        types = [dataset_type]
    else:
        names = [dataset_name]
        types = [dataset_type]

    dataset_paths = [os.path.join(path, name, type) for name in names for type in types]

    file_list = []
    file_indices = []
    index_shift = 0
    for iname, dataset_path in enumerate(dataset_paths):
        dataset_path = Path(dataset_path)
        dataset_path.mkdir(parents=True, exist_ok=True)

        if not any(dataset_path.iterdir()):
            print(f"Fetching download url for dataset {names[iname]}")
            url = get_url(names[iname], dataset_type)
            if url is None:
                raise ValueError(f"No download URL found for dataset '{dataset_name}'.")
            download_h5_files(url, dataset_path)

        h5_files = list(dataset_path.glob("*.h5")) + list(dataset_path.glob("*.hdf5"))
        file_list.extend(map(str, h5_files))  # Convert to string paths

        index_file = dataset_path / "file_index.npy"
        if index_file.is_file():
            if shuffle:
                indices = np.load(index_file, mmap_mode="r")[rank::size]
            else:
                indices = np.load(index_file, mmap_mode="r")[
                    len(np.load(index_file, mmap_mode="r")) * rank // size : len(
                        np.load(index_file, mmap_mode="r")
                    )
                    * (rank + 1)
                    // size
                ]
            file_indices.extend(
                (file_idx + index_shift, sample_idx) for file_idx, sample_idx in indices
            )
            index_shift += len(h5_files)

        else:
            print(f"Creating index list for dataset {names[iname]}")
            file_indices = []
            # Precompute indices for efficient access
            for file_idx, path in enumerate(h5_files):
                try:
                    with h5py.File(path, "r") as f:
                        num_samples = len(f["data"])
                        file_indices.extend([(file_idx, i) for i in range(num_samples)])
                except Exception as e:
                    print(f"ERROR: File {path} is likely corrupted: {e}")
            np.save(index_file, np.array(file_indices, dtype=np.int32))
            print(f"Number of events: {len(file_indices)}")

    label_shift = {
        "jetclass": 2,
        "jetclass2": 12,
        "aspen": 200,
        "cms_qcd": 201,
        "cms_bsm": 202,
    }

    data = HEPDataset(
        file_list,
        file_indices,
        use_cond=use_cond,
        use_pid=use_pid,
        pid_idx=pid_idx,
        use_add=use_add,
        num_add=num_add,
        label_shift=label_shift.get(dataset_name, 0),
        clip_inputs=clip_inputs,
        mode=mode,
        nevts=nevts,
        use_tracks=use_tracks,
        use_cells=use_cells,
        do_regression_aux_tasks=do_regression_aux_tasks,
        do_vertex_classification=do_vertex_classification,
    )

    loader_kwargs = {
        "dataset": data,
        "pin_memory": torch.cuda.is_available(),
        "num_workers": num_workers,
        "drop_last": False,
        "collate_fn": collate_point_cloud,
    }

    if shuffle:
        batch_sampler = FileGroupedBatchSampler(
            data.file_indices[: len(data)], batch, drop_last=False
        )
        loader_kwargs["batch_sampler"] = batch_sampler
        loader_kwargs["batch_size"] = 1  # required by DataLoader API but ignored
        loader_kwargs["shuffle"] = False
        loader_kwargs["sampler"] = None
    else:
        loader_kwargs["batch_size"] = batch
        loader_kwargs["shuffle"] = False
        loader_kwargs["sampler"] = None

    if num_workers > 0:
        loader_kwargs["persistent_workers"] = True
        loader_kwargs["prefetch_factor"] = 4

    loader = DataLoader(**loader_kwargs)
    return loader


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        "-d",
        "--dataset",
        default="top",
        help="Dataset name to download",
    )
    parser.add_argument(
        "-f",
        "--folder",
        default="./",
        help="Folder to save the dataset",
    )
    args = parser.parse_args()

    for tag in ["train", "test", "val"]:
        load_data(args.dataset, args.folder, dataset_type=tag, distributed=False)