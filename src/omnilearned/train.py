import json
import numpy as np
import torch
import torch.nn as nn
from omnilearned.network import PET2
from omnilearned.dataloader import load_data
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from pytorch_optimizer import Lion
from diffusers.optimization import get_cosine_schedule_with_warmup
from tqdm import tqdm

from omnilearned.utils import (
    is_master_node,
    ddp_setup,
    get_param_groups,
    CLIPLoss,
    get_checkpoint_name,
    shadow_copy,
    get_loss,
    save_checkpoint,
    restore_checkpoint,
    get_model_parameters,
)

import time
import os
import torch.amp as amp

torch.set_float32_matmul_precision("high")
torch._dynamo.config.verbose = False


def get_logs(device):
    logs_buff = torch.zeros((5), dtype=torch.float32, device=device)
    logs = {}
    logs["loss"] = logs_buff[0].view(-1)
    logs["loss_class"] = logs_buff[1].view(-1)
    logs["loss_gen"] = logs_buff[2].view(-1)
    logs["loss_clip"] = logs_buff[3].view(-1)
    logs["loss_class_event"] = logs_buff[4].view(-1)
    return logs

def map_tau_track_targets(raw_targets):
    """Map raw origin labels to 4 classes: TTT, CT, IT, FT; invalid -> -1.
    Raw labels are expected as:
    0,8=Undefined, 1=TTT, 2=CT, 3,4=IT, 5,6,7=FT.
    """
    targets = raw_targets.long()  # (B, T) or (B, T, 1)
    if targets.dim() == 3 and targets.shape[-1] == 1:
        targets = targets[..., 0]  # (B, T)

    mapped = torch.full_like(targets, -1)  # (B, T), default invalid
    mapped[targets == 1] = 0  # TTT
    mapped[targets == 2] = 1  # CT
    mapped[(targets == 3) | (targets == 4)] = 2  # IT
    mapped[(targets == 5) | (targets == 6) | (targets == 7)] = 3  # FT
    return mapped  # (B, T)

def train_step(
    model,
    dataloader,
    class_cost,
    gen_cost,
    optimizer,
    scheduler,
    mode,
    device,
    clip_loss=CLIPLoss(),
    use_clip=False,
    use_event_loss=False,
    iterations_per_epoch=-1,
    use_amp=False,
    gscaler=None,
    ema_model=None,
    ema_decay=0.9999,
    epoch_index: int | None = None,
):
    model.train()

    logs = get_logs(device)

    if iterations_per_epoch < 0:
        iterations_per_epoch = len(dataloader)

    base_model = model.module if hasattr(model, "module") else model

    dataloader_len = len(dataloader)
    if dataloader_len == 0:
        raise ValueError("Training dataloader is empty.")

    cached_batch = None
    data_iter = None
    if dataloader_len == 1 and iterations_per_epoch > 1:
        cached_batch = next(iter(dataloader))
    else:
        data_iter = iter(dataloader)

    if is_master_node():
        desc = "Train"
        if epoch_index is not None:
            desc = f"Train epoch {epoch_index + 1}"
        iterator = tqdm(range(iterations_per_epoch), desc=desc, leave=False)
    else:
        iterator = range(iterations_per_epoch)

    for batch_idx in iterator:
        if cached_batch is not None:
            batch = cached_batch
        else:
            try:
                batch = next(data_iter)
            except StopIteration:
                data_iter = iter(dataloader)
                batch = next(data_iter)

        # for batch_idx, batch in enumerate(dataloader):
        optimizer.zero_grad()  # Zero the gradients

        X, y = batch["X"].to(device, dtype=torch.float), batch["y"].to(device)
        model_kwargs = {
            key: (batch[key].to(device) if batch[key] is not None else None)
            for key in ["cond", "pid", "add_info", "tracks", "cells_per_cluster"]
            if key in batch
        }

        if batch.get("data_pid") is not None:
            data_pid = batch["data_pid"].to(device)
        else:
            data_pid = None

        # Auxiliary task labels (pid: 0=QCD, 1=tau, 2=electron)
        aux_labels = {}
        aux_masks = {}
        aux_slot_masks = None # For vertex classification task to mask zero-padded vertex slots
        
        if batch.get("decay_mode") is not None:
            aux_labels["decay_mode"] = batch["decay_mode"].to(device)
            # only taus and those with valid decay modes
            aux_masks["decay_mode"] = ((y == 1)  
                & (aux_labels["decay_mode"] != 5) & (aux_labels["decay_mode"] != 6))
        
        # Electron vs QCD: derived from pid
        aux_tasks = (model.module.aux_tasks if hasattr(model, 'module') else model.aux_tasks) or []
        if "electron_vs_qcd" in [t["name"] for t in aux_tasks]:
            aux_masks["electron_vs_qcd"] = (y == 0) | (y == 2)  # QCD and electron only
            aux_labels["electron_vs_qcd"] = (y == 2).long()     # electron=1, QCD=0
        
        # Truth-tau regression tasks — use log(pt) (index 0) to normalize MeV scale
        if batch.get("tau_targets") is not None:
            tau_mask = (y == 1)
            tau_targets = batch["tau_targets"].to(device)  # Shape: (batch, 3)
            aux_labels["tes"] = torch.log(tau_targets[:, 0])  # log(pt in MeV) ≈ [9.9, 13.8]
            aux_masks["tes"] = tau_mask
            aux_labels["tau_eta"] = tau_targets[:, 1]
            aux_masks["tau_eta"] = tau_mask
            aux_labels["tau_phi"] = tau_targets[:, 2]
            aux_masks["tau_phi"] = tau_mask

        # Tau Track Classification Labels (1 = TTT, 2 = CT, 3,4 = IT, 5,6,7 = FT, 0,8 = Undefined) # Already remapped
        if batch.get("tau_track_targets") is not None:
            raw_track_targets = batch["tau_track_targets"].to(device)
            aux_labels["tautrack_class"] = map_tau_track_targets(raw_track_targets)
            aux_masks["tautrack_class"] = ( (y[:, None] == 1) & (aux_labels["tautrack_class"] >= 0) )

        if batch.get("charged_pion_targets") is not None:
            cpt = batch["charged_pion_targets"].to(device)  # (batch, 3)
            aux_labels["charged_pion_pt"]  = torch.log(cpt[:, 0])
            aux_labels["charged_pion_eta"] = cpt[:, 1]
            aux_labels["charged_pion_phi"] = cpt[:, 2]
            
            charged_pion_mask = ((y == 1) & (cpt[:, 0] != -999))
            aux_masks["charged_pion_pt"]   = charged_pion_mask
            aux_masks["charged_pion_eta"]  = charged_pion_mask
            aux_masks["charged_pion_phi"]  = charged_pion_mask

        if batch.get("neutral_pion_targets") is not None:
            npt = batch["neutral_pion_targets"].to(device)  # (batch, 3)
            aux_labels["neutral_pion_pt"]  = torch.log(npt[:, 0])
            aux_labels["neutral_pion_eta"] = npt[:, 1]
            aux_labels["neutral_pion_phi"] = npt[:, 2]
            
            neutral_pion_mask = ((y == 1) & (npt[:, 0] != -999))
            aux_masks["neutral_pion_pt"]   = neutral_pion_mask
            aux_masks["neutral_pion_eta"]  = neutral_pion_mask
            aux_masks["neutral_pion_phi"]  = neutral_pion_mask

        if batch.get("tau_vertex_targets") is not None:
            vtx = batch["tau_vertex_targets"].to(device)  # [B, MAX_VERTICES, 1]
            # With deterministic vertex ordering and exactly one true tau vertex,
            # train a softmax head over the vertex slots.
            aux_labels["vertex_classification"] = (vtx[:, :, 0] == 1).float().argmax(dim=1)  # [B]
            aux_masks["vertex_classification"] = (y == 1)  # tau events only
            aux_slot_masks = batch["vertex_slot_mask"].to(device) if batch.get("vertex_slot_mask") is not None else None

        with amp.autocast(
            "cuda:{}".format(device) if torch.cuda.is_available() else "cpu",
            enabled=use_amp,
        ):
            outputs = model(X, y, **model_kwargs)
            loss = get_loss(
                outputs,
                y,
                mode,
                class_cost,
                gen_cost,
                use_event_loss,
                use_clip,
                clip_loss,
                logs,
                data_pid=data_pid,
                aux_labels=aux_labels if aux_labels else None,
                aux_masks=aux_masks if aux_masks else None,
                aux_tasks=aux_tasks if aux_tasks else None,
                aux_slot_masks=aux_slot_masks if aux_slot_masks is not None else None,
            )

        if use_amp and gscaler is not None:
            gscaler.scale(loss).backward()
            gscaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            gscaler.step(optimizer)
            gscaler.update()
        else:
            loss.backward()  # Backward pass
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()  # Update parameters

        scheduler.step()

        if ema_model is not None:
            with torch.no_grad():
                for ema_p, model_p in zip(
                    ema_model.parameters(), base_model.parameters()
                ):
                    ema_p.mul_(ema_decay).add_(model_p, alpha=1.0 - ema_decay)

    if dist.is_initialized():
        for key in logs:
            dist.all_reduce(logs[key].detach())
            logs[key] = float(logs[key] / dist.get_world_size() / iterations_per_epoch)

    return logs


def val_step(
    model,
    dataloader,
    class_cost,
    gen_cost,
    mode,
    device,
    clip_loss=CLIPLoss(),
    use_clip=False,
    use_event_loss=False,
    iterations_per_epoch=-1,
    epoch_index: int | None = None,
):
    model.eval()

    logs = get_logs(device)

    if iterations_per_epoch < 0:
        iterations_per_epoch = len(dataloader)

    dataloader_len = len(dataloader)
    if dataloader_len == 0:
        raise ValueError("Validation dataloader is empty.")

    cached_batch = None
    data_iter = None
    if dataloader_len == 1 and iterations_per_epoch > 1:
        cached_batch = next(iter(dataloader))
    else:
        data_iter = iter(dataloader)

    if is_master_node():
        desc = "Val"
        if epoch_index is not None:
            desc = f"Val epoch {epoch_index + 1}"
        iterator = tqdm(range(iterations_per_epoch), desc=desc, leave=False)
    else:
        iterator = range(iterations_per_epoch)

    for batch_idx in iterator:
        if cached_batch is not None:
            batch = cached_batch
        else:
            try:
                batch = next(data_iter)
            except StopIteration:
                data_iter = iter(dataloader)
                batch = next(data_iter)

        # for batch_idx, batch in enumerate(dataloader):
        X, y = batch["X"].to(device, dtype=torch.float), batch["y"].to(device)
        model_kwargs = {
            key: (batch[key].to(device) if batch[key] is not None else None)
            for key in ["cond", "pid", "add_info", "tracks", "cells_per_cluster"]
            if key in batch
        }

        if batch.get("data_pid") is not None:
            data_pid = batch["data_pid"].to(device)
        else:
            data_pid = None

        # Auxiliary task labels (pid: 0=QCD, 1=tau, 2=electron)
        aux_labels = {}
        aux_masks = {}
        aux_slot_masks = None # For vertex classification task to mask zero-padded vertex slots

        if batch.get("decay_mode") is not None:
            aux_labels["decay_mode"] = batch["decay_mode"].to(device)
            #aux_masks["decay_mode"] = (y == 1)  # Only taus
            #Isn't this better instead @Ali, @Miles?
            aux_masks["decay_mode"] = ((y == 1) # only taus and those with valid decay modes
                & (aux_labels["decay_mode"] != 5) & (aux_labels["decay_mode"] != 6))

        # Electron vs QCD: derived from pid
        aux_tasks = (model.module.aux_tasks if hasattr(model, 'module') else model.aux_tasks) or []
        if "electron_vs_qcd" in [t["name"] for t in aux_tasks]:
            aux_masks["electron_vs_qcd"] = (y == 0) | (y == 2)  # QCD and electron only
            aux_labels["electron_vs_qcd"] = (y == 2).long()     # electron=1, QCD=0

        # Tau Track Classification Labels (1 = TTT, 2 = CT, 3,4 = IT, 5,6,7 = FT, 0,8 = Undefined) # Simplify this for next prepare_data_v2.py iteration (for now just consider 1,2,3,4 as valid and mask the rest)
        if batch.get("tau_track_targets") is not None:
            raw_track_targets = batch["tau_track_targets"].to(device)
            aux_labels["tautrack_class"] = map_tau_track_targets(raw_track_targets)
            aux_masks["tautrack_class"] = ( (y[:, None] == 1) & (aux_labels["tautrack_class"] >= 0) )

        # Truth-tau regression tasks — use log(pt) (index 0) to normalize MeV scale
        if batch.get("tau_targets") is not None:
            tau_targets = batch["tau_targets"].to(device)  # Shape: (batch, 3)
            aux_labels["tes"] = torch.log(tau_targets[:, 0])  # log(pt in MeV) ≈ [9.9, 13.8]
            aux_masks["tes"] = (y == 1)  # Only tau samples
            aux_labels["tau_eta"] = tau_targets[:, 1]
            aux_masks["tau_eta"] = (y == 1)
            aux_labels["tau_phi"] = tau_targets[:, 2]
            aux_masks["tau_phi"] = (y == 1)

        if batch.get("tau_vertex_targets") is not None:
            vtx = batch["tau_vertex_targets"].to(device)  # [B, MAX_VERTICES, 1]
            aux_labels["vertex_classification"] = (vtx[:, :, 0] == 1).float().argmax(dim=1)  # [B]
            aux_masks["vertex_classification"] = (y == 1)
            aux_slot_masks = batch["vertex_slot_mask"].to(device) if batch.get("vertex_slot_mask") is not None else None

        with torch.no_grad():
            outputs = model(X, y, **model_kwargs)
            get_loss(
                outputs,
                y,
                mode,
                class_cost,
                gen_cost,
                use_event_loss,
                use_clip,
                clip_loss,
                logs,
                data_pid=data_pid,
                aux_labels=aux_labels if aux_labels else None,
                aux_masks=aux_masks if aux_masks else None,
                aux_tasks=aux_tasks if aux_tasks else None,
                aux_slot_masks=aux_slot_masks if aux_slot_masks is not None else None,
            )

    if dist.is_initialized():
        for key in logs:
            dist.all_reduce(logs[key].detach())
            logs[key] = float(logs[key] / dist.get_world_size() / iterations_per_epoch)

    return logs


def train_model(
    model,
    train_loader,
    val_loader,
    optimizer,
    lr_scheduler,
    mode,
    num_epochs=1,
    device="cpu",
    patience=500,
    loss_class=nn.CrossEntropyLoss(),
    loss_gen=nn.MSELoss(),
    use_clip=False,
    use_event_loss=False,
    output_dir="",
    save_tag="",
    iterations_per_epoch=-1,
    epoch_init=0,
    loss_init=np.inf,
    use_amp=False,
    run=None,
    ema_model=None,
    ema_decay=0.999,
):
    checkpoint_name = get_checkpoint_name(save_tag)

    losses = {
        "train_loss": [],
        "val_loss": [],
    }

    tracker = {"bestValLoss": loss_init, "bestEpoch": epoch_init}
    if use_amp:
        gscaler = amp.GradScaler()
    else:
        gscaler = None
    for epoch in range(int(epoch_init), num_epochs):
        if hasattr(train_loader.batch_sampler, "set_epoch"):
            train_loader.batch_sampler.set_epoch(epoch)
        elif isinstance(
            train_loader.sampler, torch.utils.data.distributed.DistributedSampler
        ):
            train_loader.sampler.set_epoch(epoch)

        start = time.time()
        train_logs = train_step(
            model,
            train_loader,
            loss_class,
            loss_gen,
            optimizer,
            lr_scheduler,
            mode,
            device,
            use_clip=use_clip,
            use_event_loss=use_event_loss,
            iterations_per_epoch=iterations_per_epoch,
            use_amp=use_amp,
            gscaler=gscaler,
            ema_model=ema_model,
            ema_decay=ema_decay,
            epoch_index=epoch,
        )
        val_logs = val_step(
            model,
            val_loader,
            loss_class,
            loss_gen,
            mode,
            device,
            use_clip=use_clip,
            use_event_loss=use_event_loss,
            iterations_per_epoch=iterations_per_epoch,
            epoch_index=epoch,
        )

        losses["train_loss"].append(train_logs["loss"])
        losses["val_loss"].append(val_logs["loss"])

        if is_master_node():
            print(
                f"Epoch [{epoch + 1}/{num_epochs}] Loss: {losses['train_loss'][-1]:.4f}, Val Loss: {losses['val_loss'][-1]:.4f} , lr: {lr_scheduler.get_last_lr()[0]}"
            )
            print(
                f"Class Loss: {train_logs['loss_class']:.4f}, Class Val Loss: {val_logs['loss_class']:.4f}"
            )
            if use_event_loss:
                print(
                    f"Class Event Loss: {train_logs['loss_class_event']:.4f}, Class Event Val Loss: {val_logs['loss_class_event']:.4f}"
                )
            print(
                f"Gen Loss: {train_logs['loss_gen']:.4f}, Gen Val Loss: {val_logs['loss_gen']:.4f}"
            )
            if use_clip:
                print(
                    f"CLIP loss: {train_logs['loss_clip']:.4f}, CLIP Val Loss: {val_logs['loss_clip']:.4f}"
                )
            print(
                "Time taken for epoch {} is {} sec".format(epoch, time.time() - start)
            )

        if losses["val_loss"][-1] < tracker["bestValLoss"]:
            tracker["bestValLoss"] = losses["val_loss"][-1]
            tracker["bestEpoch"] = epoch

            if is_master_node():
                print("replacing best checkpoint ...")
                save_checkpoint(
                    model,
                    ema_model,
                    epoch + 1,
                    optimizer,
                    losses["val_loss"][-1],
                    lr_scheduler,
                    output_dir,
                    checkpoint_name,
                )

        if run is not None:
            for key in train_logs:
                run.log({f"train {key}": train_logs[key]})
            for key in val_logs:
                run.log({f"val {key}": val_logs[key]})

        if epoch - tracker["bestEpoch"] > patience:
            print(f"breaking on device: {device}")
            break

    if is_master_node():
        print(
            f"Training Complete, best loss: {tracker['bestValLoss']:.5f} at epoch {tracker['bestEpoch']}!"
        )
        # save losses
        json.dump(losses, open(f"{output_dir}/training_{save_tag}.json", "w"))


def run(
    outdir: str = "",
    save_tag: str = "",
    pretrain_tag: str = "pretrain",
    dataset: str = "top",
    path: str = "/pscratch/sd/a/agarabag/OmniLearned/datasets",
    wandb=False,
    fine_tune: bool = False,
    resuming: bool = False,
    num_feat: int = 4,
    model_size: str = "small",
    interaction: bool = False,
    local_interaction: bool = False,
    num_coord: int = 2,
    K: int = 10,
    interaction_type: str = "lhc",
    conditional: bool = False,
    num_cond: bool = 3,
    use_pid: bool = False,
    pid_idx: int = -1,
    pid_dim: int = 9,
    use_add: bool = False,
    num_add: int = 4,
    zero_add: bool = False,
    use_clip: bool = False,
    use_event_loss: bool = False,
    num_classes: int = 2,
    num_gen_classes: int = 1,
    mode: str = "classifier",
    batch: int = 64,
    iterations: int = -1,
    epoch: int = 15,
    warmup_epoch: int = 1,
    use_amp: bool = False,
    optim: str = "lion",
    sched: str = "cosine",
    b1: float = 0.95,
    b2: float = 0.98,
    lr: float = 5e-4,
    lr_factor: float = 10.0,
    wd: float = 0.3,
    nevts: int = -1,
    attn_drop: float = 0.1,
    mlp_drop: float = 0.1,
    feature_drop: float = 0.0,
    num_workers: int = 16,
    clip_inputs: bool = False,
    # Auxiliary tasks: comma-separated "name:num_classes" pairs
    # e.g., "decay_mode:7,electron_vs_qcd:2"
    aux_tasks_str: str = "",
    # Regression auxiliary tasks: comma-separated names
    # e.g., "pt_regression,eta_regression"
    aux_regression_tasks_str: str = "",
    # Option B: Tracks as separate tokens
    use_tracks: bool = False,
    track_dim: int = 24,
    # Learned per-cluster cell aggregation
    use_cells: bool = False,
    cell_dim: int = 14,
    do_vertex_classification: bool = False,
):
    local_rank, rank, size = ddp_setup()

    # Derive whether to load regression aux targets from dataloader
    do_regression_aux_tasks = bool(aux_regression_tasks_str)

    # Parse regression task names
    regression_task_names = set()
    if aux_regression_tasks_str:
        regression_task_names = set(name.strip() for name in aux_regression_tasks_str.split(","))

    # Parse auxiliary tasks
    aux_tasks = None
    if aux_tasks_str:
        aux_tasks = []
        for task_def in aux_tasks_str.split(","):
            name, num_cls = task_def.strip().split(":")
            task_info = {"name": name}
            # Mark as regression or classification
            if name in regression_task_names:
                task_info["type"] = "regression"
            else:
                task_info["type"] = "classification"
                task_info["num_classes"] = int(num_cls)
            aux_tasks.append(task_info)

    # Include vertex classification task when requested
    _MAX_VERTICES_PER_TAU = 20
    if do_vertex_classification:
        vertex_task = {
            "name": "vertex_classification",
            "type": "classification",
            "num_classes": _MAX_VERTICES_PER_TAU,
        }
        if aux_tasks is None:
            aux_tasks = [vertex_task]
        elif not any(t["name"] == "vertex_classification" for t in aux_tasks):
            aux_tasks.append(vertex_task)

    model_params = get_model_parameters(model_size)
    # set up model
    model = PET2(
        input_dim=num_feat,
        use_int=interaction,
        local_int=local_interaction,
        int_type=interaction_type,
        conditional=conditional,
        cond_dim=num_cond,
        pid=use_pid,
        pid_dim=pid_dim,
        add_info=use_add,
        add_dim=num_add,
        mode=mode,
        num_classes=num_classes,
        num_gen_classes=num_gen_classes,
        mlp_drop=mlp_drop,
        attn_drop=attn_drop,
        feature_drop=feature_drop,
        num_coord=num_coord,
        K=K,
        aux_tasks=aux_tasks,
        use_tracks=use_tracks,
        track_dim=track_dim,
        use_cells=use_cells,
        cell_dim=cell_dim,
        **model_params,
    )

    if rank == 0:
        d = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        print("**** Setup ****")
        print(
            "Total params: %.2fM"
            % (sum(p.numel() for p in model.parameters()) / 1000000.0)
        )
        print(f"Training on device: {d}, with {size} GPUs")
        print("************")

    # load in train data
    train_loader = load_data(
        dataset,
        dataset_type="train",
        use_cond=conditional,
        use_pid=use_pid,
        pid_idx=pid_idx,
        use_add=use_add,
        num_add=num_add,
        path=path,
        batch=batch,
        num_workers=num_workers,
        rank=rank,
        size=size,
        clip_inputs=clip_inputs,
        mode=mode,
        nevts=nevts,
        use_tracks=use_tracks,
        use_cells=use_cells,
        do_regression_aux_tasks=do_regression_aux_tasks,
        do_vertex_classification=do_vertex_classification,
    )
    if rank == 0:
        print("**** Setup ****")
        print(f"Train dataset len: {len(train_loader)}")
        print("************")

    val_loader = load_data(
        dataset,
        dataset_type="val",
        use_cond=conditional,
        use_pid=use_pid,
        pid_idx=pid_idx,
        use_add=use_add,
        num_add=num_add,
        path=path,
        batch=batch,
        num_workers=num_workers,
        rank=rank,
        size=size,
        clip_inputs=clip_inputs,
        mode=mode,
        use_tracks=use_tracks,
        use_cells=use_cells,
        do_regression_aux_tasks=do_regression_aux_tasks,
        do_vertex_classification=do_vertex_classification,
    )

    param_groups = get_param_groups(
        model, wd, lr, lr_factor=lr_factor, fine_tune=fine_tune
    )

    if optim not in ["adam", "lion"]:
        raise ValueError(
            f"Optimizer '{optim}' not supported. Choose from adam or lion."
        )
    if sched not in ["cosine", "onecycle"]:
        raise ValueError(
            f"Scheduler '{sched}' not supported. Choose from cosine or onecycle."
        )

    if optim == "lion":
        optimizer = Lion(param_groups, betas=(b1, b2))
    elif optim == "adam":
        optimizer = torch.optim.AdamW(param_groups)

    train_steps = len(train_loader) if iterations < 0 else iterations

    if sched == "onecycle":
        lr_scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer=optimizer,
            total_steps=(train_steps * epoch),
            max_lr=lr,
            pct_start=0.1,
        )
    elif sched == "cosine":
        lr_scheduler = get_cosine_schedule_with_warmup(
            optimizer=optimizer,
            num_warmup_steps=train_steps * warmup_epoch,
            num_training_steps=(train_steps * epoch),
            last_epoch=-1,
        )

    # Transfer model to GPU if available
    kwarg = {}
    if torch.cuda.is_available():
        device = local_rank
        model.to(local_rank)
        kwarg["device_ids"] = [device]
    else:
        model.cpu()
        device = "cpu"

    if size > 1:
        model = DDP(
            model,
            find_unused_parameters=True,
            **kwarg,
        )

    # Set up EMA model
    base_model = model.module if hasattr(model, "module") else model
    ema_model = shadow_copy(base_model)

    epoch_init = 0
    loss_init = np.inf
    checkpoint_name = None

    if os.path.isfile(os.path.join(outdir, get_checkpoint_name(save_tag))) and resuming:
        if is_master_node():
            print(
                f"Continue training with checkpoint from {os.path.join(outdir, get_checkpoint_name(save_tag))}"
            )
        checkpoint_name = get_checkpoint_name(save_tag)
        fine_tune = False

    elif fine_tune:
        if is_master_node():
            print(
                f"Will fine-tune using checkpoint {os.path.join(outdir, get_checkpoint_name(pretrain_tag))}"
            )
        checkpoint_name = get_checkpoint_name(pretrain_tag)

    if checkpoint_name is not None:
        epoch_init, loss_init = restore_checkpoint(
            model,
            outdir,
            checkpoint_name,
            local_rank,
            is_main_node=is_master_node(),
            ema_model=ema_model,
            optimizer=optimizer,
            lr_scheduler=lr_scheduler,
            fine_tune=fine_tune,
        )

    if wandb:
        import wandb

        if is_master_node():
            mode_wandb = None
            wandb.login()
        else:
            mode_wandb = "disabled"

        run = wandb.init(
            # Set the project where this run will be logged
            project="OmniBoone",
            name=save_tag,
            mode=mode_wandb,
            # Track hyperparameters and run metadata
            config={
                "learning_rate": lr,
                "epochs": epoch,
                "batch size": batch,
                "mode": mode,
            },
        )
    else:
        run = None

    if mode == "regression":
        loss_class = nn.MSELoss(reduction="none")
    else:
        loss_class = nn.CrossEntropyLoss(reduction="none")

    if mode == "ftag":
        loss_gen = nn.CrossEntropyLoss(reduction="none")
    else:
        loss_gen = nn.MSELoss(reduction="none")

    train_model(
        model,
        train_loader,
        val_loader,
        optimizer,
        lr_scheduler,
        mode=mode,
        num_epochs=epoch,
        device=device,
        loss_class=loss_class,
        loss_gen=loss_gen,
        output_dir=outdir,
        save_tag=save_tag,
        use_clip=use_clip,
        use_event_loss=use_event_loss,
        iterations_per_epoch=iterations,
        epoch_init=epoch_init,
        loss_init=loss_init,
        use_amp=use_amp,
        run=run,
        ema_model=ema_model,
    )

    dist.destroy_process_group()
