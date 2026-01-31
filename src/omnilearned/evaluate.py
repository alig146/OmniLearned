import torch
from omnilearned.network import PET2
from omnilearned.dataloader import load_data
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from omnilearned.utils import (
    is_master_node,
    ddp_setup,
    get_checkpoint_name,
    restore_checkpoint,
    pad_array,
    get_model_parameters,
)
from omnilearned.diffusion import generate
import os
import numpy as np
import h5py
from tqdm.auto import tqdm

# seed_value = 30  # You can choose any integer as your seed
# torch.manual_seed(seed_value)


def eval_model(
    model,
    test_loader,
    dataset,
    mode,
    use_event_loss,
    device="cpu",
    outdir="",
    save_tag="pretrain",
    rank=0,
    aux_regression_tasks=None,
):
    prediction, cond, labels, aux_preds, decay_modes, regression_truth = test_step(model, test_loader, mode, device)

    if mode in ["classifier", "regression", "segmentation"]:
        if use_event_loss:
            np.savez(
                os.path.join(outdir, f"outputs_{save_tag}_{dataset}_{rank}.npz"),
                prediction=prediction[:, :200].softmax(-1).cpu().numpy(),
                event_prediction=prediction[:, 200:].softmax(-1).cpu().numpy(),
                pid=labels.cpu().numpy(),
                cond=cond.cpu().numpy() if cond is not None else [],
            )
        else:
            if mode == "classifier":
                prediction = prediction.softmax(-1).cpu().numpy()
            else:
                prediction = prediction.cpu().numpy()

            # Build save dictionary
            save_dict = {
                "prediction": prediction,
                "pid": labels.cpu().numpy(),
                "cond": cond.cpu().numpy() if cond is not None else [],
            }
            
            # Add auxiliary task predictions
            for task_name, task_pred in aux_preds.items():
                # Apply softmax only for classification tasks
                if aux_regression_tasks and task_name in aux_regression_tasks:
                    save_dict[f"aux_{task_name}_pred"] = task_pred.cpu().numpy()
                    if regression_truth is not None and task_name in regression_truth:
                        save_dict[f"aux_{task_name}_true"] = regression_truth[task_name].cpu().numpy()
                else:
                    save_dict[f"aux_{task_name}_pred"] = task_pred.softmax(-1).cpu().numpy()
            
            # Add decay_mode true labels if available
            if decay_modes is not None:
                save_dict["decay_mode"] = decay_modes.cpu().numpy()

            np.savez(
                os.path.join(outdir, f"outputs_{save_tag}_{dataset}_{rank}.npz"),
                **save_dict,
            )
    else:
        with h5py.File(
            os.path.join(outdir, f"generated_{save_tag}_{dataset}_{rank}.h5"),
            "w",
        ) as fh5:
            fh5.create_dataset("data", data=prediction.cpu().numpy())
            fh5.create_dataset("global", data=cond.cpu().numpy())
            fh5.create_dataset("pid", data=labels.cpu().numpy() + 1)


def test_step(
    model,
    dataloader,
    mode,
    device,
):
    model.eval()

    preds = []
    labels = []
    conds = []
    aux_preds_all = {}  # Collect auxiliary predictions
    decay_modes = []    # Collect true decay mode labels
    regression_truth_all = {}  # Collect regression truth labels

    for ib, batch in enumerate(
        tqdm(dataloader, desc="Iterating", total=len(dataloader))
        if is_master_node()
        else dataloader
    ):
        X, y = batch["X"].to(device, dtype=torch.float), batch["y"].to(device)
        npart = X.shape[1]
        model_kwargs = {
            key: (batch[key].to(device) if batch[key] is not None else None)
            for key in ["cond", "pid", "add_info", "tracks"]
            if key in batch
        }

        with torch.no_grad():
            if mode in ["classifier", "regression", "segmentation"]:
                outputs = model(X, y, **model_kwargs)
                output_name = (
                    "y_pred" if mode in ["classifier", "regression"] else "z_pred"
                )
                preds.append(outputs[output_name])
                
                # Collect auxiliary predictions
                if outputs.get("aux_preds") is not None:
                    for task_name, task_pred in outputs["aux_preds"].items():
                        if task_name not in aux_preds_all:
                            aux_preds_all[task_name] = []
                        aux_preds_all[task_name].append(task_pred)

            elif mode == "generator":
                assert "cond" in model_kwargs, (
                    "ERROR, conditioning variables not passed to model"
                )
                preds.append(generate(model, y, X.shape, **model_kwargs))
        if mode == "segmentation":
            labels.append(batch["data_pid"].to(device))
        else:
            labels.append(y)

        conds.append(batch["cond"])
        
        # Collect true decay mode labels if available
        if batch.get("decay_mode") is not None:
            decay_modes.append(batch["decay_mode"].to(device))
        
        # Collect regression truth labels if available
        if batch.get("truth_targets") is not None:
            truth_targets = batch["truth_targets"].to(device)  # Shape: (batch, NUM_TARGETS)
            tasks = ["tes"]  # Must match the task names in training
            for idx, task in enumerate(tasks):
                if task not in regression_truth_all:
                    regression_truth_all[task] = []
                regression_truth_all[task].append(truth_targets[:, idx])
        
        if mode == "generator":
            if batch["pid"] is not None:
                preds[-1] = torch.cat(
                    [preds[-1], model_kwargs["pid"].unsqueeze(-1).float()], -1
                )
            if batch["add_info"] is not None:
                preds[-1] = torch.cat([preds[-1], model_kwargs["add_info"]], -1)

    if mode == "generator":
        preds = pad_array(preds, npart)
    else:
        preds = torch.cat(preds).to(device)
    
    # Concatenate auxiliary predictions
    aux_preds_concat = {}
    for task_name, task_preds in aux_preds_all.items():
        aux_preds_concat[task_name] = torch.cat(task_preds).to(device)
    
    # Concatenate decay modes if collected
    decay_modes_concat = torch.cat(decay_modes).to(device) if decay_modes else None
    
    # Concatenate regression truth labels
    regression_truth_concat = {}
    for task_name, task_truths in regression_truth_all.items():
        regression_truth_concat[task_name] = torch.cat(task_truths).to(device)
    
    if is_master_node() and not regression_truth_all:
        print("WARNING: No regression truth labels were collected from batches.")
        print("Make sure 'truth_targets' is included in the dataloader output.")
    
    return (
        preds,
        torch.cat(conds).to(device) if conds[0] is not None else None,
        torch.cat(labels).to(device),
        aux_preds_concat,
        decay_modes_concat,
        regression_truth_concat,
    )


def run(
    indir: str = "",
    outdir: str = "",
    save_tag: str = "",
    dataset: str = "top",
    path: str = "/pscratch/sd/v/vmikuni/datasets",
    num_feat: int = 4,
    model_size: str = "small",
    interaction: bool = False,
    local_interaction: bool = False,
    num_coord: int = 2,
    K: int = 10,
    interaction_type: str = "lhc",
    conditional: bool = False,
    num_cond: int = 3,
    use_pid: bool = False,
    pid_idx: int = -1,
    use_add: bool = False,
    num_add: int = 4,
    use_event_loss: bool = False,
    num_classes: int = 2,
    num_gen_classes: int = 1,
    mode: str = "classifier",
    batch: int = 64,
    num_workers: int = 16,
    clip_inputs: bool = False,
    aux_tasks_str: str = "",
    aux_regression_tasks_str: str = "",
    use_tracks: bool = False,
    track_dim: int = 24,
):
    local_rank, rank, size = ddp_setup()

    model_params = get_model_parameters(model_size)

    # Parse regression task names
    regression_task_names = set()
    if aux_regression_tasks_str:
        regression_task_names = set(name.strip() for name in aux_regression_tasks_str.split(","))

    # Parse auxiliary tasks
    aux_tasks = None
    if aux_tasks_str:
        aux_tasks = []
        for task_def in aux_tasks_str.split(","):
            name, num_classes_aux = task_def.split(":")
            name = name.strip()
            task_info = {"name": name}
            # Mark as regression or classification
            if name in regression_task_names:
                task_info["type"] = "regression"
            else:
                task_info["type"] = "classification"
                task_info["num_classes"] = int(num_classes_aux)
            aux_tasks.append(task_info)

    # set up model
    model = PET2(
        input_dim=num_feat,
        use_int=interaction,
        local_int=local_interaction,
        int_type=interaction_type,
        conditional=conditional,
        cond_dim=num_cond,
        pid=use_pid,
        add_info=use_add,
        add_dim=num_add,
        mode=mode,
        num_classes=num_classes,
        num_gen_classes=num_gen_classes,
        num_coord=num_coord,
        K=K,
        aux_tasks=aux_tasks,
        use_tracks=use_tracks,
        track_dim=track_dim,
        **model_params,
    )

    if rank == 0:
        d = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        print("**** Setup ****")
        print(
            "Total params: %.2fM"
            % (sum(p.numel() for p in model.parameters()) / 1000000.0)
        )
        print(f"Evaluating on device: {d}, with {size} GPUs")
        print("************")

    # load in test data
    test_loader = load_data(
        dataset,
        dataset_type="test",
        use_cond=True,
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
        shuffle=False,
    )
    if rank == 0:
        print("**** Setup ****")
        print(f"Train dataset len: {len(test_loader)}")
        print("************")

    if os.path.isfile(os.path.join(indir, get_checkpoint_name(save_tag))):
        if is_master_node():
            print(
                f"Loading checkpoint from {os.path.join(indir, get_checkpoint_name(save_tag))}"
            )

        restore_checkpoint(
            model,
            indir,
            get_checkpoint_name(save_tag),
            local_rank,
            is_main_node=is_master_node(),
            restore_ema_model=mode == "generator",
        )

    else:
        raise ValueError(
            f"Error loading checkpoint: {os.path.join(indir, get_checkpoint_name(save_tag))}"
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

    model = DDP(
        model,
        **kwarg,
    )

    eval_model(
        model,
        test_loader,
        dataset,
        mode=mode,
        use_event_loss=use_event_loss,
        device=device,
        rank=rank,
        outdir=outdir,
        save_tag=save_tag,
        aux_regression_tasks=regression_task_names,
    )
    dist.barrier()
    dist.destroy_process_group()
