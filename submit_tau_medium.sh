#!/bin/bash
#SBATCH -C gpu
#SBATCH -q regular
#SBATCH -n 32
#SBATCH --ntasks-per-node 4
#SBATCH --gpus-per-task 1
#SBATCH -t 4:00:00
#SBATCH -A m2616_g
#SBATCH --gpu-bind=none
#SBATCH -o logs/%x_%j.out
#SBATCH -e logs/%x_%j.err

# Performance / correctness env vars (matches the other training scripts)
export HDF5_USE_FILE_LOCKING=FALSE
export NCCL_NET_GDR_LEVEL=PHB

export MASTER_ADDR=$(hostname)

cmd="omnilearned train \
   --dataset tau --path /pscratch/sd/m/milescb/processed_h5/ \
   --num-feat 12 --num-classes 3 --use-tracks --track-dim 16 \
   --size medium --epoch 35 --batch 512 --output_dir checkpoints \
   --aux-tasks-str='decay_mode:5,tes:1,tau_eta:1,tau_phi:1,charged_pion_pt:1,charged_pion_eta:1,charged_pion_phi:1,neutral_pion_pt:1,neutral_pion_eta:1,neutral_pion_phi:1' \
   --aux-regression-tasks-str='tes,tau_eta,tau_phi,charged_pion_pt,charged_pion_eta,charged_pion_phi,neutral_pion_pt,neutral_pion_eta,neutral_pion_phi'"

set -x
srun -l -u \
    bash -c "
    source /pscratch/sd/m/milescb/OmniTau/OmniLearned/.venv/bin/activate
    source export_ddp.sh
    $cmd
    "
