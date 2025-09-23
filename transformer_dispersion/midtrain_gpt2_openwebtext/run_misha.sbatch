#!/bin/bash

#SBATCH --job-name=hf_midtrain-1gpu
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --constraint='h100|a100'
#SBATCH --cpus-per-task=17
#SBATCH --mem=32G
#SBATCH --time=1-00:00:00
#SBATCH --output=slurm_out/hf_midtrain-1gpu-%j.out
#SBATCH --error=slurm_out/hf_midtrain-1gpu-%j.err

echo "Setting up environment variables and paths..."
# --- User-Specific Paths ---
# Please set your scratch directory before running
if [ -z "${CONTAINER_PATH}" ]; then
  echo "ERROR: Please set the CONTAINER_PATH environment variable."
  exit 1
fi

if [ -z "${OVERLAY_PATH}" ]; then
  echo "ERROR: Please set the OVERLAY_PATH environment variable."
  exit 1
fi

if [ -z "${TRITON_CACHE_DIR}" ]; then
  echo "ERROR: Please set the TRITON_CACHE_DIR environment variable."
  exit 1
fi

if [ -z "${HOST_CA_CERT_PATH}" ]; then
  echo "ERROR: Please set the HOST_CA_CERT_PATH environment variable."
  exit 1
fi

if [ -z "${CONTAINER_CA_CERT_PATH}" ]; then
  echo "ERROR: Please set the CONTAINER_CA_CERT_PATH environment variable."
  exit 1
fi

export NCCL_DEBUG=INFO
export PYTHONFAULTHANDLER=1

echo "Starting Apptainer container and executing script..."

COMMAND="accelerate launch --mixed_precision bf16 midtrain.py --model_name gpt2 --dataset_name Skylion007/openwebtext --train_tokens 1_600_000_000 --lr 1e-5 --block_size 1024 --per_device_train_batch_size 4 --gradient_accumulation_steps 32 --dispersion_coeff 1.0 --tau_infonce_l2 0.5 --tau_infonce_cos 0.5 --use_wandb --wandb_project gpt2-midtrain-dispersion-openwebtext-1e-5 --wandb_run_name baseline --num_workers 16"

srun apptainer exec \
    --nv \
    --bind ${HOST_CA_CERT_PATH}:${CONTAINER_CA_CERT_PATH} \
    --overlay ${OVERLAY_PATH}\
    ${CONTAINER_PATH} \
    bash -c "${COMMAND}"

echo "Job finished successfully."