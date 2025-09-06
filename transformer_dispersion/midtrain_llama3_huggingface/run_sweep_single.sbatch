#!/bin/bash

#SBATCH --job-name=hf_midtrain-sweep
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --constraint='h100|a100'
#SBATCH --cpus-per-task=17
#SBATCH --time=5-00:00:00
#SBATCH --output=slurm_out/hf_midtrain-sweep-%j.out
#SBATCH --error=slurm_out/hf_midtrain-sweep-%j.err

echo "Setting up environment variables and paths..."
# --- User-Specific Paths ---
if [ -z "${CONTAINER_PATH}" ]; then
  echo "ERROR: Please set the CONTAINER_PATH environment variable."
  exit 1
fi

if [ -z "${OVERLAY_PATH}" ]; then
  echo "ERROR: Please set the OVERLAY_PATH environment variable."
  exit 1
fi

if [ -z "${WORK_DIR}" ]; then
  echo "ERROR: Please set the WORK_DIR environment variable."
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

echo "Starting Apptainer container and executing dispersion sweep..."

# Base command
BASE_CMD="accelerate launch midtrain.py --train_tokens 300_000_000 --block_size 2048 --per_device_train_batch_size 2 --gradient_accumulation_steps 32 --use_wandb --wandb_project llama3-midtrain-dispersion"

# Run baseline (no dispersion) - session 1
echo "=== Running baseline (no dispersion) ==="
export OVERLAY_PATH="/gpfs/radev/home/xs272/scratch/overlay_llama_factory_session1.img"
COMMAND="$BASE_CMD --wandb_run_name llama3-baseline"
srun --chdir=${WORK_DIR} apptainer exec \
    --nv \
    --bind ${HOST_CA_CERT_PATH}:${CONTAINER_CA_CERT_PATH} \
    --overlay ${OVERLAY_PATH}\
    ${CONTAINER_PATH} \
    bash -c "${COMMAND}"

# Run with different dispersion methods
session=2
for dispersion in "infonce_l2" "infonce_cosine" "hinge" "covariance"; do
    echo "=== Running with dispersion: $dispersion (session $session) ==="
    export OVERLAY_PATH="/gpfs/radev/home/xs272/scratch/overlay_llama_factory_session${session}.img"
    COMMAND="$BASE_CMD --dispersion $dispersion --dispersion_loc all --dispersion_coeff 0.5 --wandb_run_name llama3-${dispersion}-all-0.5"
    srun --chdir=${WORK_DIR} apptainer exec \
        --nv \
        --bind ${HOST_CA_CERT_PATH}:${CONTAINER_CA_CERT_PATH} \
        --overlay ${OVERLAY_PATH}\
        ${CONTAINER_PATH} \
        bash -c "${COMMAND}"
    ((session++))
done

echo "Sweep finished successfully."
