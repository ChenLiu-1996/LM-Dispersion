#!/bin/bash

#SBATCH --job-name=plot_heatmaps-1gpu
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
## #SBATCH --constraint='h100|a100|h200'
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=1-00:00:00
#SBATCH --output=slurm_out/plot_heatmaps-1gpu-%j.out
#SBATCH --error=slurm_out/plot_heatmaps-1gpu-%j.err

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

echo "=== Generating OLMo-2 Heatmaps ==="

PLOT_COMMAND="python plot_heatmap_checkpoints.py"

echo "Starting Apptainer container and executing plot script..."
srun apptainer exec \
    --nv \
    --bind ${HOST_CA_CERT_PATH}:${CONTAINER_CA_CERT_PATH} \
    --overlay ${OVERLAY_PATH} \
    ${CONTAINER_PATH} \
    bash -c "${PLOT_COMMAND}"

echo "Plot generation completed!"
echo "Check ./figures/embedding_heatmaps_grid.png for results"

