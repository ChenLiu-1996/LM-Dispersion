#!/bin/bash

#SBATCH --job-name=pretrain-1gpu
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-task=h100:1           # 1 GPU for that process
#SBATCH --cpus-per-task=17
#SBATCH --mem=32G
#SBATCH --time=1-00:00:00
#SBATCH --output=slurm_out/pretrain-1gpu-%j.out
#SBATCH --error=slurm_out/pretrain-1gpu-%j.err

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

if [ -z "${WORK_DIR}" ]; then
  echo "ERROR: Please set the WORK_DIR environment variable."
  exit 1
fi

# Accept YAML path as command line parameter
if [ $# -eq 0 ]; then
  echo "Usage: sbatch $0 <yaml_config_file>"
  echo "Example: sbatch $0 llama_wiki_pretrain_with_dispersion.yaml"
  exit 1
fi

YAML_CONFIG_FILE=$1
YAML_PATH=${WORK_DIR}/my/${YAML_CONFIG_FILE}

# Validate that the YAML file exists
if [ ! -f "${YAML_PATH}" ]; then
  echo "ERROR: YAML config file not found: ${YAML_PATH}"
  echo "Please ensure the file exists in ${WORK_DIR}/my/"
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

srun --chdir=${WORK_DIR} apptainer exec \
    --nv \
    --bind ${HOST_CA_CERT_PATH}:${CONTAINER_CA_CERT_PATH} \
    --overlay ${OVERLAY_PATH}\
    ${CONTAINER_PATH} \
    bash -c "FORCE_TORCHRUN=1 llamafactory-cli train ${YAML_PATH}"

echo "Job finished successfully."