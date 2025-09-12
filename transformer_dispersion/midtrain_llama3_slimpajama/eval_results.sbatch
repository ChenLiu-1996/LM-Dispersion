#!/bin/bash

#SBATCH --job-name=eval_results-1gpu
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
## #SBATCH --constraint='h100'
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=1-00:00:00
#SBATCH --output=slurm_out/eval_results-1gpu-%j.out
#SBATCH --error=slurm_out/eval_results-1gpu-%j.err

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

# Set paths - modify these for your specific case
RESULTS_BASE_DIR="./results/midtrain_meta-llama"
OUTPUT_DIR="./evaluation_results"

# Example 1: Evaluate all checkpoints in all subdirectories
echo "=== Example 1: Evaluating all checkpoints in all subdirectories ==="

# Loop through all subdirectories in the results folder
for subdir in "$RESULTS_BASE_DIR"/*/ ; do
    if [ -d "$subdir" ]; then
        # Extract the subdirectory name for output naming
        subdir_name=$(basename "$subdir")
        echo "Processing subdirectory: $subdir_name"
        
        EVAL_COMMAND="python eval_checkpoints.py \
            --checkpoint_dir \"$subdir\" \
            --output_dir \"$OUTPUT_DIR/all_checkpoints_${subdir_name}\" \
            --checkpoint_patterns \"eval_ckpt_begin_*\" \"eval_ckpt_end_*\" \
            --num_fewshot 1 \
            --max_eval_samples 200"
        
        echo "Starting Apptainer container and executing evaluation script..."
        srun apptainer exec \
            --nv \
            --bind ${HOST_CA_CERT_PATH}:${CONTAINER_CA_CERT_PATH} \
            --overlay ${OVERLAY_PATH} \
            ${CONTAINER_PATH} \
            bash -c "${EVAL_COMMAND}"
        
        echo "Completed evaluation for: $subdir_name"
        echo "---"
    fi
done

# Alternative approach for all checkpoints in single directory:
# EVAL_COMMAND="python eval_checkpoints.py \
#     --checkpoint_dir \"$CHECKPOINT_BASE_DIR\" \
#     --output_dir \"$OUTPUT_DIR/all_checkpoints\" \
#     --checkpoint_pattern \"eval_ckpt_*\" \
#     --num_fewshot 1 \
#     --max_eval_samples 200"
# srun apptainer exec --nv --bind ${HOST_CA_CERT_PATH}:${CONTAINER_CA_CERT_PATH} --overlay ${OVERLAY_PATH} ${CONTAINER_PATH} bash -c "${EVAL_COMMAND}"


# # Example 2: Evaluate a single specific checkpoint
# echo "=== Example 2: Evaluating single checkpoint ==="
# SINGLE_CHECKPOINT="$CHECKPOINT_BASE_DIR/eval_ckpt_interval_step500"
# EVAL_COMMAND="python eval_checkpoints.py \
#     --checkpoint_dir \"$SINGLE_CHECKPOINT\" \
#     --output_dir \"$OUTPUT_DIR/single_checkpoint\" \
#     --single_checkpoint \
#     --num_fewshot 1 \
#     --max_eval_samples 200"
# srun apptainer exec --nv --bind ${HOST_CA_CERT_PATH}:${CONTAINER_CA_CERT_PATH} --overlay ${OVERLAY_PATH} ${CONTAINER_PATH} bash -c "${EVAL_COMMAND}"

# # Example 3: Custom task selection
# echo "=== Example 3: Custom task selection ==="
# EVAL_COMMAND="python eval_checkpoints.py \
#     --checkpoint_dir \"$CHECKPOINT_BASE_DIR\" \
#     --output_dir \"$OUTPUT_DIR/custom_tasks\" \
#     --zeroshot_tasks hellaswag piqa winogrande \
#     --fewshot_tasks arc_challenge mmlu \
#     --num_fewshot 3 \
#     --max_eval_samples 100"
# srun apptainer exec --nv --bind ${HOST_CA_CERT_PATH}:${CONTAINER_CA_CERT_PATH} --overlay ${OVERLAY_PATH} ${CONTAINER_PATH} bash -c "${EVAL_COMMAND}"

echo "Evaluation examples complete!"
echo "Check the output directories for results:"
echo "  - $OUTPUT_DIR/all_checkpoints_*/ (one for each experiment configuration)"
echo "  - $OUTPUT_DIR/single_checkpoint/ (if uncommented)"
echo "  - $OUTPUT_DIR/custom_tasks/ (if uncommented)"
