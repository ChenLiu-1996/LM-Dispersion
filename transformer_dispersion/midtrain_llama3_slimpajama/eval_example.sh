#!/bin/bash

# Example usage of eval_checkpoints.py
# This script shows how to evaluate checkpoints from a midtrain run

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
        
        python eval_checkpoints.py \
            --checkpoint_dir "$subdir" \
            --output_dir "$OUTPUT_DIR/all_checkpoints_${subdir_name}" \
            --checkpoint_patterns "eval_ckpt_begin_*" "eval_ckpt_end_*" \
            --num_fewshot 1 \
            --max_eval_samples 200
        
        echo "Completed evaluation for: $subdir_name"
        echo "---"
    fi
done

# python eval_checkpoints.py \
#     --checkpoint_dir "$CHECKPOINT_BASE_DIR" \
#     --output_dir "$OUTPUT_DIR/all_checkpoints" \
#     --checkpoint_pattern "eval_ckpt_*" \
#     --num_fewshot 1 \
#     --max_eval_samples 200


# # Example 2: Evaluate a single specific checkpoint
# echo "=== Example 2: Evaluating single checkpoint ==="
# SINGLE_CHECKPOINT="$CHECKPOINT_BASE_DIR/eval_ckpt_interval_step500"
# python eval_checkpoints.py \
#     --checkpoint_dir "$SINGLE_CHECKPOINT" \
#     --output_dir "$OUTPUT_DIR/single_checkpoint" \
#     --single_checkpoint \
#     --num_fewshot 1 \
#     --max_eval_samples 200

# # Example 3: Custom task selection
# echo "=== Example 3: Custom task selection ==="
# python eval_checkpoints.py \
#     --checkpoint_dir "$CHECKPOINT_BASE_DIR" \
#     --output_dir "$OUTPUT_DIR/custom_tasks" \
#     --zeroshot_tasks hellaswag piqa winogrande \
#     --fewshot_tasks arc_challenge mmlu \
#     --num_fewshot 3 \
#     --max_eval_samples 100

echo "Evaluation examples complete!"
echo "Check the output directories for results:"
echo "  - $OUTPUT_DIR/all_checkpoints_*/ (one for each experiment configuration)"
echo "  - $OUTPUT_DIR/single_checkpoint/ (if uncommented)"
echo "  - $OUTPUT_DIR/custom_tasks/ (if uncommented)"
