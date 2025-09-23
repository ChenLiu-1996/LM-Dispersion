# Checkpoint Evaluation Script

This directory contains `eval_checkpoints.py`, a standalone script that reuses the evaluation logic from the `LMEvalCallback` class in `midtrain.py`. It allows you to evaluate saved checkpoints using the same tasks and methodology as the training script.

## Features

- **Maximum code reuse**: Extracts and reuses the evaluation logic from `LMEvalCallback._run_evaluation`
- **Flexible checkpoint selection**: Evaluate single checkpoints or entire directories
- **Same task configuration**: Uses the same default tasks as the training script
- **Customizable**: Override tasks, few-shot settings, and evaluation parameters
- **PEFT support**: Automatically handles PEFT/LoRA models by merging adapters
- **Multi-GPU support**: Handles distributed evaluation setup

## Usage

### Basic Usage

```bash
# Evaluate all checkpoints in a directory
python eval_checkpoints.py \
    --checkpoint_dir ./results/your_training_run/ \
    --output_dir ./evaluation_results/

# Evaluate a single checkpoint
python eval_checkpoints.py \
    --checkpoint_dir ./results/your_training_run/eval_ckpt_interval_step500 \
    --output_dir ./evaluation_results/ \
    --single_checkpoint
```

### Advanced Usage

```bash
# Custom task selection and parameters
python eval_checkpoints.py \
    --checkpoint_dir ./results/your_training_run/ \
    --output_dir ./evaluation_results/ \
    --zeroshot_tasks hellaswag piqa winogrande \
    --fewshot_tasks arc_challenge mmlu \
    --num_fewshot 3 \
    --max_eval_samples 100 \
    --checkpoint_pattern "eval_ckpt_end_*"
```

## Command Line Arguments

### Required Arguments
- `--checkpoint_dir`: Directory containing checkpoints or path to single checkpoint
- `--output_dir`: Directory to save evaluation results

### Optional Arguments
- `--checkpoint_pattern`: Pattern to match checkpoint directories (default: `"eval_ckpt_*"`)
- `--single_checkpoint`: Treat `checkpoint_dir` as a single checkpoint rather than a directory
- `--num_fewshot`: Number of few-shot examples (default: 1)
- `--max_eval_samples`: Maximum samples per task (default: 200)
- `--log_path`: Path to log file (default: `output_dir/eval_log.txt`)

### Task Selection
- `--zeroshot_tasks`: Zero-shot evaluation tasks (default: same as training script)
- `--fewshot_tasks`: Few-shot evaluation tasks (default: same as training script)

### Default Tasks

**Zero-shot tasks:**
- hellaswag
- lambada  
- paloma_wikitext_103
- piqa
- truthfulqa_mc2
- winogrande

**Few-shot tasks:**
- arc_challenge
- gsm8k
- mmlu
- medmcqa

## Output

The script creates:
- Individual JSON files for each checkpoint: `lm_eval_{checkpoint_name}.json`
- Summary file with all results: `evaluation_summary.json`
- Log file with detailed evaluation progress: `eval_log.txt`

## Examples

See `eval_example.sh` for complete usage examples.

## Differences from Training Evaluation

The standalone script:
- Uses fixed random seeds (42) instead of training args seeds
- Doesn't integrate with Trainer logging/wandb
- Doesn't save checkpoints (only evaluates existing ones)
- Can be run independently of any training process

## Requirements

Same as the main training script:
- transformers
- lm_eval
- torch
- Standard Python libraries (json, tempfile, glob, etc.)
