#!/usr/bin/env python3
"""
Sweep script to evaluate GPT-2 variants at initialization (no training)
"""

import os
from pathlib import Path

# Models to evaluate
GPT2_MODELS = [
    'gpt2',           # 117M parameters
    'gpt2-medium',    # 345M parameters
    'gpt2-large',     # 774M parameters
    'gpt2-xl',        # 1.5B parameters
]

# Fixed parameters for evaluation-only run
BASE_CMD = [
    "accelerate launch --mixed_precision bf16 midtrain.py",
    "--dataset_name Salesforce/wikitext",
    "--dataset_config wikitext-103-raw-v1",
    "--train_tokens 0",  # No training, just evaluation
    "--use_wandb",
    "--wandb_project gpt2-init-eval",
    "--num_workers 4",
    "--eval_steps 1",  # Not used since no training, but required
    "--max_eval_samples 100",  # Limit eval samples for speed
    "--no_save_model",  # Don't save model checkpoints
]

def generate_command(model_name):
    """Generate command for a specific model"""
    cmd_parts = BASE_CMD.copy()
    cmd_parts.insert(2, f"--model_name {model_name}")
    cmd_parts.insert(-1, f"--wandb_run_name init-eval-{model_name.replace('/', '-')}")
    return " ".join(cmd_parts)

def main():
    print("GPT-2 Initialization Evaluation Sweep")
    print("=" * 40)

    # Create output directory
    os.makedirs("slurm_out", exist_ok=True)

    print(f"Will evaluate {len(GPT2_MODELS)} GPT-2 variants at initialization:")
    for model in GPT2_MODELS:
        print(f"  - {model}")

    print("\nCommands:")
    commands = []
    for i, model in enumerate(GPT2_MODELS):
        cmd = generate_command(model)
        commands.append(cmd)
        print(f"{i}: {cmd}\n")

    # Create SLURM array job script
    script_content = f'''#!/bin/bash

#SBATCH --job-name=gpt2-init-eval
#SBATCH --partition=gpu
#SBATCH --qos=qos_nmi
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=17
#SBATCH --mem-per-cpu=8G
#SBATCH --time=00:30:00  # Shorter time since no training
#SBATCH --array=0-{len(commands)-1}
#SBATCH --output=slurm_out/gpt2_init_eval-%A_%a.out
#SBATCH --error=slurm_out/gpt2_init_eval-%A_%a.err

echo "Job array ID: $SLURM_ARRAY_JOB_ID"
echo "Job array index: $SLURM_ARRAY_TASK_ID"

# Environment setup (same as your hypersweep)
if [ -z "${{CONTAINER_PATH}}" ]; then
  echo "ERROR: Please set the CONTAINER_PATH environment variable."
  exit 1
fi

if [ -z "${{TRITON_CACHE_DIR}}" ]; then
  echo "ERROR: Please set the TRITON_CACHE_DIR environment variable."
  exit 1
fi

if [ -z "${{HOST_CA_CERT_PATH}}" ]; then
  echo "ERROR: Please set the HOST_CA_CERT_PATH environment variable."
  exit 1
fi

if [ -z "${{CONTAINER_CA_CERT_PATH}}" ]; then
  echo "ERROR: Please set the CONTAINER_CA_CERT_PATH environment variable."
  exit 1
fi

export NCCL_DEBUG=INFO
export PYTHONFAULTHANDLER=1
export HF_ALLOW_CODE_EVAL="1"

# Calculate session number (cycle through available sessions)
SESSION_NUM=$((SLURM_ARRAY_TASK_ID % 20 + 1))
export OVERLAY_PATH="/gpfs/radev/home/xs272/scratch/overlay_llama_factory_session${{SESSION_NUM}}.img"

echo "Using overlay session: $SESSION_NUM"
echo "Overlay path: $OVERLAY_PATH"

# Define commands for each array index
case $SLURM_ARRAY_TASK_ID in
'''

    for i, cmd in enumerate(commands):
        script_content += f'''    {i})
        COMMAND="{cmd}"
        echo "Running evaluation {i}: {GPT2_MODELS[i]}"
        ;;
'''

    script_content += f'''    *)
        echo "Invalid array index: $SLURM_ARRAY_TASK_ID"
        exit 1
        ;;
esac

echo "Executing: $COMMAND"

srun apptainer exec \\
    --nv \\
    --bind ${{HOST_CA_CERT_PATH}}:${{CONTAINER_CA_CERT_PATH}} \\
    --overlay ${{OVERLAY_PATH}} \\
    ${{CONTAINER_PATH}} \\
    bash -c "$COMMAND"

echo "Job finished successfully."
'''

    # Write script
    script_path = "gpt2_init_eval.sbatch"
    with open(script_path, 'w') as f:
        f.write(script_content)

    print(f"Created SLURM array job script: {script_path}")
    print(f"To submit: sbatch {script_path}")
    print(f"To monitor: squeue -u $USER")

if __name__ == "__main__":
    main()
