#!/usr/bin/env python3
"""
Hyperparameter sweep generator for dispersion experiments
Creates SLURM array job with all parameter combinations
"""

import itertools
import os
from pathlib import Path

# Hyperparameter grid
HYPERPARAMS = {
    'dispersion_types': [
        # ('none', None),
        ('infonce_l2', 'infonce_l2'),
        ('infonce_cosine', 'infonce_cosine'),
        # ('hinge', 'hinge'),
        # ('covariance', 'covariance'),
    ],
    # 'learning_rates': [5e-6, 1e-5, 5e-5],
    'learning_rates': [1e-5],
    # 'dispersion_coeffs': [0.1, 0.5, 1.0],
    'dispersion_coeffs': [0.1, 1.0],
    'tau_infonce_l2': [0.1, 0.5, 1.0],
    'tau_infonce_cos': [0.1, 0.5, 1.0],
}

# Fixed parameters
FIXED_PARAMS = {
    'model_name': 'gpt2',
    'dataset_name': 'Salesforce/wikitext',
    'dataset_config': 'wikitext-103-raw-v1',
    'train_tokens': 300_000_000,
    'block_size': 1024,
    'per_device_train_batch_size': 38,
    'gradient_accumulation_steps': 7,
    'dispersion_loc': 'all',
    'num_workers': 16,
    'eval_steps': 5,
    'lr_scheduler_type': 'constant',
    'max_gen_tokens': 50,
    'max_grad_norm': 1.0,
    'wandb_project': 'gpt2-midtrain-dispersion-hypersweep',
}

def generate_combinations():
    """Generate all hyperparameter combinations"""
    combinations = []

    default_disp_coeff = HYPERPARAMS['dispersion_coeffs'][0]
    default_tau_l2 = HYPERPARAMS['tau_infonce_l2'][0]
    default_tau_cos = HYPERPARAMS['tau_infonce_cos'][0]
    
    for (disp_name, disp_type), lr, disp_coeff, tau_l2, tau_cos in itertools.product(
        HYPERPARAMS['dispersion_types'],
        HYPERPARAMS['learning_rates'],
        HYPERPARAMS['dispersion_coeffs'],
        HYPERPARAMS['tau_infonce_l2'],
        HYPERPARAMS['tau_infonce_cos']
    ):
        # Skip parameter combinations that don't apply to specific dispersion types
        if disp_type is None:
            # For baseline (no dispersion), skip non-default dispersion_coeff and tau values
            if disp_coeff != default_disp_coeff or tau_l2 != default_tau_l2 or tau_cos != default_tau_cos:
                continue
        elif disp_type == 'infonce_l2':
            # For infonce_l2, only vary tau_l2, keep tau_cos at default
            if tau_cos != default_tau_cos:
                continue
        elif disp_type == 'infonce_cosine':
            # For infonce_cosine, only vary tau_cos, keep tau_l2 at default
            if tau_l2 != default_tau_l2:
                continue
        elif disp_type in ['hinge', 'covariance']:
            # For these types, tau parameters don't matter, use defaults
            if tau_l2 != default_tau_l2 or tau_cos != default_tau_cos:
                continue
        
        combo = {
            'job_id': len(combinations),
            'dispersion_name': disp_name,
            'dispersion_type': disp_type,
            'learning_rate': lr,
            'dispersion_coeff': disp_coeff,
            'tau_infonce_l2': tau_l2,
            'tau_infonce_cos': tau_cos,
        }
        combinations.append(combo)
    
    return combinations

def generate_run_name(combo):
    """Generate descriptive run name (for wandb and logging)"""
    # Format learning rate to avoid periods
    lr_str = f"{combo['learning_rate']:.0e}".replace(".", "p").replace("-", "m")
    
    if combo['dispersion_type'] is None:
        return f"baseline-lr{lr_str}"
    elif combo['dispersion_type'] == 'infonce_l2':
        tau_str = f"{combo['tau_infonce_l2']}".replace(".", "p")
        coeff_str = f"{combo['dispersion_coeff']}".replace(".", "p")
        return f"infonce-l2-lr{lr_str}-c{coeff_str}-taul2-{tau_str}"
    elif combo['dispersion_type'] == 'infonce_cosine':
        tau_str = f"{combo['tau_infonce_cos']}".replace(".", "p")
        coeff_str = f"{combo['dispersion_coeff']}".replace(".", "p")
        return f"infonce-cos-lr{lr_str}-c{coeff_str}-taucos-{tau_str}"
    else:
        coeff_str = f"{combo['dispersion_coeff']}".replace(".", "p")
        return f"{combo['dispersion_name']}-lr{lr_str}-c{coeff_str}"

def generate_command(combo):
    """Generate training command for a combination"""
    base_cmd = [
        "accelerate launch --mixed_precision bf16 midtrain.py",
        f"--model_name {FIXED_PARAMS['model_name']}",
        f"--dataset_name {FIXED_PARAMS['dataset_name']}",
        f"--dataset_config {FIXED_PARAMS['dataset_config']}",
        f"--train_tokens {FIXED_PARAMS['train_tokens']}",
        f"--lr {combo['learning_rate']}",
        f"--block_size {FIXED_PARAMS['block_size']}",
        f"--per_device_train_batch_size {FIXED_PARAMS['per_device_train_batch_size']}",
        f"--gradient_accumulation_steps {FIXED_PARAMS['gradient_accumulation_steps']}",
        "--use_wandb",
        f"--wandb_project {FIXED_PARAMS['wandb_project']}",
        f"--wandb_run_name {generate_run_name(combo)}",
        f"--num_workers {FIXED_PARAMS['num_workers']}",
        f"--eval_steps {FIXED_PARAMS['eval_steps']}",
        f"--tau_infonce_l2 {combo['tau_infonce_l2']}",
        f"--tau_infonce_cos {combo['tau_infonce_cos']}",
        f"--lr_scheduler_type {FIXED_PARAMS['lr_scheduler_type']}",
        f"--max_gen_tokens {FIXED_PARAMS['max_gen_tokens']}",
        f"--max_grad_norm {FIXED_PARAMS['max_grad_norm']}",
    ]
    
    # Add dispersion-specific parameters
    if combo['dispersion_type'] is not None:
        base_cmd.extend([
            f"--dispersion {combo['dispersion_type']}",
            f"--dispersion_coeff {combo['dispersion_coeff']}",
            f"--dispersion_loc {FIXED_PARAMS['dispersion_loc']}",
        ])
    
    return " ".join(base_cmd)

def create_array_job_script(combinations):
    """Create SLURM array job script"""
    
    # Calculate session assignments (cycle through available sessions)
    max_sessions = 20  # Adjust based on how many overlay images you can create
    
    script_content = f'''#!/bin/bash

#SBATCH --job-name=hypersweep
#SBATCH --partition=gpu
#SBATCH --qos=qos_nmi
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=17
#SBATCH --mem-per-cpu=8G
#SBATCH --time=02:00:00
#SBATCH --array=0-{len(combinations)-1}
#SBATCH --output=slurm_out/hypersweep-%A_%a.out
#SBATCH --error=slurm_out/hypersweep-%A_%a.err

echo "Job array ID: $SLURM_ARRAY_JOB_ID"
echo "Job array index: $SLURM_ARRAY_TASK_ID"

# Environment setup
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
SESSION_NUM=$((SLURM_ARRAY_TASK_ID % {max_sessions} + 1))
export OVERLAY_PATH="/gpfs/radev/home/xs272/scratch/overlay_llama_factory_session${{SESSION_NUM}}.img"

echo "Using overlay session: $SESSION_NUM"
echo "Overlay path: $OVERLAY_PATH"

# Define commands for each array index
case $SLURM_ARRAY_TASK_ID in
'''

    for i, combo in enumerate(combinations):
        command = generate_command(combo)
        script_content += f'''    {i})
        COMMAND="{command}"
        echo "Running combination {i}: {generate_run_name(combo)}"
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

    return script_content

def verify_uniqueness(combinations):
    """Verify that all combinations will generate unique directory names."""
    print("\\n🔍 Verifying uniqueness of generated paths...")
    
    # We need to simulate the directory generation logic from midtrain.py
    # This is a simplified version - in practice, you'd import the function
    def simulate_output_dir(combo):
        import hashlib
        import json
        
        # Simulate the parameter dict that midtrain.py would create
        param_dict = {
            'model': FIXED_PARAMS['model_name'],
            'dataset': FIXED_PARAMS['dataset_name'],
            'dataset_config': FIXED_PARAMS['dataset_config'],
            'lr': combo['learning_rate'],
            'train_tokens': FIXED_PARAMS['train_tokens'],
            'dispersion': combo['dispersion_type'],
            'dispersion_coeff': combo['dispersion_coeff'],
            'dispersion_loc': FIXED_PARAMS['dispersion_loc'],
            'tau_infonce_l2': combo['tau_infonce_l2'],
            'tau_infonce_cos': combo['tau_infonce_cos'],
            'num_fewshot': 1,  # Default from midtrain.py
            'max_eval_samples': 100,  # Default from midtrain.py
            'seed': 1,  # Default from midtrain.py
            'block_size': FIXED_PARAMS['block_size'],
            'per_device_train_batch_size': FIXED_PARAMS['per_device_train_batch_size'],
            'gradient_accumulation_steps': FIXED_PARAMS['gradient_accumulation_steps'],
            'lr_scheduler_type': FIXED_PARAMS['lr_scheduler_type'],
        }
        
        param_str = json.dumps(param_dict, sort_keys=True)
        param_hash = hashlib.md5(param_str.encode()).hexdigest()[:8]
        return param_hash
    
    # Check for hash collisions
    hash_to_combo = {}
    collisions = []
    
    for combo in combinations:
        hash_val = simulate_output_dir(combo)
        if hash_val in hash_to_combo:
            collisions.append((combo, hash_to_combo[hash_val]))
        else:
            hash_to_combo[hash_val] = combo
    
    if collisions:
        print(f"❌ Found {len(collisions)} potential directory collisions!")
        for i, (combo1, combo2) in enumerate(collisions[:3]):  # Show first 3
            print(f"  Collision {i+1}:")
            print(f"    Job {combo1['job_id']}: {generate_run_name(combo1)}")
            print(f"    Job {combo2['job_id']}: {generate_run_name(combo2)}")
        if len(collisions) > 3:
            print(f"    ... and {len(collisions) - 3} more collisions")
        print("\\n⚠️  This could cause 'directory already exists' errors!")
        return False
    else:
        print("✅ All combinations will generate unique directory names")
        return True


def main():
    # Create output directory
    os.makedirs("slurm_out", exist_ok=True)
    
    # Generate all combinations
    combinations = generate_combinations()
    print(f"Generated {len(combinations)} hyperparameter combinations")
    
    # Verify uniqueness
    is_unique = verify_uniqueness(combinations)
    
    # Show some examples
    print("\\nExample combinations:")
    for i, combo in enumerate(combinations[:5]):
        print(f"  {i}: {generate_run_name(combo)}")
    if len(combinations) > 5:
        print(f"  ... and {len(combinations) - 5} more")
    
    # Create array job script only if uniqueness is verified
    if not is_unique:
        print("\\n❌ Stopping due to potential directory collisions.")
        print("💡 Consider:")
        print("   1. Reducing parameter combinations")
        print("   2. Using different seeds for different parameter sets")
        print("   3. Checking the generate_output_dir() function in midtrain.py")
        return
    
    # Create array job script
    script_content = create_array_job_script(combinations)
    
    # Write script
    script_path = "hypersweep_array.sbatch"
    with open(script_path, 'w') as f:
        f.write(script_content)
    
    print(f"\\n✅ Created SLURM array job script: {script_path}")
    print(f"To submit: sbatch {script_path}")
    print(f"To monitor: squeue -u $USER")
    
    # Create summary file
    summary_path = "hypersweep_combinations.txt"
    with open(summary_path, 'w') as f:
        f.write("Hyperparameter Sweep Combinations\\n")
        f.write("=" * 40 + "\\n\\n")
        for combo in combinations:
            f.write(f"Job {combo['job_id']:3d}: {generate_run_name(combo)}\\n")
            f.write(f"  Command: {generate_command(combo)}\\n\\n")
    
    print(f"Created combination summary: {summary_path}")

if __name__ == "__main__":
    main()
