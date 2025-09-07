#!/bin/bash

# Dispersion sweep script
# Usage: bash run_sweep.sh

# Base command template
# BASE_CMD="accelerate launch --mixed_precision bf16 midtrain.py --dataset_name cerebras/SlimPajama-627B --dataset_config default --train_tokens 1_600_000_000 --lr 1e-5 --block_size 2048 --per_device_train_batch_size 2 --gradient_accumulation_steps 32 --use_wandb --wandb_project llama3-midtrain-dispersion-slimpajama"
# using a subset to speed up. the token budget will not use the entire dataset anyways so I do this to speed up the data preprocessing.
BASE_CMD="accelerate launch --mixed_precision bf16 midtrain.py --dataset_name DKYoon/SlimPajama-6B --dataset_config default --train_tokens 1_600_000_000 --lr 1e-5 --block_size 2048 --per_device_train_batch_size 2 --gradient_accumulation_steps 32 --use_wandb --wandb_project llama3-midtrain-dispersion-slimpajama --num_workers 16"

# Dispersion configurations (name:dispersion:run_name:session)
declare -a CONFIGS=(
    "none:None:baseline:1"
    "infonce_l2:infonce_l2:infonce-l2-all-0.5:2"
    "infonce_cosine:infonce_cosine:infonce-cosine-all-0.5:3"
    "hinge:hinge:hinge-all-0.5:4"
    "covariance:covariance:covariance-all-0.5:5"
)

for config in "${CONFIGS[@]}"; do
    IFS=':' read -r name dispersion run_name session <<< "$config"
    
    echo "Submitting job for dispersion: $name (session $session)"
    
    if [ "$name" = "none" ]; then
        # No dispersion case
        COMMAND="$BASE_CMD --wandb_run_name $run_name"
    else
        # With dispersion
        COMMAND="$BASE_CMD --dispersion $dispersion --dispersion_loc all --dispersion_coeff 0.5 --wandb_run_name $run_name"
    fi
    
    # Create temporary sbatch file
    cat > "temp_${name}.sbatch" << EOF
#!/bin/bash

#SBATCH --job-name=hf_midtrain-${name}
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --constraint='h100'
#SBATCH --cpus-per-task=17
#SBATCH --mem-per-cpu=8G
#SBATCH --time=1-00:00:00
#SBATCH --output=slurm_out/hf_midtrain-${name}-%j.out
#SBATCH --error=slurm_out/hf_midtrain-${name}-%j.err

echo "Setting up environment variables and paths..."
# --- User-Specific Paths ---
if [ -z "\${CONTAINER_PATH}" ]; then
  echo "ERROR: Please set the CONTAINER_PATH environment variable."
  exit 1
fi

# Set session-specific overlay path
export OVERLAY_PATH="/gpfs/radev/home/xs272/scratch/overlay_llama_factory_session${session}.img"

if [ -z "\${OVERLAY_PATH}" ]; then
  echo "ERROR: Please set the OVERLAY_PATH environment variable."
  exit 1
fi

if [ -z "\${TRITON_CACHE_DIR}" ]; then
  echo "ERROR: Please set the TRITON_CACHE_DIR environment variable."
  exit 1
fi

if [ -z "\${HOST_CA_CERT_PATH}" ]; then
  echo "ERROR: Please set the HOST_CA_CERT_PATH environment variable."
  exit 1
fi

if [ -z "\${CONTAINER_CA_CERT_PATH}" ]; then
  echo "ERROR: Please set the CONTAINER_CA_CERT_PATH environment variable."
  exit 1
fi

export NCCL_DEBUG=INFO
export PYTHONFAULTHANDLER=1

echo "Starting Apptainer container and executing script..."

COMMAND="$COMMAND"

srun apptainer exec \\
    --nv \\
    --bind \${HOST_CA_CERT_PATH}:\${CONTAINER_CA_CERT_PATH} \\
    --overlay \${OVERLAY_PATH}\\
    \${CONTAINER_PATH} \\
    bash -c "\${COMMAND}"

echo "Job finished successfully."
EOF

    # Submit the job
    sbatch "temp_${name}.sbatch"
    
    # Clean up temp file
    rm "temp_${name}.sbatch"
    
    # Small delay between submissions
    sleep 2
done

echo "All jobs submitted!"
