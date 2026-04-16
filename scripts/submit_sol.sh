#!/bin/bash
#SBATCH -N 1
#SBATCH -G a100:1
#SBATCH -p public
#SBATCH -q public
#SBATCH -t 0-08:00:00
#SBATCH --mem=48G
#SBATCH -c 8
#SBATCH -o /data/amciilab/xinran/indiana_cxr/results/slurm_%j.out
#SBATCH -e /data/amciilab/xinran/indiana_cxr/results/slurm_%j.err
#SBATCH -J multimodal-fl

# -------------------------------------------------------
# Multimodal FL Experiments on ASU Sol
# Usage:
#   # Run all 10 experiments:
#   sbatch scripts/submit_sol.sh
#
#   # Run specific experiments:
#   sbatch scripts/submit_sol.sh E1 E2
#
#   # Run inside tmux for interactive monitoring:
#   tmux new -s fl-exp
#   sbatch scripts/submit_sol.sh
#   # Ctrl+b d to detach, tmux attach -t fl-exp to reattach
# -------------------------------------------------------

module load mamba/latest
source activate multimodal-fl

# Create output directory
mkdir -p /data/amciilab/xinran/indiana_cxr/results

cd /home/xzhao181/Multimodal-FL

# If specific experiments passed as args, use them; otherwise run all
if [ $# -gt 0 ]; then
    EXPERIMENTS="$@"
else
    EXPERIMENTS="E1 E2 E3 E4 E5 E6 E7 E8 E9 E10"
fi

echo "=========================================="
echo "Starting FL experiments: $EXPERIMENTS"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'N/A')"
echo "Date: $(date)"
echo "=========================================="

python run_experiment.py \
    --experiments $EXPERIMENTS \
    --prepared_csv /data/amciilab/xinran/indiana_cxr/prepared/prepared_data.csv \
    --images_dir /data/amciilab/xinran/indiana_cxr/images/images_normalized \
    --partition_dir /data/amciilab/xinran/indiana_cxr/prepared/partitions \
    --output_dir /data/amciilab/xinran/indiana_cxr/results \
    --num_rounds 50 \
    --local_epochs 5 \
    --batch_size 32 \
    --lr 1e-3 \
    --embed_dim 256 \
    --patience 10 \
    --seed 42

echo "=========================================="
echo "All experiments complete: $(date)"
echo "=========================================="
