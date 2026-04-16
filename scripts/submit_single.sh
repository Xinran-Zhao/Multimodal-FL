#!/bin/bash
#SBATCH -N 1
#SBATCH -G a100:1
#SBATCH -p public
#SBATCH -q public
#SBATCH -t 0-08:00:00
#SBATCH --mem=48G
#SBATCH -c 8
#SBATCH -o /data/amciilab/xinran/indiana_cxr/results_v3/slurm_%x_%j.out
#SBATCH -e /data/amciilab/xinran/indiana_cxr/results_v3/slurm_%x_%j.err

# -------------------------------------------------------
# Single FL experiment job
# Usage: sbatch --job-name=E1 scripts/submit_single.sh E1
# -------------------------------------------------------

EXP_ID=$1
if [ -z "$EXP_ID" ]; then
    echo "Error: must provide experiment ID (e.g., E1)"
    exit 1
fi

module load mamba/latest
source activate multimodal-fl

mkdir -p /data/amciilab/xinran/indiana_cxr/results_v3

cd /home/xzhao181/Multimodal-FL

echo "=========================================="
echo "Experiment: $EXP_ID"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'N/A')"
echo "Date: $(date)"
echo "=========================================="

python run_experiment.py \
    --experiments $EXP_ID \
    --prepared_csv /data/amciilab/xinran/indiana_cxr/prepared/prepared_data.csv \
    --images_dir /data/amciilab/xinran/indiana_cxr/images/images_normalized \
    --partition_dir /data/amciilab/xinran/indiana_cxr/prepared/partitions \
    --output_dir /data/amciilab/xinran/indiana_cxr/results_v3 \
    --num_rounds 50 \
    --local_epochs 5 \
    --batch_size 32 \
    --lr 1e-3 \
    --embed_dim 256 \
    --patience 10 \
    --seed 42

echo "Experiment $EXP_ID complete: $(date)"
