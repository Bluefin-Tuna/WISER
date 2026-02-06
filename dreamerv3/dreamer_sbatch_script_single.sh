#!/bin/bash
#SBATCH -p ei-lab
#SBATCH -A ei-lab
#SBATCH --gpus=a40:1
#SBATCH --time=00:30:00
#SBATCH --exclude=robby

set -euo pipefail

cd /coc/flash5/gzollicoffer3/SafeDreamer/CarDreamerRepo
export PYTHONPATH="/coc/flash5/gzollicoffer3/SafeDreamer/CarDreamerRepo"
export CARLA_ROOT="/coc/flash5/gzollicoffer3/carla"
export SDL_AUDIODRIVER=dummy
export AUDIODEV=null

source ~/miniconda3/etc/profile.d/conda.sh
conda activate cardreamer

GPU=0
PORT=2000  # Same port for all jobs - they're on different nodes!

declare -A BASE_CHECKPOINTS_DEFAULT_BEV=(
  ["carla_four_lane"]="./logdir/carla_four_lane_bev/checkpoint.ckpt"
  ["carla_right_turn_simple"]="./logdir/carla_right_turn_simple_bev/checkpoint.ckpt"
  ["carla_stop_sign"]="./logdir/carla_stop_sign_bev/checkpoint.ckpt"
)

SCENARIOS=("carla_stop_sign" "carla_right_turn_simple" "carla_four_lane")
AUG_TYPES=("gaussian" "jitter" "glare" "occlusion")
AUG_LEVELS=(0.625 0.75 0.875 1.0)
PROPORTION_LEVELS=(0.5 0.625 0.75 0.875)

mkdir -p logs

echo "=== Submitting Default Augmentation jobs to SLURM ==="

for scenario in "${SCENARIOS[@]}"; do
  for aug in "${AUG_TYPES[@]}"; do
    for level in "${AUG_LEVELS[@]}"; do
      for proport in "${PROPORTION_LEVELS[@]}"; do
        checkpoint="${BASE_CHECKPOINTS_DEFAULT_BEV[$scenario]}"
        variant="${aug}_filter_proportion${proport}_timestep10_${level}"
        job_name="${variant}_${scenario}"
        
        echo "Submitting job $job_name with PORT=$PORT"
        #  --gpus=a40:1 \
        
        sbatch --job-name="$job_name" \
               --output="logs/${job_name}.out" \
               --nodes=1 \
               -A ei-lab \
               -p overcap \
               --time=00:30:00 \
               --gpus='a40:1' \
               --exclusive \
               --wrap="cd /coc/flash5/gzollicoffer3/SafeDreamer/CarDreamerRepo && bash eval_dm3_sequential_jobs.sh $PORT $GPU $checkpoint $variant $scenario"
      done
    done
  done
done

echo "All jobs submitted"
# Configs
# SCENARIOS=("carla_stop_sign" "carla_right_turn_simple" "carla_four_lane")
# AUG_TYPES=("gaussian" "jitter" "glare" "occlusion")
# AUG_LEVELS=(0.625 0.75 0.875 1.0)
# PROPORTION_LEVELS=(0.5 0.625 0.75 0.875)