#!/bin/bash
export SDL_AUDIODRIVER=dummy
export AUDIODEV=null

source ~/miniconda3/etc/profile.d/conda.sh
conda activate cardreamer

# Configuration
SCENARIOS=("carla_stop_sign" "carla_right_turn_simple" "carla_four_lane")
# AUG_TYPES=("gaussian" "jitter" "glare" "occlusion")
# AUG_TYPES=("gaussian")
AUG_TYPES=("chrome")
AUG_LEVELS=(0.625 0.75 0.875 1.0)
# AUG_LEVELS=(2.00 3.00 4.00 5.00) #Used to push gaussian and occlusion further
PROPORTION_LEVELS=(0.5 0.625 0.75 0.875)

# SCENARIOS=("carla_stop_sign")
# AUG_TYPES=("jitter")
# AUG_LEVELS=(0.625)
# PROPORTION_LEVELS=(0.5)



declare -A BASE_CHECKPOINTS_DEFAULT_BEV=(
  ["carla_four_lane"]="./logdir/carla_four_lane_bev/checkpoint.ckpt"
  ["carla_right_turn_simple"]="./logdir/carla_right_turn_simple_bev/checkpoint.ckpt"
  ["carla_stop_sign"]="./logdir/carla_stop_sign_bev/checkpoint.ckpt"
)

BASE_DIR="/coc/flash5/gzollicoffer3/SafeDreamer/CarDreamerRepo/logdir/evals_finegrained"
PORT=2000
GPU=0

cd /coc/flash5/gzollicoffer3/SafeDreamer/CarDreamerRepo
export PYTHONPATH="/coc/flash5/gzollicoffer3/SafeDreamer/CarDreamerRepo"
export CARLA_ROOT="/coc/flash5/gzollicoffer3/SafeDreamer/CarDreamerRepo"

mkdir -p logs

is_folder_incomplete() {
    local folder=$1
    
    # Folder doesn't exist
    if [ ! -d "$folder" ]; then
        return 0  # True - incomplete
    fi
    
    # Folder exists but is empty
    if [ -z "$(ls -A "$folder" 2>/dev/null)" ]; then
        return 0  # True - incomplete
    fi
    
    # Check metrics.jsonl exists and has at least 30 lines
    local metrics_file="$folder/metrics.jsonl"
    if [ ! -f "$metrics_file" ]; then
        return 0  # True - incomplete
    fi

    local line_count
    line_count=$(wc -l < "$metrics_file" 2>/dev/null)
    if [ "$line_count" -lt 30 ]; then
        return 0  # True - incomplete
    fi
    
    return 1  # False - complete
}

echo "=== Checking for missing or empty folders ==="

total_expected=$((${#SCENARIOS[@]} * ${#AUG_TYPES[@]} * ${#AUG_LEVELS[@]} * ${#PROPORTION_LEVELS[@]}))
found_count=0
missing_count=0
resubmit_count=0

for scenario in "${SCENARIOS[@]}"; do
  for aug in "${AUG_TYPES[@]}"; do
    for level in "${AUG_LEVELS[@]}"; do
      for proport in "${PROPORTION_LEVELS[@]}"; do
        checkpoint="${BASE_CHECKPOINTS_DEFAULT_BEV[$scenario]}"
        dir_name=$(basename "$(dirname "$checkpoint")")
        variant="${aug}_reject5_proportion${proport}_timestep10_${level}"
        job_name="${variant}_${scenario}"
        expected_folder="${BASE_DIR}/${dir_name}_${scenario}_${variant}"
        
        if is_folder_incomplete "$expected_folder"; then
          ((missing_count++))
          echo "⟳ Resubmitting: $job_name"
          
          sbatch --job-name="$job_name" \
                 --output="logs/${job_name}.out" \
                 --nodes=1 \
                 -A ei-lab \
                 -p overcap \
                 --time=00:20:00 \
                 --gpus='a40:1' \
                 --exclusive \
                 --wrap="cd /coc/flash5/gzollicoffer3/SafeDreamer/CarDreamerRepo && bash eval_dm3_sequential_jobs.sh $PORT $GPU $checkpoint $variant $scenario"
          
          ((resubmit_count++))
        else
          ((found_count++))
          echo "✓ Complete: $job_name"
        fi
      done
    done
  done
done

echo ""
echo "========================================"
echo "Summary:"
echo "  Total expected: $total_expected"
echo "  Complete: $found_count"
echo "  Missing/Empty: $missing_count"
echo "  Resubmitted: $resubmit_count"
echo "========================================"