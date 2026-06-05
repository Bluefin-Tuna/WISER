#!/bin/bash
cd ..
set -e

PORT=2000
GPU=0


# Base checkpoints for default
declare -A BASE_CHECKPOINTS_DEFAULT_BEV=(
  ["carla_four_lane"]="./logdir/carla_four_lane_bev/checkpoint.ckpt"
  ["carla_right_turn_simple"]="./logdir/carla_right_turn_simple_bev/checkpoint.ckpt"
  ["carla_stop_sign"]="./logdir/carla_stop_sign_bev/checkpoint.ckpt"
)

# Scenarios
SCENARIOS=("carla_four_lane")
AUG_TYPES=("jitter" "gaussian" "occlusion" "glare" "chrome")
PROPORTION_LEVELS=(0.75)

AUG_LEVELS=(0.625 0.75) # Gaussian deterioration is seen at higher levels.

run_eval() {
  local checkpoint="$1"
  local variant="$2"
  local scenario="$3"
  echo "Running: $variant on $scenario"
  bash eval_dm3_sequential.sh "$PORT" "$GPU" "$checkpoint" "$variant" "$scenario"
}


echo "=== Running Reject Augmentations ==="
for scenario in "${SCENARIOS[@]}"; do
  for aug in "${AUG_TYPES[@]}"; do
    for level in "${AUG_LEVELS[@]}"; do
      for proport in "${PROPORTION_LEVELS[@]}"; do
        run_eval "${BASE_CHECKPOINTS_DEFAULT_BEV[$scenario]}" "${aug}_reject_proportion${proport}_timestep10_${level}" "$scenario"
      done
    done
  done
done


echo "=== Running Default Augmentations ==="
for scenario in "${SCENARIOS[@]}"; do
  for aug in "${AUG_TYPES[@]}"; do
    for level in "${AUG_LEVELS[@]}"; do
      for proport in "${PROPORTION_LEVELS[@]}"; do
        run_eval "${BASE_CHECKPOINTS_DEFAULT_BEV[$scenario]}" "${aug}_proportion${proport}_timestep10_${level}" "$scenario"
      done
    done
  done
done

echo "=== Running Filter Augmentations ==="
for scenario in "${SCENARIOS[@]}"; do
  for aug in "${AUG_TYPES[@]}"; do
    for level in "${AUG_LEVELS[@]}"; do
      for proport in "${PROPORTION_LEVELS[@]}"; do
        run_eval "${BASE_CHECKPOINTS_DEFAULT_BEV[$scenario]}" "${aug}_filter_proportion${proport}_timestep10_${level}" "$scenario"
      done
    done
  done
done


