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
NOMINAL_POLICY="./logdir/carla_stop_sign_bev/checkpoint.ckpt"
MANDATED_POLICY="./logdir/carla_right_turn_single_Town04/checkpoint_38000.ckpt"
SCENARIOS=("carla_right_turn_simple")
AUG_TYPES=("Default")
PROPORTION_LEVELS=(0.75)

AUG_LEVELS=(1.0)

run_eval() {
  local checkpoint="$1" #Cross Intersection
  local variant="$2"
  local scenario="$3"
  local checkpoint_2="$4" #Right Turn 
  echo "Running: $variant on $scenario"
  bash eval_dm3_sequential_switch.sh "$PORT" "$GPU" "$checkpoint" "$checkpoint_2" "$variant" "$scenario"
}

echo "=== Running Default ==="
for scenario in "${SCENARIOS[@]}"; do
  run_eval "$NOMINAL_POLICY" "Default" "$scenario" "$MANDATED_POLICY"
done
