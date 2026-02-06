#!/bin/bash
cd ..
set -e

PORT=2000
GPU=0

# Base checkpoints for augment_high
declare -A BASE_CHECKPOINTS_AUGMENT_HIGH=(
  ["carla_four_lane"]="./logdir/carla_four_lane_sensor_6_augment_high/checkpoint.ckpt"
  ["carla_right_turn_simple"]="./logdir/carla_right_turn_simple_sensor_6_augment_high/checkpoint.ckpt"
  ["carla_stop_sign"]="./logdir/carla_stop_sign_sensor_6_augment_high/checkpoint.ckpt"
)

# Base checkpoints for dropout
declare -A BASE_CHECKPOINTS_DROPOUT=(
  ["carla_four_lane"]="./logdir/carla_four_lane_sensor_6_dropout/checkpoint.ckpt"
  ["carla_right_turn_simple"]="./logdir/carla_right_turn_simple_sensor_6_dropout/checkpoint.ckpt"
  ["carla_stop_sign"]="./logdir/carla_stop_sign_sensor_6_dropout/checkpoint.ckpt"
)

# Base checkpoints for default
declare -A BASE_CHECKPOINTS_DEFAULT=(
  ["carla_four_lane"]="./logdir/carla_four_lane_sensor_6/checkpoint.ckpt"
  ["carla_right_turn_simple"]="./logdir/carla_right_turn_simple_sensor_6/checkpoint.ckpt"
  ["carla_stop_sign"]="./logdir/carla_stop_sign_sensor_6/checkpoint.ckpt"
)

# Scenarios
SCENARIOS=("carla_four_lane" "carla_right_turn_simple" "carla_stop_sign")
AUG_TYPES=("jitter" "glare" "gaussian" "occlusion" "lag")
AUG_TYPES=("lag")
AUG_LEVELS=(1 2 3 4 5)
run_eval() {
  local checkpoint="$1"
  local variant="$2"
  local scenario="$3"
  echo "Running: $variant on $scenario"
  bash eval_dm3_sequential.sh "$PORT" "$GPU" "$checkpoint" "$variant" "$scenario"
}

echo "=== Running Augment High Default ==="
for scenario in "${SCENARIOS[@]}"; do
  run_eval "${BASE_CHECKPOINTS_AUGMENT_HIGH[$scenario]}" "Default" "$scenario"
done

echo "=== Running Augment High Augmentations ==="
for scenario in "${SCENARIOS[@]}"; do
  for aug in "${AUG_TYPES[@]}"; do
    for level in "${AUG_LEVELS[@]}"; do
      run_eval "${BASE_CHECKPOINTS_AUGMENT_HIGH[$scenario]}" "${aug}_${level}" "$scenario"
    done
  done
done

echo "=== Running RME ==="
for scenario in "${SCENARIOS[@]}"; do
  run_eval "${BASE_CHECKPOINTS_DROPOUT[$scenario]}" "random_Default" "$scenario"
done

echo "=== Running RME Augmentations ==="
for scenario in "${SCENARIOS[@]}"; do
  for aug in "${AUG_TYPES[@]}"; do
    for level in "${AUG_LEVELS[@]}"; do
      run_eval "${BASE_CHECKPOINTS_DROPOUT[$scenario]}" "random_${aug}_${level}" "$scenario"
    done
  done
done

echo "=== Running Default ==="
for scenario in "${SCENARIOS[@]}"; do
  run_eval "${BASE_CHECKPOINTS_DEFAULT[$scenario]}" "Default" "$scenario"
done

echo "=== Running Default Augmentations ==="
for scenario in "${SCENARIOS[@]}"; do
  for aug in "${AUG_TYPES[@]}"; do
    for level in "${AUG_LEVELS[@]}"; do
      run_eval "${BASE_CHECKPOINTS_DEFAULT[$scenario]}" "${aug}_${level}" "$scenario"
    done
  done
done

echo "=== Running Surprise ==="
for scenario in "${SCENARIOS[@]}"; do
  run_eval "${BASE_CHECKPOINTS_DROPOUT[$scenario]}" "surprise_Default" "$scenario"
done

echo "=== Running Surprise Augmentations ==="
for scenario in "${SCENARIOS[@]}"; do
  for aug in "${AUG_TYPES[@]}"; do
    for level in "${AUG_LEVELS[@]}"; do
      run_eval "${BASE_CHECKPOINTS_DROPOUT[$scenario]}" "surprise_${aug}_${level}" "$scenario"
    done
  done
done


echo "=== Running Surprise Full ==="
for scenario in "${SCENARIOS[@]}"; do
  run_eval "${BASE_CHECKPOINTS_DROPOUT[$scenario]}" "surprise_full_Default" "$scenario"
done

echo "=== Running Surprise Full Augmentations ==="
for scenario in "${SCENARIOS[@]}"; do
  for aug in "${AUG_TYPES[@]}"; do
    for level in "${AUG_LEVELS[@]}"; do
      run_eval "${BASE_CHECKPOINTS_DROPOUT[$scenario]}" "surprise_full_${aug}_${level}" "$scenario"
    done
  done
done

# bash eval_dm3_sequential.sh 2000 0 ./logdir/carla_four_lane_sensor_6_dropout/checkpoint.ckpt surprise_jitter_3 carla_four_lane

