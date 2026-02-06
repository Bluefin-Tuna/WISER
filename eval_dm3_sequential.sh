#!/bin/bash

if [ $# -lt 5 ]; then
    echo "Usage: $0 <carla_port> <gpu_device> <checkpoint_path> <mode> <task>"
    exit 1
fi

CARLA_PORT=$1
GPU_DEVICE=$2
CHECKPOINT_PATH=$3
MODE=$4
TASK=$5

DIR_NAME=$(basename "$(dirname "$CHECKPOINT_PATH")")
LOG_FILE="logdir/eval/eval_log_${CARLA_PORT}_${MODE}.log"
LOG_DIR="logdir/eval/${DIR_NAME}_${TASK}_${MODE}"

CARLA_SERVER_COMMAND="$CARLA_ROOT/CarlaUE4.sh -RenderOffScreen -carla-port=$CARLA_PORT -benchmark -fps=10"
EVAL_SCRIPT="dreamerv3/eval.py"
COMMON_PARAMS="--env.world.carla_port $CARLA_PORT --dreamerv3.jax.policy_devices $GPU_DEVICE --dreamerv3.run.from_checkpoint $CHECKPOINT_PATH --dreamerv3.run.mode $MODE --env.mode $MODE --dreamerv3.logdir $LOG_DIR --task $TASK"
ADDITIONAL_PARAMS="${@:6}"
EVAL_COMMAND="python -u $EVAL_SCRIPT $COMMON_PARAMS $ADDITIONAL_PARAMS"

> $LOG_FILE

log_with_timestamp() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1" | tee -a $LOG_FILE
}

launch_carla() {
    pkill -f "CarlaUE4.*$CARLA_PORT" >/dev/null 2>&1
    log_with_timestamp "Starting CARLA on port $CARLA_PORT..."
    CUDA_VISIBLE_DEVICES=$GPU_DEVICE $CARLA_SERVER_COMMAND &
    CARLA_PID=$!
    while ! nc -z localhost $CARLA_PORT; do
        log_with_timestamp "Waiting for CARLA to start on port $CARLA_PORT..."
        sleep 3
    done
    log_with_timestamp "CARLA started (PID $CARLA_PID)"
}

cleanup() {
    log_with_timestamp "Cleaning up CARLA and eval..."
    pkill -f "CarlaUE4.*$CARLA_PORT" >/dev/null 2>&1
    if [ -n "$EVAL_PID" ]; then
        kill -TERM $EVAL_PID >/dev/null 2>&1
        wait $EVAL_PID 2>/dev/null
    fi
}

trap cleanup EXIT

log_with_timestamp "Starting eval..."
log_with_timestamp "Log file at ~/home/tchopra32/Programming/carla/CarDreamerRepo/$LOG_FILE"
launch_carla
sleep 5

$EVAL_COMMAND >> $LOG_FILE 2>&1 &
EVAL_PID=$!

wait $EVAL_PID
cleanup
log_with_timestamp "Eval finished on port $CARLA_PORT."
