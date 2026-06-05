#!/bin/bash
# source activate cardreamer
# Setup environment
# Activate conda and set environment
source ~/miniconda3/etc/profile.d/conda.sh
conda activate cardreamer
cd /coc/flash5/gzollicoffer3/SafeDreamer/CarDreamerRepo
export PYTHONPATH="/coc/flash5/gzollicoffer3/SafeDreamer/CarDreamerRepo"
export CARLA_ROOT="/coc/flash5/gzollicoffer3/carla"
export SDL_AUDIODRIVER=dummy
export AUDIODEV=null
rm -rf /tmp/xla_*
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_MEM_FRACTION=.9

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
LOG_FILE="logdir/evals_finegrained/eval_log_${CARLA_PORT}_${MODE}.log"
LOG_DIR="logdir/evals_finegrained/${DIR_NAME}_${TASK}_${MODE}"

CARLA_SERVER_COMMAND="$CARLA_ROOT/CarlaUE4.sh -RenderOffScreen -carla-port=$CARLA_PORT -benchmark -fps=10"
EVAL_SCRIPT="dreamerv3/eval.py"
COMMON_PARAMS="--env.world.carla_port $CARLA_PORT --dreamerv3.jax.policy_devices $GPU_DEVICE --dreamerv3.run.from_checkpoint $CHECKPOINT_PATH --dreamerv3.run.mode $MODE --env.mode $MODE --dreamerv3.logdir $LOG_DIR --task $TASK"
ADDITIONAL_PARAMS="${@:6}"
EVAL_COMMAND="python -u $EVAL_SCRIPT $COMMON_PARAMS $ADDITIONAL_PARAMS"

mkdir -p "$(dirname "$LOG_FILE")"
> "$LOG_FILE"

log_with_timestamp() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1" | tee -a "$LOG_FILE"
}

log_with_timestamp "=== GPU Information ==="
# nvidia-smi | tee -a "$LOG_FILE"
# Kill any existing CARLA process on this port
kill_carla_on_port() {
    local port=$1
    log_with_timestamp "Checking for existing CARLA on port $port..."
    
    # Find process using the port
    local pid=$(lsof -ti:$port 2>/dev/null)
    
    if [ -n "$pid" ]; then
        log_with_timestamp "Found process $pid using port $port, killing it..."
        kill -9 $pid 2>/dev/null
        sleep 2
        
        # Verify it's dead
        if lsof -ti:$port >/dev/null 2>&1; then
            log_with_timestamp "ERROR: Failed to kill process on port $port"
            return 1
        fi
        log_with_timestamp "Successfully killed process on port $port"
    else
        log_with_timestamp "Port $port is free"
    fi
    return 0
}

launch_carla() {
    # Kill any existing process on this port first
    kill_carla_on_port "$CARLA_PORT" || exit 1
    
    log_with_timestamp "Starting CARLA on port $CARLA_PORT..."
    CUDA_VISIBLE_DEVICES=$GPU_DEVICE $CARLA_SERVER_COMMAND >> "$LOG_FILE" 2>&1 &
    CARLA_PID=$!
    
    # Wait for port to become active with timeout
    local timeout=80
    local elapsed=0
    while ! nc -z localhost "$CARLA_PORT"; do
        if [ $elapsed -ge $timeout ]; then
            log_with_timestamp "ERROR: CARLA failed to start within ${timeout}s"
            kill -9 $CARLA_PID 2>/dev/null
            exit 1
        fi
        log_with_timestamp "Waiting for CARLA to start on port $CARLA_PORT... (${elapsed}s/${timeout}s)"
        sleep 3
        elapsed=$((elapsed + 3))
    done
    log_with_timestamp "CARLA started successfully (PID $CARLA_PID)"
}

cleanup() {
    log_with_timestamp "Cleaning up eval..."
    if [ -n "${EVAL_PID:-}" ]; then
        kill -TERM "$EVAL_PID" >/dev/null 2>&1
        wait "$EVAL_PID" 2>/dev/null
    fi
    
    if [ -n "${CARLA_PID:-}" ]; then
        log_with_timestamp "Stopping CARLA (PID $CARLA_PID)"
        kill -TERM "$CARLA_PID" >/dev/null 2>&1
        sleep 2
        # Force kill if still running
        if ps -p $CARLA_PID > /dev/null 2>&1; then
            kill -9 "$CARLA_PID" >/dev/null 2>&1
        fi
        wait "$CARLA_PID" 2>/dev/null
    fi
    
    # Final cleanup - kill any process still on our port
    kill_carla_on_port "$CARLA_PORT" >/dev/null 2>&1
}

trap cleanup EXIT

log_with_timestamp "Starting eval with PORT=$CARLA_PORT GPU=$GPU_DEVICE MODE=$MODE TASK=$TASK"
log_with_timestamp "Log file at $(realpath "$LOG_FILE")"

launch_carla
sleep 5

$EVAL_COMMAND >> "$LOG_FILE" 2>&1 &
EVAL_PID=$!

wait $EVAL_PID
EXIT_CODE=$?

log_with_timestamp "Eval finished on port $CARLA_PORT with exit code $EXIT_CODE."
exit $EXIT_CODE