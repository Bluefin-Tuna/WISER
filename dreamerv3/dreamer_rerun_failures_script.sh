#!/bin/bash

# Script to find failed CARLA experiments and generate rerun script

LOG_DIR="/coc/flash5/gzollicoffer3/SafeDreamer/CarDreamerRepo/logs"
FAILED_LIST="failed_experiments.txt"
RERUN_SCRIPT="rerun_failed_experiments.sh"

echo "=== Scanning logs for CARLA startup failures ==="

# Clear previous results
> "$FAILED_LIST"

# Search for the failure pattern in all log files
for logfile in "$LOG_DIR"/*.out; do
    if grep -q "ERROR: CARLA failed to start within 60s" "$logfile"; then
        # Extract just the filename without path and extension
        basename "$logfile" .out >> "$FAILED_LIST"
    fi
done

# Count failures
failure_count=$(wc -l < "$FAILED_LIST")
echo "Found $failure_count failed experiments"

if [ "$failure_count" -eq 0 ]; then
    echo "No failed experiments found. Exiting."
    exit 0
fi

echo ""
echo "Failed experiments:"
cat "$FAILED_LIST"
echo ""

# Generate the rerun script
echo "=== Generating rerun script: $RERUN_SCRIPT ==="

cat > "$RERUN_SCRIPT" << 'SCRIPT_HEADER'

SCRIPT_HEADER

# Parse each failed experiment and generate sbatch commands
while IFS= read -r job_name; do
    # Parse the job name to extract parameters
    # Format: {aug}_filter_proportion{proport}_timestep10_{level}_{scenario}
    
    # Extract scenario (last part after final underscore combination)
    if [[ "$job_name" =~ _carla_four_lane$ ]]; then
        scenario="carla_four_lane"
        variant="${job_name%_carla_four_lane}"
    elif [[ "$job_name" =~ _carla_right_turn_simple$ ]]; then
        scenario="carla_right_turn_simple"
        variant="${job_name%_carla_right_turn_simple}"
    elif [[ "$job_name" =~ _carla_stop_sign$ ]]; then
        scenario="carla_stop_sign"
        variant="${job_name%_carla_stop_sign}"
    else
        echo "Warning: Could not parse scenario from $job_name, skipping"
        continue
    fi
    
    # Add sbatch command to rerun script
    cat >> "$RERUN_SCRIPT" << SCRIPT_BODY

checkpoint="\${BASE_CHECKPOINTS_DEFAULT_BEV[$scenario]}"
variant="$variant"
job_name="$job_name"

echo "Resubmitting job \$job_name with PORT=\$PORT"

sbatch --job-name="\$job_name" \\
       --output="logs/\${job_name}.out" \\
       --nodes=1 \\
       -A ei-lab \\
       -p overcap \\
       --time=00:30:00 \\
       --gpus='a40:1' \\
       --exclusive \\
       --wrap="cd /coc/flash5/gzollicoffer3/SafeDreamer/CarDreamerRepo && bash eval_dm3_sequential_jobs.sh \$PORT \$GPU \$checkpoint \$variant $scenario"

SCRIPT_BODY

done < "$FAILED_LIST"

# Add footer
cat >> "$RERUN_SCRIPT" << 'SCRIPT_FOOTER'

echo "All failed experiments resubmitted"
SCRIPT_FOOTER

# Make the rerun script executable
chmod +x "$RERUN_SCRIPT"

echo ""
echo "=== Summary ==="
echo "Failed experiments list saved to: $FAILED_LIST"
echo "Rerun script generated: $RERUN_SCRIPT"
echo ""
echo "To resubmit the failed experiments, run:"
echo "  bash $RERUN_SCRIPT"