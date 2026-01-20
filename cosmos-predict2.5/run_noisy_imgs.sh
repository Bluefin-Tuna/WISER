# #!/bin/bash

# # Define base input directory and output directory
# INPUT_DIR="assets/noisy"
# OUTPUT_DIR="outputs/video_world/ablation"

# # Define the input variants
# INPUTS=("robot_pouring_occlusion.json" "robot_pouring_gaussian.json" "robot_pouring_glare.json")

# # Run each configuration
# for INPUT_FILE in "${INPUTS[@]}"; do
#     BASE_NAME=$(basename "$INPUT_FILE" .json)
#     echo "=== Running inference for $BASE_NAME ==="
    
#     python examples/inference.py \
#         -i "$INPUT_DIR/$INPUT_FILE" \
#         -o "$OUTPUT_DIR/$BASE_NAME"
# done

# uv run python evaluate.py \
# --mode custom_input \
# --prompt_file /storage/home/hcoda1/6/gzollicoffer3/scratch/physical-ai-bench/custom_vids/eval_vids_rl.json \
# --dimension aesthetic_quality background_consistency imaging_quality motion_smoothness overall_consistency subject_consistency i2v_background i2v_subject \
# --videos_path /storage/home/hcoda1/6/gzollicoffer3/scratch/physical-ai-bench/eval_vids \
# --output_path ./evaluation_results/

# uv run python evaluate.py \
# --mode custom_input \
# --prompt_file /storage/home/hcoda1/6/gzollicoffer3/scratch/physical-ai-bench/eval_vids/splits/json/robot_pouring_chrome_basemodel.json \
# --dimension aesthetic_quality background_consistency imaging_quality motion_smoothness overall_consistency subject_consistency i2v_background i2v_subject \
# --custom_image_folder /storage/home/hcoda1/6/gzollicoffer3/scratch/physical-ai-bench/custom_img/ \
# --videos_path /storage/home/hcoda1/6/gzollicoffer3/scratch/physical-ai-bench/eval_vids/splits/robot_pouring_chrome_basemodel \
# --output_path ./evaluation_results/

uv run python evaluate.py \
--mode custom_input \
--prompt_file /storage/home/hcoda1/6/gzollicoffer3/scratch/physical-ai-bench/custom_vids/eval_vids_rl.json \
--custom_image_folder /storage/home/hcoda1/6/gzollicoffer3/scratch/physical-ai-bench/custom_img/ \
--dimension aesthetic_quality background_consistency imaging_quality motion_smoothness overall_consistency subject_consistency i2v_background i2v_subject \
--videos_path /storage/home/hcoda1/6/gzollicoffer3/scratch/physical-ai-bench/eval_vids \
--output_path ./evaluation_results/