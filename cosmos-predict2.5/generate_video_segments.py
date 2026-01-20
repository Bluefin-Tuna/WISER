import os
import json
from moviepy.editor import VideoFileClip

# Function to split video into 10 equal segments
def split_video_into_segments(video_path, output_dir, num_segments=20):
    video = VideoFileClip(video_path)
    video_duration = video.duration
    segment_duration = video_duration / num_segments
    video_segments = []

    for i in range(num_segments):
        start_time = i * segment_duration
        end_time = (i + 1) * segment_duration if i < num_segments - 1 else video_duration

        output_video_path = os.path.join(output_dir, f"segment_{i+1}.mp4")
        video.subclip(start_time, end_time).write_videofile(output_video_path, codec="libx264", verbose=False, logger=None)

        video_id = f"{os.path.splitext(os.path.basename(video_path))[0]}_segment_{i+1}"
        segment_data = {
            "video_id": video_id,
            "seed": "12345",
            "video_path": output_video_path,
            "prompt": "A robotic arm, primarily white with black joints and cables, is shown in a clean, modern indoor setting with a white tabletop. The arm, equipped with a gripper holding a small, light green pitcher, is positioned above a clear glass containing a reddish-brown liquid and a spoon. The robotic arm is in the process of pouring a transparent liquid into the glass. To the left of the pitcher, there is an opened jar with a similar reddish-brown substance visible through its transparent body. In the background, a vase with white flowers and a brown couch are partially visible, adding to the contemporary ambiance. The lighting is bright, casting soft shadows on the table. The robotic arm's movements are smooth and controlled, demonstrating precision in its task. As the video progresses, the robotic arm completes the pour, leaving the glass half-filled with the reddish-brown liquid. The jar remains untouched throughout the sequence, and the spoon inside the glass remains stationary. The other robotic arm on the right side also stays stationary throughout the video. The final frame captures the robotic arm with the pitcher finishing the pour, with the glass now filled to a higher level, while the pitcher is slightly tilted but still held securely by the gripper."
        }
        video_segments.append(segment_data)

    video.close()
    return video_segments

def save_video_segments_to_json(video_segments, json_file_path):
    with open(json_file_path, 'w') as f:
        json.dump(video_segments, f, indent=4)

# Paths
video_dir = "/storage/home/hcoda1/6/gzollicoffer3/scratch/physical-ai-bench/eval_vids"
output_base_dir = os.path.join(video_dir, "splits")
json_output_dir = os.path.join(output_base_dir, "json")
os.makedirs(json_output_dir, exist_ok=True)

# Loop through all mp4 videos
for filename in os.listdir(video_dir):
    if filename.endswith(".mp4"):
        video_path = os.path.join(video_dir, filename)
        video_name = os.path.splitext(filename)[0]
        output_dir = os.path.join(output_base_dir, video_name)
        os.makedirs(output_dir, exist_ok=True)

        print(f"Processing {filename} ...")
        video_segments = split_video_into_segments(video_path, output_dir)

        json_file_path = os.path.join(json_output_dir, f"{video_name}.json")
        save_video_segments_to_json(video_segments, json_file_path)
        print(f"✅ Done: {filename} → {json_file_path}")
