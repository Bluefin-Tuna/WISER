import cv2
import os

# Path to your video file
video_path = "/storage/home/hcoda1/6/gzollicoffer3/scratch/cosmos-predict2.5/assets/base/robot_pouring.mp4"

# Output image path
output_path = os.path.splitext(video_path)[0] + "_last_frame.jpg"

# Open video
cap = cv2.VideoCapture(video_path)
if not cap.isOpened():
    raise IOError(f"Cannot open video: {video_path}")

# Get total frame count
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

# Set position to the last frame
cap.set(cv2.CAP_PROP_POS_FRAMES, total_frames - 1)

# Read the last frame
ret, frame = cap.read()
if not ret:
    raise ValueError("Failed to read the last frame from the video.")

# Save the frame as an image
cv2.imwrite(output_path, frame)

print(f"Last frame saved to: {output_path}")

cap.release()
