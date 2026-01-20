import cv2
import numpy as np
from pathlib import Path
from typing import Callable, List, Optional

def noise_video(
    input_path: str,
    output_path: str,
    noise_type: str,
    proportion: float = 1.0,
    noise_intensity: float = 1.0,
    seed: Optional[int] = None
) -> None:
    """
    Apply noise transformation to a portion of an MP4 video.
    
    Parameters:
    -----------
    input_path : str
        Path to input MP4 video
    output_path : str
        Path to output MP4 video
    noise_type : str
        Type of noise to apply. Options: 'jitter', 'glare', 'gaussian', 
        'occlusion', 'channelswap', 'lag', 'chromatic'
    proportion : float
        Proportion of frames to corrupt (0.0 to 1.0). Default: 1.0
    noise_intensity : float
        Intensity of the noise effect. Default: 1.0
    seed : Optional[int]
        Random seed for reproducibility. Default: None
    
    Example:
    --------
    noise_video('input.mp4', 'output.mp4', 'gaussian', proportion=0.5, noise_intensity=0.8)
    """
    if seed is not None:
        np.random.seed(seed)
    
    # Open video
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        raise ValueError(f"Could not open video: {input_path}")
    
    # Get video properties
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Setup video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    # Determine which frames to corrupt
    frames_to_corrupt = int(total_frames * proportion)
    corrupt_indices = set(np.random.choice(total_frames, frames_to_corrupt, replace=False))
    
    # Lag buffer for lag_sensor noise
    lag_buffer = []
    
    # Define noise functions
    def apply_jitter(frame, noise_intensity):
        mu = 20 * noise_intensity
        std = 20 * noise_intensity
        img = frame.astype(np.float32)
        brightness = np.random.uniform(mu, std)
        contrast = np.random.uniform(mu, std)
        img = np.clip(img * contrast + brightness * 10, 0, 255).astype(np.uint8)
        return img
    
    def apply_glare(frame, noise_intensity):
        mu = 20 * noise_intensity
        std = 20 * noise_intensity
        img = frame.astype(np.float32)
        brightness = np.random.uniform(mu, std)
        img = np.clip(img * 0 + brightness * 10, 0, 255).astype(np.uint8)
        return img
    
    def apply_gaussian(frame, noise_intensity):
        mu = 20 * noise_intensity
        std = 20 * noise_intensity
        noise = np.random.normal(mu, std, frame.shape).astype(np.float32)
        return np.clip(frame.astype(np.float32) + noise, 0, 255).astype(np.uint8)
    
    def apply_occlusion(frame, noise_intensity):
        mu = 35 * noise_intensity
        std = 40 * noise_intensity
        img = frame.copy()
        h, w, _ = img.shape
        x, y = np.random.randint(0, w//2), np.random.randint(0, h//2)
        mask_w = np.random.randint(int(mu), int(std) + 1)
        mask_h = np.random.randint(int(mu), int(std) + 1)
        img[y:y+mask_h, x:x+mask_w] = 0
        return img
    
    def apply_channelswap(frame, noise_intensity):
        return frame[..., np.random.permutation(3)]
    
    def apply_lag_sensor(frame, noise_intensity):
        if len(lag_buffer) > 1:
            max_lag = min(75, len(lag_buffer) - 1)
            lag_steps = np.random.randint(1, max_lag + 1)
            return lag_buffer[-lag_steps].copy()
        return frame
    
    def apply_chromatic_aberration(frame, noise_intensity):
        img = frame.astype(np.float32)
        h, w, _ = img.shape
        max_shift = int(2 + 10 * noise_intensity)
        
        shift_r = np.random.randint(-max_shift, max_shift + 1, size=2)
        shift_g = np.random.randint(-max_shift, max_shift + 1, size=2)
        shift_b = np.random.randint(-max_shift, max_shift + 1, size=2)
        
        def translate(channel, dx, dy):
            M = np.float32([[1, 0, dx], [0, 1, dy]])
            return cv2.warpAffine(channel, M, (w, h), borderMode=cv2.BORDER_REFLECT)
        
        r = translate(img[:, :, 0], *shift_r)
        g = translate(img[:, :, 1], *shift_g)
        b = translate(img[:, :, 2], *shift_b)
        
        merged = np.stack([r, g, b], axis=2)
        return np.clip(merged, 0, 255).astype(np.uint8)
    
    # Map noise type to function
    noise_funcs = {
        'jitter': apply_jitter,
        'glare': apply_glare,
        'gaussian': apply_gaussian,
        'occlusion': apply_occlusion,
        'channelswap': apply_channelswap,
        'lag': apply_lag_sensor,
        'chrome': apply_chromatic_aberration
    }
    
    if noise_type not in noise_funcs:
        raise ValueError(f"Unknown noise type: {noise_type}. Choose from {list(noise_funcs.keys())}")
    
    noise_func = noise_funcs[noise_type]
    
    # Process video frame by frame
    frame_idx = 0
    print(f"Processing {total_frames} frames...")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Update lag buffer
        if noise_type == 'lag':
            lag_buffer.append(frame.copy())
            if len(lag_buffer) > 100:  # Keep buffer size reasonable
                lag_buffer.pop(0)
        
        # Apply noise if this frame is selected
        if frame_idx in corrupt_indices:
            frame = noise_func(frame, noise_intensity)
        
        out.write(frame)
        frame_idx += 1
        
        if frame_idx % 100 == 0:
            print(f"Processed {frame_idx}/{total_frames} frames")
    
    # Cleanup
    cap.release()
    out.release()
    print(f"Video saved to {output_path}")


# Convenience function to apply multiple noise types
def noise_video_mixed(
    input_path: str,
    output_path: str,
    noise_types: List[str],
    proportion: float = 1.0,
    noise_intensity: float = 1.0,
    seed: Optional[int] = None
) -> None:
    """
    Apply multiple noise types randomly across frames.
    
    Parameters:
    -----------
    input_path : str
        Path to input MP4 video
    output_path : str
        Path to output MP4 video
    noise_types : List[str]
        List of noise types to randomly apply
    proportion : float
        Proportion of frames to corrupt (0.0 to 1.0)
    noise_intensity : float
        Intensity of the noise effects
    seed : Optional[int]
        Random seed for reproducibility
    """
    if seed is not None:
        np.random.seed(seed)
    
    # This would randomly select a noise type for each corrupted frame
    # Implementation similar to above but with random noise selection
    pass


if __name__ == "__main__":
    # Example usage
    noise_video(
        input_path='/home/general/Documents/work/misc/robot_pouring.mp4',
        output_path='robot_pouring_gaussian.mp4',
        noise_type='gaussian',
        proportion=0.75,  # Corrupt 50% of frames
        noise_intensity=5,  # 80% intensity
        seed=42)


# Apply Gaussian noise to 30% of frames at 50% intensity
# noise_video('input.mp4', 'output.mp4', 'gaussian', proportion=0.3, noise_intensity=0.5)

# Apply chromatic aberration to all frames at full intensity
# noise_video('input.mp4', 'output.mp4', 'chromatic', proportion=1.0, noise_intensity=1.0)

# Apply occlusion to 20% of frames at 70% intensity
# noise_video('input.mp4', 'output.mp4', 'occlusion', proportion=0.2, noise_intensity=0.7)