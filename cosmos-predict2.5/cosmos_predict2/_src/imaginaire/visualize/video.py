# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import IO, Any, Union

import cv2
import numpy as np
import torch
from einops import rearrange
from PIL import Image as PILImage
from torch import Tensor
import os

from cosmos_predict2._src.imaginaire.utils import log
from cosmos_predict2._src.imaginaire.utils.easy_io import easy_io

try:
    import ffmpegcv
except Exception as e:  # ImportError cannot catch all problems
    log.info(e)
    ffmpegcv = None


def save_video(grid, video_name, fps=30):
    grid = (grid * 255).astype(np.uint8)
    grid = np.transpose(grid, (1, 2, 3, 0))
    with ffmpegcv.VideoWriter(video_name, "h264", fps) as writer:
        for frame in grid:
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

            writer.write(frame)


def save_img_or_video(
    sample_C_T_H_W_in01: Tensor, save_fp_wo_ext: Union[str, IO[Any]], fps: int = 24, quality=None, ffmpeg_params=None
) -> None:
    """
    Save a tensor as an image or video file based on shape

        Args:
        sample_C_T_H_W_in01 (Tensor): Input tensor with shape (C, T, H, W) in [0, 1] range.
        save_fp_wo_ext (Union[str, IO[Any]]): File path without extension or file-like object.
        fps (int): Frames per second for video. Default is 24.
    """
    assert sample_C_T_H_W_in01.ndim == 4, "Only support 4D tensor"
    assert isinstance(save_fp_wo_ext, str) or hasattr(save_fp_wo_ext, "write"), (
        "save_fp_wo_ext must be a string or file-like object"
    )

    if torch.is_floating_point(sample_C_T_H_W_in01):
        sample_C_T_H_W_in01 = sample_C_T_H_W_in01.clamp(0, 1)
    else:
        assert sample_C_T_H_W_in01.dtype == torch.uint8, "Only support uint8 tensor"
        sample_C_T_H_W_in01 = sample_C_T_H_W_in01.float().div(255)

    kwargs = {}
    if quality is not None:
        kwargs["quality"] = quality
    if ffmpeg_params is not None:
        kwargs["ffmpeg_params"] = ffmpeg_params

    if sample_C_T_H_W_in01.shape[1] == 1:
        save_obj = PILImage.fromarray(
            rearrange((sample_C_T_H_W_in01.cpu().float().numpy() * 255), "c 1 h w -> h w c").astype(np.uint8),
            mode="RGB",
        )
        ext = ".jpg" if isinstance(save_fp_wo_ext, str) else ""
        easy_io.dump(
            save_obj,
            f"{save_fp_wo_ext}{ext}" if isinstance(save_fp_wo_ext, str) else save_fp_wo_ext,
            file_format="jpg",
            format="JPEG",
            quality=85,
            **kwargs,
        )
    else:
        save_obj = rearrange((sample_C_T_H_W_in01.cpu().float().numpy() * 255), "c t h w -> t h w c").astype(np.uint8)
        ext = ".mp4" if isinstance(save_fp_wo_ext, str) else ""
        easy_io.dump(
            save_obj,
            f"{save_fp_wo_ext}{ext}" if isinstance(save_fp_wo_ext, str) else save_fp_wo_ext,
            file_format="mp4",
            format="mp4",
            fps=fps,
            **kwargs,
        )

def compute_frame_mse(video_C_T_H_W_in01: Tensor, frame1_idx: int = 0, frame2_idx: int = 1) -> float:
    """
    Compute MSE between two frames in a video tensor.
    
    Args:
        video_C_T_H_W_in01 (Tensor): Video tensor with shape (C, T, H, W) in [0, 1] range.
        frame1_idx (int): Index of first frame. Default is 0.
        frame2_idx (int): Index of second frame. Default is 1.
    
    Returns:
        float: MSE between the two frames
    """
    assert video_C_T_H_W_in01.ndim == 4, "Only support 4D tensor"
    assert frame1_idx < video_C_T_H_W_in01.shape[1], f"frame1_idx {frame1_idx} out of range"
    assert frame2_idx < video_C_T_H_W_in01.shape[1], f"frame2_idx {frame2_idx} out of range"
    
    frame1 = video_C_T_H_W_in01[:, frame1_idx, :, :]  # C, H, W
    frame2 = video_C_T_H_W_in01[:, frame2_idx, :, :]  # C, H, W
    mse = torch.mean((frame1 - frame2) ** 2).item()
    
    return mse

def save_frame_comparison(
    video_C_T_H_W_in01: Tensor,
    save_dir: str,
    frame1_idx: int = 0,
    frame2_idx: int = 1,
    prefix: str = "frame"
) -> None:
    """
    Save two frames from a video for comparison.
    
    Args:
        video_C_T_H_W_in01 (Tensor): Video tensor with shape (C, T, H, W) in [0, 1] range.
        save_dir (str): Directory to save frames.
        frame1_idx (int): Index of first frame.
        frame2_idx (int): Index of second frame.
        prefix (str): Prefix for saved filenames.
    """
    if torch.is_floating_point(video_C_T_H_W_in01):
        video_C_T_H_W_in01 = video_C_T_H_W_in01.clamp(0, 1)
    else:
        video_C_T_H_W_in01 = video_C_T_H_W_in01.float().div(255)
    
    os.makedirs(save_dir, exist_ok=True)
    
    # Save frame 1
    frame1 = video_C_T_H_W_in01[:, frame1_idx, :, :]
    frame1_img = PILImage.fromarray(
        rearrange((frame1.cpu().float().numpy() * 255), "c h w -> h w c").astype(np.uint8),
        mode="RGB",
    )
    frame1_img.save(os.path.join(save_dir, f"{prefix}_{frame1_idx}.jpg"), quality=95)
    
    # Save frame 2
    frame2 = video_C_T_H_W_in01[:, frame2_idx, :, :]
    frame2_img = PILImage.fromarray(
        rearrange((frame2.cpu().float().numpy() * 255), "c h w -> h w c").astype(np.uint8),
        mode="RGB",
    )
    frame2_img.save(os.path.join(save_dir, f"{prefix}_{frame2_idx}.jpg"), quality=95)