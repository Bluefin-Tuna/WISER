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
from pathlib import Path

import numpy as np
import torch

from cosmos_predict2._src.imaginaire.auxiliary.guardrail.common import presets as guardrail_presets
from cosmos_predict2._src.imaginaire.flags import SMOKE
from cosmos_predict2._src.imaginaire.lazy_config.lazy import LazyConfig
from cosmos_predict2._src.imaginaire.utils import distributed, log
from cosmos_predict2._src.imaginaire.visualize.video import save_img_or_video, compute_frame_mse, save_frame_comparison
from cosmos_predict2._src.predict2.inference.video2world import Video2WorldInference
from cosmos_predict2.config import InferenceArguments, SetupArguments, path_to_str


class Inference:
    def __init__(self, args: SetupArguments):
        log.debug(f"{args.__class__.__name__}({args})")

        torch.enable_grad(False)  # Disable gradient calculations for inference

        self.rank0 = distributed.is_rank0()
        self.setup_args = args
        self.offload_diffusion_model = args.offload_diffusion_model
        self.offload_tokenizer = args.offload_tokenizer
        self.offload_text_encoder = args.offload_text_encoder
        self.pipe = Video2WorldInference(
            # pyrefly: ignore  # bad-argument-type
            experiment_name=args.experiment,
            # pyrefly: ignore  # bad-argument-type
            ckpt_path=args.checkpoint_path,
            s3_credential_path="",
            # pyrefly: ignore  # bad-argument-type
            context_parallel_size=args.context_parallel_size,
            config_file=args.config_file,
        )
        if self.rank0:
            args.output_dir.mkdir(parents=True, exist_ok=True)
            config_path = args.output_dir / "config.yaml"
            # pyrefly: ignore  # bad-argument-type
            LazyConfig.save_yaml(self.pipe.config, config_path)
            log.info(f"Saved config to {config_path}")

        self.guardrail_enabled = not args.disable_guardrails

        if self.rank0 and self.guardrail_enabled:
            self.text_guardrail_runner = guardrail_presets.create_text_guardrail_runner(
                offload_model_to_cpu=args.offload_guardrail_models
            )
            self.video_guardrail_runner = guardrail_presets.create_video_guardrail_runner(
                offload_model_to_cpu=args.offload_guardrail_models
            )
        else:
            # pyrefly: ignore  # bad-assignment
            self.text_guardrail_runner = None
            # pyrefly: ignore  # bad-assignment
            self.video_guardrail_runner = None

    def generate(self, samples: list[InferenceArguments], output_dir: Path) -> list[str]:
        if SMOKE:
            samples = samples[:1]

        sample_names = [sample.name for sample in samples]
        log.info(f"Generating {len(samples)} samples: {sample_names}")

        output_paths: list[str] = []
        for i_sample, sample in enumerate(samples):
            log.info(f"[{i_sample + 1}/{len(samples)}] Processing sample {sample.name}")
            output_path = self._generate_sample(sample, output_dir)
            if output_path is not None:
                output_paths.append(output_path)
        return output_paths

    def _generate_sample(self, sample: InferenceArguments, output_dir: Path) -> str | None:
        log.debug(f"{sample.__class__.__name__}({sample})")
        output_path = output_dir / sample.name

        if self.rank0:
            output_dir.mkdir(parents=True, exist_ok=True)
            open(f"{output_path}.json", "w").write(sample.model_dump_json())
            log.info(f"Saved arguments to {output_path}.json")

            # run text guardrail on the prompt
            if self.text_guardrail_runner is not None:
                if not guardrail_presets.run_text_guardrail(sample.prompt, self.text_guardrail_runner):
                    message = f"Guardrail blocked text2world generation. Prompt: {sample.prompt}"
                    log.critical(message)
                    if self.setup_args.keep_going:
                        return None
                    else:
                        raise Exception(message)
                else:
                    log.success("Passed guardrail on prompt")
            elif self.text_guardrail_runner is None:
                log.warning("Guardrail checks on prompt are disabled")
        print('Getting video..')
        noise_reject_alg = False
        
        if noise_reject_alg == True:
            frames_to_extract = 4 * (sample.num_input_frames - 1) + 1
            clean_context = 0
            for i in range(frames_to_extract):
                #Compute M(x)
                video: torch.Tensor = self.pipe.generate_vid2world_compute_Mx(
                    prompt=sample.prompt,
                    input_path=path_to_str(sample.input_path),
                    guidance=sample.guidance,
                    num_video_frames=2,
                    frame_index = i,
                    num_latent_conditional_frames=1,#sample.num_input_frames,
                    resolution=sample.resolution,
                    seed=sample.seed,
                    negative_prompt=sample.negative_prompt,
                    num_steps=1,#This looks to be the diffusion steps..#sample.num_steps,
                    offload_diffusion_model=self.offload_diffusion_model,
                    offload_text_encoder=self.offload_text_encoder,
                    offload_tokenizer=self.offload_tokenizer,
                )
                #Select this as starting frame if M passes.:
                if self.rank0:
                    video = (1.0 + video[0]) / 2

                # save_img_or_video(video, str(output_path)+'_'+str(i), fps=16)
                mx_score = compute_frame_mse(video)
                print('mx_score: ',mx_score)
                save_frame_comparison(video, prefix=f'frame_{i}', save_dir= str(output_path)+'_'+str(i))
                
                tau = .000420 # Precalculated, fit with tau based on mu and std.
                tau_2 = .0000420
                if mx_score > tau:
                    # Try to denoise x.
                    print('Trying to Denoise...')
                    video: torch.Tensor = self.pipe.generate_vid2world_attempt_denoise(
                        prompt=sample.prompt,
                        input_path=path_to_str(sample.input_path),
                        guidance=sample.guidance,
                        num_video_frames=2,
                        frame_index = i,
                        num_latent_conditional_frames=1,#sample.num_input_frames,
                        resolution=sample.resolution,
                        seed=sample.seed,
                        negative_prompt=sample.negative_prompt,
                        num_steps=2,
                        offload_diffusion_model=self.offload_diffusion_model,
                        offload_text_encoder=self.offload_text_encoder,
                        offload_tokenizer=self.offload_tokenizer,
                    )
                    if self.rank0:
                        video = (1.0 + video[0]) / 2
                    mx_score = compute_frame_mse(video, frame1_idx=1, frame2_idx=2)
                    save_frame_comparison(video, frame1_idx=1, frame2_idx=2, prefix=f'frame_denoise_{i}', save_dir= str(output_path)+'_'+str(i))
                    print('Denoise mx_score: ',mx_score)
                    if mx_score > tau_2:
                        #Reject...
                        continue
                    else: 
                        #Accept:
                        clean_context = i
                else: #Accept:
                    clean_context = i

            #Use the frame as the starting point for video. If all 5 pass, then use normal gen.
            video: torch.Tensor = self.pipe.generate_vid2world_compute_Mx(
                prompt=sample.prompt,
                input_path=path_to_str(sample.input_path),
                guidance=sample.guidance,
                frame_index=clean_context,
                num_video_frames=sample.num_output_frames,
                num_latent_conditional_frames=1,#sample.num_input_frames,
                resolution=sample.resolution,
                seed=sample.seed,
                negative_prompt=sample.negative_prompt,
                num_steps=sample.num_steps,
                offload_diffusion_model=self.offload_diffusion_model,
                offload_text_encoder=self.offload_text_encoder,
                offload_tokenizer=self.offload_tokenizer,
            )
        else:
            video: torch.Tensor = self.pipe.generate_vid2world(
                prompt=sample.prompt,
                input_path=path_to_str(sample.input_path),
                guidance=sample.guidance,
                num_video_frames=sample.num_output_frames,
                num_latent_conditional_frames=2,#sample.num_input_frames, 
                resolution=sample.resolution,
                seed=sample.seed,
                negative_prompt=sample.negative_prompt,
                num_steps=sample.num_steps,
                offload_diffusion_model=self.offload_diffusion_model,
                offload_text_encoder=self.offload_text_encoder,
                offload_tokenizer=self.offload_tokenizer,
            )

        if self.rank0:
            video = (1.0 + video[0]) / 2

            # run video guardrail on the video
            if self.video_guardrail_runner is not None:
                log.info("Running guardrail check on video...")
                frames = (video * 255.0).clamp(0.0, 255.0).to(torch.uint8)
                frames = frames.permute(1, 2, 3, 0).cpu().numpy().astype(np.uint8)  # (T, H, W, C)
                processed_frames = guardrail_presets.run_video_guardrail(frames, self.video_guardrail_runner)
                if processed_frames is None:
                    message = "Guardrail blocked video2world generation."
                    log.critical(message)
                    if self.setup_args.keep_going:
                        return None
                    else:
                        raise Exception(message)
                else:
                    log.success("Passed guardrail on generated video")
                # Convert processed frames back to tensor format
                processed_video = torch.from_numpy(processed_frames).float().permute(3, 0, 1, 2) / 255.0
                video = processed_video.to(video.device, dtype=video.dtype)
            else:
                log.warning("Guardrail checks on video are disabled")
            if noise_reject_alg == False:
                save_img_or_video(video, str(output_path)+'_basemodel', fps=16)
            else:
                save_img_or_video(video, str(output_path)+'_improved', fps=16)
            log.success(f"Saved video to {output_path}.mp4")
        return f"{output_path}.mp4"
