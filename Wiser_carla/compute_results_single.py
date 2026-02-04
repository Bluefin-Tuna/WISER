import os
import re
from collections import defaultdict
import json
import numpy as np
import matplotlib.pyplot as plt

def gen_score(file_path):
    scores = []

    # with open("pointbutton_1_geigh_3_occlusion_surprise_any/metrics.jsonl", "r") as f:
    with open(file_path, "r") as f:
        for line in f:
            data = json.loads(line)
            scores.append(data["episode/score"])

    scores = np.array(scores)
    mean = np.mean(scores)
    std = np.std(scores)

    return mean, std

# print(gen_score('/home/general/Documents/work/Trolls/SafeRL/CarDreamer/CarDreamer/logdir/evals/carla_stop_sign_bev_recon_carla_stop_sign_sample_glare_1/metrics.jsonl'))
print(gen_score('/home/general/Documents/work/Trolls/SafeRL/CarDreamer/CarDreamer/CarDreamerRepo/logdir/evals/carla_right_turn_simple_bev_carla_right_turn_simple_jitter_sample_proportion0.75_timestep10_1.0/metrics.jsonl'))
# print(gen_score('/home/general/Documents/work/Trolls/SafeRL/CarDreamer/CarDreamer/logdir/evals/carla_stop_sign_bev_pixel_carla_stop_sign_gaussian_1/metrics.jsonl'))
# print(gen_score('/home/general/Documents/work/Trolls/SafeRL/CarDreamer/CarDreamer/logdir/evals/carla_stop_sign_bev_carla_stop_sign_gaussian_1/metrics.jsonl'))
# print(gen_score('/home/general/Documents/work/Trolls/SafeRL/CarDreamer/CarDreamer/logdir/evals/carla_stop_sign_bev_pixel_carla_stop_sign_sample_glare_1/metrics.jsonl'))