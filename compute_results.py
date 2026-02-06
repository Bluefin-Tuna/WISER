import os
import re
from collections import defaultdict
import json
import numpy as np
import matplotlib.pyplot as plt

def plot_multiple_diamonds(data_dict, title=None):
    labels = ["gaussian", "glare", "jitter", "occlusion"]
    """
    Plot multiple diamond radar plots on the same polar axes.

    Parameters:
    - data_dict: Dictionary from make_data_template().
                 Keys are internal category names, values are dicts with:
                   'label': display name
                   'values': list of values
    - labels: List of axis labels (same for all radar plots).
    """
    num_vars = len(labels)

    # Angles for each axis
    angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
    angles += angles[:1]  # close loop

    # Rotate to make diamond
    offset = np.pi / 4
    angles = [(angle + offset) % (2 * np.pi) for angle in angles]

    # Setup polar plot
    fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(polar=True))

    # Plot each dataset
    for cat_key, info in data_dict.items():
        if not info['values']:  # skip empty
            continue
        values = info['values'] + info['values'][:1]  # close loop
        ax.plot(angles, values, label=info['label'], linewidth=2)
        ax.fill(angles, values, alpha=0.3)

    # Customize axes
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels)
    ax.set_yticklabels([])
    ax.spines['polar'].set_visible(False)

    # Dynamic y-limits based on title
    if title:
        if "carla_four_lane" in title:
            ax.set_ylim(-300, 1100)
        elif "carla_right_turn_simple" in title:
            ax.set_ylim(-300, 400)
        elif "carla_stop_sign" in title:
            ax.set_ylim(-300, 250)
        else:
            ax.set_ylim(0, 1)  # default
    else:
        title = "Multiple Diamond Radar Plots"
        ax.set_ylim(0, 1)

    # Title + Legend
    plt.title(title, va='bottom')
    plt.legend(loc='upper right', bbox_to_anchor=(1.2, 1.1))
    plt.show()

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

    return mean#, std

# Settings
eval_dir = "/home/general/Documents/work/Trolls/SafeRL/CarDreamer/CarDreamer/logdir/evals/"

SCENARIOS = ["carla_four_lane", "carla_right_turn_simple", "carla_stop_sign"]
AUG_TYPES = ["jitter", "glare", "gaussian", "occlusion"]

# List all folders in eval
all_folders = [f for f in os.listdir(eval_dir) if os.path.isdir(os.path.join(eval_dir, f))]

# Group structure: groups[number][scenario][augmentation] = list of folders
groups = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))

# Regex to get number suffix
number_pattern = re.compile(r"_(\d+)$")

for folder in all_folders:
    if folder.endswith("_Default"):
        if 'carla_four_lane' in folder:
            groups["Default"]["carla_four_lane"]["Default"].append(folder)
        elif  "carla_right_turn_simple" in folder:
            groups["Default"]["carla_right_turn_simple"]["Default"].append(folder)
        elif  "carla_stop_sign" in folder:
            groups["Default"]["carla_stop_sign"]["Default"].append(folder)
        continue

    # Match number
    num_match = number_pattern.search(folder)
    number = num_match.group(1) if num_match else None

    # Match scenario
    scenario = next((s for s in SCENARIOS if s in folder), None)

    # Match augmentation
    aug = next((a for a in AUG_TYPES if a in folder), "unknown")

    if number and scenario:
        groups[number][scenario][aug].append(folder)
    else:
        groups["Misc"]["Unmatched"]["unknown"].append(folder)

def make_data_template():
    return {
        'augment_high': {'label': 'Augmented', 'values': []},
        'surprise': {'label': 'Controlled Representation', 'values': []},
        'random': {'label': 'RME', 'values': []},
        'base': {'label': 'Baseline', 'values': []},
        # 'full': {'label': 'Max', 'values': []},
    }

# Print results
for group_key in sorted(groups, key=lambda x: (x != "Default", x)):
    if group_key != 'Default':
        print(f"\nGroup: {group_key}")
        for scenario_key in sorted(groups[group_key]):
            print(f"  Scenario: {scenario_key}")
            data = make_data_template()

            for aug_key in sorted(groups[group_key][scenario_key]):
                print(f"    Augmentation: {aug_key}")
                for folder in sorted(groups[group_key][scenario_key][aug_key]):
                    print(f"      - {folder}")
                    metrics_path = os.path.join(eval_dir+folder, "metrics.jsonl")
                    score = gen_score(metrics_path)
                    if 'augment_high' in folder:
                        data['augment_high']['values'].append(score)
                    elif 'random' in folder:
                        metrics_path = os.path.join(eval_dir+folder, "metrics.jsonl")
                        data['random']['values'].append(score)
                    elif 'surprise' in folder and 'full' not in folder:
                        metrics_path = os.path.join(eval_dir+folder, "metrics.jsonl")
                        data['surprise']['values'].append(score)
                    elif 'full' not in folder:
                        metrics_path = os.path.join(eval_dir+folder, "metrics.jsonl")
                        data['base']['values'].append(score)

            print(data)
            plot_multiple_diamonds(data, title=f'{scenario_key}_{group_key}')
    else:
        for scenario_key in sorted(groups[group_key]):
            print(f"  Scenario: {scenario_key}")
            print('Group key: ',group_key)
            data = make_data_template()
            for folder in sorted(groups[group_key][scenario_key]['Default']):
                print(f"      - {folder}")
                metrics_path = os.path.join(eval_dir+folder, "metrics.jsonl")
                score = gen_score(metrics_path)
                if 'augment_high' in folder:
                    metrics_path = os.path.join(eval_dir+folder, "metrics.jsonl")
                    data['augment_high']['values'].extend([score] * 4)
                elif 'random' in folder:
                    metrics_path = os.path.join(eval_dir+folder, "metrics.jsonl")
                    data['random']['values'].extend([score] * 4)
                elif 'surprise' in folder and 'full' not in folder:
                    metrics_path = os.path.join(eval_dir+folder, "metrics.jsonl")
                    data['surprise']['values'].extend([score] * 4)
                elif 'full' not in folder:
                    metrics_path = os.path.join(eval_dir+folder, "metrics.jsonl")
                    data['base']['values'].extend([score] * 4)
            print(data)
            plot_multiple_diamonds(data, title=f'{scenario_key}_{group_key}')
            



