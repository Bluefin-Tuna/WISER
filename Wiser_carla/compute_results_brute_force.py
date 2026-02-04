import os
import re
from collections import defaultdict
import json
import numpy as np
import matplotlib.pyplot as plt


def plot_line_per_aug(data_per_aug, default_data, scenario_name):
    """
    Plots one line chart per augmentation type.
    X-axis: augmentation intensity (0 = default)
    Y-axis: score
    - Points for different methods are slightly offset to avoid overlap.
    - Points are connected by lines.
    - Vertical dashed lines separate each intensity group.
    """
    import matplotlib.pyplot as plt
    import numpy as np

    aug_types = ["gaussian", "glare", "jitter", "lag"]
    method_labels = {
        "surprise": "$O(n \log n)$",
        "full": '$2^N$'
    }
    colors = {
        "surprise": "tab:green",
        "full": "tab:red"
    }

    fig, axes = plt.subplots(2, 2, figsize=(14, 6), sharey=True)
    axes = axes.flatten()

    method_list = list(method_labels.keys())
    n_methods = len(method_list)
    offset_width = 0.2  # horizontal spread of methods within a group

    for i, aug in enumerate(aug_types):
        ax = axes[i]

        for j, method in enumerate(method_list):
            # Offset so points don't overlap
            offset = (j - (n_methods - 1) / 2) * offset_width

            xs, means, stds = [], [], []

            # Add default at x=0
            if default_data and aug in default_data and method in default_data[aug]:
                default_mean, default_std = default_data[aug][method]
                xs.append(0 + offset)
                means.append(default_mean)
                stds.append(default_std)

            # Add other intensities starting from 1
            if aug in data_per_aug and method in data_per_aug[aug]:
                group_nums = sorted(data_per_aug[aug][method].keys(), key=lambda x: int(x))
                xs.extend([int(g) + offset for g in group_nums])
                means.extend([data_per_aug[aug][method][g][0] for g in group_nums])
                stds.extend([data_per_aug[aug][method][g][1] for g in group_nums])
                max_err = 75  # Limit the visualization
                stds = [min(s, max_err) for s in stds]

            # Plot line + points with error bars
            if xs:
                ax.errorbar(
                    xs, means, yerr=stds,
                    marker='o', capsize=4, elinewidth=1.5,
                    linewidth=2, label=method_labels[method],
                    color=colors[method]
                )

        # Set integer ticks
        all_intensities = [0] + sorted(
            {int(g) for m in data_per_aug.get(aug, {}) for g in data_per_aug[aug][m]}
        )
        ax.set_xticks(all_intensities)
        ax.set_xticklabels(all_intensities)

        # Draw vertical separators between groups
        for x in range(len(all_intensities) - 1):
            ax.axvline(x + 0.5, color="black", linestyle="--", alpha=0.9)
        # if 'four_lane' in scenario_name:
        ax.set_title(f"{aug.capitalize()}", fontweight='bold', fontsize=22)
        ax.set_xlabel("Augmentation Intensity")
        # if i == 0:
            # ax.set_ylabel("Score")
        ax.grid(True, linestyle='--', alpha=0.5)

        # if 'four_lane' in scenario_name:
        #     bottom, _ = ax.get_ylim()
        #     ax.set_ylim(bottom, 1500)
        # if 'right_turn' in scenario_name:
        #     bottom, _ = ax.get_ylim()
        #     ax.set_ylim(bottom, 400)
        #     if 'gaussian' in aug:
        #         bottom, _ = ax.get_ylim()
        #         ax.set_ylim(200, 400)

    # Single legend at bottom
    handles, labels = axes[0].get_legend_handles_labels()
   
    fig.legend(handles, labels, loc='lower center', ncol=4)
    # plt.suptitle(scenario_name.replace("_", " ").title(),fontsize=22)
    plt.tight_layout(rect=[0, 0.05, 1, 0.92])

    plt.savefig(f'/home/general/Documents/work/Trolls/SafeRL/CarDreamer/CarDreamer/logdir/plots/brute_{scenario_name}.png', bbox_inches='tight',pad_inches=0.1)

def plot_line_per_aug(data_per_aug, default_data, scenario_name):
    """
    Plots one line chart per augmentation type.
    X-axis: augmentation intensity (0 = default)
    Y-axis: score
    - Points for different methods are slightly offset to avoid overlap.
    - Points are connected by lines.
    - Vertical dashed lines separate each intensity group.
    """
    import matplotlib.pyplot as plt
    import numpy as np

    aug_types = ["gaussian", "glare", "jitter", "lag"]
    method_labels = {
        "surprise": "$O(n \log n)$",
        "full": '$2^N$'
    }
    colors = {
        "surprise": "tab:green",
        "full": "tab:red"
    }

    fig, axes = plt.subplots(2, 2, figsize=(14, 6), sharey=True)
    axes = axes.flatten()

    method_list = list(method_labels.keys())
    n_methods = len(method_list)
    offset_width = 0.2  # horizontal spread of methods within a group
    

    for i, aug in enumerate(aug_types):
        ax = axes[i]

        for j, method in enumerate(method_list):
            offset = (j - (n_methods - 1) / 2) * offset_width
            xs, means, stds = [], [], []

            if default_data and aug in default_data and method in default_data[aug]:
                default_mean, default_std = default_data[aug][method]
                xs.append(0 + offset)
                means.append(default_mean)
                stds.append(default_std)

            if aug in data_per_aug and method in data_per_aug[aug]:
                group_nums = sorted(data_per_aug[aug][method].keys(), key=lambda x: int(x))
                xs.extend([int(g) + offset for g in group_nums])
                means.extend([data_per_aug[aug][method][g][0] for g in group_nums])
                stds.extend([min(data_per_aug[aug][method][g][1], 75) for g in group_nums])

            if xs:
                ax.errorbar(
                    xs, means, yerr=stds,
                    marker='o', capsize=4, elinewidth=1.5,
                    linewidth=2, label=method_labels[method],
                    color=colors[method]
                )

        all_intensities = [0] + sorted(
            {int(g) for m in data_per_aug.get(aug, {}) for g in data_per_aug[aug][m]}
        )
        # ax.set_xticks(all_intensities)
        # ax.set_xticklabels(all_intensities)

        for x in range(len(all_intensities) - 1):
            ax.axvline(x + 0.5, color="black", linestyle="--", alpha=0.9)
            ax.set_yticklabels([])

        ax.set_title(f"{aug.capitalize()}", fontweight='bold', fontsize=22)
        # ax.set_xlabel("Augmentation Intensity")
        ax.grid(True, linestyle='--', alpha=0.5)
        ax.set_yticklabels([])


        # ---------- SAVE IDENTICAL FORMATTING ----------
        fig_single, ax_single = plt.subplots(figsize=(7, 5))
        for j, method in enumerate(method_list):
            offset = (j - (n_methods - 1) / 2) * offset_width
            xs, means, stds = [], [], []

            if default_data and aug in default_data and method in default_data[aug]:
                default_mean, default_std = default_data[aug][method]
                xs.append(0 + offset)
                means.append(default_mean)
                stds.append(default_std)

            if aug in data_per_aug and method in data_per_aug[aug]:
                group_nums = sorted(data_per_aug[aug][method].keys(), key=lambda x: int(x))
                xs.extend([int(g) + offset for g in group_nums])
                means.extend([data_per_aug[aug][method][g][0] for g in group_nums])
                stds.extend([min(data_per_aug[aug][method][g][1], 75) for g in group_nums])

            if xs:
                ax_single.errorbar(
                    xs, means, yerr=stds,
                    marker='o', capsize=4, elinewidth=1.5,
                    linewidth=2, label=method_labels[method],
                    color=colors[method]
                )

        # ax_single.set_xticks(all_intensities)
        # ax_single.set_xticklabels(all_intensities)
        for x in range(len(all_intensities) - 1):
            ax_single.axvline(x + 0.5, color="black", linestyle="--", alpha=0.9)

        ax_single.set_title(f"{aug.capitalize()}",
                            fontweight='bold', fontsize=22)
        ax_single.set_xlabel("Augmentation Intensity", fontweight='bold', fontsize=22)
        ax_single.set_ylabel("Score", fontweight='bold', fontsize=22)
        ax_single.set_yticklabels([])
        # Set bold and larger tick labels
        for label in ax_single.get_xticklabels():
            label.set_fontsize(14)
            label.set_fontweight('bold')

        # Set bold axis label
        ax_single.set_xlabel("Augmentation Intensity", fontsize=22, fontweight='bold')
        ax_single.legend(prop={'size':30})

        fig_single.savefig(
            f"/home/general/Documents/work/Trolls/SafeRL/CarDreamer/CarDreamer/logdir/plots/brute_{scenario_name}_{aug}.png",
            bbox_inches='tight', pad_inches=0.1
        )
        plt.close(fig_single)
# ---------- Score function ----------
def gen_score(file_path):
    scores = []
    
    with open(file_path, "r") as f:
        for line in f:
            data = json.loads(line)
            scores.append(data["episode/score"])
    scores = np.array(scores)
    if 'Default' in file_path:
        print(scores)
        print(np.mean(scores))
    return np.mean(scores), np.std(scores)

# ---------- Config ----------
eval_dir = "/home/general/Documents/work/Trolls/SafeRL/CarDreamer/CarDreamer/logdir/evals/"
SCENARIOS = ["carla_four_lane", "carla_right_turn_simple", "carla_stop_sign"]
AUG_TYPES = ["jitter", "glare", "gaussian", "lag"]

# ---------- Group folders ----------
all_folders = [f for f in os.listdir(eval_dir) if os.path.isdir(os.path.join(eval_dir, f))]
groups = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
number_pattern = re.compile(r"_(\d+)$")

for folder in all_folders:
    if folder.endswith("_Default"):
        if 'carla_four_lane' in folder:
            groups["Default"]["carla_four_lane"]["Default"].append(folder)
        elif "carla_right_turn_simple" in folder:
            groups["Default"]["carla_right_turn_simple"]["Default"].append(folder)
        elif "carla_stop_sign" in folder:
            groups["Default"]["carla_stop_sign"]["Default"].append(folder)
        continue

    num_match = number_pattern.search(folder)
    number = num_match.group(1) if num_match else None
    scenario = next((s for s in SCENARIOS if s in folder), None)
    aug = next((a for a in AUG_TYPES if a in folder), "unknown")

    if number and scenario:
        groups[number][scenario][aug].append(folder)
    else:
        groups["Misc"]["Unmatched"]["unknown"].append(folder)

# ---------- Main loop ----------
for scenario_key in SCENARIOS:
    data_per_aug = defaultdict(lambda: defaultdict(dict))  # aug -> method -> {group_number: (mean, std)}
    
    # Process default group data for this scenario
    default_data = defaultdict(dict)  # aug -> method -> (mean, std)
    if "Default" in groups and scenario_key in groups["Default"]:
        for aug_key in groups["Default"][scenario_key]:
            if aug_key == "Default":
                # Handle case where default folders don't have augmentation type in name
                # Assume these are baseline results for all augmentation types
                for folder in groups["Default"][scenario_key][aug_key]:
                    print(folder)
                    metrics_path = os.path.join(eval_dir, folder, "metrics.jsonl")
                    mean, std = gen_score(metrics_path)
                    
                    # Determine method type from folder name
                    if 'augment_high' in folder:
                        method = 'augment_high'
                    elif 'random' in folder:
                        method = 'random'
                    elif 'surprise' in folder and 'full' not in folder:
                        method = 'surprise'
                    elif 'full' not in folder:
                        method = 'base'
                    elif 'full' in folder:
                        method = 'full'
                    else:
                        continue
                    
                    # Add to all augmentation types for baseline comparison
                    for aug_type in AUG_TYPES:
                        default_data[aug_type][method] = (mean, std)
            else:
                # Handle case where default folders have specific augmentation types
                for folder in groups["Default"][scenario_key][aug_key]:
                    metrics_path = os.path.join(eval_dir, folder, "metrics.jsonl")
                    mean, std = gen_score(metrics_path)
                    
                    if 'augment_high' in folder:
                        default_data[aug_key]['augment_high'] = (mean, std)
                    elif 'random' in folder:
                        default_data[aug_key]['random'] = (mean, std)
                    elif 'surprise' in folder and 'full' not in folder:
                        default_data[aug_key]['surprise'] = (mean, std)
                    elif 'full' not in folder:
                        default_data[aug_key]['base'] = (mean, std)
                    elif 'full' in folder:
                        default_data[aug_key]['full'] = (mean, std)

    # Process intensity group data
    for group_key in sorted(groups, key=lambda x: (x != "Default", int(x) if x.isdigit() else float('inf'))):
        if group_key == 'Default':
            continue  # Skip default in intensity plots (already processed above)

        if scenario_key not in groups[group_key]:
            continue

        for aug_key in sorted(groups[group_key][scenario_key]):
            for folder in sorted(groups[group_key][scenario_key][aug_key]):
                metrics_path = os.path.join(eval_dir, folder, "metrics.jsonl")
                mean, std = gen_score(metrics_path)

                if 'augment_high' in folder:
                    data_per_aug[aug_key]['augment_high'][group_key] = (mean, std)
                elif 'random' in folder:
                    data_per_aug[aug_key]['random'][group_key] = (mean, std)
                elif 'surprise' in folder and 'full' not in folder:
                    data_per_aug[aug_key]['surprise'][group_key] = (mean, std)
                elif 'full' not in folder:
                    data_per_aug[aug_key]['base'][group_key] = (mean, std)
                elif 'full' in folder:
                    data_per_aug[aug_key]['full'][group_key] = (mean, std)

    # Plot for this scenario with default data
    plot_line_per_aug(data_per_aug, default_data, scenario_key)