import os
import re
from collections import defaultdict
import json
import numpy as np
import matplotlib.pyplot as plt

# ---------- Plotting function ----------
def plot_line_per_aug(data_per_aug, default_data, scenario_name):
    """
    Plots one line chart per augmentation type.
    X-axis: augmentation intensity (group number)
    Y-axis: score
    One line per method with error bands.
    Adds horizontal dashed lines for default group values.
    """
    aug_types = ["gaussian", "glare", "jitter", "lag"]
    method_labels = {
        "augment_high": "Augmented",
        "random": "RME",
        "surprise": "Controlled Representation",
        "base": "Baseline"
    }
    colors = {
        "augment_high": "tab:blue",
        "random": "tab:orange",
        "surprise": "tab:green",
        "base": "tab:red"
    }

    fig, axes = plt.subplots(1, 4, figsize=(10, 8))
    axes = axes.flatten()

    for i, aug in enumerate(aug_types):
        ax = axes[i]
        
        # Plot intensity lines for each method
        for method, label in method_labels.items():
            if aug in data_per_aug and method in data_per_aug[aug]:
                group_nums = sorted(data_per_aug[aug][method].keys(), key=lambda x: int(x))
                means = [data_per_aug[aug][method][g][0] for g in group_nums]
                stds = [data_per_aug[aug][method][g][1] for g in group_nums]
                # stds = [0 for g in group_nums]

                ax.plot(group_nums, means, marker='o', linewidth=2,
                        label=label, color=colors[method])
                ax.fill_between(group_nums,
                                [m - s for m, s in zip(means, stds)],
                                [m + s for m, s in zip(means, stds)],
                                color=colors[method], alpha=0.2)
        
        # Add horizontal dashed lines for default group values
        if default_data and aug in default_data:
            for method, label in method_labels.items():
                if method in default_data[aug]:
                    default_mean, default_std = default_data[aug][method]
                    # default_std = 0
                    ax.axhline(y=default_mean, color=colors[method], 
                              linestyle='--', alpha=0.7, linewidth=1.5)
                    # Optionally add error band for default as well
                    ax.axhspan(default_mean - default_std, default_mean + default_std,
                              color=colors[method], alpha=0.1)
        
        ax.set_title(f"{aug.capitalize()}")
        ax.set_xlabel("Augmentation Intensity")
        ax.set_ylabel("Score")
        ax.grid(True, linestyle='--', alpha=0.5)
        if 'four_lane' in scenario_name:
            # ax.set_yticks([-1000, -500, 0, 500, 1000, 1500])  # example custom ticks
            # ax.set_ylim(-1000, 1500)  # optional, adjust limits
            bottom, _ = ax.get_ylim()
      
            ax.set_ylim(bottom, 1500)
        if 'right_turn' in scenario_name:
            # ax.set_yticks([-1000, -500, 0, 500, 1000, 1500])  # example custom ticks
            bottom, _ = ax.get_ylim()
            ax.set_ylim(bottom, 400)

    # Single legend at top
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', ncol=4)
    plt.suptitle(scenario_name.replace("_", " ").title())
    plt.tight_layout(rect=[0, 0, 1, 0.92])
    # plt.show()
    plt.savefig(f'/home/general/Documents/work/Trolls/SafeRL/CarDreamer/CarDreamer/CarDreamerRepo/logdir/plots/{scenario_name}.png')

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
        "augment_high": "Augmented",
        "random": "RME",
        "surprise": "Confident Representation",
        "base": "Baseline"
    }
    colors = {
        "augment_high": "tab:blue",
        "random": "tab:orange",
        "surprise": "black",
        "base": "tab:red"
    }

    fig, axes = plt.subplots(1, 4, figsize=(14, 6), sharey=False)
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
                max_err = 200  # Limit the visualization
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
        if aug =='lag':
            ax.set_title(f"Latency", fontweight='bold', fontsize=22)
        else:
            ax.set_title(f"{aug.capitalize()}", fontweight='bold', fontsize=22)
            # ax.set_xlabel("Augmentation Intensity")
        if i == 0:
            ax.set_ylabel("Score", fontsize=22)
        ax.grid(True, linestyle='--', alpha=0.5)

        # if 'four_lane' in scenario_name:
        #     bottom, _ = ax.get_ylim()
        #     ax.set_ylim(bottom, 1500)
        if 'right_turn' in scenario_name:
            bottom, _ = ax.get_ylim()
            ax.set_ylim(bottom, 400)
            if 'gaussian' in aug:
                bottom, _ = ax.get_ylim()
                ax.set_ylim(200, 400)

    # Single legend at bottom
    handles, labels = axes[0].get_legend_handles_labels()
    # if 'stop_sign' in scenario_name:
    fig.legend(handles, labels, loc='lower center', ncol=4, prop={'size': 14})
        # plt.suptitle(scenario_name.replace("_", " ").title())
    plt.tight_layout(rect=[0, 0.05, 1, 0.92])

    plt.savefig(f'/home/general/Documents/work/Trolls/SafeRL/CarDreamer/CarDreamer/CarDreamerRepo/logdir/plots/{scenario_name}.png', bbox_inches='tight',pad_inches=0.1)
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
eval_dir = "/home/general/Documents/work/Trolls/SafeRL/CarDreamer/CarDreamer/CarDreamerRepo/logdir/evals/"
SCENARIOS = ["carla_four_lane", "carla_right_turn_simple", "carla_stop_sign"]
AUG_TYPES = ["jitter", "glare", "gaussian", "lag"]

# ---------- Group folders ----------
all_folders = [f for f in os.listdir(eval_dir) if os.path.isdir(os.path.join(eval_dir, f))]
groups = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
number_pattern = re.compile(r"_(\d+)$")

for folder in all_folders:
    if 'bev' in folder or 'sample' in folder:
        continue
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
    print('Data per Aug',data_per_aug)
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

                if 'augment_high' in folder and 'bev' not in folder:
                    data_per_aug[aug_key]['augment_high'][group_key] = (mean, std)
                elif 'random' in folder:
                    data_per_aug[aug_key]['random'][group_key] = (mean, std)
                elif 'surprise' in folder and 'full' not in folder:
                    data_per_aug[aug_key]['surprise'][group_key] = (mean, std)
                elif 'full' not in folder:
                    data_per_aug[aug_key]['base'][group_key] = (mean, std)

    # Plot for this scenario with default data
    plot_line_per_aug(data_per_aug, default_data, scenario_key)