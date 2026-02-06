import os
import json
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict

def gen_score(file_path):
    scores = []
    with open(file_path, "r") as f:
        for line in f:
            data = json.loads(line)
            scores.append(data["episode/score"])
    scores = np.array(scores)
    mean = np.mean(scores)
    std = np.std(scores)
    return mean, std

# Abbreviation dictionaries
SCENARIOS = {
    "carla_right_turn_simple": "RTS",
    "carla_stop_sign": "STOP",
    "carla_four_lane": "FOUR",
}
AUG_TYPES = {
    "jitter": "JIT",
    "glare": "GLA",
    "gaussian": "GAU",
    "occlusion": "OCC",
}

def abbreviate(name: str):
    """Map folder names into compact tags like STOP_JIT_75 and return scenario too"""
    scenario_tag = None
    parts = []
    for long, short in SCENARIOS.items():
        if long in name:
            parts.append(short)
            scenario_tag = short
            break
    for long, short in AUG_TYPES.items():
        if long in name:
            parts.append(short)
            break
    if "0.75" in name:
        parts.append("75")
    elif "0.875" in name:
        parts.append("875")
    return scenario_tag, "_".join(parts) if parts else name[:6]

folder = "/home/general/Documents/work/Trolls/SafeRL/CarDreamer/CarDreamer/CarDreamerRepo/logdir/evals_finegrained"
subdirs = sorted(os.listdir(folder))

# Collect results by scenario → task → group
results = defaultdict(lambda: defaultdict(dict))  
# results[scenario][task][group] = (mean, std)

for subdir in subdirs:
    metrics_path = os.path.join(folder, subdir, "metrics.jsonl")
    if os.path.isfile(metrics_path):
        mean, std = gen_score(metrics_path)
        scenario, task = abbreviate(subdir)
        if not scenario:
            continue
        if "sample" in subdir.lower():
            group = "sample"
        elif "filter" in subdir.lower():
            group = "filter"
        elif "reject" in subdir.lower():
            group = 'reject'
        elif "rejv0" in subdir.lower():
            group = 'rejv0'
        else:
            group = "other"
        results[scenario][task][group] = (mean, std)

# Plot: one subplot per scenario
groups = ["other", "reject", "rejv0"]
fig, axes = plt.subplots(1, len(SCENARIOS), figsize=(16, 5), sharey=False)

for ax, (scenario, tasks_dict) in zip(axes, results.items()):
    tasks = sorted(tasks_dict.keys())
    x = np.arange(len(tasks))
    width = 0.25

    for i, group in enumerate(groups):
        means = [tasks_dict[t].get(group, (np.nan, np.nan))[0] for t in tasks]
        stds = [tasks_dict[t].get(group, (np.nan, np.nan))[1] for t in tasks]
        ax.bar(x + i*width - width, means, width, yerr=stds, capsize=4, label=group)

    ax.set_xticks(x)
    ax.set_xticklabels(tasks, rotation=45, ha="right", fontsize=9)
    ax.set_title(scenario)
    ax.set_ylabel("Score (mean ± std)")

fig.suptitle("Sample vs Filter vs Other by Scenario", fontsize=14)
fig.legend(groups, loc="upper right")
plt.tight_layout()
plt.savefig('/home/general/Documents/work/Trolls/SafeRL/CarDreamer/CarDreamer/CarDreamerRepo/logdir/plots/folder_results.png')
