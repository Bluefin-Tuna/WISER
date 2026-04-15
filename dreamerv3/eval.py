import re
import warnings
import os
import embodied
import numpy as np
import ruamel.yaml as yaml

import car_dreamer
import dreamerv3

import torch
import torch.nn as nn
from torchvision import datasets, transforms, models
from deepinv.models import Restormer

import signal
import sys
import time
import wandb

wandb.login(key="17cf136c31b41699d0b7abe62648964e787fd06c")

warnings.filterwarnings("ignore", ".*truncated to dtype int32.*")


def build_model(device="cuda"):
    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, 2)  # clean vs noisy
    return model.to(device)


def wrap_env(env, config):
    args = config.wrapper
    env = embodied.wrappers.InfoWrapper(env)
    for name, space in env.act_space.items():
        if name == "reset":
            continue
        elif space.discrete:
            env = embodied.wrappers.OneHotAction(env, name)
        elif args.discretize:
            env = embodied.wrappers.DiscretizeAction(env, name, args.discretize)
        else:
            env = embodied.wrappers.NormalizeAction(env, name)
    env = embodied.wrappers.ExpandScalars(env)
    if args.length:
        env = embodied.wrappers.TimeLimit(env, args.length, args.reset)
    if args.checks:
        env = embodied.wrappers.CheckSpaces(env)
    for name, space in env.act_space.items():
        if not space.discrete:
            env = embodied.wrappers.ClipAction(env, name)
    return env


import subprocess
import re


def get_pids(port):
    command = f"lsof -i :{port} | awk '{{print $2}}'"
    pids = subprocess.check_output(command, shell=True, text=True)  # decode to str
    pids = pids.strip()
    if pids:
        for pid in pids.split("\n"):
            try:
                yield int(pid)
            except ValueError:
                pass


def kill_processes_on_ports(ports):
    for port in ports:
        pids = set(get_pids(port))
        if pids:
            # Try graceful shutdown first
            subprocess.run(["kill", "-15"] + list(map(str, pids)))
            time.sleep(2)
            # Force kill if still alive
            remaining = set(get_pids(port))
            if remaining:
                subprocess.run(["kill", "-9"] + list(map(str, remaining)))


def eval_only(agent, env, logger, args):
    print("Start evaluation.")
    print("args:", args)
    logdir = embodied.Path(args.logdir)
    logdir.mkdirs()
    print("Logdir", logdir)
    step = logger.step
    metrics = embodied.Metrics()
    print("Observation space:", env.obs_space)
    print("Action space:", env.act_space)

    timer = embodied.Timer()
    timer.wrap("agent", agent, ["policy"])
    timer.wrap("env", env, ["step"])
    timer.wrap("logger", logger, ["write"])

    nonzeros = set()

    def per_episode(ep, ep_info):
        length = len(ep["reward"]) - 1
        score = float(ep["reward"].astype(np.float64).sum())
        logger.add({"length": length, "score": score}, prefix="episode")
        print(f"Episode has {length} steps and return {score:.1f}.")
        stats = {}
        # UNCOMMENT FOR VIDEOS TO WANDB
        # for key in ep:
        #     if 'custom' in key:
        #         stats[key] = ep[key]
        # for key in args.log_keys_video:
        #     if key in ep:
        #         stats[f"policy_{key}"] = ep[key]
        custom_values = [
            "stages",
            "condition_1",
            "condition_2",
            "condition_3",
            "gradients_exact",
            "mu_gradients",
        ]

        def log(key, value):
            if key == "log_surprise_mean":
                stats["log_surprise_mean"] = value[10]  # Set value index to wanted.
            if key in custom_values:
                stats[key] = np.round(value, decimals=4)
            if re.match(args.log_keys_sum, key):
                stats[f"sum_{key}"] = value.sum()
            if re.match(args.log_keys_mean, key):
                stats[f"mean_{key}"] = value.mean()
            if re.match(args.log_keys_max, key):
                stats[f"max_{key}"] = value.max(0).mean()

        debug_info = {
            "stages": [],
            "reconstruction_error_1": [],
            "reconstruction_error_2": [],
            "reconstruction_error_3": [],
        }
        for key, value in ep.items():
            if not args.log_zeros and key not in nonzeros and (value == 0).all():
                continue
            nonzeros.add(key)
            log(key, value)

        for key, value in ep_info.items():
            log(key, value)

        logger.add(metrics.result())
        logger.add(timer.stats(), prefix="timer")
        logger.write(fps=True)

        metrics.add(stats, prefix="stats")

    def per_step(tran):
        step.increment()

    driver = embodied.Driver(env)
    driver.on_episode(lambda ep, ep_info, worker: per_episode(ep, ep_info))
    driver.on_step(lambda tran, info, _: per_step(step))
    driver.off_shelf_mx = args.off_shelf_mx
    driver.denoise_method = args.denoise_method
    if driver.off_shelf_mx:
        model = build_model()
        state_dict = torch.load(checkpoint, map_location="cuda")
        model.load_state_dict(state_dict)
        model.eval()
        driver.rejection_score_model = model
    if driver.denoise_method == "denoiser":
        driver.denoiser_model = Restormer(pretrained="denoising").cuda().eval()

    checkpoint = embodied.Checkpoint()
    checkpoint.agent = agent
    if args.from_checkpoint:
        checkpoint.load(args.from_checkpoint, keys=["agent"])
    else:
        raise ValueError("No checkpoint specified.")

    print("Start evaluation loop.")
    print(args.mode)
    if any(keyword in args.mode for keyword in ("sample", "random", "filter")):
        raise ValueError(
            "Unsupported eval mode. 'sample', 'random', and 'filter' have been removed."
        )

    if "surprise" in args.mode:
        if "full" in args.mode:
            policy = lambda *args: agent.policy(*args, mode="surprise_full")
        else:
            policy = lambda *args: agent.policy(*args, mode="surprise")
    elif "reject" in args.mode:
        policy = lambda *args: agent.policy(*args, mode="reject")
    else:
        policy = lambda *args: agent.policy(*args, mode="eval")

    while step < args.steps:
        driver(policy, steps=100)
    logger.write()


def get_tau(name, mode):
    # Extract the number after "reject" in mode
    match = re.search(r"reject(\d+)", mode)
    if match:
        n = int(match.group(1))
        print(n)
    else:
        print("No reject number found...")
        n = 1  # default if no reject number found

    # Set mean and std based on the environment name
    if name == "carla_four_lane":
        mean = 0.0143
        std = 0.0094
    elif name == "carla_stop_sign":
        mean = 0.0104
        std = 0.0039
    elif name == "carla_right_turn_simple":
        mean = 0.0100
        std = 0.0043
    else:
        raise ValueError(f"Unknown environment name: {name}")

    return mean + n * std


def main(argv=None):
    model_configs = yaml.YAML(typ="safe").load(
        (embodied.Path(__file__).parent / "dreamerv3.yaml").read()
    )
    config = embodied.Config({"dreamerv3": model_configs["defaults"]})
    config = config.update({"dreamerv3": model_configs["small"]})

    parsed, other = embodied.Flags(task=["carla_navigation"]).parse_known(argv)
    for name in parsed.task:
        print("Using task: ", name)
        env, env_config = car_dreamer.create_task(name, argv)
        config = config.update(env_config)
    config = embodied.Flags(config).parse(other)

    logdir = embodied.Path(config.dreamerv3.logdir)
    step = embodied.Counter()
    logger = embodied.Logger(
        step,
        [
            embodied.logger.TerminalOutput(),
            embodied.logger.JSONLOutput(logdir, "metrics.jsonl"),
            embodied.logger.TensorBoardOutput(logdir),
            embodied.logger.WandBOutput(logdir.name, config),
        ],
    )

    from embodied.envs import from_gym

    dreamerv3_config = config.dreamerv3
    env = from_gym.FromGym(env)
    env = wrap_env(env, dreamerv3_config)
    env = embodied.BatchEnv([env], parallel=False)

    if "reject" in dreamerv3_config.run.mode:  # Compute tau:
        tau = get_tau(name, dreamerv3_config.run.mode)
        dreamerv3_config = dreamerv3_config.update({"run.reject_tau": tau})

    dreamerv3_config = dreamerv3_config.update(
        {
            "run.log_keys_sum": "(travel_distance|destination_reached|out_of_lane|time_exceeded|is_collision|timesteps)",
            "run.log_keys_mean": "(travel_distance|ttc|speed_norm|wpt_dis)",
            "run.log_keys_max": "(travel_distance|ttc|speed_norm|wpt_dis)",
            "run.steps": 10000,  # 15000,
        }
    )

    agent = dreamerv3.Agent(env.obs_space, env.act_space, step, dreamerv3_config)
    args = embodied.Config(
        **dreamerv3_config.run,
        logdir=dreamerv3_config.logdir,
        batch_steps=dreamerv3_config.batch_size * dreamerv3_config.batch_length,
    )
    eval_only(agent, env, logger, args)
    print("Done with Eval. Killing Carla.")
    env.close()
    kill_processes_on_ports([2000, 8000, 9000])

    # import sys
    # sys.exit(0)


if __name__ == "__main__":
    main()
