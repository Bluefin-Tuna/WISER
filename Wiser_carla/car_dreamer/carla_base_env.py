from abc import abstractmethod
from typing import Dict, Tuple

import carla
import gym
import numpy as np
from gym import spaces
import random
from .toolkit import EnvMonitorOpenCV, Observer, WorldManager


class CarlaBaseEnv(gym.Env):
    def __init__(self, config):
        self._config = config

        self._monitor = EnvMonitorOpenCV(self._config)
        self._world = WorldManager(self._config)
        self._world.on_reset(self.on_reset)
        self._world.on_step(self.on_step)
        self._observer = Observer(self._world, self._config.observation)

        self.action_space = self._get_action_space()
        self.observation_space = self._get_observation_space()

        # Lag buffer for multi-frame lag simulation
        self._lag_buffer = {}  # {key: [frame1, frame2, ...]}

    @abstractmethod
    def on_reset(self) -> None:
        """
        Override this method to perform additional reset operations.
        Specifically, you can spawn actors and plan routes here.
        """
        pass

    @abstractmethod
    def apply_control(self, action) -> None:
        """
        Override this method to apply control to actors.
        This method will be called before the simulator ticks.
        """
        pass

    @abstractmethod
    def on_step(self) -> None:
        """
        Override this method to perform additional operations at each step.
        Specifically, you can update the planner and the route here.
        This method will be called after the simulator ticks.
        """
        pass

    @abstractmethod
    def reward(self) -> Tuple[float, Dict]:
        """
        Override this method to define the reward function.
        """
        pass

    @abstractmethod
    def get_terminal_conditions(self) -> Dict[str, bool]:
        """
        Override this method to define the terminal condition.
        If one of the keys in the returned dictionary gives True, the episode will be terminated.
        """
        pass

    def get_ego_vehicle(self) -> carla.Actor:
        """
        Override this method to return the ego vehicle.
        The default behavior is to return self.ego
        """
        return self.ego

    def get_state(self) -> Dict:
        """Return the environment state. Implement this method to define the env state."""
        return self._state

    def _get_action_space(self):
        action_config = self._config.action
        if action_config.discrete:
            self.n_steer = len(action_config.discrete_steer)
            self.n_acc = len(action_config.discrete_acc)
            return spaces.Discrete(self.n_steer * self.n_acc)
        else:
            return spaces.Box(
                low=np.array([action_config.continuous_acc[0], action_config.continuous_steer[0]]),
                high=np.array([action_config.continuous_acc[1], action_config.continuous_steer[1]]),
                dtype=np.float32,
            )

    def _get_observation_space(self):
        return self._observer.get_observation_space()

    def reset(self):
        print("[CARLA] Reset environment")

        self._observer.destroy()
        self._world.reset()
        self._observer.reset(self.get_ego_vehicle())

        self._time_step = 0

        print("[CARLA] Environment reset")
        self.obs, _ = self._observer.get_observation(self.get_state())
        return self.obs

    def get_vehicle_control(self, action):
        """
        Convert actions in the action space to vehicle control in CARLA
        """
        action_config = self._config.action
        # Calculate acceleration and steering
        if action_config.discrete:
            acc = action_config.discrete_acc[action // self.n_steer]
            steer = action_config.discrete_steer[action % self.n_steer]
        else:
            acc = action[0]
            steer = action[1]
        # Convert acceleration to throttle and brake
        if acc > 0:
            throttle = np.clip(acc / 3, 0, 1)
            brake = 0
        else:
            throttle = 0
            brake = np.clip(-acc / 3, 0, 1)

        return carla.VehicleControl(throttle=float(throttle), steer=float(-steer), brake=float(brake))

    def _is_terminal(self):
        terminal_conds = self.get_terminal_conditions()
        terminal = False
        for k, v in terminal_conds.items():
            if v:
                print(f"[CARLA] Terminal condition triggered: {k}")
                terminal = True
            terminal_conds[k] = np.array([v], dtype=np.bool_)
        if terminal:
            terminal_conds["episode_timesteps"] = self._time_step
        terminal_conds["terminal"] = terminal
        return terminal, terminal_conds

    def step(self, action):
        self.apply_control(action)
        self._world.step()
        self._time_step += 1

        env_state = self.get_state()
        is_terminal, terminal_conds = self._is_terminal()
        self.obs, obs_info = self._observer.get_observation(env_state)
        reward, reward_info = self.reward()

        info = {
            **env_state,
            **terminal_conds,
            **obs_info,
            **reward_info,
            "action": action,
        }
        # print('Is eval true?: ',self._config.eval)
        # print('Obs keys: ',self.obs.keys())
        # print('Mode being used: ',self._config.mode)
        if self._config.mode != '' and 'Default' not in self._config.mode:
             # --- Store current obs in lag buffer ---
            if 'lag' in self._config.mode:
                for k, v in self.obs.items():
                    if k not in self._lag_buffer:
                        self._lag_buffer[k] = []
                    self._lag_buffer[k].append(v.copy())
                    # Keep only the last 10 frames (or any desired max lag)
                    if len(self._lag_buffer[k]) > 100:
                        self._lag_buffer[k].pop(0)

            
            self._simulate_failure()

        if self._config.eval:
            info = {f"eval_{k}": v for k, v in info.items()}
            self.obs = {**self.obs, **info}

        if self._config.display.enable:
            self._render(self.obs, info)

        return (self.obs, reward, is_terminal, info)

    def is_collision(self):
        """
        Check if the ego vehicle is in collision.
        You must include 'collsion' in observation.names to use this method.
        """
        return self.obs["collision"][0] > 0

    def _render(self, obs, info):
        self._monitor.render(obs, info)

    def _simulate_failure(self):
        np.random.seed(0)

        number_of_failures = int(self._config.mode.split('_')[-1])
        # available_keys = [
        #     "birdeye_gt",
        #     "birdeye_raw",
        #     "birdeye_with_traffic_lights",
        #     "birdeye_wpt",
        #     "camera",
        #     "lidar"
        # ]
        
        available_keys = [
            "birdeye_wpt"
        ]

        # Always sample a list of keys
        nov_keys = random.sample(available_keys, number_of_failures)

        # Helper: apply transformation to a single key
        def apply_jitter(key):
            img = self.obs[key].astype(np.float32)
            brightness = np.random.uniform(20, 30)
            contrast = np.random.uniform(20, 30)
            img = np.clip(img * contrast + brightness * 10, 0, 255).astype(np.uint8)
            self.obs[key] = img

        def apply_glare(key):
            img = self.obs[key].astype(np.float32)
            brightness = np.random.uniform(20, 30)
            img = np.clip(img * 0 + brightness * 10, 0, 255).astype(np.uint8)
            self.obs[key] = img

        def apply_gaussian(key):
            noise = np.random.normal(20, 30, self.obs[key].shape).astype(np.uint8)
            self.obs[key] = np.clip(self.obs[key] + noise, 0, 255)

        def apply_gaussianlite(key):
            noise = np.random.normal(20, 30, self.obs[key].shape).astype(np.uint8)
            self.obs[key] = np.clip(self.obs[key] + noise, 0, 255)

        def apply_occlusion(key):
            img = self.obs[key].copy()
            h, w, _ = img.shape
            x, y = np.random.randint(0, w//2), np.random.randint(0, h//2)
            mask_w, mask_h = np.random.randint(35, 40), np.random.randint(35, 40)
            img[y:y+mask_h, x:x+mask_w] = 0
            self.obs[key] = img

        def apply_channelswap(key):
            self.obs[key] = self.obs[key][..., np.random.permutation(3)]
        
        def apply_lag_sensor(key):
            """Simulate multi-frame lag by using an older observation."""
            if key in self._lag_buffer and len(self._lag_buffer[key]) > 1:
                max_lag = min(75, len(self._lag_buffer[key]) - 1)  # up to 5 steps back
                lag_steps = np.random.randint(1, max_lag + 1)
                self.obs[key] = self._lag_buffer[key][-lag_steps].copy()


        # Loop over keys and modes
        for key in nov_keys:
            if 'jitter' in self._config.mode:
                apply_jitter(key)
            if 'glare' in self._config.mode:
                apply_glare(key)
            if 'gaussian' in self._config.mode:
                apply_gaussian(key)
            if 'occlusion' in self._config.mode:
                apply_occlusion(key)
            if 'channelswap' in self._config.mode:
                apply_channelswap(key)
            if 'lag' in self._config.mode:
                apply_lag_sensor(key)
            if 'gaulite' in self._config.mode:
                apply_gaussianlite(key)
