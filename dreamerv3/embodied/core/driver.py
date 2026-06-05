import collections

import numpy as np

from .basics import convert
import torch
import torch.nn as nn
from torchvision import datasets, transforms, models
from deepinv.models import Restormer
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")



class Driver:
    _CONVERSION = {
        np.floating: np.float32,
        np.signedinteger: np.int32,
        np.uint8: np.uint8,
        bool: bool,
    }

    def __init__(self, env, **kwargs):
        assert len(env) > 0
        self._env = env
        self._kwargs = kwargs
        self._on_steps = []
        self._on_episodes = []
        self.reset()
        checkpoint = ""
        self.rejection_score_model = None
        self.denoiser_model = None

        # model = build_model()
        # state_dict = torch.load(checkpoint, map_location=device)
        # model.load_state_dict(state_dict)
        # model.eval()
        # self.model = model
        # print("Loaded model:", checkpoint)
        # self.restormer = Restormer(pretrained='denoising').cuda().eval()

    def reset(self):
        self._acts = {k: convert(np.zeros((len(self._env),) + v.shape, v.dtype)) for k, v in self._env.act_space.items()}
        self._acts["reset"] = np.ones(len(self._env), bool)
        self._eps = [collections.defaultdict(list) for _ in range(len(self._env))]
        self._eps_info = [collections.defaultdict(list) for _ in range(len(self._env))]
        self._state = None

    def on_step(self, callback):
        self._on_steps.append(callback)

    def on_episode(self, callback):
        self._on_episodes.append(callback)

    def __call__(self, policy, steps=0, episodes=0):
        step, episode = 0, 0
        while step < steps or episode < episodes:
            step, episode = self._step(policy, step, episode)

    def _step(self, policy, step, episode):
        assert all(len(x) == len(self._env) for x in self._acts.values())
        acts = {k: v for k, v in self._acts.items() if not k.startswith("log_")}
        obs, info = self._env.step(acts)
        if self.off_shelf_mx:
            obs['mx'] = self.compute_mx(obs)
        if self.denoise_method == 'denoiser': # Set yoyur own denoiser here.
            obs['denoised'] = self.denoise_restormer(obs['birdeye_wpt'])
        obs = {k: convert(v) for k, v in obs.items()}
        info = {k: convert(v) for k, v in info.items()}
        assert all(len(x) == len(self._env) for x in obs.values()), obs
        acts, self._state = policy(obs, self._state, **self._kwargs)
        acts = {k: convert(v) for k, v in acts.items()}
        if obs["is_last"].any():
            mask = 1 - obs["is_last"]
            acts = {k: v * self._expand(mask, len(v.shape)) for k, v in acts.items()}
        acts["reset"] = obs["is_last"].copy()
        self._acts = acts
        trns = {**obs, **acts}
        if obs["is_first"].any():
            for i, first in enumerate(obs["is_first"]):
                if first:
                    self._eps[i].clear()
                    self._eps_info[i].clear()
        for i in range(len(self._env)):
            trn = {k: v[i] for k, v in trns.items()}
            inf = {k: v[i] for k, v in info.items()}
            [self._eps[i][k].append(v) for k, v in trn.items()]
            [self._eps_info[i][k].append(v) for k, v in inf.items()]
            [fn(trn, inf, i, **self._kwargs) for fn in self._on_steps]
            step += 1
        if obs["is_last"].any():
            for i, done in enumerate(obs["is_last"]):
                if done:
                    ep = {k: convert(v) for k, v in self._eps[i].items()}
                    ep_info = {k: convert(v) for k, v in self._eps_info[i].items()}
                    [fn(ep.copy(), ep_info.copy(), i, **self._kwargs) for fn in self._on_episodes]
                    episode += 1
        return step, episode

    def _expand(self, value, dims):
        while len(value.shape) < dims:
            value = value[..., None]
        return value



    def compute_mx(self, obs):
        x = obs['birdeye_wpt']

        # Convert numpy → tensor
        if isinstance(x, np.ndarray):
            x = torch.from_numpy(x).float()

        # Ensure batch dim exists
        if x.ndim == 3:
            # HWC → CHW
            x = x.permute(2, 0, 1)
            x = x.unsqueeze(0)
        elif x.ndim == 4:
            # NHWC → NCHW
            x = x.permute(0, 3, 1, 2)

        x = x.to('cuda')

        with torch.no_grad():
            out = self.rejection_score_model(x)
            preds = torch.argmax(out, dim=1)

        return preds.cpu().numpy()
    
    def denoise_restormer(self, x):
        """
        Denoise with Restormer but return SAME SHAPE as input.
        """
        original_shape = x.shape
        return_HWC = False
        return_NHWC = False

        # Convert numpy → tensor
        if isinstance(x, np.ndarray):
            x = torch.from_numpy(x).float()

        # Track what format to return
        if x.ndim == 3:          # HWC
            return_HWC = True
            x = x.permute(2, 0, 1).unsqueeze(0)  # → NCHW
        elif x.ndim == 4:
            if x.shape[-1] == 3:  # NHWC
                return_NHWC = True
                x = x.permute(0, 3, 1, 2)        # → NCHW

        # Now guaranteed NCHW
        assert x.ndim == 4 and x.shape[1] == 3

        # Normalize if needed
        if x.max() > 1.0:
            x = x / 255.0

        x = x.to('cuda')

        with torch.no_grad():
            y = self.denoiser_model(x)  # NCHW

        # Undo NCHW → original format
        if return_HWC:
            y = y.squeeze(0).permute(1, 2, 0)  # → HWC
        elif return_NHWC:
            y = y.permute(0, 2, 3, 1)          # → NHWC

        return y.cpu().numpy()
    
    
