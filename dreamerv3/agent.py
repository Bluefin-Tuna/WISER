import jax
import numpy as np
import jax.numpy as jnp
import jax.nn as jnn
import random as py_rand
from jax import random
# tree_map = jax.tree_util.tree_map
try:
    from jax import tree
    tree_map = tree.map
except (ImportError, AttributeError):
    from jax import tree_util
    tree_map = tree_util.tree_map

sg = lambda x: tree_map(jax.lax.stop_gradient, x)
import itertools
import logging

logger = logging.getLogger()


class CheckTypesFilter(logging.Filter):
    def filter(self, record):
        return "check_types" not in record.getMessage()


logger.addFilter(CheckTypesFilter())

from . import behaviors, jaxagent, jaxutils, nets
from . import ninjax as nj


@jaxagent.Wrapper
class Agent(nj.Module):
    def __init__(self, obs_space, act_space, step, config):
        self.config = config
        self.obs_space = obs_space
        self.act_space = act_space["action"]
        self.step = step
        self.wm = WorldModel(obs_space, act_space, config, name="wm")
        self.task_behavior = getattr(behaviors, config.task_behavior)(self.wm, self.act_space, self.config, name="task_behavior")
        if config.expl_behavior == "None":
            self.expl_behavior = self.task_behavior
        else:
            self.expl_behavior = getattr(behaviors, config.expl_behavior)(self.wm, self.act_space, self.config, name="expl_behavior")

    def policy_initial(self, batch_size):
        return (
            self.wm.initial(batch_size),
            self.task_behavior.initial(batch_size),
            self.expl_behavior.initial(batch_size),
        )

    def train_initial(self, batch_size):
        return self.wm.initial(batch_size)
    
    def get_recon_imgs(self, obs, posterior, prior, key='birdeye_wpt'):
        reconstruct_prior = self.wm.heads['decoder'](prior)[key].mode()
        reconstruct_post = self.wm.heads['decoder'](posterior)[key].mode()
        truth = obs[key]
        
        return truth,  reconstruct_post, reconstruct_prior
    
    def get_single_recon(self, latent, key='birdeye_wpt'):
        reconstruct = self.wm.heads['decoder'](latent)[key].mode()
        return reconstruct
    
    def interpolate(self, prev_latent, prev_action, obs, image_key, prior_orig):
        def tanh(gradients):
            # normalized = (gradients - gradients.min()) / (gradients.max() - gradients.min() + 1e-8)
            # normalized = jnn.sigmoid(gradients - 0.5)
            normalized = 2.0 * jnp.power(gradients - 0.5, 3.0) + 0.5
            mean = jnp.max(gradients)
            return normalized, mean
        
        def activation_experimental(gradients):
            # normalized = (gradients - gradients.min()) / (gradients.max() - gradients.min() + 1e-8)
            # normalized = jnn.sigmoid(gradients - 0.5)
            normalized = jnn.sigmoid(gradients + 1)
            # normalized = 2.0 * jnp.power(nor - 0.5, 3.0) + 0.5
            mean = jnp.mean(gradients)
            return normalized, mean
        
        def compute_surprise_for_obs(obs_input):
            """Compute surprise for given observation - used for gradient computation"""
            embed = self.wm.encoder(obs_input)
            temp_latent, temp_prior = self.wm.rssm.obs_step(
                prev_latent, prev_action, embed, obs_input['is_first']
            )
            surprise = self.wm.rssm.get_dist(temp_latent).kl_divergence(
                self.wm.rssm.get_dist(temp_prior)
            )
            return surprise.mean()  # Return scalar for gradient computation
        surprise_grad_fn = jax.grad(compute_surprise_for_obs)
        gradients = surprise_grad_fn(obs)
        normalized_gradients, mean_gradients = tanh(jnp.abs(gradients[image_key]))
        prior_img = self.get_single_recon(prior_orig, image_key)
        
        # interpolation = reconstruct_post#(1 - normalized_gradients) * obs[image_key] + normalized_gradients * prior_img
        interpolation = (1 - normalized_gradients) * obs[image_key] + normalized_gradients * prior_img
        return interpolation

    def policy(self, obs, state, mode="train"):

        self.config.jax.jit and print("Tracing policy function.")
        obs = self.preprocess(obs)
        if len(state) == 5:
            (prev_latent, prev_action), task_state, expl_state, stage, step = state
            step+=1
        else:
            (prev_latent, prev_action), task_state, expl_state = state
            stage = jnp.array(0) # or some default value
            step = 0
            print(step)

        embed = self.wm.encoder(obs)
        latent, prior_orig = self.wm.rssm.obs_step(prev_latent, prev_action, embed, obs["is_first"])
        if mode not in ['train', 'eval', 'explore']:
            available_keys = ["birdeye_gt","birdeye_raw","birdeye_with_traffic_lights","birdeye_wpt","camera","lidar"]
            if 'surprise' in mode:
                used_keys = ['None'] + available_keys
                used_keys = np.array(used_keys)
                total_newlat_seperate = []
                surprises = []
                # Compute baseline surprise
                base_surprise = self.wm.rssm.get_dist(latent).kl_divergence(self.wm.rssm.get_dist(prior_orig))
                if 'full' in mode:
                    # Iterate over all combinations of keys (masking 1, 2, ..., N keys)
                    for r in range(1, len(available_keys) + 1):
                        for combo in itertools.combinations(available_keys, r):
                            obs_masked = jax.tree_map(lambda x: x, obs)
                            
                            # Mask all keys in the current combination
                            for key in combo:
                                obs_masked[key] = jnp.zeros(obs[key].shape, dtype=float)

                            # Encode and compute surprise
                            embed = self.wm.encoder(obs_masked)
                            temp_latent, temp_prior_orig = self.wm.rssm.obs_step(
                                prev_latent, prev_action, embed, obs['is_first']
                            )
                            surprise = self.wm.rssm.get_dist(temp_latent).kl_divergence(
                                self.wm.rssm.get_dist(temp_prior_orig)
                            )

                            surprises.append(surprise)
                            total_newlat_seperate.append(temp_latent)
                else: #Run N version
                    for ite, image_key in enumerate(available_keys): #Keeps only one sensor at a time:
                        # Start with all observations as zeros
                        obs_masked = jax.tree_map(lambda x: jnp.zeros_like(x), obs)
                        
                        # Keep only the current image_key unmasked (use original data)
                        obs_masked[image_key] = obs[image_key]
                        
                        embed = self.wm.encoder(obs_masked)

                        temp_latent, temp_prior_orig = self.wm.rssm.obs_step(
                            prev_latent, prev_action, embed, obs['is_first'])

                        surprise = self.wm.rssm.get_dist(temp_latent).kl_divergence(
                            self.wm.rssm.get_dist(temp_prior_orig))

                        surprises.append(surprise)
                        total_newlat_seperate.append(temp_latent)

                    #Sort sensors by surprise (lowest surprise first)
                    surprises_init = jnp.array(surprises)
                    surprises_init = surprises_init.flatten()
                    sorted_indices = jnp.argsort(surprises_init, descending=True)

                    #Create N more latents by iteratively masking according to the order of sorted indices:
                    obs_iter = tree_map(lambda x: x, obs)
                    for i in range(len(sorted_indices[:self.config.run.surprise_depth])):
                        idx = sorted_indices[i] # 
                        # Use conditional masking for each key based on sorted order
                        for j, key_name in enumerate(available_keys):
                            # Check if this key should be masked at this step (idx matches the key's position)
                            should_mask = (idx == j)
                            obs_iter[key_name] = jnp.where(
                                should_mask,
                                jnp.zeros_like(obs_iter[key_name]),
                                obs_iter[key_name]
                            )
                        
                        embed = self.wm.encoder(obs_iter)
                        temp_latent, temp_prior_orig = self.wm.rssm.obs_step(
                            prev_latent, prev_action, embed, obs['is_first'])
                        surprise = self.wm.rssm.get_dist(temp_latent).kl_divergence(
                            self.wm.rssm.get_dist(temp_prior_orig))
                        
                        surprises.append(surprise)
                        total_newlat_seperate.append(temp_latent)
                
                total_newlat_seperate.append(latent)
                surprises.append(base_surprise)
                # Find index of minimum surprise
                surprises = jnp.array(surprises)
                min_idx = jnp.argmin(surprises)
                
                # Stack candidates and select using advanced indexing
                stacked_latents = jax.tree_map(
                    lambda *candidates: jnp.stack(candidates, axis=0),
                    *total_newlat_seperate
                )
            
                # Select the latent with lowest surprise
                latent = jax.tree_map(
                    lambda stacked: stacked[min_idx],
                    stacked_latents
                )
                latent = {k: v[min_idx] for k, v in stacked_latents.items() if k != "image_key"} # I dont think this does anything.

            elif 'random' in mode: # mode == 'random':
                k = py_rand.randint(0, len(available_keys))  # actual number of keys chosen
                chosen_keys = py_rand.sample(available_keys, k)
                obs_masked = jax.tree_map(lambda x: x, obs)
                for ite, image_key in enumerate(chosen_keys):
                    obs_masked[image_key] = jnp.zeros(obs[image_key].shape, dtype=float)
                embed = self.wm.encoder(obs_masked)
                latent, temp_prior_orig = self.wm.rssm.obs_step(
                    prev_latent, prev_action, embed, obs['is_first'])
                
            elif 'reject' in mode:
                # Smart gradient-based dropout instead of brute force search
                surprises = []
                total_newlat_seperate = []
                # First, get baseline surprise with original observation
                base_surprise = self.wm.rssm.get_dist(latent).kl_divergence(self.wm.rssm.get_dist(prior_orig))
                # Get image key
                image_key = self.wm.config.encoder.cnn_keys#[0]
    
                batch_size = 1  # or however you determine batch size
                is_first = jnp.ones((batch_size,), dtype=jnp.float32)
                
                #P(z|x,h_0)
                dropped_residual_latent, dropped_prior_orig = self.wm.rssm.obs_step(
                    prev_latent, prev_action, embed, is_first
                )
                #X_po = E[P(x|z_t,h_0)] where z_t ~ P(z|x,h_0)
                truth, reconstruct_post_dropped, reconstruct_post = self.get_recon_imgs(obs, dropped_residual_latent, latent, image_key)
                prior_img = self.get_single_recon(prior_orig, image_key)

                #Check if the image looks clean
                #||X_t - X_po||
                tau = self.config.run.reject_tau 
                recon_score = jnp.mean(jnp.abs(obs[image_key] - reconstruct_post_dropped))
                ablation_m = self.config.run.off_shelf_mx #True or false.
                
                if self.config.run.denoise_method == 'denoiser':
                    interpolated_img = obs['denoised']
                elif self.config.run.denoise_method == 'interpolate':
                    interpolated_img = self.interpolate(prev_latent,prev_action,obs,image_key,prior_orig)
                else: #Default
                    interpolated_img = reconstruct_post

                if ablation_m == False:
                    condition = recon_score < tau
                else:#ABLATION M(X):
                    condition = base_surprise[0] < tau #obs['mx'][0]
                

                sampled_obs = tree_map(lambda x: x, obs)
                sampled_obs[image_key] = interpolated_img
                interpolated_img = jnp.array(interpolated_img)

                embed = self.wm.encoder(sampled_obs)
                interpolated_img_post_dropped, temp_prior_orig_dropped = self.wm.rssm.obs_step(
                    prev_latent, prev_action, embed, is_first
                )
                
                recon_interpolated_img_dropped = self.get_single_recon(interpolated_img_post_dropped,image_key)

                interpolated_img_post, temp_prior_orig = self.wm.rssm.obs_step(
                    prev_latent, prev_action, embed, obs['is_first']
                )
                recon_interpolated_img = self.get_single_recon(interpolated_img_post,image_key)

                #M(\bar{X})
                recon_score_2 = jnp.mean(jnp.abs(interpolated_img - recon_interpolated_img_dropped))
                condition_2 = recon_score_2 < tau

                #Does h_t align with x_bar?
                recon_score_3 = jnp.mean(jnp.abs(interpolated_img - recon_interpolated_img))
                condition_3 = recon_score_3 < tau
                latent, action_latent, stage = jax.lax.cond(
                    stage == jnp.array(0),
                    # When stage == 0
                    lambda: jax.lax.cond(
                        condition,  # If truth is good keep using.
                        lambda: (latent, latent, jnp.array(0)),
                        lambda: jax.lax.cond(
                            jnp.logical_and(condition_3, condition_2),  # Does h_t align with x_bar? Can we at least use x_bar?
                            lambda: (interpolated_img_post, interpolated_img_post, jnp.array(1)), #(prior_orig, interpolated_img_post, jnp.array(1)),
                            lambda: (prior_orig, prior_orig, jnp.array(2))
                        )
                    ),
                    # When stage != 0 (assuming stage == 1)
                    lambda: jax.lax.cond(
                        condition,  # Can we recover to x_t
                        lambda: (dropped_residual_latent, dropped_residual_latent, jnp.array(0)),
                        lambda: jax.lax.cond(
                            jnp.logical_and(condition_3, condition_2),  # Does h_t still align with x_bar?
                            lambda: (interpolated_img_post, interpolated_img_post, jnp.array(3)),
                            lambda: jax.lax.cond(  # h_t and x_bar are different, can we recover to x_bar?
                                condition_2,
                                lambda: (interpolated_img_post_dropped, interpolated_img_post_dropped, jnp.array(4)),
                                lambda: (prior_orig, prior_orig, jnp.array(5))
                            )
                        )
                    )
                )

                
        
                video = jnp.concatenate([truth, prior_img, reconstruct_post, reconstruct_post_dropped, interpolated_img], axis=2)
                
            elif 'filter' in mode:
                # Smart gradient-based dropout instead of brute force search
                surprises = []
                total_newlat_seperate = []
                # First, get baseline surprise with original observation
                base_surprise = self.wm.rssm.get_dist(latent).kl_divergence(self.wm.rssm.get_dist(prior_orig))

                def median_blur(image: jnp.ndarray, ksize: int) -> jnp.ndarray:
                    """
                    Apply median blur to a batch of images (NHWC format).

                    Args:
                        image: jnp.ndarray of shape (N, H, W, C)
                        ksize: odd integer kernel size (e.g. 3, 5, 7)

                    Returns:
                        Blurred image of same shape as input.
                    """
                    assert ksize % 2 == 1, "ksize must be odd"
                    pad = ksize // 2
                    N, H, W, C = image.shape

                    # Pad with reflect to mimic OpenCV behavior
                    padded = jnp.pad(
                        image, ((0, 0), (pad, pad), (pad, pad), (0, 0)), mode="reflect"
                    )

                    # Extract patches using dynamic slicing
                    def get_patch(n, i, j):
                        return jax.lax.dynamic_slice(
                            padded, (n, i, j, 0), (1, ksize, ksize, C)
                        ).reshape(-1, C)

                    # Meshgrid over image coords
                    ii, jj = jnp.meshgrid(jnp.arange(H), jnp.arange(W), indexing="ij")

                    # Vectorize over batch, height, width
                    patches = jax.vmap(
                        lambda n: jax.vmap(
                            lambda i_row, j_row: jax.vmap(
                                get_patch, in_axes=(None, 0, 0)
                            )(n, i_row, j_row),
                            in_axes=(0, 0),
                        )(ii, jj),
                        in_axes=(0,),
                    )(jnp.arange(N))  # (N, H, W, ksize*ksize, C)

                    # Take median across neighborhood axis
                    result = jnp.median(patches, axis=3)  # (N, H, W, C)
                    return result
                
                # Get image key
                image_key = self.wm.config.encoder.cnn_keys#[0]

                obs[image_key] = median_blur(obs[image_key],ksize=3)
                embed = self.wm.encoder(obs)
                temp_latent, temp_prior_orig = self.wm.rssm.obs_step(
                    prev_latent, prev_action, embed, obs['is_first']
                )
                latent = temp_latent

        if mode == 'reject':
            self.expl_behavior.policy(action_latent, expl_state)
            task_outs, task_state = self.task_behavior.policy(action_latent, task_state)
            expl_outs, expl_state = self.expl_behavior.policy(action_latent, expl_state)
            
        else:
            self.expl_behavior.policy(latent, expl_state)
            task_outs, task_state = self.task_behavior.policy(latent, task_state)
            expl_outs, expl_state = self.expl_behavior.policy(latent, expl_state)

        def reshape_outputs(surprises_dict):
            reshaped = {}
            for k, v in surprises_dict.items():
                if k.startswith('log_surprise'):
                    # reshape scalar [] or any shape to [1]
                    reshaped[k] = jnp.reshape(v, (1,))
                if k.startswith('mu_gradients'):
                    reshaped[k] = jnp.reshape(v, (1,))
                else:
                    reshaped[k] = v
            return reshaped

        if mode == "eval":
            outs = task_outs
            outs["action"] = outs["action"].sample(seed=nj.rng())
            outs["log_entropy"] = jnp.zeros(outs["action"].shape[:1])
            surprise = lambda post, prior: self.wm.rssm.get_dist(post).kl_divergence(
                self.wm.rssm.get_dist(prior)
            )
            image_key = 'birdeye_wpt'
            truth, reconstruct_post, prior_img  = self.get_recon_imgs(obs, latent, prior_orig, image_key)
            video = jnp.concatenate([truth, prior_img, reconstruct_post], axis=2)
            video = jnp.expand_dims(video, 0)
            # metrics = {}
            # outs.update({f"openl_{key}" : jaxutils.video_grid(video)})
            # outs.update(jaxutils.tensorstats(surprise(latent, prior_orig), "log_surprise"))
            # print(outs)

            outs = reshape_outputs(outs)
            print({k: v.shape for k, v in outs.items()})

            outs.update({f"openl_custom_{image_key}" : jaxutils.video_grid(video)})

        elif mode == "explore":
            outs = expl_outs
            outs["log_entropy"] = outs["action"].entropy()
            outs["action"] = outs["action"].sample(seed=nj.rng())
        elif mode == "train":
            outs = task_outs
            outs["log_entropy"] = outs["action"].entropy()
            outs["action"] = outs["action"].sample(seed=nj.rng())
        else: #Run eval by default
            outs = task_outs
            if mode in ['sample','reject']:
                video = jnp.expand_dims(video, 0)
                # interpolated_img = jnp.expand_dims(interpolated_img, 0)
                # mean_gradients = reshape_outputs({"mu_gradients": mean_gradients})["mu_gradients"]
                # outs.update({f"openl_custom_interpolated_{image_key}": jaxutils.video_grid(interpolated_img)})
                outs.update({f"openl_custom_{image_key}" : jaxutils.video_grid(video)})
                print(stage, recon_score, recon_score_2, recon_score_3)
                outs.update({f"stages":jnp.reshape(stage, (1,))})
                outs.update({f"condition_1":jnp.reshape(recon_score, (1,))})
                outs.update({f"condition_2":jnp.reshape(recon_score_2, (1,))})
                outs.update({f"condition_3":jnp.reshape(recon_score_3, (1,))})


            outs["action"] = outs["action"].sample(seed=nj.rng())
            outs["log_entropy"] = jnp.zeros(outs["action"].shape[:1])

        state = ((latent, outs["action"]), task_state, expl_state, stage, step)
        return outs, state

    def train(self, data, state):
        self.config.jax.jit and print("Tracing train function.")
        metrics = {}
        data = self.preprocess(data)
        state, wm_outs, mets = self.wm.train(data, state)
        metrics.update(mets)
        context = {**data, **wm_outs["post"]}
        start = tree_map(lambda x: x.reshape([-1] + list(x.shape[2:])), context)
        _, mets = self.task_behavior.train(self.wm.imagine, start, context)
        metrics.update(mets)
        if self.config.expl_behavior != "None":
            _, mets = self.expl_behavior.train(self.wm.imagine, start, context)
            metrics.update({"expl_" + key: value for key, value in mets.items()})

        if "keyA" in data.keys():
            outs = {
                "key": data["key"],
                "env_step": data["env_step"],
                "model_loss": metrics["model_loss_raw"].copy(),
                "td_error": metrics["td_error"].copy(),
            }

        else:
            outs = {}

        # Don't need the full model_loss_raw or td_error after the priority calculation, summarize it.
        metrics.update({"model_loss_raw": metrics["model_loss_raw"].mean()})
        metrics.update({"td_error": metrics["td_error"].mean()})

        return outs, state, metrics

    def report(self, data):
        self.config.jax.jit and print("Tracing report function.")
        data = self.preprocess(data)
        report = {}
        report.update(self.wm.report(data))
        mets = self.task_behavior.report(data)
        report.update({f"task_{k}": v for k, v in mets.items()})
        if self.expl_behavior is not self.task_behavior:
            mets = self.expl_behavior.report(data)
            report.update({f"expl_{k}": v for k, v in mets.items()})
        return report

    def preprocess(self, obs):
        obs = obs.copy()
        for key, value in obs.items():
            if key.startswith("log_") or key in ("key", "env_step"):
                continue
            if len(value.shape) > 3 and value.dtype == jnp.uint8:
                value = jaxutils.cast_to_compute(value) / 255.0
            else:
                value = value.astype(jnp.float32)
            obs[key] = value
        obs["cont"] = 1.0 - obs["is_terminal"].astype(jnp.float32)
        return obs


class WorldModel(nj.Module):
    def __init__(self, obs_space, act_space, config):
        self.obs_space = obs_space
        self.act_space = act_space["action"]
        self.config = config
        shapes = {k: tuple(v.shape) for k, v in obs_space.items()}
        shapes = {k: v for k, v in shapes.items() if not k.startswith("log_")}
        self.encoder = nets.MultiEncoder(shapes, **config.encoder, name="enc")
        self.rssm = nets.RSSM(**config.rssm, name="rssm")
        self.heads = {
            "decoder": nets.MultiDecoder(shapes, **config.decoder, name="dec"),
            "reward": nets.MLP((), **config.reward_head, name="rew"),
            "cont": nets.MLP((), **config.cont_head, name="cont"),
        }
        self.opt = jaxutils.Optimizer(name="model_opt", **config.model_opt)
        scales = self.config.loss_scales.copy()
        image, vector = scales.pop("image"), scales.pop("vector")
        scales.update({k: image for k in self.heads["decoder"].cnn_shapes})
        scales.update({k: vector for k in self.heads["decoder"].mlp_shapes})
        self.scales = scales
        self.stage = 0

    def initial(self, batch_size):
        prev_latent = self.rssm.initial(batch_size)
        prev_action = jnp.zeros((batch_size, *self.act_space.shape))
        return prev_latent, prev_action

    def train(self, data, state):
        modules = [self.encoder, self.rssm, *self.heads.values()]
        mets, (state, outs, metrics) = self.opt(modules, self.loss, data, state, has_aux=True)
        metrics.update(mets)
        return state, outs, metrics
    
    def randomly_mask_images_per_timestep(self, data, key, mask_value=0.0):
        """
        Randomly mask 0 to n-1 images for each timestep, ensuring at least one image remains unmasked.
        
        Args:
            data: Dictionary containing the dataset with image keys
            key: JAX random key
            mask_value: Value to use for masked pixels (default: 0.0)
        
        Returns:
            Modified data dictionary with randomly masked images
        """
        # Create a copy of the data to avoid modifying the original
        masked_data = data.copy()
        
        # Image keys to process
        image_keys = [
            "birdeye_gt",
            "birdeye_raw",
            "birdeye_with_traffic_lights",
            "birdeye_wpt",
            "camera",
            "lidar"
        ]
        available_keys = [k for k in image_keys if k in data]
        # print('Loss on Keys: ',available_keys)
        if len(available_keys) == 0:
            return masked_data
        
        n_images = len(available_keys)
        batch_size, seq_len = data[available_keys[0]].shape[0], data[available_keys[0]].shape[1]  # 16, 64
        
        # Split keys
        key1, key2 = random.split(key)
        
        # For each (batch, timestep), randomly choose how many images to mask (0 to n-1)
        num_to_mask = random.randint(key1, shape=(batch_size, seq_len), minval=0, maxval=n_images)
        
        # Generate random values for each image at each (batch, timestep)
        # Shape: [batch_size, seq_len, n_images]
        random_vals = random.uniform(key2, shape=(batch_size, seq_len, n_images))
        
        # Sort the random values to get rankings (0 = smallest, n_images-1 = largest)
        rankings = jnp.argsort(random_vals, axis=-1)
        
        # Create a mask where we mask images with ranking < num_to_mask
        # This gives us a random selection of num_to_mask images
        mask_matrix = jnp.zeros((batch_size, seq_len, n_images), dtype=bool)
        
        for i in range(n_images):
            # For each image position, check if its ranking is less than num_to_mask
            image_rank = jnp.where(rankings == i, 
                                jnp.arange(n_images)[None, None, :], 
                                n_images)  # Set to n_images if not this image
            min_rank = jnp.min(image_rank, axis=-1)  # Get the ranking for image i
            should_mask = min_rank < num_to_mask
            mask_matrix = mask_matrix.at[:, :, i].set(should_mask)
        
        # Apply masks to each image
        for i, img_key in enumerate(available_keys):
            images = data[img_key]
            
            # Expand mask to match image dimensions
            mask = mask_matrix[:, :, i].reshape(batch_size, seq_len, 1, 1, 1)
            
            # Apply mask: where mask is True, replace with mask_value
            masked_images = jnp.where(mask, mask_value, images)
            
            # Update the data dictionary
            masked_data[img_key] = masked_images
            # print('Masking...')
        
        return masked_data
  
    def randomly_noise_images_per_timestep(self, data, key, mask_value=0.0):
        """
        Randomly mask 0 to n-1 images for each timestep, ensuring at least one image remains unmasked.
        
        Args:
            data: Dictionary containing the dataset with image keys
            key: JAX random key
            mask_value: Value to use for masked pixels (default: 0.0)
        
        Returns:
            Modified data dictionary with randomly masked images
        """
        # Create a copy of the data to avoid modifying the original
        masked_data = data.copy()
        
        # Image keys to process
        image_keys = [
            "birdeye_gt",
            "birdeye_raw",
            "birdeye_with_traffic_lights",
            "birdeye_wpt",
            "camera",
            "lidar"
        ]
        
        available_keys = [k for k in image_keys if k in data]
        
        if len(available_keys) == 0:
            return masked_data
        
        n_images = len(available_keys)
        batch_size, seq_len = data[available_keys[0]].shape[0], data[available_keys[0]].shape[1]  # 16, 64
        
        # Split keys
        key1, key2 = random.split(key)
        
        # For each (batch, timestep), randomly choose how many images to mask (0 to n-1)
        num_to_mask = random.randint(key1, shape=(batch_size, seq_len), minval=0, maxval=n_images)
        
        # Generate random values for each image at each (batch, timestep)
        # Shape: [batch_size, seq_len, n_images]
        random_vals = random.uniform(key2, shape=(batch_size, seq_len, n_images))
        
        # Sort the random values to get rankings (0 = smallest, n_images-1 = largest)
        rankings = jnp.argsort(random_vals, axis=-1)
        
        # Create a mask where we mask images with ranking < num_to_mask
        # This gives us a random selection of num_to_mask images
        mean = 20.0
        std = 30.0
        mask_matrix = mean + std * random.normal(key2, (batch_size, seq_len, n_images))  # mean=0, std=1 #jnp.zeros((batch_size, seq_len, n_images), dtype=bool)
        
        for i in range(n_images):
            # For each image position, check if its ranking is less than num_to_mask
            image_rank = jnp.where(rankings == i, 
                                jnp.arange(n_images)[None, None, :], 
                                n_images)  # Set to n_images if not this image
            min_rank = jnp.min(image_rank, axis=-1)  # Get the ranking for image i
            should_mask = min_rank < num_to_mask
            mask_matrix = mask_matrix.at[:, :, i].set(should_mask)
        
        # Apply masks to each image
        for i, img_key in enumerate(available_keys):
            images = data[img_key]
            
            # Expand mask to match image dimensions
            mask = mask_matrix[:, :, i].reshape(batch_size, seq_len, 1, 1, 1)
            
            # Apply mask: where mask is True, replace with mask_value
            masked_images = jnp.where(mask, mask_value, images)
            
            # Update the data dictionary
            masked_data[img_key] = masked_images
            
        
        return masked_data

    def loss(self, data, state):
        key = random.PRNGKey(42)
        if self.config.run.dropout_training: # Set to true to do dropout representation training
            enc_data = self.randomly_mask_images_per_timestep(data, key, mask_value=0.0)
        else:
            enc_data = data
        embed = self.encoder(enc_data)
        prev_latent, prev_action = state
        prev_actions = jnp.concatenate([prev_action[:, None], data["action"][:, :-1]], 1)
        post, prior = self.rssm.observe(embed, prev_actions, data["is_first"], prev_latent)
        dists = {}
        feats = {**post, "embed": embed}
        for name, head in self.heads.items():
            out = head(feats if name in self.config.grad_heads else sg(feats))
            out = out if isinstance(out, dict) else {name: out}
            dists.update(out)
        losses = {}
        losses["dyn"] = self.rssm.dyn_loss(post, prior, **self.config.dyn_loss)
        losses["rep"] = self.rssm.rep_loss(post, prior, **self.config.rep_loss)
        for key, dist in dists.items():
            loss = -dist.log_prob(data[key].astype(jnp.float32))
            assert loss.shape == embed.shape[:2], (key, loss.shape)
            losses[key] = loss
        scaled = {k: v * self.scales[k] for k, v in losses.items()}
        model_loss = sum(scaled.values())
        out = {"embed": embed, "post": post, "prior": prior}
        out.update({f"{k}_loss": v for k, v in losses.items()})
        last_latent = {k: v[:, -1] for k, v in post.items()}
        last_action = data["action"][:, -1]
        state = last_latent, last_action
        metrics = self._metrics(data, dists, post, prior, losses, model_loss)
        metrics["model_loss_raw"] = model_loss  # Store model loss for Curious Replay prioritization
        return model_loss.mean(), (state, out, metrics)

    def imagine(self, policy, start, horizon):
        first_cont = (1.0 - start["is_terminal"]).astype(jnp.float32)
        keys = list(self.rssm.initial(1).keys())
        start = {k: v for k, v in start.items() if k in keys}
        start["action"] = policy(start)

        def step(prev, _):
            prev = prev.copy()
            state = self.rssm.img_step(prev, prev.pop("action"))
            return {**state, "action": policy(state)}

        traj = jaxutils.scan(step, jnp.arange(horizon), start, self.config.imag_unroll)
        traj = {k: jnp.concatenate([start[k][None], v], 0) for k, v in traj.items()}
        cont = self.heads["cont"](traj).mode()
        traj["cont"] = jnp.concatenate([first_cont[None], cont[1:]], 0)
        discount = 1 - 1 / self.config.horizon
        traj["weight"] = jnp.cumprod(discount * traj["cont"], 0) / discount
        return traj

    def imagine_carry(self, policy, start, horizon, carry):
        first_cont = (1.0 - start["is_terminal"]).astype(jnp.float32)
        keys = list(self.rssm.initial(1).keys())
        start = {k: v for k, v in start.items() if k in keys}
        outs, carry = policy(start, carry)
        start["action"] = outs
        start["carry"] = carry

        def step(prev, _):
            prev = prev.copy()
            carry = prev.pop("carry")
            state = self.rssm.img_step(prev, prev.pop("action"))
            outs, carry = policy(state, carry)
            return {**state, "action": outs, "carry": carry}

        traj = jaxutils.scan(step, jnp.arange(horizon), start, self.config.imag_unroll)
        traj = {k: jnp.concatenate([start[k][None], v], 0) for k, v in traj.items() if k != "carry"}
        cont = self.heads["cont"](traj).mode()
        traj["cont"] = jnp.concatenate([first_cont[None], cont[1:]], 0)
        discount = 1 - 1 / self.config.horizon
        traj["weight"] = jnp.cumprod(discount * traj["cont"], 0) / discount
        return traj

    def report(self, data):
        state = self.initial(len(data["is_first"]))
        report = {}
        report.update(self.loss(data, state)[-1][-1])
        context, _ = self.rssm.observe(self.encoder(data)[:6, :5], data["action"][:6, :5], data["is_first"][:6, :5])
        start = {k: v[:, -1] for k, v in context.items()}
        recon = self.heads["decoder"](context)
        openl = self.heads["decoder"](self.rssm.imagine(data["action"][:6, 5:], start))
        for key in self.heads["decoder"].cnn_shapes.keys():
            truth = data[key][:6].astype(jnp.float32)
            model = jnp.concatenate([recon[key].mode()[:, :5], openl[key].mode()], 1)
            error = (model - truth + 1) / 2
            video = jnp.concatenate([truth, model, error], 2)
            report[f"openl_{key}"] = jaxutils.video_grid(video)
        return report

    def _metrics(self, data, dists, post, prior, losses, model_loss):
        entropy = lambda feat: self.rssm.get_dist(feat).entropy()
        metrics = {}
        metrics.update(jaxutils.tensorstats(entropy(prior), "prior_ent"))
        metrics.update(jaxutils.tensorstats(entropy(post), "post_ent"))
        metrics.update({f"{k}_loss_mean": v.mean() for k, v in losses.items()})
        metrics.update({f"{k}_loss_std": v.std() for k, v in losses.items()})
        metrics["model_loss_mean"] = model_loss.mean()
        metrics["model_loss_std"] = model_loss.std()
        metrics["reward_max_data"] = jnp.abs(data["reward"]).max()
        metrics["reward_max_pred"] = jnp.abs(dists["reward"].mean()).max()
        if "reward" in dists and not self.config.jax.debug_nans:
            stats = jaxutils.balance_stats(dists["reward"], data["reward"], 0.1)
            metrics.update({f"reward_{k}": v for k, v in stats.items()})
        if "cont" in dists and not self.config.jax.debug_nans:
            stats = jaxutils.balance_stats(dists["cont"], data["cont"], 0.5)
            metrics.update({f"cont_{k}": v for k, v in stats.items()})
        return metrics


class ImagActorCritic(nj.Module):
    def __init__(self, critics, scales, act_space, config):
        critics = {k: v for k, v in critics.items() if scales[k]}
        for key, scale in scales.items():
            assert not scale or key in critics, key
        self.critics = {k: v for k, v in critics.items() if scales[k]}
        self.scales = scales
        self.act_space = act_space
        self.config = config
        disc = act_space.discrete
        self.grad = config.actor_grad_disc if disc else config.actor_grad_cont
        self.actor = nets.MLP(
            name="actor",
            dims="deter",
            shape=act_space.shape,
            **config.actor,
            dist=config.actor_dist_disc if disc else config.actor_dist_cont,
        )
        self.retnorms = {k: jaxutils.Moments(**config.retnorm, name=f"retnorm_{k}") for k in critics}
        self.opt = jaxutils.Optimizer(name="actor_opt", **config.actor_opt)

    def initial(self, batch_size):
        return {}

    def policy(self, state, carry):
        return {"action": self.actor(state)}, carry

    def train(self, imagine, start, context):
        def loss(start):
            policy = lambda s: self.actor(sg(s)).sample(seed=nj.rng())
            traj = imagine(policy, start, self.config.imag_horizon)
            loss, metrics = self.loss(traj)
            return loss, (traj, metrics)

        mets, (traj, metrics) = self.opt(self.actor, loss, start, has_aux=True)
        metrics.update(mets)
        for key, critic in self.critics.items():
            mets = critic.train(traj, self.actor)
            metrics.update({f"{key}_critic_{k}": v for k, v in mets.items()})
        return traj, metrics

    def loss(self, traj):
        metrics = {}
        advs = []
        total = sum(self.scales[k] for k in self.critics)
        for key, critic in self.critics.items():
            rew, ret, base = critic.score(traj, self.actor)
            offset, invscale = self.retnorms[key](ret)
            normed_ret = (ret - offset) / invscale
            normed_base = (base - offset) / invscale
            advs.append((normed_ret - normed_base) * self.scales[key] / total)
            metrics.update(jaxutils.tensorstats(rew, f"{key}_reward"))
            metrics.update(jaxutils.tensorstats(ret, f"{key}_return_raw"))
            metrics.update(jaxutils.tensorstats(normed_ret, f"{key}_return_normed"))
            metrics[f"{key}_return_rate"] = (jnp.abs(ret) >= 0.5).mean()

        # if len(self.critics) != 1:
        #  raise NotImplementedError('Must have exactly one critic for TD error calculation.')

        r = jnp.reshape(rew[0], (self.config.batch_size, self.config.batch_length))
        v = jnp.reshape(base[0], (self.config.batch_size, self.config.batch_length))
        disc = jnp.reshape(traj["cont"][0], (self.config.batch_size, self.config.batch_length)) * (1 - 1 / self.config.horizon)
        td_error = r[:, :-1] + disc[:, 1:] * v[:, 1:] - v[:, :-1]
        metrics["td_error"] = td_error  # Store TD error for PER prioritization

        adv = jnp.stack(advs).sum(0)
        policy = self.actor(sg(traj))
        logpi = policy.log_prob(sg(traj["action"]))[:-1]
        loss = {"backprop": -adv, "reinforce": -logpi * sg(adv)}[self.grad]
        ent = policy.entropy()[:-1]
        loss -= self.config.actent * ent
        loss *= sg(traj["weight"])[:-1]
        loss *= self.config.loss_scales.actor
        metrics.update(self._metrics(traj, policy, logpi, ent, adv))
        return loss.mean(), metrics

    def _metrics(self, traj, policy, logpi, ent, adv):
        metrics = {}
        ent = policy.entropy()[:-1]
        rand = (ent - policy.minent) / (policy.maxent - policy.minent)
        rand = rand.mean(range(2, len(rand.shape)))
        act = traj["action"]
        act = jnp.argmax(act, -1) if self.act_space.discrete else act
        metrics.update(jaxutils.tensorstats(act, "action"))
        metrics.update(jaxutils.tensorstats(rand, "policy_randomness"))
        metrics.update(jaxutils.tensorstats(ent, "policy_entropy"))
        metrics.update(jaxutils.tensorstats(logpi, "policy_logprob"))
        metrics.update(jaxutils.tensorstats(adv, "adv"))
        metrics["imag_weight_dist"] = jaxutils.subsample(traj["weight"])
        return metrics


class VFunction(nj.Module):
    def __init__(self, rewfn, config):
        self.rewfn = rewfn
        self.config = config
        self.net = nets.MLP((), name="net", dims="deter", **self.config.critic)
        self.slow = nets.MLP((), name="slow", dims="deter", **self.config.critic)
        self.updater = jaxutils.SlowUpdater(
            self.net,
            self.slow,
            self.config.slow_critic_fraction,
            self.config.slow_critic_update,
        )
        self.opt = jaxutils.Optimizer(name="critic_opt", **self.config.critic_opt)

    def train(self, traj, actor):
        target = sg(self.score(traj)[1])
        mets, metrics = self.opt(self.net, self.loss, traj, target, has_aux=True)
        metrics.update(mets)
        self.updater()
        return metrics

    def loss(self, traj, target):
        metrics = {}
        traj = {k: v[:-1] for k, v in traj.items()}
        dist = self.net(traj)
        loss = -dist.log_prob(sg(target))
        if self.config.critic_slowreg == "logprob":
            reg = -dist.log_prob(sg(self.slow(traj).mean()))
        elif self.config.critic_slowreg == "xent":
            reg = -jnp.einsum("...i,...i->...", sg(self.slow(traj).probs), jnp.log(dist.probs))
        else:
            raise NotImplementedError(self.config.critic_slowreg)
        loss += self.config.loss_scales.slowreg * reg
        loss = (loss * sg(traj["weight"])).mean()
        loss *= self.config.loss_scales.critic
        metrics = jaxutils.tensorstats(dist.mean())
        return loss, metrics

    def score(self, traj, actor=None):
        rew = self.rewfn(traj)
        assert len(rew) == len(traj["action"]) - 1, "should provide rewards for all but last action"
        discount = 1 - 1 / self.config.horizon
        disc = traj["cont"][1:] * discount
        value = self.net(traj).mean()
        vals = [value[-1]]
        interm = rew + disc * value[1:] * (1 - self.config.return_lambda)
        for t in reversed(range(len(disc))):
            vals.append(interm[t] + disc[t] * self.config.return_lambda * vals[-1])
        ret = jnp.stack(list(reversed(vals))[:-1])
        return rew, ret, value[:-1]
