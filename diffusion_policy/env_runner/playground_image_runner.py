from copy import deepcopy

import wandb
import numpy as np
import torch
import collections
import pathlib
import tqdm
import dill
import wandb.sdk.data_types.video as wv
from omegaconf import OmegaConf

# from diffusion_policy.env.pusht.pusht_image_env import PushTImageEnv
# from diffusion_policy.gym_util.async_vector_env import AsyncVectorEnv
# from diffusion_policy.gym_util.sync_vector_env import SyncVectorEnv
from diffusion_policy.gym_util.multistep_wrapper import MultiStepWrapper
from diffusion_policy.gym_util.video_recording_wrapper import (
    VideoRecordingWrapper,
    VideoRecorder,
)

from diffusion_policy.policy.base_image_policy import BaseImagePolicy
from diffusion_policy.common.pytorch_util import dict_apply
from diffusion_policy.env_runner.base_image_runner import BaseImageRunner

from playground.envs import make_point_maze_env


class PlayGroundImageRunner(BaseImageRunner):
    def __init__(
        self,
        config_path,
        output_dir,
        n_train=10,
        n_train_vis=3,
        train_start_seed=0,
        n_test=22,
        n_test_vis=6,
        test_start_seed=10000,
        max_steps=200,
        n_obs_steps=8,
        n_action_steps=8,
        fps=10,
        crf=22,
        tqdm_interval_sec=5.0,
    ):
        super().__init__(output_dir)
        n_envs = n_train + n_test

        steps_per_render = max(10 // fps, 1)
        config = OmegaConf.load(config_path)
        eval_config = OmegaConf.merge(config, config.evaluation)  # eval mode

        def env_fn():
            return MultiStepWrapper(
                VideoRecordingWrapper(
                    make_point_maze_env(eval_config),
                    video_recoder=VideoRecorder.create_h264(
                        fps=fps,
                        codec="h264",
                        input_pix_fmt="rgb24",
                        crf=crf,
                        thread_type="FRAME",
                        thread_count=1,
                    ),
                    file_path=None,
                    steps_per_render=steps_per_render,
                ),
                n_obs_steps=n_obs_steps,
                n_action_steps=n_action_steps,
                max_episode_steps=max_steps,
            )

        # env_fns = [env_fn] * n_envs
        env_seeds = list()
        env_prefixs = list()
        env_init_fn_dills = list()
        # train
        for i in range(n_train):
            seed = train_start_seed + i
            enable_render = i < n_train_vis

            def init_fn(env, seed=seed, enable_render=enable_render):
                # setup rendering
                # video_wrapper
                assert isinstance(env.env, VideoRecordingWrapper)
                env.env.video_recoder.stop()
                env.env.file_path = None
                if enable_render:
                    filename = pathlib.Path(output_dir).joinpath(
                        "media", wv.util.generate_id() + ".mp4"
                    )
                    filename.parent.mkdir(parents=False, exist_ok=True)
                    filename = str(filename)
                    env.env.file_path = filename

                # set seed
                assert isinstance(env, MultiStepWrapper)
                env.seed(seed)

            env_seeds.append(seed)
            env_prefixs.append("train/")
            env_init_fn_dills.append(dill.dumps(init_fn))

        # test
        for i in range(n_test):
            seed = test_start_seed + i
            enable_render = i < n_test_vis

            def init_fn(env, seed=seed, enable_render=enable_render):
                # setup rendering
                # video_wrapper
                assert isinstance(env.env, VideoRecordingWrapper)
                env.env.video_recoder.stop()
                env.env.file_path = None
                if enable_render:
                    filename = pathlib.Path(output_dir).joinpath(
                        "media", wv.util.generate_id() + ".mp4"
                    )
                    filename.parent.mkdir(parents=False, exist_ok=True)
                    filename = str(filename)
                    env.env.file_path = filename

                # set seed
                assert isinstance(env, MultiStepWrapper)
                env.seed(seed)

            env_seeds.append(seed)
            env_prefixs.append("test/")
            env_init_fn_dills.append(dill.dumps(init_fn))

        envs = [env_fn() for _ in range(n_envs)]
        # env = SyncVectorEnv(env_fns)

        # test env
        # env.reset(seed=env_seeds)
        # x = env.step(env.action_space.sample())
        # imgs = env.call('render')
        # import pdb; pdb.set_trace()

        self.envs = envs
        # self.env = env
        # self.env_fns = env_fns
        self.env_seeds = env_seeds
        self.env_prefixs = env_prefixs
        self.env_init_fn_dills = env_init_fn_dills
        self.fps = fps
        self.crf = crf
        self.n_obs_steps = n_obs_steps
        self.n_action_steps = n_action_steps
        self.max_steps = max_steps
        self.tqdm_interval_sec = tqdm_interval_sec

    def run(self, policy: BaseImagePolicy):
        device = policy.device
        dtype = policy.dtype
        envs = self.envs

        # plan for rollout
        n_envs = len(self.envs)

        # allocate data
        all_video_paths = [None] * n_envs
        all_rewards = [None] * n_envs

        # init envs
        for this_env, this_init_fn in zip(envs, self.env_init_fn_dills):
            this_env.run_dill_function(this_init_fn)

        pbar = tqdm.tqdm(
            total=self.max_steps,
            desc=f"Eval PlayGroundImageRunner",
            leave=False,
            mininterval=self.tqdm_interval_sec,
        )

        def aggregate_obs(obs_list, init_obs_list):
            if not len(obs_list) == len(init_obs_list):
                raise (
                    ValueError(
                        "obs_list and init_obs_list must have the same length."
                    )
                )
            image_list = []
            for idx in range(len(obs_list)):
                obs = obs_list[idx]
                np_image = np.moveaxis(
                    obs["rgb_stack"].astype(np.float32) / 255,
                    -1,
                    1,
                )
                init_obs = init_obs_list[idx]
                init_np_image = np.moveaxis(
                    init_obs["rgb_stack"].astype(np.float32) / 255,
                    -1,
                    1,
                )
                np_image = np.concatenate([np_image, init_np_image], axis=-3)
                image_list.append(np_image)
            np_image = np.stack(image_list, axis=0)
            obs = {"image": np_image}
            return obs

        # start rollout
        obs_list = []
        for this_env in envs:
            this_obs = this_env.reset()
            obs_list.append(this_obs)
        init_obs_list = deepcopy(obs_list)
        obs_aggregated = aggregate_obs(obs_list, init_obs_list)
        policy.reset()

        done_list = [False for _ in range(n_envs)]
        reward_list = [0 for _ in range(n_envs)]
        while not np.all(done_list):
            # create obs dict
            np_obs_dict = dict(obs_aggregated)
            # device transfer
            obs_dict = dict_apply(
                np_obs_dict,
                lambda x: torch.from_numpy(x).to(device=device),
            )
            # run policy
            with torch.no_grad():
                action_dict = policy.predict_action(obs_dict)
            # device_transfer
            np_action_dict = dict_apply(
                action_dict, lambda x: x.detach().to("cpu").numpy()
            )
            action = np_action_dict["action"]
            for env_idx, this_env in enumerate(envs):
                n_act = this_env.action_space.shape[-1]
                this_action = action[env_idx][..., :n_act]
                # step env
                obs, reward, done, _info = this_env.step(this_action)
                done_list[env_idx] = done
                obs_list[env_idx] = obs
                reward_list[env_idx] += reward
            obs_aggregated = aggregate_obs(obs_list, init_obs_list)

            # update pbar
            pbar.update(action.shape[1])
            # pbar.update(action.shape[1])
        pbar.close()

        for env_idx, this_env in enumerate(envs):
            all_video_paths[env_idx] = this_env.render()
            all_rewards[env_idx] = reward_list[env_idx]
        # clear out video buffer
        for this_env in envs:
            _ = this_env.reset()

        # log
        max_rewards = collections.defaultdict(list)
        log_data = dict()
        # results reported in the paper are generated using the commented out line below
        # which will only report and average metrics from first n_envs initial condition and seeds
        # fortunately this won't invalidate our conclusion since
        # 1. This bug only affects the variance of metrics, not their mean
        # 2. All baseline methods are evaluated using the same code
        # to completely reproduce reported numbers, uncomment this line:
        # for i in range(len(self.env_fns)):
        # and comment out this line
        for i in range(n_envs):
            seed = self.env_seeds[i]
            prefix = self.env_prefixs[i]
            max_reward = np.max(all_rewards[i])
            max_rewards[prefix].append(max_reward)
            log_data[prefix + f"sim_max_reward_{seed}"] = max_reward

            # visualize sim
            video_path = all_video_paths[i]
            if video_path is not None:
                sim_video = wandb.Video(video_path)
                log_data[prefix + f"sim_video_{seed}"] = sim_video

        # log aggregate metrics
        for prefix, value in max_rewards.items():
            name = prefix + "mean_score"
            value = np.mean(value)
            log_data[name] = value

        return log_data
