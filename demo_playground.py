import argparse
from copy import deepcopy

# from functools import partial
import os

import numpy as np
from omegaconf import OmegaConf
from stable_baselines3 import PPO, SAC
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.monitor import Monitor

from diffusion_policy.common.replay_buffer import ReplayBuffer
from playground.envs import make_point_maze_env


def main(env_config_path, policy_model_dir, output_dir, num_episodes):
    """
    Collect demonstration for the Playground task.

    Usage: python demo_playground.py -o data/pusht_demo.zarr

    This script is compatible with both Linux and MacOS.
    """

    # Load config
    config = OmegaConf.load(env_config_path)
    eval_config = OmegaConf.merge(config, config.evaluation)  # eval mode.

    # Create replay buffer in read-write mode
    replay_buffer = ReplayBuffer.create_from_path(output_dir, mode="a")

    # Create env
    eval_env = make_point_maze_env(eval_config)

    # Load policy model
    vec_env = DummyVecEnv(
        [lambda: Monitor(make_point_maze_env(config, use_new_gym_api=True))]
    )
    vec_env = VecNormalize.load(
        os.path.join(policy_model_dir, "vec_normalize.pkl"), vec_env
    )
    if config.rl.training.algorithm == "sac":
        model = SAC.load(os.path.join(policy_model_dir, "model.zip"), vec_env)
    elif config.rl.training.algorithm == "ppo":
        model = PPO.load(os.path.join(policy_model_dir, "model.zip"), vec_env)
    else:
        raise (NotImplementedError(config.rl.training.algorithm))

    # episode-level while loop
    while replay_buffer.n_episodes < num_episodes:
        episode = list()
        # record in seed order, starting with 0
        seed = replay_buffer.n_episodes
        print(f"starting seed {seed}")

        # reset env and get observations (including info and render for recording)
        obs = eval_env.reset(seed=seed)
        img = deepcopy(obs["rgb_stack"])
        if not config.env.return_rgb_observation:
            # Do not feed the image if it is not used during training
            del obs["rgb_stack"]
        if vec_env.normalize_obs:
            obs = vec_env.normalize_obs(obs)

        # loop state
        done = False
        # step-level while loop
        while not done:
            # get action from model
            act, _ = model.predict(obs, deterministic=True)
            state = np.concatenate(
                [obs["observation"], obs["achieved_goal"], obs["desired_goal"]]
            )
            data = {
                "img": img,
                "state": np.float32(state),
                "action": np.float32(act),
            }
            episode.append(data)

            # step env and render
            obs, _reward, done, _info = eval_env.step(act)
            img = deepcopy(obs["rgb_stack"])
            if not config.env.return_rgb_observation:
                # Do not feed the image if it is not used during trining
                del obs["rgb_stack"]
            if vec_env.normalize_obs:
                obs = vec_env.normalize_obs(obs)

        # save episode buffer to replay buffer (on disk)
        data_dict = dict()
        for key in episode[0].keys():
            data_dict[key] = np.stack([x[key] for x in episode])
        replay_buffer.add_episode(data_dict, compressors="disk")
        print(f"saved seed {seed}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Script to generate demonstraiton data for diffusion policy."
    )
    parser.add_argument(
        "--rl_model_name",
        type=str,
        required=True,
        help=(
            "Model name of the RL policy. The model must be located under "
            "`playground/logs/<model_name>`"
        ),
    )
    parser.add_argument(
        "--zarr_name",
        type=str,
        required=True,
        help=(
            "Name of the Zarr file to save demonstration data. The data will "
            "be created at `data/playground/<zarr_name>_<model_name>.zarr`."
        ),
    )
    parser.add_argument(
        "--num_episodes",
        type=int,
        default=200,
        help=("Number of demonstration trajectories. Defaults to 200."),
    )
    args = parser.parse_args()
    policy_model_name = args.rl_model_name
    zarr_name = f"{args.zarr_name}_{args.rl_model_name}.zarr"
    num_episodes = args.num_episodes

    current_dir = os.path.dirname(os.path.realpath(__file__))
    env_config_path = os.path.join(
        current_dir, "../playground/configs/config.yaml"
    )
    policy_model_dir = os.path.join(
        current_dir, f"../playground/logs/{policy_model_name}/models"
    )
    output_dir = os.path.join(current_dir, f"../data/playground/{zarr_name}")
    main(env_config_path, policy_model_dir, output_dir, num_episodes)
