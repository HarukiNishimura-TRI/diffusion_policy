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


def filter_this_state(state: np.ndarray):
    # TODO: make this function configurable.
    if not len(state) >= 2:
        raise ValueError("In-valid state specification.")
    x = state[0]
    y = state[1]
    # filter = (0.0 < x < 2.0) & (-2.0 < y < 0.0)
    filter = (0.0 < x < 2.0) & (0.0 < y < 2.0)
    return filter


def main(
    env_config_path: str,
    policy_model_dir: str,
    output_dir: str,
    num_episodes: str,
    action_includes_progress: bool = False,
    filter_data: bool = False,
    init_seed: int = 0,
):
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
    seed = init_seed
    while replay_buffer.n_episodes < num_episodes:
        episode = list()
        # record in seed order, starting with 0
        print(f"starting seed {seed}")

        # reset env and get observations (including info and render for recording)
        obs = eval_env.reset(seed=seed)
        if filter_this_state(obs["achieved_goal"]):
            print(
                "Skipping this seed where initial (x, y) == "
                f"({obs['achieved_goal'][0]}, {obs['achieved_goal'][1]})"
            )
            seed += 1
            continue
        init_img = deepcopy(obs["rgb_stack"])
        img = deepcopy(init_img)
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
                "img": np.concatenate([img, init_img], axis=-1),
                "achieved_goal": np.float32(obs["achieved_goal"]),
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

        if action_includes_progress:
            # Add normalized task progress to the action.
            for idx, progress in enumerate(
                np.linspace(0.0, 1.0, len(episode))
            ):
                data = episode[idx]
                data["action"] = np.concatenate([data["action"], [progress]])

        if filter_data:
            # Filter out data
            filtered_episode = list()
            for data in episode:
                if not filter_this_state(data["achieved_goal"]):
                    filtered_episode.append(data)
            if len(filtered_episode) < len(episode):
                filtered_steps = len(episode) - len(filtered_episode)
                print(
                    f"Filtered {filtered_steps} steps "
                    f"({round(filtered_steps / len(episode) * 100, 1)}% reduction)."
                )
            episode = filtered_episode

        # save episode buffer to replay buffer (on disk)
        data_dict = dict()
        for key in episode[0].keys():
            data_dict[key] = np.stack([x[key] for x in episode])
        replay_buffer.add_episode(data_dict, compressors="disk")
        print(f"saved seed {seed}")
        seed += 1


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
    parser.add_argument(
        "--action_includes_progress",
        action="store_true",
        default=False,
        help=(
            "If True, action includes normalized task progress. Defaults to False."
        ),
    )
    parser.add_argument(
        "--filter_data",
        action="store_true",
        default=False,
        help=("If True, data is filtered so certain states become OOD."),
    )
    parser.add_argument(
        "--init_seed",
        type=int,
        default=0,
        help=("Initial seed value for demonstration. Defaults to 0."),
    )
    args = parser.parse_args()
    policy_model_name = args.rl_model_name
    action_includes_progress = args.action_includes_progress
    if action_includes_progress:
        zarr_name = f"{args.zarr_name}_{args.rl_model_name}_with_progress.zarr"
    else:
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
    filter_data = args.filter_data
    init_seed = args.init_seed
    main(
        env_config_path,
        policy_model_dir,
        output_dir,
        num_episodes,
        action_includes_progress,
        filter_data,
        init_seed,
    )
