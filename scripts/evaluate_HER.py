import pprint
import glob

import os
import sys
ROOT_DIR = os.path.abspath('../')
sys.path.append(ROOT_DIR)
print("********* cwd {} *********".format(ROOT_DIR))

import argparse

import gym
import gym_uwrt_arm

import numpy as np

from gym.wrappers import FlattenObservation
from gym.wrappers import TimeLimit
from gym_uwrt_arm.wrappers.discrete_to_continuous_dict_action_wrapper import DiscreteToContinuousDictActionWrapper
from gym_uwrt_arm.wrappers.flatten_action_space import FlattenAction

from stable_baselines3 import HER, HER
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.vec_env import VecEnv

import config


def _load_latest_model(env):
    saved_model_file_mtime = None
    last_modified_checkpoint_file_mtime = None

    saved_model_file_path = config.BASE_SAVE_PATH / f'{config.GYM_ID}-her-trained-model.zip'
    if saved_model_file_path.is_file():
        saved_model_file_mtime = os.path.getmtime(saved_model_file_path)

    checkpoints_dir = config.BASE_SAVE_PATH / config.CHECKPOINTS_DIR
    if checkpoints_dir.is_dir():
        checkpoint_files = glob.glob(f'{checkpoints_dir}/{config.GYM_ID}-her-trained-model_*_steps.zip')

        if checkpoint_files:
            checkpoint_files.sort(key=os.path.getmtime)
            last_modified_checkpoint_file_path = checkpoint_files[-1]
            last_modified_checkpoint_file_mtime = os.path.getmtime(last_modified_checkpoint_file_path)

    if saved_model_file_mtime and last_modified_checkpoint_file_mtime:
        if saved_model_file_mtime >= last_modified_checkpoint_file_mtime:
            choice = 'saved_model'
        else:
            choice = 'saved_checkpoint'
    elif saved_model_file_mtime:
        choice = 'saved_model'
    elif last_modified_checkpoint_file_mtime:
        choice = 'saved_checkpoint'
    else:
        raise

    # TODO
    # saved_model_file_path = '/home/akeaveny/git/uwrt_arm_rl/trained_models/HER_1/best_mean_reward_model.zip'
    # last_modified_checkpoint_file_path = '/home/akeaveny/git/uwrt_arm_rl/trained_models/HER_1/best_mean_reward_model.zip'
    if choice == 'saved_model':
        print(f'Loading Saved Model from {saved_model_file_path}')
        model = HER.load(path=saved_model_file_path, env=env, verbose=1)
    elif choice == 'saved_checkpoint':
        print(f'Loading Checkpoint from {last_modified_checkpoint_file_path}')
        model = HER.load(path=last_modified_checkpoint_file_path, env=env, verbose=1)

    return model


def main(args):
    pp = pprint.PrettyPrinter()  # TODO: update to python 3.8 to use sort_dicts = False
    #################
    # Vec Norm
    #################

    if args.vecnorm:
        print("Evaluating with Vec Normalize!")
        evaluation_env = DummyVecEnv([lambda:
                                    (TimeLimit(FlattenAction(
                                        gym.make(config.GYM_ID, key_position=config.KEY_POSITION,
                                                 key_orientation=config.KEY_ORIENTATION,
                                                 max_steps=config.MAX_STEPS_PER_EPISODE, enable_render=True)),
                                        max_episode_steps=config.MAX_STEPS_PER_EPISODE))])

        saved_vecenv_file_path = (config.BASE_SAVE_PATH / config.STATISTICS_LOG_DIR / 'vec_normalize.pkl')
        assert saved_vecenv_file_path.is_file(), "Do not have a save stats file .."
        evaluation_env = VecNormalize.load(saved_vecenv_file_path, evaluation_env)
        #  do not update them at test time
        evaluation_env.training = False
        # reward normalization is not needed at test time
        evaluation_env.norm_reward = False

    else:
        evaluation_env = TimeLimit(FlattenAction(
                            gym.make(config.GYM_ID, key_position=config.KEY_POSITION,
                                     key_orientation=config.KEY_ORIENTATION,
                                     max_steps=config.MAX_STEPS_PER_EPISODE, enable_render=True)),
                            max_episode_steps=int(config.MAX_STEPS_PER_EPISODE))

    model = _load_latest_model(evaluation_env)

    while True:
        obs = evaluation_env.reset()
        done, state, total_reward, rewards = False, None, 0, []
        distances = []
        while not done:
            action, state = model.predict(obs, state=state, deterministic=True)
            obs, reward, done, info = evaluation_env.step(action)
            if args.vecnorm:
                distance = info[0]["desired_goal"]["distance_to_target"] * 1000
                print('reward: {:.2f},  distance left: {:.2f} [mm], Rotation Error: {:.2f} %'
                      .format(reward[0], info[0]["desired_goal"]["distance_to_target"] * 1000,
                              (info[0]["desired_goal"]['percentage_rotation_error']*100)))
            else:
                distance = info["desired_goal"]["distance_to_target"] * 1000
                print('reward: {:.2f}, distance left: {:.2f} [mm], Rotation Error: {:.2f} %'
                      .format(reward, info["desired_goal"]["distance_to_target"] * 1000,
                              (info["desired_goal"]['percentage_rotation_error']*100)))
            distances.append(distance)
            rewards.append(reward)
            total_reward += reward
        print(f'average_distance_to_target: {np.mean(distances)}, min: {np.min(distances)}, max: {np.max(distances)}')
        print(f'average_action_reward: {np.mean(rewards)}')
        print(f'total_reward: {total_reward}')
        input()

if __name__ == '__main__':

    ############################################################
    #  Parse command line arguments
    ############################################################
    parser = argparse.ArgumentParser(description='Eval DQN with UWRT_ARM_ENV')

    parser.add_argument('--vecnorm', '-vecn', required=False, default=False,
                        type=bool,
                        metavar="Train DQN with normalized obs and rewards")

    args = parser.parse_args()

    main(args)
