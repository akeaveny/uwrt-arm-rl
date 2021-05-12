import glob
import os
import sys

import argparse

import gym
import gym_uwrt_arm

import numpy as np

from gym.wrappers import FlattenObservation
from gym_uwrt_arm.wrappers.discrete_to_continuous_dict_action_wrapper import DiscreteToContinuousDictActionWrapper
from gym_uwrt_arm.wrappers.flatten_action_space import FlattenAction

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CallbackList, CheckpointCallback, BaseCallback
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.monitor import Monitor, load_results
from stable_baselines3.ppo import MlpPolicy

from torchsummary import summary
from torch.utils.tensorboard import SummaryWriter

import config

###########################
###########################

class TensorboardCallback(BaseCallback):
    """
    Custom callback for plotting additional values in tensorboard.
    """
    def __init__(self, monitor_log_dir, save_best_model_path, tensorboard_log_dir, check_freq=1000, verbose=0):
        super(TensorboardCallback, self).__init__(verbose)
        self.monitor_log_dir = monitor_log_dir
        self.check_freq = check_freq
        self.verbose = verbose

        self.writer = SummaryWriter(tensorboard_log_dir, comment='PPO_GRADS_and_WEIGHTS')

        self.save_best_model_path = os.path.join(save_best_model_path, 'best_mean_reward_model')

        self.best_mean_reward = -np.inf
        self.history_for_best_mean_reward = 100

        self.policy = None

    # from https://stable-baselines.readthedocs.io/en/master/guide/examples.html#using-callback-monitoring-training
    def _on_step(self) -> bool:

        if self.policy == None:
            self.policy = self.model.policy.mlp_extractor.policy_net
            summary(self.policy, (config.BATCH_SIZE,))

        if self.n_calls % self.check_freq == 0:
            # TODO: policy
            for tag, value in self.policy.named_parameters():
                tag = tag.replace('.', '/')
                if value.grad is None:
                    # print('No Grad Layer: ', tag.split('/'))
                    self.writer.add_histogram('weights/ppo/policy/' + tag, value.data.cpu().numpy(), self.n_calls)
                    pass
                else:
                    # print('Layer: ', tag.split('/'))
                    self.writer.add_histogram('weights/ppo/policy/' + tag, value.data.cpu().numpy(), self.n_calls)
                    self.writer.add_histogram('grads/ppo/policy/' + tag, value.grad.data.cpu().numpy(), self.n_calls)

            monitor_csv_dataframe = load_results(self.monitor_log_dir)
            # dataframe is loaded as: [index], [r], [l], [t]
            index = monitor_csv_dataframe['index'].to_numpy()
            rewards = monitor_csv_dataframe['r'].to_numpy()
            episode_lengths = monitor_csv_dataframe['l'].to_numpy()

            if len(rewards) > self.history_for_best_mean_reward:
                # Mean training reward over the last 100 episodes
                mean_reward = np.mean(rewards[-self.history_for_best_mean_reward:]) / \
                              np.mean(episode_lengths[-self.history_for_best_mean_reward:])
                if mean_reward > self.best_mean_reward:
                    self.best_mean_reward = mean_reward
                    # saving best
                    # if self.verbose > 0:
                    #     print("Saving new best model to {}".format(self.save_best_model_path))
                    self.model.save(self.save_best_model_path)

                self.logger.record('Best Mean Reward', self.best_mean_reward)
                if self.verbose > 0:
                    print("Best mean reward: {:.2f}\nLast mean reward per episode: {:.2f}"
                            .format(self.best_mean_reward, mean_reward))

        return True

def _load_latest_model(training_env):
    '''
    Chooses between the latest checkpoint and the latest save to load. If neither exist, a new model is returned.
    '''

    saved_model_file_mtime = None
    last_modified_checkpoint_file_mtime = None

    saved_model_file_path = config.BASE_SAVE_PATH / f'{config.GYM_ID}-ppo-trained-model.zip'
    if saved_model_file_path.is_file():
        saved_model_file_mtime = os.path.getmtime(saved_model_file_path)

    checkpoints_dir = config.BASE_SAVE_PATH / config.CHECKPOINTS_DIR
    if checkpoints_dir.is_dir():
        checkpoint_files = glob.glob(f'{checkpoints_dir}/{config.GYM_ID}-ppo-trained-model_*_steps.zip')

        if checkpoint_files:
            checkpoint_files.sort(key=os.path.getmtime)
            last_modified_checkpoint_file_path = checkpoint_files[-1]
            last_modified_checkpoint_file_mtime = os.path.getmtime(last_modified_checkpoint_file_path)

    ################################################
    # TODO (ak): Don't load any checkpoint or model
    ################################################
    print('Training from Scratch!')
    model = PPO(MlpPolicy, env=training_env, verbose=1, tensorboard_log=str(config.BASE_SAVE_PATH / config.TENSORBOARD_LOG_DIR),
                learning_rate=config.LEARNING_RATE, batch_size=config.BATCH_SIZE, gamma=config.GAMMA)

    # Clear all unused checkpoints
    if last_modified_checkpoint_file_mtime:
        for file_path in checkpoint_files:
            if file_path != last_modified_checkpoint_file_path:
                os.remove(file_path)

    return model

def main(args):

    #################
    # Vec Norm
    #################

    if args.vecnorm:
        print("Normalizing Input Obs and Rewards!")
        training_env = DummyVecEnv([lambda:
                                    Monitor(FlattenObservation(FlattenAction(
                                        gym.make(config.GYM_ID, key_position=config.KEY_POSITION,
                                                 key_orientation=config.KEY_ORIENTATION,
                                                 max_steps=config.MAX_STEPS_PER_EPISODE, enable_render=False))),
                                        filename=str(config.BASE_SAVE_PATH))])

        # Automatically normalize the input features and reward
        training_env = VecNormalize(training_env, norm_obs=True, norm_reward=True)

    else:
        training_env = Monitor(FlattenObservation(FlattenAction(
                            gym.make(config.GYM_ID, key_position=config.KEY_POSITION, key_orientation=config.KEY_ORIENTATION,
                            max_steps=config.MAX_STEPS_PER_EPISODE, enable_render=False))),
                        filename=str(config.BASE_SAVE_PATH))

    #################
    #################

    model = _load_latest_model(training_env=training_env)

    ###################
    # callbacks
    ###################

    checkpoint_callback = CheckpointCallback(save_freq=config.ESTIMATED_STEPS_PER_MIN_1080TI * 5,
                                             save_path=str(config.BASE_SAVE_PATH / config.CHECKPOINTS_DIR),
                                             name_prefix=f'{config.GYM_ID}-ppo-trained-model')

    # TODO: tensorboard not getting updated when evalcallback is used
    # evaluation_callback = EvalCallback(eval_env=evaluation_env,
    #                                    best_model_save_path=str(config.BASE_SAVE_PATH / config.BEST_MODEL_SAVE_DIR),
    #                                    log_path=str(config.BASE_SAVE_PATH / config.BEST_MODEL_LOG_DIR),
    #                                    eval_freq=ESTIMATED_STEPS_PER_MIN_1080TI * 10)

    custom_callback = TensorboardCallback(monitor_log_dir=str(config.BASE_SAVE_PATH), save_best_model_path=str(config.BASE_SAVE_PATH),
                                          tensorboard_log_dir=str(config.BASE_SAVE_PATH / config.TENSORBOARD_LOG_DIR),
                                            check_freq=1000, verbose=0)

    ppo_callbacks = CallbackList([checkpoint_callback, custom_callback])

    ###################
    ###################
    # TODO: autosave on KEYBOARD_INTERRUPT
    try:
        model.learn(total_timesteps=int(config.TOTAL_TRAINING_ENV_STEPS), callback=ppo_callbacks)
        model.save(path=config.BASE_SAVE_PATH / f'{config.GYM_ID}-ppo-trained-model')
    except KeyboardInterrupt:
        model.save(path=config.BASE_SAVE_PATH / f'{config.GYM_ID}-ppo-INTERRUPTED-model')
        print('\n*** Saved INTERRUPTED Model ***')
        try:
            sys.exit(0)
        except SystemExit:
            os._exit(0)

    #################
    #################

    if args.vecnorm:
        if not (os.path.exists(config.BASE_SAVE_PATH / config.STATISTICS_LOG_DIR)):
            os.makedirs(config.BASE_SAVE_PATH / config.STATISTICS_LOG_DIR)
        training_env.save(config.BASE_SAVE_PATH / config.STATISTICS_LOG_DIR / 'vec_normalize.pkl')

    training_env.close()


if __name__ == '__main__':

    ############################################################
    #  Parse command line arguments
    ############################################################

    parser = argparse.ArgumentParser(description='Train DQN with UWRT_ARM_ENV')

    parser.add_argument('--vecnorm', '-vecn', required=False, default=False,
                        type=bool,
                        metavar="Train DQN with normalized obs and rewards")

    args = parser.parse_args()

    print(f'Beginning to train for about {config.NUM_TRAINING_EPISODES} episodes ({config.TOTAL_TRAINING_ENV_STEPS} time steps)')

    main(args)
