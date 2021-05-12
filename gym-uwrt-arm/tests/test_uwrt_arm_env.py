import pprint
import time

import gym
import numpy as np
from gym.wrappers import FlattenObservation
from stable_baselines3.common import env_checker

import pybullet as pb

# noinspection PyUnresolvedReferences
import gym_uwrt_arm.envs.uwrt_arm_env
import gym_uwrt_arm.envs.uwrt_arm_goal_env
from gym_uwrt_arm.wrappers.discrete_to_continuous_dict_action_wrapper import DiscreteToContinuousDictActionWrapper
from gym_uwrt_arm.wrappers.flatten_action_space import FlattenAction

GYM_ID = 'uwrt-arm-v0'
# GYM_ID = 'uwrt-arm-v1'
# GYM_ID = 'gen3-arm-v0'

class TestClass:
    NUM_EPISODES = 9
    MAX_STEPS = 100
    KEY_POSITION = np.array([0.6, 0.6, 0.6])
    KEY_ORIENTATION = np.array([0, 0, 0, 1])

    def __run_test(self, env):
        pp = pprint.PrettyPrinter()  # TODO: update to python 3.8 to use sort_dicts = False

        rewards, total_reward = [], 0
        for episode in range(self.NUM_EPISODES):
            initial_observation = env.reset()
            # self.__draw_keyboard()
            # print('Initial Observation:')
            # pp.pprint(initial_observation)

            start = time.time()
            for sim_step in range(self.MAX_STEPS):
                action = env.action_space.sample()
                observation, reward, done, info = env.step(action)

                # print()
                # print('Action:')
                # pp.pprint(action)
                # print('Observation: len():', len(observation))
                # pp.pprint(observation)
                # print('Info:')
                # pp.pprint(info)
                # print('Reward:')
                # pp.pprint(reward)
                rewards.append(reward)
                total_reward += reward

                # import time
                # time.sleep(1)

                if done:
                    print()
                    # print("total time taken this loop: ", time.time() - start)
                    print(f'Episode #{episode} finished after {info["sim"]["steps_executed"]} steps!')
                    print(f'Episode #{episode} exit condition was {info["sim"]["end_condition"]}')
                    print()
                    break
            print(f'average_action_reward: {np.mean(rewards)}')
            print(f'total_reward: {total_reward}\n')

    def __draw_keyboard(self):

        heights = np.arange(start=0.70, stop=0.90, step=(0.90-0.70)/1000)

        for height in heights:
            ### drawing 'black' keyboard
            pb.addUserDebugLine(lineFromXYZ=[0.85, -0.3, height],
                                lineToXYZ=[0.85, 0.3, height],
                                lineColorRGB=[0, 0, 0], lineWidth=10)

    def test_env(self):
        env = gym.make(GYM_ID, key_position=self.KEY_POSITION, key_orientation=self.KEY_ORIENTATION,
                       max_steps=self.MAX_STEPS, enable_render=True,
                       # is_keyboard_demo_env=False
                       )
        self.__run_test(env)

        # Implicitly closes the environment
        env_checker.check_env(env=env, warn=True, skip_render_check=False)

    def test_discrete_action_wrapper_env(self):
        env = DiscreteToContinuousDictActionWrapper(
            gym.make(GYM_ID, key_position=self.KEY_POSITION, key_orientation=self.KEY_ORIENTATION,
                     max_steps=self.MAX_STEPS, enable_render=True))
        self.__run_test(env)

        # Implicitly closes the environment
        env_checker.check_env(env=env, warn=True, skip_render_check=False)

    def test_flatten_observation_wrapper_env(self):
        env = FlattenObservation(
            gym.make(GYM_ID, key_position=self.KEY_POSITION, key_orientation=self.KEY_ORIENTATION,
                     max_steps=self.MAX_STEPS, enable_render=True))
        self.__run_test(env)

        # TODO: Broken because of dict action. fix upstream in sb3
        # Implicitly closes the environment
        env_checker.check_env(env=env, warn=True, skip_render_check=False)

    def test_flatten_action_wrapper_env(self):
        env = FlattenAction(
            gym.make(GYM_ID, key_position=self.KEY_POSITION, key_orientation=self.KEY_ORIENTATION,
                     max_steps=self.MAX_STEPS, enable_render=True))
        self.__run_test(env)

        # TODO: Broken because of dict action. fix upstream in sb3
        # Implicitly closes the environment
        env_checker.check_env(env=env, warn=True, skip_render_check=False)

    def test_gym_api_compliance_for_dqn_wrapper_setup(self):
        env = FlattenObservation(DiscreteToContinuousDictActionWrapper(
            gym.make(GYM_ID, key_position=self.KEY_POSITION, key_orientation=self.KEY_ORIENTATION,
                     max_steps=self.MAX_STEPS, enable_render=True)))
        self.__run_test(env)

        # TODO: Broken because of dict action. fix upstream in sb3
        # Implicitly closes the environment
        env_checker.check_env(env=env, warn=True, skip_render_check=False)

if __name__ == '__main__':
    test = TestClass()

    test.test_env()
    # test.test_discrete_action_wrapper_env()
    # test.test_flatten_observation_wrapper_env()
    # test.test_flatten_action_wrapper_env()
    # test.test_gym_api_compliance_for_dqn_wrapper_setup()
