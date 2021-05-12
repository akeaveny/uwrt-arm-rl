import math
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Union

import gym
import numpy as np
import pybullet as pb
import pybullet_data
import requests
from gym import spaces
from urdfpy import URDF

from scipy.spatial.transform import Rotation as R

import config

class UWRTArmEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    # UWRT URDF
    ARM_URDF = '/home/akeaveny/git/uwrt_arm_rl/gym-uwrt-arm/urdfs/uwrt_arm/new_urdf/robot.urdf'
    ARM_URDF_FILE_NAME = 'robot.urdf'

    # Workspace Analysis
    STARTING_ARM_POSITION = config.STARTING_ARM_POSITION       # np.array([0.5, 0.0, 0.6])
    STARTING_ARM_ORIENTATION = config.STARTING_ARM_ORIENTATION # np.array([0, 0, 0, 1])

    # Pybullet Constants
    DEFAULT_PYBULLET_TIME_STEP = 1 / 240

    # Reward Constants
    GOAL_POSITION_DISTANCE_THRESHOLD = 1 / 1000 # 1 mm
    REWARD_MAX = 100
    reward_range = (-float('inf'), float(REWARD_MAX))

    @dataclass(frozen=True)
    class InitOptions:
        __slots__ = ['key_position', 'key_orientation', 'sim_step_duration', 'max_steps', 'enable_render', 'is_val_env', 'is_keyboard_demo_env',  'tmp_dir']
        key_position: np.ndarray
        key_orientation: np.ndarray

        sim_step_duration: float
        max_steps: int
        enable_render: bool

        is_val_env: bool
        is_keyboard_demo_env: bool

        tmp_dir: tempfile.TemporaryDirectory

    @dataclass
    class PyBulletInfo:
        __slots__ = ['key_uid', 'arm_uid']
        key_uid: Union[int, None]
        arm_uid: Union[int, None]

    def __init__(self, key_position, key_orientation, max_steps, desired_sim_step_duration=1/100,
                 enable_render=False, is_val_env=False, is_keyboard_demo_env=False):

        # Chose closest time step duration that's multiple of pybullet time step duration and greater than or equal to
        # desired_sim_step_duration
        sim_step_duration = math.ceil(
            desired_sim_step_duration / UWRTArmEnv.DEFAULT_PYBULLET_TIME_STEP) * UWRTArmEnv.DEFAULT_PYBULLET_TIME_STEP

        self.init_options = self.InitOptions(key_position=key_position, key_orientation=key_orientation,
                                             max_steps=max_steps, sim_step_duration=sim_step_duration,
                                             enable_render=enable_render,
                                             is_val_env=is_val_env, is_keyboard_demo_env=is_keyboard_demo_env,
                                             tmp_dir=tempfile.TemporaryDirectory())

        self.__initialize_gym()
        self.__initialize_sim()

    def __initialize_gym(self):
        arm_urdf = URDF.load(UWRTArmEnv.ARM_URDF)
        urdf_arm_joint_idxs = [1, 4, 9, 10, 11] # these joint idxs are used to initalize joint limits

        num_actuated_joints = len(arm_urdf.actuated_joint_names)
        actuated_joints_names = arm_urdf.actuated_joint_names

        # TODO (ak): these are the joints we want to control with pybullet. tested with workspace script.
        arm_joint_idxs = [1, 4, 5, 6, 7]
        num_joints = len(arm_joint_idxs)
        ee_link_idx = 11

        joint_limits = []
        for joint_idx in range(num_actuated_joints):
            if joint_idx in urdf_arm_joint_idxs:
                if arm_urdf.actuated_joints[joint_idx].joint_type == 'continuous': # wrist_rotate
                    joint_limits.append((-75*np.pi/180.0, 75*np.pi/180.0))
                else:
                    joint_limits.append((arm_urdf.actuated_joints[joint_idx].limit.lower,
                                         arm_urdf.actuated_joints[joint_idx].limit.upper))

        joint_vel_limits = []
        for joint_idx in range(num_actuated_joints):
            if joint_idx in urdf_arm_joint_idxs:
                # joint_vel_limits.append((-1, 1)) # set all joint limits to 1 m/s
                if arm_urdf.actuated_joints[joint_idx].joint_type == 'continuous': # wrist_rotate
                    joint_vel_limits.append((-0.5, 0.5))
                else:
                    joint_vel_limits.append((-1*arm_urdf.actuated_joints[joint_idx].limit.velocity,
                                             arm_urdf.actuated_joints[joint_idx].limit.velocity))


        # All joint limit switch states are either NOT_TRIGGERED[0], LOWER_TRIGGERED[1], UPPER_TRIGGERED[2]
        # The exception is roll which only has NOT_TRIGGERED[0]
        # TODO: this is wrong. wrist joints flipped
        joint_limit_switch_dims = np.concatenate((np.full(num_joints - 1, 3), np.array([1])))

        # TODO: Load mechanical limits from something (ex. pull info from config in uwrt_mars_rover thru git)
        self.observation_space = spaces.Dict({
            'goal': spaces.Dict({
                'desired_key_pose_in_world_frame': spaces.Dict({
                    'position': spaces.Box(low=np.full(3, -np.inf), high=np.full(3, np.inf), shape=(3,),
                                           dtype=np.float32),
                    'orientation': spaces.Box(low=np.full(4, -np.inf), high=np.full(4, np.inf), shape=(4,),
                                              dtype=np.float32),
                }),
                'initial_distance_to_target': spaces.Box(low=0, high=np.inf, shape=(), dtype=np.float32),
                'initial_orientation_difference': spaces.Box(low=np.full(4, -np.inf), high=np.full(4, np.inf),
                                                             shape=(4,), dtype=np.float32)
            }),
            'joint_sensors': spaces.Dict({
                # Order of array is [turntable, shoulder, elbow, wrist pitch, wrist roll]
                # TODO: this is wrong. wrist joints flipped
                'position': spaces.Box(low=np.full(num_joints, -180), high=np.full(num_joints, 180),
                                       shape=(num_joints,), dtype=np.float32),
                'velocity': spaces.Box(low=np.full(num_joints, -np.inf), high=np.full(num_joints, np.inf),
                                       shape=(num_joints,), dtype=np.float32),
                'effort': spaces.Box(low=np.full(num_joints, -np.inf), high=np.full(num_joints, np.inf),
                                     shape=(num_joints,), dtype=np.float32),
                'joint_limit_switches': spaces.MultiDiscrete(joint_limit_switch_dims),
                'joint_vel_limit_switches': spaces.MultiDiscrete(joint_limit_switch_dims),
            }),
        })

        self.action_space = spaces.Dict({
            'joint_commands': spaces.Box(low=np.full(num_joints, -3), high=np.full(num_joints, 1),
                                                  shape=(num_joints,), dtype=np.float32)
        })

        self.observation = {
            'goal': {
                'desired_key_pose_in_world_frame': {
                    'position': self.init_options.key_position,
                    'orientation': self.init_options.key_orientation,
                },
                'initial_distance_to_target': np.array(np.inf),
                'initial_orientation_difference': np.full(4, np.inf),
            },
            'joint_sensors': {
                'position': np.zeros(num_joints),
                'velocity': np.zeros(num_joints),
                'effort': np.zeros(num_joints),
                'joint_limit_switches': np.zeros(num_joints),
                'joint_vel_limit_switches': np.zeros(num_joints),
            }
        }

        self.info = {
            'sim': {
                'step_duration': self.init_options.sim_step_duration,
                'max_steps': self.init_options.max_steps,
                'steps_executed': 0,
                'seconds_executed': 0,
                'end_condition': 'Not Done',
                'keys_hit': 0,
            },
            'goal': {
                'distance_to_target': 0,
                'previous_distance_to_target': 0,
                'distance_moved_towards_target': 0,
                'orientation_difference': [0, 0, 0, 0],
            },
            'arm': {
                'allen_key_pose_in_world_frame': {
                    'position': [0, 0, 0],
                    'orientation': [0, 0, 0, 0],
                },
                'num_joints': num_joints,
                'actuated_joints_names': actuated_joints_names,
                'num_actuated_joints': num_actuated_joints,
                'arm_joint_idxs': arm_joint_idxs,
                'ee_link_idx': ee_link_idx,
                'joint_limits': joint_limits,
                'joint_vel_limits': joint_vel_limits,
                # TODO: UPDATE MAX FORCE
                # 'max_force': max_force,
            },
        }


    def __initialize_sim(self):
        self.py_bullet_info = UWRTArmEnv.PyBulletInfo(None, None)

        if not self.init_options.enable_render:
            self.physicsClient = pb.connect(pb.DIRECT)
        else:
            self.physicsClient = pb.connect(pb.GUI)

            # Set default camera viewing angle
            # front view
            # pb.resetDebugVisualizerCamera(cameraDistance=1, cameraYaw=-90, cameraPitch=-15,
            #                               cameraTargetPosition=[0.85, 0, 0.75])
            # side view
            pb.resetDebugVisualizerCamera(cameraDistance=1.5, cameraYaw=-165, cameraPitch=-15,
                                          cameraTargetPosition=[0.85-0.75, 0, 0.75-0.25])

            pb.configureDebugVisualizer(pb.COV_ENABLE_SEGMENTATION_MARK_PREVIEW, 0)
            pb.configureDebugVisualizer(pb.COV_ENABLE_DEPTH_BUFFER_PREVIEW, 0)
            pb.configureDebugVisualizer(pb.COV_ENABLE_RGB_BUFFER_PREVIEW, 0)

    def __spawn_uwrt_arm(self):
        self.py_bullet_info.arm_uid = pb.loadURDF(UWRTArmEnv.ARM_URDF, useFixedBase=True,
            flags=pb.URDF_MAINTAIN_LINK_ORDER | pb.URDF_MERGE_FIXED_LINKS)

        # TODO: Randomize arm starting configuration
        # TODO: limit to valid configurations using nullspace?
        joint_home_poses = pb.calculateInverseKinematics(self.py_bullet_info.arm_uid,
                                                         endEffectorLinkIndex=self.info['arm']['ee_link_idx'],
                                                         targetPosition=UWRTArmEnv.STARTING_ARM_POSITION,
                                                         targetOrientation=UWRTArmEnv.STARTING_ARM_ORIENTATION,
                                                         )

        _joint_home_poses = np.array(joint_home_poses)
        _joint_home_poses[-6] = -0.05  # gripper left
        _joint_home_poses[-7] = 0.05   # gripper right
        _joint_home_poses[-8] = 0      # servo joint
        # Move joints to starting position
        for joint_index in range(self.info['arm']['num_actuated_joints']):
            pb.resetJointState(self.py_bullet_info.arm_uid,
                               jointIndex=joint_index,
                               targetValue=_joint_home_poses[joint_index],
                               targetVelocity=0)

        # TODO(add toggle using addUserDebugParameter)
        # Draw Coordinate Frames. These are the inertial frames.
        # axis_length = 0.15
        # for joint_index in range(self.info['arm']['num_actuated_joints']):
        #     link_name = pb.getJointInfo(self.py_bullet_info.arm_uid, joint_index)[12].decode('ascii')
        #     pb.addUserDebugText(link_name, [0, 0, 0], textColorRGB=[0, 1, 1], textSize=0.75,
        #                         parentObjectUniqueId=self.py_bullet_info.arm_uid, parentLinkIndex=joint_index)
        #     pb.addUserDebugLine(lineFromXYZ=[0, 0, 0], lineToXYZ=[axis_length, 0, 0], lineColorRGB=[1, 0, 0],
        #                         parentObjectUniqueId=self.py_bullet_info.arm_uid, parentLinkIndex=joint_index)
        #     pb.addUserDebugLine(lineFromXYZ=[0, 0, 0], lineToXYZ=[0, axis_length, 0], lineColorRGB=[0, 1, 0],
        #                         parentObjectUniqueId=self.py_bullet_info.arm_uid, parentLinkIndex=joint_index)
        #     pb.addUserDebugLine(lineFromXYZ=[0, 0, 0], lineToXYZ=[0, 0, axis_length], lineColorRGB=[0, 0, 1],
        #                         parentObjectUniqueId=self.py_bullet_info.arm_uid, parentLinkIndex=joint_index)

    def __draw_keyboard(self):

        _start, _stop = 0.825, 0.675
        heights = np.arange(start=_stop, stop=_start, step=(_start - _stop)/1000)

        for height in heights:
            ### drawing 'black' keyboard
            pb.addUserDebugLine(lineFromXYZ=[0.85, -0.25, height],
                                lineToXYZ=[0.85, 0.25, height],
                                lineColorRGB=[0, 0, 0], lineWidth=10)


    def __spawn_key(self):

        self.keyboard_orientation = np.array([0, 0, 0, 1])

        if self.init_options.is_keyboard_demo_env:
            """ Select a subset of key locations that we know the arm can hit """
            TEST_KEY_POSITIONS = np.array([[0.85,-0.1, 0.775],     # O
                                          [0.85, -0.0, 0.725],     # N
                                          [0.85, +0.1, 0.775],     # T
                                          [0.85, -0.1, 0.775],     # O
                                          [0.85, -0.05, 0.725],    # M
                                          [0.85, +0.175, 0.750],   # A
                                          [0.85, +0.125, 0.775],   # R
                                          [0.85, +0.155, 0.750],   # S
                                          [0.85, +0.185, 0.775],   # !
                                          [0.85, +0.185, 0.800],   # On To Mars!
                                           ])
            TEST_KEY_TEXT = np.array([ 'O',
                                       'N',
                                       'T',
                                       'O',
                                       'M',
                                       'A',
                                       'R',
                                       'S',
                                       '!',
                                       'On To Mars!'])
            self.keyboard_position = np.array(TEST_KEY_POSITIONS[self.info['sim']['keys_hit']]).flatten()
            self.keyboard_text = np.str(TEST_KEY_TEXT[self.info['sim']['keys_hit']])
            self.info['sim']['keys_hit'] += 1

            ### drawing 'green' key
            pb.addUserDebugLine(lineFromXYZ=self.keyboard_position - np.array([0.01, 1 / 100, 0]),
                                lineToXYZ=self.keyboard_position + np.array([0.00, 1 / 100, 0]),
                                lineColorRGB=[1, 1, 1], lineWidth=1)

            ### drawing 'letter'
            pb.addUserDebugText(text=self.keyboard_text,
                                textPosition=self.keyboard_position - np.array([0.05, 0, 0]),
                                textColorRGB=[1, 1, 1],
                                textSize=3)

        elif self.init_options.is_val_env:
            """ Select a subset of key locations that we know the arm can hit """
            VAL_KEY_POSITIONS = np.array([[0.85, 0, 0.8],
                                          [0.85, 0.2, 0.7],
                                          [0.85, -0.2, 0.7],
                                          [0.9, 0, 0.7]])
            random_idx = np.random.randint(low=0, high=4, size=1)
            self.keyboard_position = np.array(VAL_KEY_POSITIONS[random_idx]).flatten()

            # drawing 'green' key
            pb.addUserDebugLine(lineFromXYZ=self.keyboard_position - np.array([0, 2 / 100, 0]),
                                lineToXYZ=self.keyboard_position + np.array([0, 2 / 100, 0]),
                                lineColorRGB=[1, 0, 0], lineWidth=10)

        else:
            """ Randomize keyboard based on workspace analysis """
            self.keyboard_position = np.array([np.random.uniform(0.8, 0.9),
                                          np.random.uniform(-0.30, 0.30),
                                          np.random.uniform(0.65, 0.85)])
            # drawing 'magenta' key
            pb.addUserDebugLine(lineFromXYZ=self.keyboard_position - np.array([0, 1 / 100, 0]),
                                lineToXYZ=self.keyboard_position + np.array([0, 1 / 100, 0]),
                                lineColorRGB=[1, 0, 1], lineWidth=5)

        self.__draw_keyboard()

        self.observation = {
            'goal': {
                'desired_key_pose_in_world_frame': {
                    'position': self.keyboard_position,
                    'orientation': self.keyboard_orientation,
                }
            }
        }

    def __update_observation_and_info(self, reset=False):
        joint_states = pb.getJointStates(self.py_bullet_info.arm_uid,
                                         np.arange(pb.getNumJoints(self.py_bullet_info.arm_uid)))
        # joint_states for the entire robot
        robot_joint_positions = np.array([joint_state[0] for joint_state in joint_states], dtype=np.float32)
        robot_joint_velocities = np.array([joint_state[1] for joint_state in joint_states], dtype=np.float32)
        robot_joint_torques = np.array([joint_state[3] for joint_state in joint_states], dtype=np.float32)

        # joint_states for the arm
        arm_joint_positions = robot_joint_positions[self.info['arm']['arm_joint_idxs']]
        arm_joint_velocities = robot_joint_velocities[self.info['arm']['arm_joint_idxs']]
        arm_joint_torques = robot_joint_torques[self.info['arm']['arm_joint_idxs']]
        arm_joint_limit_states = [1 if arm_joint_positions[joint_index] <= self.info['arm']['joint_limits'][joint_index][0] else
                              2 if arm_joint_positions[joint_index] >= self.info['arm']['joint_limits'][joint_index][1] else
                              0 for joint_index in range(self.info['arm']['num_joints'])]
        arm_joint_vel_limit_states = [1 if arm_joint_velocities[joint_index] <= self.info['arm']['joint_vel_limits'][joint_index][0] else
                              2 if arm_joint_velocities[joint_index] >= self.info['arm']['joint_vel_limits'][joint_index][1] else
                              0 for joint_index in range(self.info['arm']['num_joints'])]

        self.observation['joint_sensors'] = {
            'position': arm_joint_positions,
            'velocity': arm_joint_velocities,
            'effort': arm_joint_torques,
            'joint_limit_switches': arm_joint_limit_states,
            'joint_vel_limit_switches': arm_joint_vel_limit_states,
        }

        # joint_states for the ee_link_idx
        ee_link_state = pb.getLinkState(self.py_bullet_info.arm_uid, self.info['arm']['ee_link_idx'])
        allen_key_tip_position_world_frame = ee_link_state[4]
        allen_key_tip_orientation_world_frame = ee_link_state[5]
        self.info['arm']['allen_key_pose_in_world_frame'] = {
            'position': allen_key_tip_position_world_frame,
            'orientation': allen_key_tip_orientation_world_frame,
        }

        # drawing allen_key_pose_in_world_frame
        pb.addUserDebugLine(lineFromXYZ=allen_key_tip_position_world_frame,
                            lineToXYZ=allen_key_tip_position_world_frame + np.array([0, 0, 5 / 1000]),
                            lineColorRGB=[1, 1, 0], lineWidth=50)

        distance_to_target = np.array(np.linalg.norm(
                                        allen_key_tip_position_world_frame - \
                                        self.observation['goal']['desired_key_pose_in_world_frame']['position']),
                                dtype=np.float32)

        self.info['goal']['previous_distance_to_target'] = self.info['goal']['distance_to_target']
        self.info['goal']['distance_to_target'] = distance_to_target
        self.info['goal']['distance_moved_towards_target'] = self.info['goal']['previous_distance_to_target'] - \
                                                             self.info['goal']['distance_to_target']

        # Difference in Quaternion
        difference_quaternion = np.array(pb.getDifferenceQuaternion(allen_key_tip_orientation_world_frame,
                                                                    self.observation['goal']['desired_key_pose_in_world_frame']
                                                                    ['orientation']), dtype=np.float32)

        # Difference in Rotation Matrix
        current_rotation_matrix = R.from_quat(allen_key_tip_orientation_world_frame).as_matrix()
        goal_rotation_matrix = R.from_quat(self.observation['goal']['desired_key_pose_in_world_frame']
                                               ['orientation']).as_matrix()

        # Now R*R' should produce eye(3)
        rotation_vector = R.from_matrix(current_rotation_matrix.dot(goal_rotation_matrix.T)).as_rotvec()
        rotation_error = np.pi - np.linalg.norm(rotation_vector)    # in rads
        percentage_rotation_error = rotation_error / np.pi          # normalized from 0 to 1 as a %

        self.info['goal']['orientation_difference'] = difference_quaternion
        self.info['goal']['percentage_rotation_error'] = percentage_rotation_error

        if reset:
            self.observation['goal']['initial_distance_to_target'] = self.info['goal']['distance_to_target']
            self.observation['goal']['initial_orientation_difference'] = self.info['goal']['orientation_difference']

            self.info['sim']['steps_executed'] = 0
            self.info['sim']['seconds_executed'] = 0
        else:
            self.info['sim']['steps_executed'] += 1
            self.info['sim']['seconds_executed'] += self.info['sim']['step_duration']

    def __execute_action(self, action):
        # from network
        action = action['joint_commands'] if isinstance(action, dict) else action

        ##########################
        # enfore joint limits
        ##########################

        for i, arm_joint_idx in enumerate(self.info['arm']['arm_joint_idxs']):
            pb.changeDynamics(bodyUniqueId=self.py_bullet_info.arm_uid,
                              linkIndex=arm_joint_idx,
                              maxJointVelocity=self.info['arm']['joint_vel_limits'][i][1],
                              jointLowerLimit=self.info['arm']['joint_limits'][i][0],
                              jointUpperLimit=self.info['arm']['joint_limits'][i][1],
                              )

        ####################
        # pos control
        ####################

        clipped_action = []
        for joint_index in range(self.info['arm']['num_joints']):
            clipped_action.append(np.clip(action[joint_index],
                                          self.info['arm']['joint_limits'][joint_index][0],
                                          self.info['arm']['joint_limits'][joint_index][1]))
        clipped_action = np.array(clipped_action)

        pb.setJointMotorControlArray(bodyUniqueId=self.py_bullet_info.arm_uid,
                                     jointIndices=self.info['arm']['arm_joint_idxs'],
                                     controlMode=pb.POSITION_CONTROL,
                                     targetPositions=clipped_action,
                                     # forces = np.full(len(self.info['arm']['arm_joint_idxs']), 400),
                                     )

        ####################
        # vel control
        ####################

        # clipped_action = []
        # for joint_index in range(self.info['arm']['num_joints']):
        #     clipped_action.append(np.clip(action[joint_index],
        #                                   self.info['arm']['joint_vel_limits'][joint_index][0],
        #                                   self.info['arm']['joint_vel_limits'][joint_index][1]))
        # clipped_action = np.array(clipped_action)
        #
        # pb.setJointMotorControlArray(bodyUniqueId=self.py_bullet_info.arm_uid,
        #                              jointIndices=self.info['arm']['arm_joint_idxs'],
        #                              controlMode=pb.VELOCITY_CONTROL,
        #                              targetVelocities=clipped_action,
        #                              # forces = np.full(len(self.info['arm']['arm_joint_idxs']), 400),
        #                              )

        ####################
        ####################

        pb_steps_per_sim_step = int(self.info['sim']['step_duration'] / UWRTArmEnv.DEFAULT_PYBULLET_TIME_STEP)
        for pb_sim_step in range(pb_steps_per_sim_step):
            pb.stepSimulation()

    def __calculate_reward(self):
        percent_time_used = self.info['sim']['steps_executed'] / self.info['sim']['max_steps']
        percent_distance_remaining = self.info['goal']['distance_to_target'] / \
                                     self.observation['goal']['initial_distance_to_target']

        # TODO: scale based off max speed to normalize
        # TODO: investigate weird values
        distance_moved = self.info['goal']['distance_moved_towards_target'] / self.observation['goal']['initial_distance_to_target']

        distance_weight = 1
        time_weight = 1 - distance_weight

        # TODO: investigate weird values
        # reward = distance_moved * UWRTArmEnv.REWARD_MAX / 2
        reward = (1 - percent_distance_remaining) * UWRTArmEnv.REWARD_MAX / 2

        # TODO (ak): tweak reward formula to reward more for orientation thats closer to perpendicular to surface of key
        percentage_rotation_error = self.info['goal']['percentage_rotation_error']
        reward -= percentage_rotation_error * UWRTArmEnv.REWARD_MAX / 10

        if self.info['goal']['distance_to_target'] < UWRTArmEnv.GOAL_POSITION_DISTANCE_THRESHOLD:
            self.info['sim']['end_condition'] = 'Key Reached'
            done = True
            reward += UWRTArmEnv.REWARD_MAX / 2

        elif self.info['sim']['steps_executed'] >= self.info['sim']['max_steps']:
            self.info['sim']['end_condition'] = 'Max Sim Steps Executed'
            done = True
            reward -= UWRTArmEnv.REWARD_MAX / 2
        else:
            done = False

        # TODO: add penalty for hitting anything that's not the desired key

        return reward, done

    def step(self, action):
        # TODO: is this required? does speed increase if used in non-gui mode? does speed slow increaenvse if not used in gui mode?
        pb.configureDebugVisualizer(pb.COV_ENABLE_SINGLE_STEP_RENDERING)

        self.__execute_action(action)

        self.__update_observation_and_info()

        reward, done = self.__calculate_reward()

        return self.observation, reward, done, self.info

    def reset(self):
        pb.resetSimulation()
        pb.setGravity(0, 0, -9.81)

        # Disable rendering while assets being loaded
        pb.configureDebugVisualizer(pb.COV_ENABLE_RENDERING, False)

        self.__spawn_uwrt_arm()
        self.__spawn_key()

        # Re-enable rendering if enabled in init_options
        pb.configureDebugVisualizer(pb.COV_ENABLE_RENDERING, self.init_options.enable_render)

        self.__update_observation_and_info(reset=True)

        return self.observation

    def render(self, mode='human'):
        if not self.init_options.enable_render:
            raise UserWarning('This environment was initialized with rendering disabled')
        return

    def close(self):
        if self.init_options.enable_render:
            pb.disconnect()