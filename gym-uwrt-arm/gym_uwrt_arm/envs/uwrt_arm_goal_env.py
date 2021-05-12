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

class UWRTArmGoalEnv(gym.GoalEnv):
    metadata = {'render.modes': ['human']}

    # Arm Constants
    ARM_URDF_URL = 'https://raw.githubusercontent.com/uwrobotics/uwrt_arm_rl/DQN/UWRTArmGym/urdf/uwrt_arm.urdf'  # TODO: Change to specific commit on master of uwrt_mars_rover. Warn on outdated
    ### for loading with allen key
    ARM_URDF = '/home/akeaveny/git/uwrt_arm_rl/gym-uwrt-arm/urdfs/uwrt_arm.urdf'
    ARM_URDF_FILE_NAME = 'uwrt_arm.urdf'
    ALLEN_KEY_LENGTH = 0.10

    # Pybullet Constants
    DEFAULT_PYBULLET_TIME_STEP = 1 / 240

    # Reward Constants
    GOAL_POSITION_DISTANCE_THRESHOLD = 1 / 1000 # 1 mm
    REWARD_MAX = 100
    reward_range = (-float('inf'), float(REWARD_MAX))

    OBS_DIM = 37  # keep same as og observation
    GOAL_DIM = 3  # TODO: changed this to 7 for pose: (position + orientation)

    @dataclass(frozen=True)
    class InitOptions:
        __slots__ = ['key_position', 'key_orientation', 'sim_step_duration', 'max_steps', 'enable_render', 'tmp_dir']
        key_position: np.ndarray
        key_orientation: np.ndarray

        sim_step_duration: float
        max_steps: int
        enable_render: bool

        tmp_dir: tempfile.TemporaryDirectory

    @dataclass
    class PyBulletInfo:
        __slots__ = ['key_uid', 'arm_uid']
        key_uid: Union[int, None]
        arm_uid: Union[int, None]

    def __init__(self, key_position, key_orientation, max_steps, desired_sim_step_duration=1 / 100,
                 enable_render=False):

        # Chose closest time step duration that's multiple of pybullet time step duration and greater than or equal to
        # desired_sim_step_duration
        sim_step_duration = math.ceil(
            desired_sim_step_duration / UWRTArmGoalEnv.DEFAULT_PYBULLET_TIME_STEP) * UWRTArmGoalEnv.DEFAULT_PYBULLET_TIME_STEP

        self.init_options = self.InitOptions(key_position=key_position, key_orientation=key_orientation,
                                             max_steps=max_steps, sim_step_duration=sim_step_duration,
                                             enable_render=enable_render, tmp_dir=tempfile.TemporaryDirectory())

        self.__initialize_urdf_file()
        self.__initialize_gym()
        self.__initialize_sim()

    def __initialize_urdf_file(self):
        with open(Path(self.init_options.tmp_dir.name) / UWRTArmGoalEnv.ARM_URDF_FILE_NAME, 'x') as arm_urdf_file:
            arm_urdf_file.write(requests.get(self.ARM_URDF_URL).text)

    def __initialize_gym(self):
        arm_urdf = URDF.load(str(Path(self.init_options.tmp_dir.name) / UWRTArmGoalEnv.ARM_URDF_FILE_NAME))
        ### for loading with allen key
        # arm_urdf = URDF.load(UWRTArmGoalEnv.ARM_URDF)
        num_joints = len(arm_urdf.actuated_joints)

        joint_limits = []
        for joint_idx in range(num_joints):
            joint_limits.append((arm_urdf.actuated_joints[joint_idx].limit.lower,
                                 arm_urdf.actuated_joints[joint_idx].limit.upper))

        joint_vel_limits = []
        for joint_idx in range(num_joints):
            joint_vel_limits.append((-1 * arm_urdf.actuated_joints[joint_idx].limit.velocity,
                                     arm_urdf.actuated_joints[joint_idx].limit.velocity))

        # TODO: UPDATE MAX FORCE
        max_force = np.full(5, 181.437)

        # All joint limit switch states are either NOT_TRIGGERED[0], LOWER_TRIGGERED[1], UPPER_TRIGGERED[2]
        # The exception is roll which only has NOT_TRIGGERED[0]
        joint_limit_switch_dims = np.concatenate(
            (np.full(num_joints - 1, 3), np.array([1])))  # TODO: this is wrong. wrist joints flipped

        self.action_space = spaces.Dict({
            # 'joint_velocity_commands': spaces.Box(low=np.full(num_joints, -np.inf), high=np.full(num_joints, np.inf),
            #                                       shape=(num_joints,), dtype=np.float32),
            'joint_velocity_commands': spaces.Box(low=np.full(num_joints, -1.5), high=np.full(num_joints, 1.5),
                                                  shape=(num_joints,), dtype=np.float32),
        })

        self.observation_space = spaces.Dict(
            {
                "achieved_goal": spaces.Box(low=-np.inf, high=np.inf, shape=(UWRTArmGoalEnv.GOAL_DIM,), dtype=np.float32),
                "desired_goal": spaces.Box(low=-np.inf, high=np.inf, shape=(UWRTArmGoalEnv.GOAL_DIM,), dtype=np.float32),
                # TODO: convert to dict with ObsDictWrapper
                "observation": spaces.Box(low=-np.inf, high=np.inf, shape=(UWRTArmGoalEnv.OBS_DIM,), dtype=np.float32),
            }
        )

        self.observation = self.observation_space.sample()

        self.info = {
            'sim': {
                'step_duration': self.init_options.sim_step_duration,
                'max_steps': self.init_options.max_steps,
                'steps_executed': 0,
                'seconds_executed': 0,
                'end_condition': 'Not Done',
            },
            'desired_goal': {
                'key_pose_world_frame': {
                    'position': self.init_options.key_position,
                    'orientation': self.init_options.key_orientation,
                },
                'initial_distance_to_target': 0,
                'initial_orientation_difference': [0, 0, 0, 0],
                'distance_to_target': 0,
                'previous_distance_to_target': 0,
                'distance_moved_towards_target': 0,
                'orientation_difference': [0, 0, 0, 0],
                'is_success': False,
            },
            'arm': {
                'allen_key_tip_pose_world_frame': {
                    'position': [0, 0, 0],
                    'orientation': [0, 0, 0, 0],
                },
                'joint_sensors': {
                    'position': [0, 0, 0, 0, 0],
                    'velocity': [0, 0, 0, 0, 0],
                    'effort':  [0, 0, 0, 0, 0],
                    'joint_limit_switches':  [0, 0, 0, 0, 0],
                    'joint_vel_limit_switches': [0, 0, 0, 0, 0],
                },
                'num_joints': num_joints,
                'joint_limits': joint_limits,
                'joint_vel_limits': joint_vel_limits,
                'max_force': max_force,
            },
        }

    def __initialize_sim(self):
        self.py_bullet_info = UWRTArmGoalEnv.PyBulletInfo(None, None)

        if not self.init_options.enable_render:
            pb.connect(pb.DIRECT)
        else:
            pb.connect(pb.GUI)

            # Set default camera viewing angle
            pb.resetDebugVisualizerCamera(cameraDistance=1.5, cameraYaw=0, cameraPitch=-40,
                                          cameraTargetPosition=[0.55, -0.35, 0.2])
            pb.configureDebugVisualizer(pb.COV_ENABLE_SEGMENTATION_MARK_PREVIEW, 0)
            pb.configureDebugVisualizer(pb.COV_ENABLE_DEPTH_BUFFER_PREVIEW, 0)
            pb.configureDebugVisualizer(pb.COV_ENABLE_RGB_BUFFER_PREVIEW, 0)

    def __spawn_uwrt_arm(self):
        self.py_bullet_info.arm_uid = pb.loadURDF(
            str(Path(self.init_options.tmp_dir.name) / UWRTArmGoalEnv.ARM_URDF_FILE_NAME), useFixedBase=True,
            flags=pb.URDF_MAINTAIN_LINK_ORDER | pb.URDF_MERGE_FIXED_LINKS)
        ### for loading with allen key
        # self.py_bullet_info.arm_uid = pb.loadURDF(UWRTArmGoalEnv.ARM_URDF, useFixedBase=True,
        #     flags=pb.URDF_MAINTAIN_LINK_ORDER | pb.URDF_MERGE_FIXED_LINKS)

        # TODO: Randomize arm starting configuration
        # TODO: Calculate Claw link pose from desired allen key tip pose

        # TODO: limit to valid configurations using nullspace?
        joint_home_poses = pb.calculateInverseKinematics(self.py_bullet_info.arm_uid,
                                                         endEffectorLinkIndex=self.info['arm']['num_joints'] - 1,
                                                         targetPosition=[0.3, 0.0, 0.8],
                                                         targetOrientation=pb.getQuaternionFromEuler(
                                                             [0, np.pi / 3.5, 0])
                                                         )

        # Move joints to starting position
        for joint_index in range(self.info['arm']['num_joints']):
            pb.resetJointState(self.py_bullet_info.arm_uid, jointIndex=joint_index,
                               targetValue=joint_home_poses[joint_index], targetVelocity=0)

        # Draw Coordinate Frames. These are the inertial frames. # TODO(add toggle using addUserDebugParameter)
        # axis_length = 0.15
        # for joint_index in range(self.info['arm']['num_joints']):
        #     link_name = pb.getJointInfo(self.py_bullet_info.arm_uid, joint_index)[12].decode('ascii')
        #     pb.addUserDebugText(link_name, [0, 0, 0], textColorRGB=[0, 1, 1], textSize=0.75,
        #                         parentObjectUniqueId=self.py_bullet_info.arm_uid, parentLinkIndex=joint_index)
        #     pb.addUserDebugLine(lineFromXYZ=[0, 0, 0], lineToXYZ=[axis_length, 0, 0], lineColorRGB=[1, 0, 0],
        #                         parentObjectUniqueId=self.py_bullet_info.arm_uid, parentLinkIndex=joint_index)
        #     pb.addUserDebugLine(lineFromXYZ=[0, 0, 0], lineToXYZ=[0, axis_length, 0], lineColorRGB=[0, 1, 0],
        #                         parentObjectUniqueId=self.py_bullet_info.arm_uid, parentLinkIndex=joint_index)
        #     pb.addUserDebugLine(lineFromXYZ=[0, 0, 0], lineToXYZ=[0, 0, axis_length], lineColorRGB=[0, 0, 1],
        #                         parentObjectUniqueId=self.py_bullet_info.arm_uid, parentLinkIndex=joint_index)

        # Draw Allen Key Offset # TODO(melvinw): transform to link frame and draw from front of box to allen key
        claw_visual_shape_data = pb.getVisualShapeData(self.py_bullet_info.arm_uid)[self.info['arm']['num_joints']]
        claw_visual_box_z_dim = claw_visual_shape_data[3][2]
        # Box geometry origin is defined at the center of the box
        allen_key_tip_position_visual_frame = [0, 0, (claw_visual_box_z_dim / 2 + UWRTArmGoalEnv.ALLEN_KEY_LENGTH)]
        pb.addUserDebugLine(lineFromXYZ=[0, 0, 0], lineToXYZ=allen_key_tip_position_visual_frame,
                            lineColorRGB=[1, 1, 1], lineWidth=5,
                            parentObjectUniqueId=self.py_bullet_info.arm_uid,
                            parentLinkIndex=self.info['arm']['num_joints'] - 1)

    def __spawn_key(self):
        """ Randomize keyboard """
        # np.random.seed(0) ### uncomment to spawn in same location
        self.info['desired_goal']['key_pose_world_frame']['position'] = np.array([np.random.uniform(0.625, 0.675),
                                                                           np.random.uniform(-0.30, 0.30),
                                                                           np.random.uniform(0.65, 0.675)])
        # we want the key vertical (should be -90 deg)
        self.info['desired_goal']['key_pose_world_frame']['orientation'] = R.from_euler('y', -90, degrees=True).as_quat()

        pb.addUserDebugLine(lineFromXYZ=self.info['desired_goal']['key_pose_world_frame']['position'],
                            lineToXYZ=self.info['desired_goal']['key_pose_world_frame']['position'] + np.array([0, 0, 15 / 1000]),
                            lineColorRGB=[0, 0, 0], lineWidth=50)

        # Currently, there is only a single key represented as a cube
        # TODO: when this spawns a full keyboard (and the backboard of the equipment servicing unit) we need to make sure its orientation relative to the arm is reachable

    def __get_allen_key_tip_in_world_frame(self):
        # TODO: Fix mismatch between collision box and visual box in urdf. it looks like the collision box has the wrong origin
        claw_visual_shape_data = pb.getVisualShapeData(self.py_bullet_info.arm_uid)[self.info['arm']['num_joints']]
        claw_visual_box_z_dim = claw_visual_shape_data[3][2]
        visual_frame_position_link_frame = claw_visual_shape_data[5]
        visual_frame_orientation_link_frame = claw_visual_shape_data[6]

        # Box geometry origin is defined at the center of the box
        allen_key_tip_position_visual_frame = [0, 0, (claw_visual_box_z_dim / 2 + UWRTArmGoalEnv.ALLEN_KEY_LENGTH)]

        claw_link_state = pb.getLinkState(self.py_bullet_info.arm_uid, self.info['arm']['num_joints'] - 1)
        claw_link_position_world_frame = claw_link_state[4]
        claw_link_orientation_world_frame = claw_link_state[5]

        allen_key_tip_position_link_frame, allen_key_tip_orientation_link_frame = pb.multiplyTransforms(
            visual_frame_position_link_frame, visual_frame_orientation_link_frame,
            allen_key_tip_position_visual_frame, [0, 0, 0, 1])
        allen_key_tip_position_world_frame, allen_key_tip_orientation_world_frame = pb.multiplyTransforms(
            claw_link_position_world_frame, claw_link_orientation_world_frame,
            allen_key_tip_position_link_frame, allen_key_tip_orientation_link_frame)

        return allen_key_tip_position_world_frame, allen_key_tip_orientation_world_frame

    def __dict_to_list(self, parent_dict):
        local_list = []
        for key, value in parent_dict.items():
            if isinstance(value, dict):
                local_list.extend(self.__dict_to_list(value))
            else:
                local_list.append(int(value)) if len(np.array(value).shape) == 0 else local_list.extend(value)
        return local_list

    def __update_observation_and_info(self, reset=False):
        joint_states = pb.getJointStates(self.py_bullet_info.arm_uid,
                                         np.arange(pb.getNumJoints(self.py_bullet_info.arm_uid)))
        joint_positions = np.array([joint_state[0] for joint_state in joint_states], dtype=np.float32)
        joint_velocities = np.array([joint_state[1] for joint_state in joint_states], dtype=np.float32)
        joint_torques = np.array([joint_state[3] for joint_state in joint_states], dtype=np.float32)
        joint_limit_states = [1 if joint_positions[joint_index] <= self.info['arm']['joint_limits'][joint_index][0] else
                              2 if joint_positions[joint_index] >= self.info['arm']['joint_limits'][joint_index][1] else
                              0 for joint_index in range(self.info['arm']['num_joints'])]
        joint_vel_limit_states = [
            1 if joint_velocities[joint_index] <= self.info['arm']['joint_vel_limits'][joint_index][0] else
            2 if joint_velocities[joint_index] >= self.info['arm']['joint_vel_limits'][joint_index][1] else
            0 for joint_index in range(self.info['arm']['num_joints'])]
        self.info['arm']['joint_sensors'] = {
            'position': joint_positions,
            'velocity': joint_velocities,
            'effort': joint_torques,
            'joint_limit_switches': joint_limit_states,
            'joint_vel_limit_switches': joint_vel_limit_states,
        }

        allen_key_tip_position_world_frame, allen_key_tip_orientation_world_frame = self.__get_allen_key_tip_in_world_frame()
        self.info['arm']['allen_key_tip_pose_world_frame'] = {
            'position': allen_key_tip_position_world_frame,
            'orientation': allen_key_tip_orientation_world_frame,
        }

        distance_to_target = np.array(np.linalg.norm(
                                        allen_key_tip_position_world_frame - \
                                        self.info['desired_goal']['key_pose_world_frame']['position']),
                                dtype=np.float32)

        difference_quaternion = np.array(pb.getDifferenceQuaternion(allen_key_tip_orientation_world_frame,
                                                                    self.info['desired_goal']['key_pose_world_frame']
                                                                    ['orientation']), dtype=np.float32)

        current_rotation_matrix = R.from_quat(np.asarray(allen_key_tip_orientation_world_frame)).as_matrix()
        goal_rotation_matrix = R.from_quat(self.info['desired_goal']['key_pose_world_frame']['orientation']).as_matrix()

        # Now R*R' should produce eye(3)
        rotation_vector = R.from_matrix(current_rotation_matrix.dot(goal_rotation_matrix.T)).as_rotvec()
        rotation_error = np.pi - np.linalg.norm(rotation_vector)    # in rads
        percentage_rotation_error = rotation_error / np.pi          # normalized from 0 to 1 as a %

        self.info['desired_goal']['previous_distance_to_target'] = self.info['desired_goal']['distance_to_target']
        self.info['desired_goal']['distance_to_target'] = distance_to_target
        self.info['desired_goal']['distance_moved_towards_target'] = self.info['desired_goal']['previous_distance_to_target'] - \
                                                             self.info['desired_goal']['distance_to_target']

        self.info['desired_goal']['orientation_difference'] = difference_quaternion
        self.info['desired_goal']['percentage_rotation_error'] = percentage_rotation_error

        # TODO (ak): more elegant way of transferring nested dict to flattened list,
        #            self.__dict_to_list() is ordered in reverse
        desired_goal = []
        desired_goal.extend(list(self.info['desired_goal']['key_pose_world_frame']['position']))
        # desired_goal.extend(list(self.info['desired_goal']['key_pose_world_frame']['orientation']))
        self.observation['desired_goal'] = np.array(desired_goal)

        achieved_goal = []
        achieved_goal.extend(list(self.info['arm']['allen_key_tip_pose_world_frame']['position']))
        # achieved_goal.extend(list(self.info['arm']['allen_key_tip_pose_world_frame']['orientation']))
        self.observation['achieved_goal'] = np.array(achieved_goal)

        observation = []
        observation.extend(list(self.info['desired_goal']['key_pose_world_frame']['position']))
        observation.extend(list(self.info['desired_goal']['key_pose_world_frame']['orientation']))
        observation.append(self.info['desired_goal']['initial_distance_to_target'])
        observation.extend(list(self.info['desired_goal']['initial_orientation_difference']))
        observation.extend(list(self.info['arm']['joint_sensors']['position']))
        observation.extend(list(self.info['arm']['joint_sensors']['velocity']))
        observation.extend(list(self.info['arm']['joint_sensors']['effort']))
        observation.extend(list(self.info['arm']['joint_sensors']['joint_limit_switches']))
        observation.extend(list(self.info['arm']['joint_sensors']['joint_vel_limit_switches']))
        self.observation['observation'] = np.array(observation)

        if reset:
            self.info['desired_goal']['initial_distance_to_target'] = self.info['desired_goal']['distance_to_target']
            self.info['desired_goal']['initial_orientation_difference'] = self.info['desired_goal']['orientation_difference']

            self.info['sim']['steps_executed'] = 0
            self.info['sim']['seconds_executed'] = 0
        else:
            self.info['sim']['steps_executed'] += 1
            self.info['sim']['seconds_executed'] += self.info['sim']['step_duration']

    def __execute_action(self, action):
        # from network
        action = action['joint_velocity_commands'] if isinstance(action, dict) else action
        # URDF cmd vel limits
        clipped_action = []
        for joint_index in range(self.info['arm']['num_joints']):
            clipped_action.append(np.clip(action[joint_index],
                                          self.info['arm']['joint_vel_limits'][joint_index][0],
                                          self.info['arm']['joint_vel_limits'][joint_index][1]))
        clipped_action = np.array(clipped_action)

        pb.setJointMotorControlArray(bodyUniqueId=self.py_bullet_info.arm_uid,
                                     jointIndices=range(0, self.info['arm']['num_joints']),
                                     controlMode=pb.VELOCITY_CONTROL,
                                     targetVelocities=clipped_action,
                                     # forces=self.info['arm']['max_force']
                                     )

        pb_steps_per_sim_step = int(self.info['sim']['step_duration'] / UWRTArmGoalEnv.DEFAULT_PYBULLET_TIME_STEP)
        for pb_sim_step in range(pb_steps_per_sim_step):
            pb.stepSimulation()

    def compute_reward(self, achieved_goal, desired_goal, info):
        percent_time_used = self.info['sim']['steps_executed'] / self.info['sim']['max_steps']
        percent_distance_remaining = self.info['desired_goal']['distance_to_target'] / \
                                     self.info['desired_goal']['initial_distance_to_target']

        # TODO: scale based off max speed to normalize
        # TODO: investigate weird values
        distance_moved = self.info['desired_goal']['distance_moved_towards_target'] / self.info['desired_goal']['initial_distance_to_target']

        # TODO (ak): changed reward to use percent_distance_remaining
        # reward = distance_moved * UWRTArmGoalEnv.REWARD_MAX / 2
        reward = (1 - percent_distance_remaining) * UWRTArmGoalEnv.REWARD_MAX / 2

        # TODO (ak): tweak reward formula for orientation
        percentage_rotation_error = self.info['desired_goal']['percentage_rotation_error']
        reward -= percentage_rotation_error * UWRTArmGoalEnv.REWARD_MAX / 10

        if self.info['desired_goal']['distance_to_target'] < UWRTArmGoalEnv.GOAL_POSITION_DISTANCE_THRESHOLD:
            self.info['sim']['end_condition'] = 'Key Reached'
            reward += UWRTArmGoalEnv.REWARD_MAX / 2

        elif self.info['sim']['steps_executed'] >= self.info['sim']['max_steps']:
            self.info['sim']['end_condition'] = 'Max Sim Steps Executed'
            reward -= UWRTArmGoalEnv.REWARD_MAX / 2

        # TODO: add penalty for hitting anything that's not the desired key

        return reward

    # TODO: try sparse rewards for HER
    # def compute_reward(self, achieved_goal, desired_goal, info):
    #     assert achieved_goal.shape == desired_goal.shape
    #     ### SPARSE
    #     # return -(self.info['desired_goal']['distance_to_target']
    #     #          > UWRTArmGoalEnv.GOAL_POSITION_DISTANCE_THRESHOLD).astype(np.float32)
    #     ### SIMPLE DENSE
    #     return -self.info['desired_goal']['distance_to_target']

    def _is_success(self, achieved_goal, desired_goal):
        assert achieved_goal.shape == desired_goal.shape
        return (self.info['desired_goal']['distance_to_target']
                < UWRTArmGoalEnv.GOAL_POSITION_DISTANCE_THRESHOLD).astype(np.float32)

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

    def step(self, action):
        # TODO: is this required? does speed increase if used in non-gui mode? does speed slow increaenvse if not used in gui mode?
        pb.configureDebugVisualizer(pb.COV_ENABLE_SINGLE_STEP_RENDERING)

        self.__execute_action(action)

        self.__update_observation_and_info()

        self.info['desired_goal']['is_success'] = self._is_success(achieved_goal=self.observation['achieved_goal'],
                                                                    desired_goal=self.observation['desired_goal'])

        reward = self.compute_reward(achieved_goal=self.observation['achieved_goal'],
                                       desired_goal=self.observation['desired_goal'],
                                       info=self.info['desired_goal']['is_success'])

        return self.observation, reward, self.info['desired_goal']['is_success'], self.info

    def render(self, mode='human'):
        if not self.init_options.enable_render:
            raise UserWarning('This environment was initialized with rendering disabled')
        return

    def close(self):
        if self.init_options.enable_render:
            pb.disconnect()
