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

class Gen3Lite2FArmEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    # Arm Constants
    ARM_URDF = '/home/akeaveny/git/uwrt_arm_rl/gym-uwrt-arm/urdfs/gen3_lite/gen3_lite_gen3_lite_2f.urdf'
    ARM_URDF_FILE_NAME = 'gen3_lite_gen3_lite_2f.urdf'
    ALLEN_KEY_LENGTH = 0.10
    EE_LINK_OFFSET = 0.10

    # Pybullet Constants
    DEFAULT_PYBULLET_TIME_STEP = 1 / 240

    # Reward Constants
    GOAL_POSITION_DISTANCE_THRESHOLD = 1 / 1000 # 1 mm
    REWARD_MAX = 100
    reward_range = (-float('inf'), float(REWARD_MAX))

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
            desired_sim_step_duration / Gen3Lite2FArmEnv.DEFAULT_PYBULLET_TIME_STEP) * Gen3Lite2FArmEnv.DEFAULT_PYBULLET_TIME_STEP

        self.init_options = self.InitOptions(key_position=key_position, key_orientation=key_orientation,
                                             max_steps=max_steps, sim_step_duration=sim_step_duration,
                                             enable_render=enable_render, tmp_dir=tempfile.TemporaryDirectory())

        self.__initialize_gym()
        self.__initialize_sim()

    def __initialize_gym(self):
        arm_urdf = URDF.load(Gen3Lite2FArmEnv.ARM_URDF)
        num_joints = len(arm_urdf.actuated_joints) - 1 ### TODO: we execlude right_finger_bottom_joint
        ee_link = 5                                    ### TODO: from link_names

        actuated_joints_names = {}
        for joint_idx in range(len(arm_urdf.actuated_joints)):
            actuated_joints_names["actuated_joint_"+str(joint_idx)] = (
                                                                        arm_urdf.actuated_joints[joint_idx].name,
                                                                        arm_urdf.actuated_joints[joint_idx].parent,
                                                                        arm_urdf.actuated_joints[joint_idx].child,
                                                                        )
        joint_names = {}
        for joint_idx in range(len(arm_urdf.joints)):
            joint_names["joint_" + str(joint_idx)] = (
                                                       arm_urdf.joints[joint_idx].name,
                                                       arm_urdf.joints[joint_idx].parent,
                                                       arm_urdf.joints[joint_idx].child,
                                                       )

        link_names = {}
        for link_idx in range(len(arm_urdf.links)):
            link_names["link_" + str(link_idx)] = (
                                                    arm_urdf.links[link_idx].name,
                                                    )

        joint_limits = []
        for joint_idx in range(num_joints):
            joint_limits.append((arm_urdf.actuated_joints[joint_idx].limit.lower,
                                 arm_urdf.actuated_joints[joint_idx].limit.upper))

        joint_vel_limits = []
        for joint_idx in range(num_joints):
            joint_vel_limits.append((-1*arm_urdf.actuated_joints[joint_idx].limit.velocity,
                                        arm_urdf.actuated_joints[joint_idx].limit.velocity))

        # All joint limit switch states are either NOT_TRIGGERED[0], LOWER_TRIGGERED[1], UPPER_TRIGGERED[2]
        # The exception is roll which only has NOT_TRIGGERED[0]
        joint_limit_switch_dims = np.concatenate(
            (np.full(num_joints - 1, 3), np.array([1])))  # TODO: this is wrong. wrist joints flipped

        # TODO: Load mechanical limits from something (ex. pull info from config in uwrt_mars_rover thru git)
        self.observation_space = spaces.Dict({
            'goal': spaces.Dict({
                'key_pose_world_frame': spaces.Dict({
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
                # Order of array is [turntable, shoulder, elbow, wrist pitch, wrist roll] # TODO: this is wrong. wrist joints flipped
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
            'joint_velocity_commands': spaces.Box(low=np.full(num_joints, -1.5), high=np.full(num_joints, 1.5),
                                                  shape=(num_joints,), dtype=np.float32)
        })

        self.observation = {
            'goal': {
                'key_pose_world_frame': {
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
                'end_condition': 'Not Done'
            },
            'goal': {
                'distance_to_target': 0,
                'previous_distance_to_target': 0,
                'distance_moved_towards_target': 0,
                'orientation_difference': [0, 0, 0, 0],
            },
            'arm': {
                'allen_key_tip_pose_world_frame': {
                    'position': [0, 0, 0],
                    'orientation': [0, 0, 0, 0],
                },
                'num_joints': num_joints,
                'ee_link': ee_link,
                'joint_limits': joint_limits,
                'joint_vel_limits': joint_vel_limits,
            },
        }

    def __initialize_sim(self):
        self.py_bullet_info = Gen3Lite2FArmEnv.PyBulletInfo(None, None)

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

    def __spawn_gen3_lite_2f_arm(self):
        self.py_bullet_info.arm_uid = pb.loadURDF(Gen3Lite2FArmEnv.ARM_URDF, useFixedBase=True,
            flags=pb.URDF_MAINTAIN_LINK_ORDER | pb.URDF_MERGE_FIXED_LINKS)

        # TODO: Randomize arm starting configuration
        # TODO: Calculate Claw link pose from desired allen key tip pose

        # TODO: limit to valid configurations using nullspace?
        joint_home_poses = pb.calculateInverseKinematics(self.py_bullet_info.arm_uid,
                                                         endEffectorLinkIndex=self.info['arm']['ee_link'],
                                                         targetPosition=[0.3, 0.0, 0.8],
                                                         targetOrientation=pb.getQuaternionFromEuler(
                                                             [0, np.pi / 3.5, 0])
                                                         )

        # Move joints to starting position
        for joint_index in range(self.info['arm']['num_joints']):
            pb.resetJointState(self.py_bullet_info.arm_uid, jointIndex=joint_index,
                               targetValue=joint_home_poses[joint_index], targetVelocity=0)

        # Draw Coordinate Frames. These are the inertial frames. # TODO(add toggle using addUserDebugParameter)
        axis_length = 0.15
        for joint_index in range(self.info['arm']['num_joints']+1):
            link_name = pb.getJointInfo(self.py_bullet_info.arm_uid, joint_index)[12].decode('ascii')
            pb.addUserDebugText(link_name, [0, 0, 0], textColorRGB=[0, 1, 1], textSize=0.75,
                                parentObjectUniqueId=self.py_bullet_info.arm_uid, parentLinkIndex=joint_index)
            pb.addUserDebugLine(lineFromXYZ=[0, 0, 0], lineToXYZ=[axis_length, 0, 0], lineColorRGB=[1, 0, 0],
                                parentObjectUniqueId=self.py_bullet_info.arm_uid, parentLinkIndex=joint_index)
            pb.addUserDebugLine(lineFromXYZ=[0, 0, 0], lineToXYZ=[0, axis_length, 0], lineColorRGB=[0, 1, 0],
                                parentObjectUniqueId=self.py_bullet_info.arm_uid, parentLinkIndex=joint_index)
            pb.addUserDebugLine(lineFromXYZ=[0, 0, 0], lineToXYZ=[0, 0, axis_length], lineColorRGB=[0, 0, 1],
                                parentObjectUniqueId=self.py_bullet_info.arm_uid, parentLinkIndex=joint_index)

            # Draw Allen Key Offset # TODO(melvinw): transform to link frame and draw from front of box to allen key
            claw_visual_shape_data = pb.getVisualShapeData(self.py_bullet_info.arm_uid)[self.info['arm']['num_joints']]
            claw_visual_box_z_dim = claw_visual_shape_data[3][2]
            # Box geometry origin is defined at the center of the box
            allen_key_tip_position_visual_frame = [0, 0, (Gen3Lite2FArmEnv.EE_LINK_OFFSET / 2 + Gen3Lite2FArmEnv.ALLEN_KEY_LENGTH)]
            pb.addUserDebugLine(lineFromXYZ=[0, 0, 0], lineToXYZ=allen_key_tip_position_visual_frame,
                                lineColorRGB=[1, 1, 1], lineWidth=5,
                                parentObjectUniqueId=self.py_bullet_info.arm_uid,
                                parentLinkIndex=self.info['arm']['ee_link'])

    def __spawn_key(self):
        """ Randomize keyboard """
        # np.random.seed(0) ### uncomment to spawn in same location
        self.keyboard_position = np.array([np.random.uniform(0.625, 0.675),
                                              np.random.uniform(-0.30, 0.30),
                                              np.random.uniform(0.65, 0.675)])

        # we want the key vertical (should be -90 deg)
        self.keyboard_orientation = R.from_euler('y', -90,degrees=True).as_quat()

        # drawing rectangle
        pb.addUserDebugLine(lineFromXYZ=self.keyboard_position, lineToXYZ=self.keyboard_position + np.array([0, 0, 15/1000]),
                            lineColorRGB=[0, 0, 0], lineWidth=50)

        self.observation = {
            'goal': {
                'key_pose_world_frame': {
                    'position': self.keyboard_position,
                    'orientation': self.keyboard_orientation,
                }
            }
        }

    def __get_allen_key_tip_in_world_frame(self):

        ee_link_state_pose = pb.getLinkState(self.py_bullet_info.arm_uid, self.info['arm']['ee_link'])
        ee_link_state_position = ee_link_state_pose[4]
        ee_link_state_orientation = ee_link_state_pose[5]

        allen_key_offset = np.array([0, 0, Gen3Lite2FArmEnv.EE_LINK_OFFSET + Gen3Lite2FArmEnv.ALLEN_KEY_LENGTH])[np.newaxis].T

        # transformation
        from scipy.spatial.transform import Rotation as R
        ee_link_rot_matrix = R.from_quat([ee_link_state_orientation[0],
                                               ee_link_state_orientation[1],
                                               ee_link_state_orientation[2],
                                               ee_link_state_orientation[3]]).as_matrix()
        allen_key_trans = ee_link_rot_matrix @ allen_key_offset

        allen_key_tip_position_world_frame = ee_link_state_position + allen_key_trans.T[0]
        allen_key_tip_orientation_world_frame = ee_link_state_orientation

        # drawing allen_key_tip_pose_world_frame
        pb.addUserDebugLine(lineFromXYZ=allen_key_tip_position_world_frame,
                            lineToXYZ=allen_key_tip_position_world_frame + np.array([0, 0, 5 / 1000]),
                            lineColorRGB=[1, 1, 0], lineWidth=50)

        return allen_key_tip_position_world_frame, allen_key_tip_orientation_world_frame

    def __update_observation_and_info(self, reset=False):
        joint_states = pb.getJointStates(self.py_bullet_info.arm_uid,
                                         np.arange(pb.getNumJoints(self.py_bullet_info.arm_uid)))
        ### TODO: WE EXECLUDE THE ALLEN KEY TIP
        joint_states = joint_states[0:self.info['arm']['num_joints']]

        joint_positions = np.array([joint_state[0] for joint_state in joint_states], dtype=np.float32)
        joint_velocities = np.array([joint_state[1] for joint_state in joint_states], dtype=np.float32)
        joint_torques = np.array([joint_state[3] for joint_state in joint_states], dtype=np.float32)

        joint_limit_states = [1 if joint_positions[joint_index] <= self.info['arm']['joint_limits'][joint_index][0] else
                              2 if joint_positions[joint_index] >= self.info['arm']['joint_limits'][joint_index][1] else
                              0 for joint_index in range(self.info['arm']['num_joints'])]
        joint_vel_limit_states = [1 if joint_velocities[joint_index] <= self.info['arm']['joint_vel_limits'][joint_index][0] else
                              2 if joint_velocities[joint_index] >= self.info['arm']['joint_vel_limits'][joint_index][1] else
                              0 for joint_index in range(self.info['arm']['num_joints'])]

        self.observation['joint_sensors'] = {
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
                                        self.observation['goal']['key_pose_world_frame']['position']),
                                dtype=np.float32)

        difference_quaternion = np.array(pb.getDifferenceQuaternion(allen_key_tip_orientation_world_frame,
                                                                    self.observation['goal']['key_pose_world_frame']
                                                                    ['orientation']), dtype=np.float32)

        current_rotation_matrix = R.from_quat(allen_key_tip_orientation_world_frame).as_matrix()
        goal_rotation_matrix = R.from_quat(self.observation['goal']['key_pose_world_frame']
                                               ['orientation']).as_matrix()

        # Now R*R' should produce eye(3)
        rotation_vector = R.from_matrix(current_rotation_matrix.dot(goal_rotation_matrix.T)).as_rotvec()
        rotation_error = np.pi - np.linalg.norm(rotation_vector)    # in rads
        percentage_rotation_error = rotation_error / np.pi          # normalized from 0 to 1 as a %

        self.info['goal']['previous_distance_to_target'] = self.info['goal']['distance_to_target']
        self.info['goal']['distance_to_target'] = distance_to_target
        self.info['goal']['distance_moved_towards_target'] = self.info['goal']['previous_distance_to_target'] - \
                                                             self.info['goal']['distance_to_target']

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
                                     )

        pb_steps_per_sim_step = int(self.info['sim']['step_duration'] / Gen3Lite2FArmEnv.DEFAULT_PYBULLET_TIME_STEP)
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
        # reward = distance_moved * Gen3Lite2FArmEnv.REWARD_MAX / 2
        reward = (1 - percent_distance_remaining) * Gen3Lite2FArmEnv.REWARD_MAX / 2

        # TODO (ak): tweak reward formula to reward more for orientation thats closer to perpendicular to surface of key
        percentage_rotation_error = self.info['goal']['percentage_rotation_error']
        # reward -= percentage_rotation_error * Gen3Lite2FArmEnv.REWARD_MAX / 10

        if self.info['goal']['distance_to_target'] < Gen3Lite2FArmEnv.GOAL_POSITION_DISTANCE_THRESHOLD:
            self.info['sim']['end_condition'] = 'Key Reached'
            done = True
            reward += Gen3Lite2FArmEnv.REWARD_MAX / 2

        elif self.info['sim']['steps_executed'] >= self.info['sim']['max_steps']:
            self.info['sim']['end_condition'] = 'Max Sim Steps Executed'
            done = True
            reward -= Gen3Lite2FArmEnv.REWARD_MAX / 2
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

        self.__spawn_gen3_lite_2f_arm()
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
