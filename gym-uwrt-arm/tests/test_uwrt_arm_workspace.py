import config

import time

import gym
import numpy as np

import pybullet as pb

import gym_uwrt_arm.envs.uwrt_arm_env
GYM_ID = 'uwrt-arm-v0'

####################
####################

class TestClass:
    NUM_EPISODES = 1
    MAX_STEPS = int(1e7)
    DISCRETIZED_JOINT_POSITIONS = int(1e3)

    XYZ_File = 'workspace_analysis/uwrt_arm_workspace_' + np.str(MAX_STEPS) + '.xyz'

    def __draw_keyboard(self):

        heights = np.arange(start=0.70, stop=0.90, step=(0.90-0.70)/1000)

        for height in heights:
            ### drawing 'black' keyboard
            pb.addUserDebugLine(lineFromXYZ=[0.85, -0.3, height],
                                lineToXYZ=[0.85, 0.3, height],
                                lineColorRGB=[0, 0, 0], lineWidth=10)

    def __draw_configs(self, env, arm_uid):

        # Move joints to starting position
        joint_home_poses = pb.calculateInverseKinematics(arm_uid,
                                                         endEffectorLinkIndex=env.info['arm']['ee_link_idx'],
                                                         targetPosition=config.STARTING_ARM_POSITION,
                                                         targetOrientation=config.STARTING_ARM_ORIENTATION,
                                                         )
        _joint_home_poses = np.array(joint_home_poses)
        _joint_home_poses[-6] = -0.05   # gripper left
        _joint_home_poses[-7] = 0.05    # gripper right
        _joint_home_poses[-8] = 0       # servo joint
        for joint_index in range(env.info['arm']['num_actuated_joints']):
            pb.resetJointState(arm_uid,
                               jointIndex=joint_index,
                               targetValue=_joint_home_poses[joint_index],
                               targetVelocity=0)

        # starting arm position
        ee_link_state = pb.getLinkState(arm_uid, env.info['arm']['ee_link_idx'])
        allen_key_tip_position_world_frame = np.asarray(ee_link_state[4])
        allen_key_tip_orientation_world_frame = np.asarray(ee_link_state[5])
        pb.addUserDebugLine(lineFromXYZ=allen_key_tip_position_world_frame,
                            lineToXYZ=allen_key_tip_position_world_frame + np.array([0, 0, 5 / 100]),
                            lineColorRGB=[0, 0, 1], lineWidth=15)

        # val_env Keys
        for key_position in config.VAL_KEY_POSITIONS:
            pb.addUserDebugLine(lineFromXYZ=key_position - np.array([0, 2 / 100, 0]),
                                lineToXYZ=key_position + np.array([0, 2 / 100, 0]),
                                lineColorRGB=[0, 1, 0], lineWidth=10)

        # self.__draw_keyboard()
        time.sleep(10)

    def __run_test(self, env, arm_uid):

        arm_joint_idxs = np.array(env.info['arm']['arm_joint_idxs'])
        arm_joint_names = np.array(env.info['arm']['actuated_joints_names'])[arm_joint_idxs]
        _joint_limits = env.info['arm']['joint_limits']

        joint_limits = {}
        for i, arm_joint_idx in enumerate(arm_joint_idxs):
            # print(f'{str(arm_joint_names[i])}')
            _joint_limit = _joint_limits[i]
            joint_limits[str(arm_joint_names[i])] = np.arange(start=_joint_limit[0], stop=_joint_limit[1],
                                                              step=((_joint_limit[1]-_joint_limit[0])/TestClass.DISCRETIZED_JOINT_POSITIONS))

        xyz_points = []
        for step in range(TestClass.MAX_STEPS):
            print(f'{step}/{TestClass.MAX_STEPS}')

            ### testing each joint limit
            # arm_joint_positions = []
            # for i, arm_joint_idx in enumerate(arm_joint_idxs):
            #     if i == 4:
            #         random_idx = np.random.randint(low=0, high=TestClass.DISCRETIZED_JOINT_POSITIONS, size=1)
            #         arm_joint_positions.extend(joint_limits[str(arm_joint_names[i])][random_idx])
            #     else:
            #         random_idx = np.random.randint(low=0, high=1, size=1)
            #         arm_joint_positions.extend(joint_limits[str(arm_joint_names[i])][random_idx])

            ### selecting random configs
            arm_joint_positions = []
            for i, arm_joint_idx in enumerate(arm_joint_idxs):
                random_idx = np.random.randint(low=0, high=TestClass.DISCRETIZED_JOINT_POSITIONS, size=1)
                arm_joint_positions.extend(joint_limits[str(arm_joint_names[i])][random_idx])

            # moving arm with positon control
            for i, arm_joint_idx in enumerate(arm_joint_idxs):
                pb.setJointMotorControl2(bodyIndex=arm_uid,
                                         jointIndex=arm_joint_idx,
                                         controlMode=pb.POSITION_CONTROL,
                                         targetPosition=arm_joint_positions[i],
                                         )

            pb_steps_per_sim_step = int(env.info['sim']['step_duration'] / env.DEFAULT_PYBULLET_TIME_STEP)
            for pb_sim_step in range(pb_steps_per_sim_step):
                pb.stepSimulation()

            # drawing point from ee_link_frame
            ee_link_state = pb.getLinkState(arm_uid, env.info['arm']['ee_link_idx'])
            allen_key_tip_position_world_frame = np.asarray(ee_link_state[4])
            allen_key_tip_orientation_world_frame = np.asarray(ee_link_state[5])
            pb.addUserDebugLine(lineFromXYZ=allen_key_tip_position_world_frame,
                                lineToXYZ=allen_key_tip_position_world_frame + np.array([0, 0, 5 / 1000]),
                                lineColorRGB=[1, 1, 0], lineWidth=3)

            # appending to global xyz points
            xyz_points.append(allen_key_tip_position_world_frame)

            if step % 100 == 0:
                key_position = np.array([np.random.uniform(0.8, 0.9),
                                         np.random.uniform(-0.30, 0.30),
                                         np.random.uniform(0.65, 0.85)])
                pb.addUserDebugLine(lineFromXYZ=key_position - np.array([0, 1 / 100, 0]),
                                    lineToXYZ=key_position + np.array([0, 1 / 100, 0]),
                                    lineColorRGB=[1, 0, 1], lineWidth=5)

        # writing xyz points
        xyzfile = open(TestClass.XYZ_File, "w")
        for xyz_point in xyz_points:
            xyzfile.write(f'{xyz_point[0]} {xyz_point[1]} {xyz_point[2]}\n')
        xyzfile.close()

    def test_env(self):
        env = gym.make(GYM_ID, key_position=config.KEY_POSITION, key_orientation=config.KEY_ORIENTATION, is_val_env=False,
                       max_steps=self.MAX_STEPS, enable_render=True)
        arm_uid = pb.loadURDF(env.ARM_URDF, useFixedBase=True,
                              flags=pb.URDF_MAINTAIN_LINK_ORDER | pb.URDF_MERGE_FIXED_LINKS)
        self.__draw_configs(env, arm_uid)
        self.__run_test(env, arm_uid)

if __name__ == '__main__':
    test = TestClass()
    test.test_env()
