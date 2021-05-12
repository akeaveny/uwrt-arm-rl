from gym.envs.registration import register

register(
    id='uwrt-arm-v0',
    entry_point='gym_uwrt_arm.envs:UWRTArmEnv',
)

register(
    id='uwrt-arm-v1',
    entry_point='gym_uwrt_arm.envs:UWRTArmGoalEnv',
)

register(
    id='gen3-arm-v0',
    entry_point='gym_uwrt_arm.envs:Gen3Lite2FArmEnv',
)

