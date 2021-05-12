#####################
# 1. register OpenAI gym env
#####################
env = gym.make("uwrt-arm") # initiate sim
#####################
# 2. Init simulator (e.g. PyBullet)
#####################
observation = env.reset()
for _ in range(num_episodes):
    #####################
    # 3. Learn optimal actions from RL Algorithm (e.g. StabeBaseline3)
    #####################
    action = sample_action(observation)
    #####################
    # 4. Transition to new state with current action
    #####################
    observation, reward, done, info = step(action)
    if done: # hit key or max timesteps
        observation = env.reset()
