import numpy as np

from pathlib import Path

ROOT_DIR_PATH = Path(__file__).parent.absolute().resolve(strict=True)

MODELS_DIR_PATH = (ROOT_DIR_PATH / 'trained_models').resolve(strict=True)
BASE_SAVE_PATH = (MODELS_DIR_PATH / 'TD3_keyboard_demo_pos_control').resolve(strict=True)

# Relative Paths
CHECKPOINTS_DIR = 'checkpoints'
BEST_MODEL_SAVE_DIR = 'best_model'
BEST_MODEL_LOG_DIR = 'best_model'
TENSORBOARD_LOG_DIR = 'tensorboard_logs'
STATISTICS_LOG_DIR = 'norm_input_stats'
STATISTICS_PKL_FILE = 'vec_normalize.pkl'

#################################
# PRELIMS
#################################

# Env params
GYM_ID = 'uwrt-arm-v0'   # new urdf
# GYM_ID = 'uwrt-arm-v1' # GoalEnv
# GYM_ID = 'gen3-arm-v0'

''' below is from workspace analysis '''
STARTING_ARM_POSITION = np.array([0.6, 0.0, 0.9])
STARTING_ARM_ORIENTATION = np.array([0, 0, 0, 1])

# val_env Keys
VAL_KEY_POSITIONS = np.array([[0.85, 0, 0.8],
                          [0.85, 0.2, 0.7],
                          [0.85, -0.2, 0.7],
                          [0.9, 0, 0.7]])

# training_env Keys
RAND_KEY_POSITION = np.array([np.random.uniform(0.8, 0.9),
                         np.random.uniform(-0.30, 0.30),
                         np.random.uniform(0.65, 0.85)])

# Starting Key Pose
KEY_POSITION = VAL_KEY_POSITIONS[0].flatten()
KEY_ORIENTATION = np.array([0, 0, 0, 1])

#################################
# TIME FOR TRAINING
#################################

MAX_SIM_SECONDS_PER_EPISODE = 10      ### og value: 10
DESIRED_TRAINING_TIME_HOURS = 1.5     ### og value: 0.5

# System params
ESTIMATED_STEPS_PER_SECOND_1080TI = 600

'''
Calculate max sim steps per episode from desired max episode sim time
'''
# TODO: remove constants here:
PYBULLET_STEPS_PER_ENV_STEP = 1  # This is based on a desired_sim_step_duration of 1/100s (100hz control loop)
PYBULLET_SECONDS_PER_PYBULLET_STEP = 1 / 240

MAX_STEPS_PER_EPISODE = MAX_SIM_SECONDS_PER_EPISODE / PYBULLET_SECONDS_PER_PYBULLET_STEP / PYBULLET_STEPS_PER_ENV_STEP

'''
Calculate env steps from desired training time
'''
DESIRED_TRAINING_TIME_MINS = DESIRED_TRAINING_TIME_HOURS * 60
ESTIMATED_STEPS_PER_MIN_1080TI = ESTIMATED_STEPS_PER_SECOND_1080TI * 60
TOTAL_TRAINING_ENV_STEPS = DESIRED_TRAINING_TIME_MINS * ESTIMATED_STEPS_PER_MIN_1080TI

NUM_TRAINING_EPISODES = TOTAL_TRAINING_ENV_STEPS // MAX_STEPS_PER_EPISODE

##################
# HYPER PARAMS
##################

LEARNING_RATE = 0.0001
BATCH_SIZE = 37
# BATCH_SIZE = 43 # GoalEnv
# BATCH_SIZE = 42 # Gen3

GAMMA = 0.99  # Discount factor

##################
# DQN
##################

# Model Training
TRAIN_FREQ = 1  # minimum number of env time steps between model training
GRADIENT_STEPS = -1  # number of gradient steps to execute. -1 matches the number of steps in the rollout
N_EPISODES_ROLLOUT = -1  # minimum number of episodes between model training

# Target network syncing
TARGET_UPDATE_INTERVAL = 1000  # number of env time steps between target network updates

# Exploration
EXPLORATION_FRACTION = 0.3  # fraction of entire training period over which the exploration rate is reduced
EXPLORATION_INITIAL_EPS = 1.0  # initial value of random action probability
EXPLORATION_FINAL_EPS = 0.3  # final value of random action probability

##############
# PPO
##############

# TODO: Just used LR, BATCH_SIZE, GAMMA

##############
# TD3
##############

# TODO: Just used LR, BATCH_SIZE, GAMMA, TRAIN_FREQ, GRADIENT_STEPS, action_noise (in script)

LEARNING_STARTS = 100

##############
# HER
##############

# Available strategies (cf paper): future, final, episode
GOAL_SELECTION_STRATEGY = 'future'  # equivalent to GoalSelectionStrategy.FUTURE

# If True the HER transitions will get sampled online
ONLINE_SAMPLING = True

BUFFER_SIZE = 2000
