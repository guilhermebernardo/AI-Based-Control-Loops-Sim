from stable_baselines3 import A2C, DDPG, DQN, HER, PPO, SAC, TD3
from sb3_contrib import QRDQN, TQC

AGENTS = {
    'a2c': A2C,
    'ddpg': DDPG,
    'dqn': DQN,
    'her': HER,
    'ppo': PPO,
    'qrdqn': QRDQN,
    'sac': SAC,
    'td3': TD3,
    'tqc': TQC
}
