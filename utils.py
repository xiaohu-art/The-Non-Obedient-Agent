import os
import random
import gymnasium as gym
import numpy as np
import torch

def set_seed_everywhere(env: gym.Env, seed=0):
    """
    Set seed for all randomness sources
    """
    env.action_space.seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

def map_transform():
    pass