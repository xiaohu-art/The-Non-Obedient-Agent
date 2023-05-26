import os
import random
import gymnasium as gym
import numpy as np
import torch
from gymnasium.spaces import Discrete, Box

def set_seed_everywhere(env: gym.Env, seed=0):
    """
    Set seed for all randomness sources
    """
    env.action_space.seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

def get_grid(env_size, map: list):
    find_hole = lambda line, s: [i for i in range(len(line)) if line[i] == s]

    grid = np.zeros((env_size, env_size))
    grid[-1, -1] = 5
    for i in range(env_size):
        grid[i, find_hole(map[i], "H")] = -1

    return grid

def get_space_shape(space):
    """
    Return the shape of the gym.Space object
    """
    if isinstance(space, Discrete):
        return space.n
    elif isinstance(space, Box):
        if len(space.shape) == 1:
            return space.shape[0]
        else:
            return space.shape
    else:
        raise ValueError(f"Space not supported: {space}")
    
def get_epsilon(step, eps_min, eps_max, eps_steps):
    """
    Return the linearly descending epsilon of the current step for the epsilon-greedy policy. After eps_steps, epsilon will keep at eps_min
    """
    
    if step >= eps_steps:
        return eps_min
    else:
        return eps_max - (eps_max - eps_min) * step / eps_steps
    
    