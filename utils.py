import os
import random
import gymnasium as gym
import numpy as np
import torch
from gymnasium.spaces import Discrete, Box
import matplotlib.pyplot as plt

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

def get_slippery(env_size):
    return np.random.random((env_size, env_size))

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
    
def visualize(uplosses, lowlosses, up_Q, low_Q, upreward, lowreward):

    plt.figure(figsize=(20, 6))

    plt.subplot(1, 3, 1)
    plt.plot(uplosses, label="Upper Loss")
    plt.plot(lowlosses, label="Lower Loss")
    plt.legend()
    plt.title("Loss")

    plt.subplot(1, 3, 2)
    plt.plot(up_Q, label="Upper Q")
    plt.plot(low_Q, label="Lower Q")
    plt.legend()
    plt.title("Q")

    plt.subplot(1, 3, 3)
    plt.plot(upreward, label="upper Reward")
    plt.plot(lowreward, label="lower Reward")
    plt.legend()
    plt.title("Reward over the whole buffer")

    plt.savefig("result.png")
    plt.close()
    
    