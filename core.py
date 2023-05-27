import random
import numpy as np
from utils import get_epsilon
from scipy.stats import entropy as KL

def train(env, upper, lower, buffer, grid, slippery, cfg, seed=0):
    '''
    upper: -> UpperAgent
    lower: -> LowerAgent
    '''
    
    done, truncated = False, False
    state, _ = env.reset(seed = seed)

    for step in range(1, cfg.timesteps + 1):
        if done or truncated:
            state, _ = env.reset()
            done, truncated = False, False

        upper_obs = upper.get_observation(state)
        message = upper.get_message(upper_obs)

        lower_obs = lower.get_observation(state, grid, slippery, message)
        lower_action = lower.get_action(lower_obs)

        eps = get_epsilon(step-1, cfg.eps_min, cfg.eps_max, cfg.eps_steps)

        if random.random() < eps:
            action = env.action_space.sample()
        else:
            action = lower_action

        x, y = state // 8, state % 8
        if random.random() < slippery[x, y]:
            action = env.action_space.sample()
        else:
            action = lower_action

        next_state, reward, done, truncated, info = env.step(action)
        
        upper_reward = KL(lower_action, message) + reward
        lower_reward = reward

        state = next_state
        upper_next_obs = upper.get_observation(state)
        upper_next_message = upper.get_message(upper_next_obs)
        lower_next_obs = lower.get_observation(state, grid, slippery, upper_next_message)

        break