import random
import numpy as np
from utils import get_epsilon

def train(env, upper, lower, buffer, grid, cfg, seed=0):
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

        lower_obs = lower.get_observation(grid, state)
        belief, lower_action = lower.get_action(lower_obs)

        eps = get_epsilon(step-1, cfg.eps_min, cfg.eps_max, cfg.eps_steps)

        if random.random() < eps:
            action = env.action_space.sample()
        else:
            if random.random() < belief:
                action = lower_action
            else:
                action = message

        next_state, reward, done, truncated, info = env.step(action)
        state = next_state

        break