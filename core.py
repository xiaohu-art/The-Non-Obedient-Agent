import random
import torch
from scipy.stats import entropy as KL

def train(env, agent, buffer, grid, slippery, cfg, seed=0):
    '''
    upper: -> UpperAgent
    lower: -> LowerAgent
    '''
    upper_buffer, lower_buffer = buffer
    upper, lower = agent

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

        action = torch.distributions.Categorical(probs=lower_action).sample().item()

        x, y = state // 8, state % 8
        if random.random() < slippery[x, y]:
            action = env.action_space.sample()

        next_state, reward, done, truncated, info = env.step(action)
        
        upper_reward = KL(lower_action, message) + reward
        lower_reward = reward

        upper_next_obs = upper.get_observation(next_state)
        upper_next_message = upper.get_message(upper_next_obs)
        lower_next_obs = lower.get_observation(next_state, grid, slippery, upper_next_message)

        upper_buffer.add((upper_obs, message, upper_reward, upper_next_obs, int(done)))
        lower_buffer.add((lower_obs, lower_action, lower_reward, lower_next_obs, int(done)))
        
        state = next_state

        break