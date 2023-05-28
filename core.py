import random
import numpy as np
import logging
from utils import get_epsilon
logger = logging.getLogger(__name__)

def eval(env, agent, grid, slippery, episodes, seed):
    upper, lower = agent
    returns = []
    for episode in range(episodes):
        state, _ = env.reset(seed = seed)
        done, truncated = False, False
        while not (done or truncated):
            upper_obs = upper.get_observation(state)
            message = upper.get_action(upper_obs)

            lower_obs = lower.get_observation(state, grid, slippery, message)
            lower_action = lower.get_action(lower_obs)

            state, reward, done, truncated, info = env.step(lower_action)
            returns.append(reward)

    return np.mean(returns), np.std(returns)

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
        message = upper.get_action(upper_obs)

        lower_obs = lower.get_observation(state, grid, slippery, message)
        lower_action = lower.get_action(lower_obs)

        eps = get_epsilon(step-1, cfg.eps_min, cfg.eps_max, cfg.eps_steps)
        if random.random() < eps:
            lower_action = env.action_space.sample()
        
        # x, y = state // 8, state % 8
        # if random.random() < slippery[x, y]:
        #     action = env.action_space.sample()

        next_state, reward, done, truncated, info = env.step(lower_action)

        upper_reward = int(lower_action==message) * 0.01 + reward
        lower_reward = int(lower_action!=message) * 0.00 + reward

        upper_next_obs = upper.get_observation(next_state)
        upper_next_message = upper.get_action(upper_next_obs)
        lower_next_obs = lower.get_observation(next_state, grid, slippery, upper_next_message)

        upper_buffer.add((upper_obs, message, upper_reward, upper_next_obs, int(done)))
        lower_buffer.add((lower_obs, lower_action, lower_reward, lower_next_obs, int(done)))
        
        state = next_state

        if step > cfg.batch_size + cfg.nstep:
            upper_batch = upper_buffer.sample(cfg.batch_size)
            lower_batch = lower_buffer.sample(cfg.batch_size)

            upper_loss, _, upper_Q = upper.update(upper_batch, step)
            lower_loss, _, lower_Q = lower.update(lower_batch, step)

            lower_buffer.reward
            if step % 50 == 0:
                logger.info(f"Step: {step}, Upper Loss: {upper_loss}, Lower Loss: {lower_loss}, Upper Q: {upper_Q}, Lower Q: {lower_Q}, Reward: {lower_buffer.reward.mean().item()}")

        # if step % cfg.eval_interval == 0:
        #     eval_mean, eval_std = eval(env, agent, grid, slippery, cfg.eval_episodes, seed=seed)
        #     logger.info(f"Step: {step}, Eval Mean: {eval_mean}, Eval Std: {eval_std}")

    upper_map = np.zeros((8, 8))
    lower_map = np.zeros((8, 8))

    for i in range(8):
        for j in range(8):
            upper_obs = upper.get_observation(i*8+j)
            message = upper.get_action(upper_obs)

            lower_obs = lower.get_observation(i*8+j, grid, slippery, message)
            lower_action = lower.get_action(lower_obs)

            upper_map[i, j] = message
            lower_map[i, j] = lower_action
            if grid[i, j] == -1:
                upper_map[i, j] = -1
                lower_map[i, j] = -1

    print(grid)
    print(upper_map)
    print(lower_map)