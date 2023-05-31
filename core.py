import random
import numpy as np
import logging
from utils import get_epsilon, visualize, visualize_
from agent import DQNAgent
logger = logging.getLogger(__name__)

def train(env, agent, buffer, grid, cfg, seed=0):
    '''
    upper: -> UpperAgent
    lower: -> LowerAgent
    '''

    if isinstance(agent, tuple):

        upper, lower = agent
        upper_buffer, lower_buffer = buffer
        upper, lower = agent

        done, truncated = False, False
        state, _ = env.reset(seed = seed)

        uplosses = []
        lowlosses = []
        up_Q = []
        low_Q = []
        uprewards = []
        lowrewards = []

        for step in range(1, cfg.timesteps + 1):
            if done or truncated:
                state, _ = env.reset()
                done, truncated = False, False

            upper_obs = upper.get_observation(state)
            message = upper.get_action(upper_obs)

            lower_obs = lower.get_observation(state, grid, message)
            lower_action = lower.get_action(lower_obs)

            eps = get_epsilon(step-1, cfg.eps_min, cfg.eps_max, cfg.eps_steps)
            if random.random() < eps:
                lower_action = env.action_space.sample()
            
            # x, y = state // 8, state % 8
            # if random.random() < slippery[x, y]:
            #     action = env.action_space.sample()

            next_state, reward, done, truncated, info = env.step(lower_action)

            upper_reward = int(lower_action==message) * cfg.weight + reward
            lower_reward = int(lower_action!=message) * 0.00 + reward

            upper_next_obs = upper.get_observation(next_state)
            upper_next_message = upper.get_action(upper_next_obs)
            lower_next_obs = lower.get_observation(next_state, grid, upper_next_message)

            upper_buffer.add((upper_obs, message, upper_reward, upper_next_obs, int(done)))
            lower_buffer.add((lower_obs, lower_action, lower_reward, lower_next_obs, int(done)))
            
            state = next_state

            if step > cfg.batch_size + cfg.nstep:
                upper_batch = upper_buffer.sample(cfg.batch_size)
                lower_batch = lower_buffer.sample(cfg.batch_size)

                upper_loss, _, upper_Q = upper.update(upper_batch, step)
                lower_loss, _, lower_Q = lower.update(lower_batch, step)

                uplosses.append(upper_loss)
                lowlosses.append(lower_loss)
                up_Q.append(upper_Q)
                low_Q.append(lower_Q)
                uprewards.append(upper_buffer.reward.mean().item())
                lowrewards.append(lower_buffer.reward.mean().item())
                if step % 100 == 0:
                    logger.info(f"Step: {step}, Upper Loss: {upper_loss}, Lower Loss: {lower_loss}, Upper Q: {upper_Q}, Lower Q: {lower_Q}")
                    visualize(uplosses, lowlosses, up_Q, low_Q, uprewards, lowrewards)
        
        
        state, _ = env.reset(seed = seed)
        done, truncated = False, False
        while not (done or truncated):
            upper_obs = upper.get_observation(state)
            message = upper.get_action(upper_obs)

            lower_obs = lower.get_observation(state, grid, message)
            lower_action = lower.get_action(lower_obs)

            state, reward, done, truncated, info = env.step(lower_action)
            env.render()

        upper_map = np.zeros((8, 8))
        lower_map = np.zeros((8, 8))

        for i in range(8):
            for j in range(8):
                upper_obs = upper.get_observation(i*8+j)
                message = upper.get_action(upper_obs)

                lower_obs = lower.get_observation(i*8+j, grid, message)
                lower_action = lower.get_action(lower_obs)

                upper_map[i, j] = message
                lower_map[i, j] = lower_action
                if grid[i, j] == -1:
                    upper_map[i, j] = -1
                    lower_map[i, j] = -1
        
        np.save(f"grid.npy", grid)
        np.save(f"lower_map_{seed}.npy", lower_map)
        np.save(f"upper_map_{seed}.npy", upper_map)

    elif isinstance(agent, DQNAgent):
        done, truncated = False, False
        state, _ = env.reset(seed = seed)

        losses = []
        Qs = []
        rewards = []

        for step in range(1, cfg.timesteps + 1):
            if done or truncated:
                state, _ = env.reset()
                done, truncated = False, False

            obs = agent.get_observation(state, grid)
            action = agent.get_action(obs)

            eps = get_epsilon(step-1, cfg.eps_min, cfg.eps_max, cfg.eps_steps)
            if random.random() < eps:
                action = env.action_space.sample()

            next_state, reward, done, truncated, info = env.step(action)
            next_obs = agent.get_observation(next_state, grid)

            buffer.add((obs, action, reward, next_obs, int(done)))

            state = next_state

            if step > cfg.batch_size + cfg.nstep:
                batch = buffer.sample(cfg.batch_size)
                
                loss, _, Q = agent.update(batch, step)
                losses.append(loss)
                Qs.append(Q)
                rewards.append(buffer.reward.mean().item())
                if step % 100 == 0:
                    logger.info(f"Step: {step}, Loss: {loss}, Q: {Q}")
                    visualize_(losses, Qs, rewards)

        a_map = np.zeros((8, 8))
        for i in range(8):
            for j in range(8):
                obs = agent.get_observation(i*8+j, grid)
                action = agent.get_action(obs)
                a_map[i, j] = action
                if grid[i, j] == -1:
                    a_map[i, j] = -1

        print( a_map)
        np.save("a_map.npy", a_map)