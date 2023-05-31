import hydra
import torch
import utils
import gymnasium as gym
from core import train
from buffer import get_buffer
from agent import UpperAgent, LowerAgent, DQNAgent

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

@hydra.main(config_path="cfgs", config_name="config", version_base="1.3")
def main(cfg):
    env = gym.make(cfg.env_name, render_mode="ansi", map_name="8x8", is_slippery=cfg.slippery)
    env.reset(seed = cfg.seed)
    map = env.render().split("\n")

    utils.set_seed_everywhere(env, cfg.seed)
    grid = utils.get_grid(cfg.env_size, map[1:])

    upper_state_size = cfg.upper_agent.state_size
    upper_action_size = cfg.upper_agent.action_size
    lower_state_size = cfg.lower_agent.state_size
    lower_action_size = cfg.lower_agent.action_size

    upper_buffer = get_buffer(cfg.buffer, 
                              state_size = upper_state_size, 
                              action_size = 1, 
                              device=device)
    lower_buffer = get_buffer(cfg.buffer, 
                              state_size = lower_state_size, 
                              action_size = 1, 
                              device=device)
    buffer = (upper_buffer, lower_buffer)

    uagent = UpperAgent(upper_state_size, upper_action_size, cfg.upper_agent, device=device)
    lagent = LowerAgent(lower_state_size, lower_action_size, cfg.lower_agent, device=device)
    agent = (uagent, lagent)
    
    if not cfg.hierarchical:
        state_size = cfg.agent.state_size
        action_size = cfg.agent.action_size
        agent = DQNAgent(state_size, action_size, cfg.agent, device=device)
        buffer = get_buffer(cfg.buffer, 
                            state_size=state_size, 
                            action_size = 1, 
                            device=device)

    train(env, agent, buffer, grid, cfg.train, seed=cfg.seed)
    
    if not cfg.hierarchical:
        env = gym.make(cfg.env_name, render_mode="human", map_name="8x8", is_slippery=cfg.slippery)
        state, _ = env.reset(seed = cfg.seed)
        done, truncated = False, False
        while not (done or truncated):
            obs = agent.get_observation(state, grid)
            action = agent.get_action(obs)

            state, reward, done, truncated, info = env.step(action)
            env.render()
        return

    env = gym.make(cfg.env_name, render_mode="human", map_name="8x8", is_slippery=cfg.slippery)
    state, _ = env.reset(seed = cfg.seed)
    done, truncated = False, False
    while not (done or truncated):
        upper_obs = uagent.get_observation(state)
        message = uagent.get_action(upper_obs)

        lower_obs = lagent.get_observation(state, grid, message)
        lower_action = lagent.get_action(lower_obs)

        state, reward, done, truncated, info = env.step(lower_action)
        env.render()

if __name__ == "__main__":
    main()