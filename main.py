import hydra
import torch
import utils
import gymnasium as gym
from core import train
from buffer import get_buffer
from agent import UpperAgent, LowerAgent

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

@hydra.main(config_path="cfgs", config_name="config", version_base="1.3")
def main(cfg):
    env = gym.make(cfg.env_name, render_mode="ansi", map_name="8x8", is_slippery=False)
    env.reset(seed = cfg.seed)
    map = env.render().split("\n")

    utils.set_seed_everywhere(env, cfg.seed)
    grid = utils.get_grid(cfg.env_size, map[1:])
    slippery = utils.get_slippery(cfg.env_size)

    upper_state_size = cfg.upper_agent.state_size
    upper_action_size = cfg.upper_agent.action_size
    lower_state_size = cfg.lower_agent.state_size
    if cfg.lower_agent.slippery:
        lower_state_size = cfg.lower_agent.state_size_slippery
    
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
    
    train(env, agent, buffer, grid, slippery, cfg.train, seed=cfg.seed)

if __name__ == "__main__":
    main()