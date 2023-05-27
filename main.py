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

    state_size = utils.get_space_shape(env.observation_space)
    action_size = utils.get_space_shape(env.action_space)

    upper_buffer = get_buffer(cfg.upper_buffer, device=device)
    lower_buffer = get_buffer(cfg.lower_buffer, device=device)

    buffer = [upper_buffer, lower_buffer]
    uagent = UpperAgent()
    lagent = LowerAgent()
    
    train(env, uagent, lagent, buffer, grid, slippery, cfg.train, seed=cfg.seed)

if __name__ == "__main__":
    main()