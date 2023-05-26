import hydra
import torch
import utils
import gymnasium as gym
from core import train
from agent import UpperAgent, LowerAgent

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

@hydra.main(config_path="cfgs", config_name="config", version_base="1.3")
def main(cfg):
    env = gym.make(cfg.env_name, render_mode="ansi", map_name="8x8", is_slippery=False)
    env.reset(seed = cfg.seed)
    map = env.render().split("\n")

    utils.set_seed_everywhere(env, cfg.seed)
    grid = utils.get_grid(cfg.env_size, map[1:])

    state_size = utils.get_space_shape(env.observation_space)
    action_size = utils.get_space_shape(env.action_space)

    buffer = None
    if cfg.multi_agent == True:
        buffer = None
    else:
        upper_buffer = None
        lower_buffer = None
        buffer = [upper_buffer, lower_buffer]

    uagent = UpperAgent()
    lagent = LowerAgent()
    
    train(env, uagent, lagent, buffer, grid, cfg.train, seed=cfg.seed)

if __name__ == "__main__":
    main()