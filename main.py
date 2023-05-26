import hydra
import torch
import utils
import random
import numpy as np
from agent import UpperAgent, LowerAgent
import gymnasium as gym

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

@hydra.main(config_path="cfgs", config_name="config", version_base="1.3")
def main(cfg):
    env = gym.make(cfg.env_name, render_mode="ansi", map_name="8x8", is_slippery=False)
    utils.set_seed_everywhere(env, cfg.seed)

    done, truncated = False, False
    state, _ = env.reset(seed = cfg.seed)

    upper = UpperAgent
    lower = LowerAgent

    for step in range(1, cfg.train.timesteps + 1):
        if done or truncated:
            state, _ = env.reset()
            done, truncated = False, False

        upper_obs = upper.get_observation(state)
        message = upper.get_message(upper_obs)

        lower_obs = lower.get_observation(state)
        belief, lower_action = lower.get_action(lower_obs)

        if random.random() < cfg.train.epsilon:
            action = env.action_space.sample()
        else:
            if random.random() < belief:
                action = lower_action
            else:
                action = torch.argmax(message)

        next_state, reward, done, truncated, info = env.step(action)
        state = next_state
        
        break

if __name__ == "__main__":
    main()