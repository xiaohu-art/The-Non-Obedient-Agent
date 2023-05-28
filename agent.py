import numpy as np
import torch
import torch.optim as optim
from model import QNetwork
from copy import deepcopy

class DQNAgent:
    def __init__(self, state_size, action_size, cfg, device="cuda"):
        self.device = device

        self.target_update_interval = cfg.target_update_interval
        
        q_model = QNetwork
        self.q_net = q_model(state_size, action_size, cfg.hidden_size, cfg.activation).to(self.device)
        self.target_net = deepcopy(self.q_net).to(self.device)
        self.optimizer = optim.AdamW(self.q_net.parameters(), lr=cfg.lr)

        self.tau = cfg.tau
        self.gamma = cfg.gamma ** cfg.nstep

    def soft_update(self, target, source):
        for target_param, param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(target_param.data * (1.0 - self.tau) + param.data * self.tau)

    @torch.no_grad()
    def get_observation():
        pass
    
    @torch.no_grad()
    def get_action(self, state):
        state = torch.tensor(state).to(self.device)
        action = self.q_net(state).argmax().item()

        return action
    
    @torch.no_grad()
    def get_Q_target(self, state, action, reward, done, next_state) -> torch.Tensor:
        Q_target = reward + self.gamma * self.target_net(next_state).max(dim=1)[0] * (1 - done)
        return Q_target
    
    def get_Q(self, state, action) -> torch.Tensor:
        action = action.long()
        Q = self.q_net(state).gather(1, action).squeeze(1)
        return Q
    
    def update(self, batch, step, weights = None):
        state, action, reward, next_state, done = batch
        Q_target = self.get_Q_target(state, action, reward, done, next_state)
        Q = self.get_Q(state, action)

        if weights is None:
            weights = torch.ones_like(Q).to(self.device)
        td_error = torch.abs(Q - Q_target).detach()
        loss = torch.mean((Q - Q_target)**2 * weights)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        if not step % self.target_update_interval:
            with torch.no_grad():
                self.soft_update(self.target_net, self.q_net)

        return loss.item(), td_error, Q.mean().item()

class UpperAgent(DQNAgent):
    def __init__(self, state_size, action_size, cfg, device="cuda"):
        super().__init__(state_size, action_size, cfg, device)
        self.map_size = 8

    def get_observation(self, state):
        col = state // self.map_size
        row = state % self.map_size

        return np.array([col, row, self.map_size, self.map_size], dtype=np.float32)

class LowerAgent(DQNAgent):
    def __init__(self, state_size, action_size, cfg, device="cuda"):
        super().__init__(state_size, action_size, cfg, device)
        self.window_size = 3
        self.map_size = 8
        
        self.slippery = cfg.slippery

    def get_observation(self, state, grid, slippery, message):
        loc = [state // 8, state % 8]

        grid_padding = np.pad(grid, ((1, 1), (1, 1)), 'constant', constant_values=-1)
        grid_window = grid_padding[loc[0]:loc[0]+self.window_size, loc[1]:loc[1]+self.window_size]
        
        slippery_padding = np.pad(slippery, ((1, 1), (1, 1)), 'constant', constant_values=0)
        slippery_window = slippery_padding[loc[0]:loc[0]+self.window_size, loc[1]:loc[1]+self.window_size]

        if self.slippery:
            return np.concatenate((grid_window.flatten(), slippery_window.flatten(), np.array([message])), dtype=np.float32)
        else:
            return np.concatenate((grid_window.flatten(), np.array([message])), dtype=np.float32)