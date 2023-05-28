import torch
import numpy as np

def get_buffer(cfg, **args):
    assert type(cfg.nstep) == int and cfg.nstep > 0, 'nstep must be a positive integer'
    if not cfg.use_per:
        if cfg.nstep == 1:
            return ReplayBuffer(cfg.capacity, **args)
        else:
            return NStepReplayBuffer(cfg.capacity, cfg.nstep, cfg.gamma, **args)
    else:
        if cfg.nstep == 1:
            return PrioritizedReplayBuffer(cfg.capacity, cfg.per_eps, cfg.per_alpha, cfg.per_beta, **args)
        else:
            return PrioritizedNStepReplayBuffer(cfg.capacity, cfg.per_eps, cfg.per_alpha, cfg.per_beta, cfg.nstep, cfg.gamma, **args)

class ReplayBuffer:
    def __init__(self, capacity, state_size, action_size, device):
        self.device = device
        self.state = torch.zeros((capacity, state_size), dtype=torch.float32)
        self.action = torch.zeros((capacity, action_size), dtype=torch.float32)
        self.reward = torch.zeros(capacity, dtype=torch.float32)
        self.next_state = torch.zeros((capacity, state_size), dtype=torch.float32)
        self.done = torch.zeros(capacity, dtype=torch.int)

        self.idx = 0
        self.size = 0
        self.capacity = capacity

    def __repr__(self) -> str:
        return "NormalReplayBuffer"
    
    def add(self, transition):
        state, action, reward, next_state, done = transition

        self.state[self.idx] = torch.tensor(state, dtype=torch.float32)
        self.action[self.idx] = torch.tensor(action, dtype=torch.float32)
        self.reward[self.idx] = torch.tensor(reward, dtype=torch.float32)
        self.next_state[self.idx] = torch.tensor(next_state, dtype=torch.float32)
        self.done[self.idx] = torch.tensor(done, dtype=torch.int)

        self.idx = (self.idx + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size):
        sample_idx = np.random.choice(self.size, batch_size, replace=False)
        batch = ()

        state = self.state[sample_idx].to(self.device)
        action = self.action[sample_idx].to(self.device)
        reward = self.reward[sample_idx].to(self.device)
        next_state = self.next_state[sample_idx].to(self.device)
        done = self.done[sample_idx].to(self.device)
        batch = (state, action, reward, next_state, done)

        return batch


class NStepReplayBuffer(ReplayBuffer):
    pass

class PrioritizedReplayBuffer(ReplayBuffer):
    pass

class PrioritizedNStepReplayBuffer(NStepReplayBuffer):
    pass