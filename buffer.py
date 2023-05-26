import torch
import numpy as np
from starr import SumTreeArray
from collections import deque


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
    def __init__(self, capacity, state_size, device):
        self.device = device
        self.state = torch.empty(capacity, state_size, dtype=torch.float)
        self.action = torch.empty(capacity, 1, dtype=torch.float)
        self.reward = torch.empty(capacity, dtype=torch.float)
        self.next_state = torch.empty(capacity, state_size, dtype=torch.float)
        self.done = torch.empty(capacity, dtype=torch.int)

        self.idx = 0
        self.size = 0
        self.capacity = capacity

    def __repr__(self) -> str:
        return 'NormalReplayBuffer'

    def add(self, transition):
        state, action, reward, next_state, done = transition

        # store transition in the buffer and update the index and size of the buffer
        # you may need to convert the data type to torch.tensor
        
        self.state[self.idx] = torch.tensor(state)
        self.action[self.idx] = torch.tensor(action)
        self.reward[self.idx] = torch.tensor(reward)
        self.next_state[self.idx] = torch.tensor(next_state)
        self.done[self.idx] = torch.tensor(done)

        self.idx = (self.idx + 1) % self.capacity
        self.size = self.size + 1 if self.size < self.capacity else self.capacity

    def sample(self, batch_size):
        # sample batch_size data from the buffer without replacement
        sample_idxs = np.random.choice(self.size, batch_size, replace=False)
        batch = ()
        # get a batch of data from the buffer according to the sample_idxs
        # please transfer the data to the corresponding device before return
      
        state = self.state[sample_idxs].to(self.device)
        action = self.action[sample_idxs].to(self.device)
        reward = self.reward[sample_idxs].to(self.device)
        next_state = self.next_state[sample_idxs].to(self.device)
        done = self.done[sample_idxs].to(self.device)
        batch = (state, action, reward, next_state, done)

        return batch


class NStepReplayBuffer(ReplayBuffer):
    def __init__(self, capacity, n_step, gamma, state_size, device):
        super().__init__(capacity, state_size, device=device)
        self.n_step = n_step
        self.n_step_buffer = deque([], maxlen=n_step)
        self.gamma = gamma

    def __repr__(self) -> str:
        return f'{self.n_step}StepReplayBuffer'

    def n_step_handler(self):
        """Get n-step state, action, reward and done for the transition, discard those rewards after done=True"""
        
        state = self.n_step_buffer[0][0]
        action = self.n_step_buffer[0][1]
        reward = 0

        for idx, transition in enumerate(self.n_step_buffer):
            reward += transition[2] * self.gamma ** idx

        done = self.n_step_buffer[-1][3]
        
        return state, action, reward, done

    def add(self, transition):
        state, action, reward, next_state, done = transition
        self.n_step_buffer.append((state, action, reward, done))
        if len(self.n_step_buffer) < self.n_step:
            return
        state, action, reward, done = self.n_step_handler()
        super().add((state, action, reward, next_state, done))


class PrioritizedReplayBuffer_bak(ReplayBuffer):
    def __init__(self, capacity, eps, alpha, beta, state_size, device):
        self.weights = np.zeros(capacity, dtype=np.float32) # stores weights for importance sampling
        self.eps = eps  # minimal priority for stability
        self.alpha = alpha  # determines how much prioritization is used, α = 0 corresponding to the uniform case
        self.beta = beta  # determines the amount of importance-sampling correction, b = 1 fully compensate for the non-uniform probabilities
        self.max_priority = eps  # priority for new samples, init as eps
        super().__init__(capacity, state_size, device=device)

    def add(self, transition):
        """
        Add a new experience to memory, and update it's priority to the max_priority.
        """
        state, action, reward, next_state, done = transition
        self.state[self.idx] = torch.tensor(state)
        self.action[self.idx] = torch.tensor(action)
        self.reward[self.idx] = torch.tensor(reward)
        self.next_state[self.idx] = torch.tensor(next_state)
        self.done[self.idx] = torch.tensor(done)

        self.weights[self.idx] = self.max_priority

        self.idx = (self.idx + 1) % self.capacity
        self.size = self.size + 1 if self.size < self.capacity else self.capacity


    def sample(self, batch_size):
        """
        Sample a batch of experiences from the buffer with priority, and calculates the weights used for the correction of bias used in the Q-learning update
        Returns:
            batch: a batch of experiences as in the normal replay buffer
            weights: torch.Tensor (batch_size, ), importance sampling weights for each sample
            sample_idxs: numpy.ndarray (batch_size, ), the indexes of the sample in the buffer
        """
        
        sample_idxs = np.random.choice(self.capacity, batch_size, replace=False, p=self.weights/self.weights.sum())
        batch = ()
        state = self.state[sample_idxs].to(self.device)
        action = self.action[sample_idxs].to(self.device)
        reward = self.reward[sample_idxs].to(self.device)
        next_state = self.next_state[sample_idxs].to(self.device)
        done = self.done[sample_idxs].to(self.device)
        batch = (state, action, reward, next_state, done)

        weights = (self.size * self.weights[sample_idxs] / self.weights.sum()) ** (-self.beta)
        weights = torch.tensor(weights / weights.max(), dtype=torch.float32).to(self.device)
        
        return batch, weights, sample_idxs

    def update_priorities(self, data_idxs, priorities: np.ndarray):
        priorities = (priorities + self.eps) ** self.alpha
        self.weights[data_idxs] = priorities
        self.max_priority = max(self.weights)

    def __repr__(self) -> str:
        return 'PrioritizedReplayBuffer'

class PrioritizedReplayBuffer(ReplayBuffer):
    def __init__(self, capacity, eps, alpha, beta, state_size, device):
        super().__init__(capacity, state_size, device=device)
        self.weights = np.zeros(capacity, dtype=np.float32) # stores weights for importance sampling
        self.tree = SumTreeArray(capacity, dtype='float32')
        self.eps = eps  # minimal priority for stability
        self.alpha = alpha  # determines how much prioritization is used, α = 0 corresponding to the uniform case
        self.beta = beta  # determines the amount of importance-sampling correction, b = 1 fully compensate for the non-uniform probabilities
        self.max_priority = eps  # priority for new samples, init as eps
        

    def add(self, transition):
        """
        Add a new experience to memory, and update it's priority to the max_priority.
        """
        state, action, reward, next_state, done = transition
        self.state[self.idx] = torch.tensor(state)
        self.action[self.idx] = torch.tensor(action)
        self.reward[self.idx] = torch.tensor(reward)
        self.next_state[self.idx] = torch.tensor(next_state)
        self.done[self.idx] = torch.tensor(done)

        self.weights[self.idx] = self.max_priority

        self.idx = (self.idx + 1) % self.capacity
        self.size = self.size + 1 if self.size < self.capacity else self.capacity


    def sample(self, batch_size):
        """
        Sample a batch of experiences from the buffer with priority, and calculates the weights used for the correction of bias used in the Q-learning update
        Returns:
            batch: a batch of experiences as in the normal replay buffer
            weights: torch.Tensor (batch_size, ), importance sampling weights for each sample
            sample_idxs: numpy.ndarray (batch_size, ), the indexes of the sample in the buffer
        """
        
        p = self.tree/ self.tree.sum()
        sample_idxs = np.random.choice(self.capacity, batch_size, replace=False, p=p)
        batch = ()
        state = self.state[sample_idxs].to(self.device)
        action = self.action[sample_idxs].to(self.device)
        reward = self.reward[sample_idxs].to(self.device)
        next_state = self.next_state[sample_idxs].to(self.device)
        done = self.done[sample_idxs].to(self.device)
        batch = (state, action, reward, next_state, done)

        weights = (self.size * self.weights[sample_idxs] / self.weights.sum()) ** (-self.beta)
        weights = torch.tensor(weights / weights.max(), dtype=torch.float32).to(self.device)
        
        return batch, weights, sample_idxs

    def update_priorities(self, data_idxs, priorities: np.ndarray):
        priorities = (priorities + self.eps) ** self.alpha
        self.weights[data_idxs] = priorities
        self.max_priority = max(self.weights)

    def __repr__(self) -> str:
        return 'PrioritizedReplayBuffer'

# Avoid Diamond Inheritance
class PrioritizedNStepReplayBuffer_bak(NStepReplayBuffer):
    def __init__(self, capacity, eps, alpha, beta, n_step, gamma, state_size, device):
        
        super().__init__(capacity, n_step, gamma, state_size, device)
        self.eps = eps  # minimal priority for stability
        self.alpha = alpha  # determines how much prioritization is used, α = 0 corresponding to the uniform case
        self.beta = beta  # determines the amount of importance-sampling correction, b = 1 fully compensate for the non-uniform probabilities
        self.max_priority = eps  # priority for new samples, init as eps
        self.weights = np.zeros(capacity, dtype=np.float32) # stores weights for importance sampling
        
    def __repr__(self) -> str:
        return f'Prioritized{self.n_step}StepReplayBuffer'

    def add(self, transition):
        
        state, action, reward, next_state, done = transition
        super().add((state, action, reward, next_state, done))
        self.weights[self.idx] = self.max_priority


    # def the other necessary class methods as your need
    def sample(self, batch_size):
        sample_idxs = np.random.choice(self.capacity, batch_size, replace=False, p=self.weights/self.weights.sum())
        batch = ()
        state = self.state[sample_idxs].to(self.device)
        action = self.action[sample_idxs].to(self.device)
        reward = self.reward[sample_idxs].to(self.device)
        next_state = self.next_state[sample_idxs].to(self.device)
        done = self.done[sample_idxs].to(self.device)
        batch = (state, action, reward, next_state, done)

        weights = (self.size * self.weights[sample_idxs] / self.weights.sum()) ** (-self.beta)
        weights = torch.tensor(weights / weights.max(), dtype=torch.float32).to(self.device)
        
        return batch, weights, sample_idxs
    
    def update_priorities(self, data_idxs, priorities: np.ndarray):
        priorities = (priorities + self.eps) ** self.alpha
        self.weights[data_idxs] = priorities
        self.max_priority = max(self.weights)

class PrioritizedNStepReplayBuffer(PrioritizedReplayBuffer):
    def __init__(self, capacity, eps, alpha, beta, n_step, gamma, state_size, device):
        
        super().__init__(capacity, eps, alpha, beta, state_size, device)
        
        self.gamma = gamma
        self.n_step = n_step
        self.n_step_buffer = deque([], maxlen=n_step)
        
    def __repr__(self) -> str:
        return f'Prioritized{self.n_step}StepReplayBuffer'

    def add(self, transition):
        
        state, action, reward, next_state, done = transition
        self.n_step_buffer.append((state, action, reward, done))
        if len(self.n_step_buffer) < self.n_step:
            return
        state, action, reward, done = self.n_step_handler()
        self.state[self.idx] = torch.tensor(state)
        self.action[self.idx] = torch.tensor(action)
        self.reward[self.idx] = torch.tensor(reward)
        self.next_state[self.idx] = torch.tensor(next_state)
        self.done[self.idx] = torch.tensor(done)
        self.weights[self.idx] = self.max_priority

        self.idx = (self.idx + 1) % self.capacity
        self.size = self.size + 1 if self.size < self.capacity else self.capacity

    # def the other necessary class methods as your need
    def n_step_handler(self):
        state = self.n_step_buffer[0][0]
        action = self.n_step_buffer[0][1]
        reward = 0

        for idx, transition in enumerate(self.n_step_buffer):
            reward += transition[2] * self.gamma ** idx

        done = self.n_step_buffer[-1][3]
        
        return state, action, reward, done