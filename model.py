
from hydra.utils import instantiate
import torch
import torch.nn as nn

class QNetwork(nn.Module):
    def __init__(self, state_size, action_size, hidden_size, activation):
        super(QNetwork, self).__init__()
        self.q_head = nn.Sequential(
            nn.Linear(state_size, hidden_size),
            instantiate(activation),
            nn.Linear(hidden_size, hidden_size),
            instantiate(activation),
            nn.Linear(hidden_size, action_size)
        )

    def forward(self, state):
        Qs = self.q_head(state)
        return Qs