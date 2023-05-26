import numpy as np

class DQNAgent:
    def get_observation(state):
        pass

class UpperAgent(DQNAgent):
    def get_observation(state):
        col = state // 8
        row = state % 8

        return np.array([col, row, 15, 15])
    
    def get_message(observation):
        return np.array([1, 0, 0, 0])

class LowerAgent(DQNAgent):
    def get_observation(state):
        return 0
    
    def get_action(observation):
        return 0.5, 0