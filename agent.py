import numpy as np

class DQNAgent:
    pass

class UpperAgent(DQNAgent):
    def __init__(self):
        pass

    def get_observation(self, state):
        col = state // 8
        row = state % 8

        return np.array([col, row, 15, 15])
    
    def get_message(self, observation):
        '''
        not implemented
        '''
        return np.argmax(np.array([1, 0, 0, 0]))

class LowerAgent(DQNAgent):
    def __init__(self):
        self.window_size = 3
        self.map_size = 8


    def get_observation(self, grid, state):
        loc = [state // 8, state % 8]

        padding = np.pad(grid, ((1, 1), (1, 1)), 'constant', constant_values=-1)
        window = padding[loc[0]:loc[0]+self.window_size, loc[1]:loc[1]+self.window_size]
        
        return window.flatten()

    def get_action(self, observation):
        '''
        not implemented
        '''
        return 0.5, 0