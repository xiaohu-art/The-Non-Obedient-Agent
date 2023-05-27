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
        return np.array([1, 0, 0, 0])

class LowerAgent(DQNAgent):
    def __init__(self):
        self.window_size = 3
        self.map_size = 8


    def get_observation(self, state, grid, slippery, message):
        loc = [state // 8, state % 8]

        grid_padding = np.pad(grid, ((1, 1), (1, 1)), 'constant', constant_values=-1)
        grid_window = grid_padding[loc[0]:loc[0]+self.window_size, loc[1]:loc[1]+self.window_size]
        
        slippery_padding = np.pad(slippery, ((1, 1), (1, 1)), 'constant', constant_values=0)
        slippery_window = slippery_padding[loc[0]:loc[0]+self.window_size, loc[1]:loc[1]+self.window_size]

        return np.concatenate((grid_window.flatten(), slippery_window.flatten(), message))

    def get_action(self, observation):
        '''
        not implemented
        '''
        return np.array([1, 0, 0, 0])