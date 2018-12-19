from RL2048.Game.Game import ACTION, Grid
import numpy as np
from functools import reduce # For cumulated product

ACT_DICT = {0: ACTION.LEFT, 1: ACTION.UP, 2: ACTION.RIGHT, 3: ACTION.DOWN}

class OnehotGridPreprocessing:
    def __init__(self, grid_obj, num_one_hot_matrices=16):
        self.__grid_obj = grid_obj

        # Maximum number of encode value of grid
        # e.g. 16 => 0, 2, 4, 8, ..., 2**15
        self.__num_one_hot_matrices = num_one_hot_matrices

    def __SearchHelper(self, depth, lastGrid, grid_batch):
        for temp_action in ACT_DICT.values():
            tempGrid = lastGrid.getCopyGrid()
            tempGrid.shift(temp_action)
            grid_batch.append(tempGrid.getState())
            if depth > 1:
                self.__SearchHelper(depth-1, tempGrid, grid_batch)

    # DFS to constrct a batch of state
    def GridSearchBatch(self, depth=3):
        grid_batch = [] # Store every step status of grids
        for temp_action in ACT_DICT.values(): # Get each direction
            tempGrid = self.__grid_obj.getCopyGrid() # Copy new grid
            tempGrid.shift(temp_action) # Take action
            grid_batch.append(tempGrid.getState())
            if depth > 1:
                self.__SearchHelper(depth-1, tempGrid, grid_batch)       
        
        return np.array(grid_batch)
    
    # Encode a grid to one-hot
    def __OneHotEncodingHelper(self, grid):
        grid_onehot = np.zeros(shape=(self.__num_one_hot_matrices, 4, 4))
        for power in range(self.__num_one_hot_matrices):
            indices = np.where(grid == power) # Find tiles with the power
            new_indices = np.vstack(indices).T
            for x_pos, y_pos in new_indices:
                grid_onehot[power, x_pos, y_pos] = 1
        
        return grid_onehot

    # Return One-hot encoding GridSearchBatch
    def OneHotEncodingBatch(self, depth=2):
        search_grids = self.GridSearchBatch(depth) # Get grids
        grid_onehot_batch = []
        for grid in search_grids: # Encode for each grid
            grid_onehot_batch.append(self.__OneHotEncodingHelper(grid))
        
        return np.array(grid_onehot_batch)

    # Return a one-dimension flatten batch
    def FlattenBatch(self, depth=2, onehot=True):
        if onehot: # Return one-hot batch
            onehot_results = self.OneHotEncodingBatch(depth)
            return np.reshape(onehot_results, (1, reduce(lambda x, y: x*y, onehot_results.shape)))
        else: # Return normal batch
            search_grids = self.GridSearchBatch(depth)
            return np.reshape(search_grids, (1, reduce(lambda x, y: x*y, search_grids.shape)))

if __name__ == "__main__":
    # GridSearch Test
    inputGrid = np.array([
        [2, 1, 0, 2],
        [4, 3, 0, 0],
        [2, 0, 2, 1],
        [8, 3, 2, 1]
    ])

    testGrid = Grid(initGrid=np.copy(inputGrid), initScore=10)
    result = OnehotGridPreprocessing(testGrid).GridSearchBatch(2)

    print(len(result))
    print(result)

    onehot_result = OnehotGridPreprocessing(testGrid).OneHotEncodingBatch(2)
    
    print(np.shape(onehot_result))
    print(onehot_result)

    flatten_result = OnehotGridPreprocessing(testGrid).FlattenBatch(2, onehot=True)
    print(np.shape(flatten_result))
