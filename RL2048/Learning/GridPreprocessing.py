from RL2048.Game.Game import ACTION, Grid
import numpy as np

ACT_DICT = {0: ACTION.LEFT, 1: ACTION.UP, 2: ACTION.RIGHT, 3: ACTION.DOWN}

class OnehotGridPreprocessing:
    def __init__(self, grid_obj):
        self.__grid_obj = grid_obj

    def __SearchHelper(self, depth, lastGrid, grid_batch):
        for temp_action in ACT_DICT.values():
            tempGrid = lastGrid.getCopyGrid()
            tempGrid.shift(temp_action)
            grid_batch.append(tempGrid.getState())
            if depth > 1:
                self.__SearchHelper(depth-1, tempGrid, grid_batch)

    # DFS to constrct a batch of state
    def getGridSearchBatch(self, depth=3):
        grid_batch = [] # Store every step status of grids
        for temp_action in ACT_DICT.values(): # Get each direction
            tempGrid = self.__grid_obj.getCopyGrid() # Copy new grid
            tempGrid.shift(temp_action) # Take action
            grid_batch.append(tempGrid.getState())
            if depth > 1:
                self.__SearchHelper(depth-1, tempGrid, grid_batch)
        
        return np.array(grid_batch)

if __name__ == "__main__":
    # GridSearch Test
    inputGrid = np.array([
        [2, 1, 0, 2],
        [4, 3, 0, 0],
        [2, 0, 2, 1],
        [8, 3, 2, 1]
    ])

    testGrid = Grid(initGrid=np.copy(inputGrid), initScore=10)
    result = OnehotGridPreprocessing(testGrid).getGridSearchBatch(2)
    print(len(result))
    print(result)
