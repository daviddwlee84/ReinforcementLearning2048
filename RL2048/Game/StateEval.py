from RL2048.Game.Game import ACTION, GRID_LENGTH
from RL2048.Game.Game import Grid
import numpy as np

# Human Heuristic
# Return weighted score by some metrics
class Metrics:

    def __init__(self, grid_obj):
        self.__grid_obj = grid_obj
        self.__grid = grid_obj.getState()

# Shared single metric

    def __Monotonicity(self, grid):
        """Monotonicity
        
        Increasing or Decreasing order of board
        """

        mono = [0]*4 # For four direction
        indices = np.where(grid != 0) # Index of all tile
        # Accending order by x
        vertical_indices = np.vstack(indices).T
        # Accending order by y
        parallel_indices = np.array(sorted(vertical_indices, key=lambda y: y[1]))

        # Left / Right
        for i, (curr_x, curr_y) in enumerate(vertical_indices[:-1]):
            (next_x, next_y) = vertical_indices[i+1]
            curr_val = grid[curr_x, curr_y]
            next_val = grid[next_x, next_y]
            if curr_x == next_x: # In the same row
                if curr_val > next_val:
                    mono[0] += curr_val - next_val # From left to right
                elif next_val > curr_val:
                    mono[1] += next_val - curr_val # From right to left

        # Up / Down
        for i, (curr_x, curr_y) in enumerate(parallel_indices[:-1]):
            (next_x, next_y) = parallel_indices[i+1]
            curr_val = grid[curr_x, curr_y]
            next_val = grid[next_x, next_y]
            if curr_y == next_y: # In the same row
                if curr_val > next_val:
                    mono[2] += curr_val - next_val # From top to down
                elif next_val > curr_val:
                    mono[3] += next_val - curr_val # From bottom to up
        return np.max(mono[0:2]) + np.max(mono[2:4]) # Select max for each direction
    
    def __Smoothness(self, grid):
        """Smoothness

        Number of possible merge.
        
        A negative metric that means how grid can merge.
        The more it can't, the smaller the value is.
        """

        smooth = 0
        indices = np.where(grid != 0)
        
        new_indices = np.vstack(indices).T
        to_ignore = [] # i.e. remove items from new_indices

        for i, (x_pos, y_pos) in enumerate(new_indices):
            to_ignore.append(i)
            foundXdir = False
            foundYdir = False
            for target_x, target_y in np.delete(new_indices, to_ignore, axis=0):
                if target_x == x_pos and not foundXdir:
                    smooth -= np.abs(grid[x_pos, y_pos] - grid[target_x, target_y])
                    foundXdir = True
                elif target_y == y_pos and not foundYdir:
                    smooth -= np.abs(grid[x_pos, y_pos] - grid[target_x, target_y])
                    foundYdir = True
        return smooth
    
    def __FreeTiles(self, grid):
        """Free Tiles
        
        Empty space of board.
        (intuition: if the board is full then it may die)
        """

        x_pos, _ = np.where(grid == 0)
        return len(x_pos)
    
    def __ZShapeMonotonicity(self, grid):
        """Z-shape Monotonicity
        
        Put Largest tile in the corner and then follow the z-shape
        """

        pass

# Combinations

    def ThreeEvalValue(self, monotonicity=1, smoothness=1, free_tiles=1):
        score = 0
        score += monotonicity * self.__Monotonicity(self.__grid)
        score += smoothness * self.__Smoothness(self.__grid)
        score += free_tiles * self.__FreeTiles(self.__grid)
        return score

    def ThreeEvalValueWithScore(self, scoreWeight=1, monotonicity=1, smoothness=1, free_tiles=1):
        score = scoreWeight * self.__grid_obj.getScore()
        score += self.ThreeEvalValue(monotonicity, smoothness, free_tiles)
        return score

# Return action to take based on strategy
class Strategy(Metrics):
    pass

if __name__ == "__main__":
    inputGrid = np.array([
        [2, 0, 0, 2],
        [4, 0, 0, 0],
        [2, 0, 0, 0],
        [8, 0, 0, 0]
    ])
    testGrid = Grid(initGrid=inputGrid, initScore=10)
    print(Metrics(testGrid).ThreeEvalValue(8, 2, 7))
    print(Metrics(testGrid).ThreeEvalValueWithScore(10, 8, 2, 7))
