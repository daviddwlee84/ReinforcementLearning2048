from RL2048.Game.Game import ACTION, GRID_LENGTH
from RL2048.Game.Game import Grid
import numpy as np


# Return weighted score by some metrics
class Metrics:
    # Human Heuristic
    def ThreeEvalValue(self, grid_obj, monotonicity=8, smoothness=2, free_tiles=7):
        def Monotonicity(grid):
            """Monotonicity
            
            Increasing or Decreasing 
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
        
        def Smoothness(grid):
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
        
        def FreeTiles(grid):
            x_pos, _ = np.where(grid == 0)
            return len(x_pos)
        
        return monotonicity * Monotonicity(grid_obj.getState()) + smoothness * Smoothness(grid_obj.getState()) + free_tiles * FreeTiles(grid_obj.getState())

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
    testGrid = Grid(initGrid=inputGrid)
    print(Metrics().ThreeEvalValue(testGrid, 1, 1, 1))
