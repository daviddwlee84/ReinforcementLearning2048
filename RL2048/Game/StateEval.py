from RL2048.Game.Game import ACTION, GRID_LENGTH, Grid
# import RL2048.Learning.forward as forward
import numpy as np
import operator

MODEL_SAVE_PATH = "./model/"

ACT_DICT = {0: ACTION.LEFT, 1: ACTION.UP, 2: ACTION.RIGHT, 3: ACTION.DOWN}

# Human Heuristic
# Return weighted score by some metrics
class Metrics:

    def __init__(self, grid_obj):
        self.__grid_obj = grid_obj
        self.__grid = grid_obj.getState() # This is in-place np.array

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
    
    def __FreeTiles(self, grid, log=False):
        """Free Tiles
        
        Empty space of board.
        (intuition: if the board is full then it may die)
        """

        x_pos, _ = np.where(grid == 0)
        return len(x_pos)
    
    def __ZShapeMonotonicity(self, grid, weightMatrixApproach=False):
        """Z-shape Monotonicity
        
        Put Largest tile in the corner and then follow the z-shape
        """

        if weightMatrixApproach:

            maxScore = 0

            # weightMat = np.array([
            #     [100, 90, 70, 50],
            #     [8, 10, 20, 30],
            #     [7, 6, 5, 4],
            #     [0, 1, 2, 3]
            # ])

            weightMat = np.array([
                [15, 14, 13, 12],
                [8, 9, 10, 11],
                [7, 6, 5, 4],
                [0, 1, 2, 3]
            ])

            weightMat = 2**weightMat

            for direction in range(4):
                mat = np.rot90(weightMat, direction)
                maxScore = max(np.sum(np.dot(mat, grid)), maxScore)
            weightMat = np.transpose(weightMat)
            for direction in range(4):
                mat = np.rot90(weightMat, direction)
                maxScore = max(np.sum(np.dot(mat, grid)), maxScore)
            return maxScore

        else:

            maxScore = 0

            def indexToRowCol(index):
                row = index // grid.shape[0]
                idx = index % grid.shape[0]
                if row % 2 == 0: # row 0, 2
                    col = idx
                else: # row 1, 3
                    col = grid.shape[0] - idx - 1
                return row, col

            def calculateScore(tempGrid):
                """
                ---->
                    |
                <----
                """
                tempScore = 0
                for i in range(grid.size-1):
                    thisRow, thisCol = indexToRowCol(i)
                    nextRow, nextCol = indexToRowCol(i+1)

                    if tempGrid[thisRow, thisCol] > tempGrid[nextRow, nextCol]:
                        tempScore += grid.size-i
                    else:
                        break

                return tempScore

            for direction in range(4):
                # Search twice for each corner
                
                tempGrid = np.copy(grid)
                tempGrid = np.rot90(tempGrid, direction)
                tempScore = calculateScore(tempGrid)
                maxScore = max(maxScore, tempScore)

                tempGrid = tempGrid.T
                tempScore = calculateScore(tempGrid)
                maxScore = max(maxScore, tempScore)
                
            return maxScore

    def __MaxTile(self, grid, normalize=True):
        """Max Tile
        """
        if normalize:
            return np.max(grid)
        else:
            return self.__grid_obj.getMaxTail()

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

    def NewEvalValue(self, monotonicity=1, smoothness=1, free_tiles=1, zshape=1):
        score = 0
        score += self.ThreeEvalValue(monotonicity, smoothness, free_tiles)
        score += zshape * self.__ZShapeMonotonicity(self.__grid)
        return score

# Return action to take based on strategy
class Strategy:

    def __init__(self, grid_obj):
        self.__grid_obj = grid_obj
        self.__grid = grid_obj.getState()

    def RandomValidMove(self):
        isShift = False
        while not isShift:
            action = ACT_DICT[np.random.randint(0, 4)]
            isShift = self.__grid_obj.shift(action)
        return action # Return only valid action

    # There is bug now (can't predict multiple action)
    # def PolicyGradientModel(self, ckpt_path=MODEL_SAVE_PATH):
    #     import tensorflow as tf
    #     state_batch_placeholder = tf.placeholder(tf.float32, shape=[None, forward.INPUT_NODE], name="state_batch")
    #     _, action_prob = forward.forward(state_batch_placeholder)
    #     get_action_num_op = tf.argmax(action_prob, 1)
    #     saver = tf.train.Saver()
    #     with tf.Session() as sess:
    #         ckpt = tf.train.latest_checkpoint(ckpt_path)
    #         if ckpt:
    #             saver.restore(sess, ckpt)
    #         else:
    #             raise ValueError('ckpt not found')
    #         state_batch = np.reshape(self.__grid, (1, forward.INPUT_NODE))
    #         actionNum = sess.run([get_action_num_op],
    #                     feed_dict={state_batch_placeholder: state_batch})
            
    #     return ACT_DICT[np.max(actionNum)] # Extract scalar from np array

    # A recursive call helper
    # Keep search and update reward
    def __MCTSDFShelper(self, depth, lastGrid, rewardDict, subDir):
        # print(depth, rewardDict, subDir)
        # print(lastGrid.getState())

        for temp_action in ACT_DICT.values():
            tempGrid = lastGrid.getCopyGrid()
            isShift = tempGrid.shift(temp_action)
            reward = Metrics(tempGrid).ThreeEvalValueWithScore(10, 10, 2, 7) * isShift
            rewardDict[subDir] = max(rewardDict[subDir], reward) # Update the maximum reward
            if depth > 1 and isShift:
                self.__MCTSDFShelper(depth-1, tempGrid, rewardDict, subDir)

    # Basic version of MCTS using DFS
    def MCTSDFS(self, depth=3):
        rewardDict = {} # Store the maximum reward for each possible direction
        for temp_action in ACT_DICT.values(): # Get each direction
            tempGrid = self.__grid_obj.getCopyGrid() # Copy new grid
            isShift = tempGrid.shift(temp_action) # Take action
            reward = Metrics(tempGrid).ThreeEvalValueWithScore(10, 10, 2, 7) * isShift # if not isShift -> 0
            rewardDict[temp_action] = reward
            if depth > 1 and isShift: # Skip invalid search
                self.__MCTSDFShelper(depth-1, tempGrid, rewardDict, temp_action)
        
        return max(rewardDict.items(), key=operator.itemgetter(1))[0]

    # Monte Carlo Tree Search with Minimax
    def MCTSMinimax(self, depth):
        pass
    
    # Monte Carlo Tree Seach with Alpha-beta pruning (using DFS)
    def MCTSAlphaBeta(self, depth, alpha, beta):
        bestReward = 0
        bestAction = None
        # Alpha: Player turn

        # Beta: Computer turn, do heavy pruning to keep the branching factor low.
        # (try a 2 and 4 in each cell and measure how annoying it is with metrics)

        # Pick out the most annoying moves
        
        # Search on each candidate
        pass

if __name__ == "__main__":
    inputGrid = np.array([
        [2, 0, 0, 2],
        [4, 0, 0, 0],
        [2, 0, 0, 0],
        [8, 0, 0, 0]
    ])
    testGrid = Grid(initGrid=np.copy(inputGrid), initScore=10)
    print(Metrics(testGrid).ThreeEvalValue(8, 2, 7))
    print(Metrics(testGrid).ThreeEvalValueWithScore(10, 8, 2, 7))
    print(Metrics(testGrid).NewEvalValue(1, 1, 1, 1))

    # Random test
    print(Strategy(testGrid).RandomValidMove())

    # Network restore test
    # print(Strategy(testGrid).PolicyGradientModel(MODEL_SAVE_PATH))
    # print(Strategy(testGrid).PolicyGradientModel(MODEL_SAVE_PATH))

    # MCTS DFS test
    inputGrid = np.array([
        [2, 1, 0, 2],
        [4, 3, 0, 0],
        [2, 0, 2, 1],
        [8, 3, 2, 1]
    ])

    testGrid = Grid(initGrid=np.copy(inputGrid), initScore=10)
    print(Strategy(testGrid).MCTSDFS(depth=3))
