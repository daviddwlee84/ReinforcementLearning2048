"""
Main 2048 Game play and some execution management.
"""

import numpy as np
from enum import Enum
import time

GRID_LENGTH = 4
INITIAL_TILE_NUM = 2

class ACTION(Enum):
    LEFT = 0
    UP = 1
    RIGHT = 2
    DOWN = 3

class Grid:
    def __init__(self, initGrid=None, initScore=0):
        if initGrid is not None:
            self.__grid = initGrid
        else:
            self.__grid = np.zeros((GRID_LENGTH, GRID_LENGTH), dtype=np.int)
        self.__score = initScore

    def shift(self, action):
        isShifted = False
        scoreGain = 0
        
        # LEFT:  Original
        # UP:    Transpose
        # RIGHT: Flip
        # DOWN:  Flip and Transpose
        temp_grid = np.rot90(self.__grid, action.value)

        # Merge candidate to record whether the col of merge_candidate is merged
        merged = np.zeros((GRID_LENGTH, GRID_LENGTH), dtype=np.bool)
        # Merge to left
        for row in range(GRID_LENGTH):
            merge_candidate = -1
            for col in range(GRID_LENGTH):
                if temp_grid[row, col] == 0:
                    continue
                
                if (merge_candidate != -1 and
                   not merged[row, merge_candidate] and
                   temp_grid[row, merge_candidate] == temp_grid[row, col]):
                    isShifted = True
                    # Merge tile with merge_candidate
                    temp_grid[row, col] = 0
                    merged[row, merge_candidate] = True
                    temp_grid[row, merge_candidate] += 1
                    scoreGain += 2**temp_grid[row, merge_candidate]
                else:
                    # Move tile to the left
                    merge_candidate += 1
                    if col != merge_candidate:
                        isShifted = True
                        temp_grid[row, merge_candidate] = temp_grid[row, col]
                        temp_grid[row, col] = 0
        
        # print(np.rot90(merged, -action.value)) # To see which tile has been merged (debug)
        self.__score += scoreGain # Update score
        return isShifted

    def genNewTile(self):
        """Generate New Tile in random position"""

        x_pos, y_pos = np.where(self.__grid == 0)
        assert len(x_pos) != 0, "No space to generate new tile"
        empty_index = np.random.choice(len(x_pos))
        # 2048 will generate tile in random position based on probability
        # Normal tile    (2): 90%
        # Wild card tile (4): 10%
        value = np.random.choice([1, 2], p=[0.9, 0.1])

        self.__grid[x_pos[empty_index], y_pos[empty_index]] = value

    def isGameOver(self):
        x_pos, _ = np.where(self.__grid == 0)
        if len(x_pos) > 0:
            # There is empty space
            return False
        
        for row in range(GRID_LENGTH):
            for col in range(GRID_LENGTH):
                # There is something can be merged
                if row < GRID_LENGTH-1 and self.__grid[row, col] == self.__grid[row+1, col]:
                    return False
                if col < GRID_LENGTH-1 and self.__grid[row, col] == self.__grid[row, col+1]:
                    return False
        return True

## Get status
        
    def getState(self, original=False):
        if original:
            # In original format
            # i.e. Tiles: 2, 4, 8, 16, ...
            temp_grid = np.full((GRID_LENGTH, GRID_LENGTH), 2, dtype=np.int)
            temp_grid = temp_grid**self.__grid
            temp_grid[temp_grid==1] = 0
            return temp_grid
        else:
            # In log format
            # i.e. 1, 2, 3, 4, 5, ...
            return self.__grid

    def getCopyGrid(self):
        return Grid(initGrid=np.copy(self.__grid), initScore=self.__score)

    def getScore(self):
        return self.__score
    
    def getMaxTail(self):
        return 2**np.max(self.__grid)

class Game:
    def __init__(self, initGrid=None):
        self.__nRound = 0
        self.__lastMoveCount = 0
        self.__lastScore = 0
        self.__maxScore = 0
        self.__scoreSum = 0
        self.__startTime = time.time()
        self.__lastDuration = 0
        self.__maxTile = 0
        if initGrid:
            self.__currGrid = initGrid
        else:
            self.newGame()
        
    def __updateStatus(self):
        currScore = self.__currGrid.getScore()
        self.__lastScore = currScore
        self.__scoreSum += currScore
        self.__maxScore = max(self.__maxScore, currScore)
        self.__maxTile = max(self.__currGrid.getMaxTail(), self.__maxTile)
        self.__lastDuration = time.time() - self.__startTime

    def newGame(self):
        self.__lastMoveCount = 0
        self.__startTime = time.time()
        self.__nRound += 1

        self.gameOver = False
        self.__currGrid = Grid()
        for _ in range(INITIAL_TILE_NUM):
            self.__currGrid.genNewTile()
        
    def doAction(self, action):
        if not self.gameOver:
            isShift = self.__currGrid.shift(action)
            # If shift then it is a valid movement
            if isShift:
                self.__lastMoveCount += 1
                self.__currGrid.genNewTile()
            self.gameOver = self.__currGrid.isGameOver()
            if self.gameOver:
                self.__updateStatus()
            return isShift
        else:
            return False
    
    def getCurrentScore(self):
        return self.__currGrid.getScore()

    def printGrid(self):
        print(self.__currGrid.getState(original=True))
    
    def getCopyGrid(self):
        return self.__currGrid.getCopyGrid()

    def printStatus(self):
        print("Round:", self.__nRound)
        print("Round Time:", self.__lastDuration)
        print("Current Score:", self.__lastScore)
        print("Highest Score:", self.__maxScore)
        print("Average Score:", self.__scoreSum/self.__nRound)
        print("Max Tile:", self.__maxTile)
        print("Move Count:", self.__lastMoveCount)

    def dumpLog(self, filename):
        """Dump log to a file"""
        with open(filename, 'a') as log:
            log.write(f"\n=== Round: {self.__nRound} ===\n")
            log.write(f"Round Time: {self.__lastDuration}\n")
            log.write(f"Current Score: {self.__lastScore}\n")
            log.write(f"Highest Score: {self.__maxScore}\n")
            log.write(f"Average Score: {self.__scoreSum/self.__nRound}\n")
            log.write(f"Max Tile: {self.__maxTile}\n")
            log.write(f"Move Count: {self.__lastMoveCount}\n")
            log.write(f"Detail:\n {self.__currGrid.getState(original=True)}\n\n")
