import re
import math
import os
import numpy as np
from collections import defaultdict
import matplotlib.pyplot as plt
from RL2048.Game.Play import Play

REPORT_PATH = 'report'

# Helper function
def findTitleGetNumList(subtitle, log):
    """Extract integer followed after a subtitle in log content dump by Game.dumpLog()
    
    Arguments:
        subtitle {string} -- Subtitle of an item
        log {string} -- String stream of log file
    
    Returns:
        list -- list contain all the integer
    """

    pattern = subtitle + r': \d+'
    candidate_strings = re.findall(pattern, log)

    resultList = []

    for string in candidate_strings:
        resultList.append(int(string.replace(subtitle + ':', '')))

    return resultList

class Performance:
    def __init__(self, trainingLog):
        self.__inputLogPath = trainingLog
        with open(trainingLog, 'r') as log:
            self.__content = log.read()
    
    def __MaxTileSuccessRate(self):
        
        MaxTiles = findTitleGetNumList('Current Max Tile', self.__content)

        rounds = len(MaxTiles)

        MaxTilesDict = defaultdict(int)

        MaxTileOfAllTime = max(MaxTiles)

        tileList = [2**x for x in range(4, int(math.log2(MaxTileOfAllTime))+1)] # skip 2, 4, 8

        # Cumulated Result
        for maxtile in MaxTiles: # max tile for each round
            for tile in tileList: # compare tile for each possible tile shown
                if tile <= maxtile:
                    MaxTilesDict[tile] += 1
                else:
                    break

        MaxTileRateDict = {}
        
        for tile, count in MaxTilesDict.items():
            MaxTileRateDict[tile] = count/rounds

        return MaxTileRateDict
    
    def __ScoreDiagram(self):
        
        Scores = findTitleGetNumList('Current Score', self.__content)
        
        rounds = len(Scores)

        x = np.linspace(1, rounds, num=rounds, dtype=np.int)

        plt.figure(figsize=(10, 6))        
        plt.plot(x, Scores)
        plt.xticks(np.linspace(1, rounds, num=min(rounds, 10)))
        plt.title('Score Diagram')
        plt.xlabel('Rounds')
        plt.ylabel('Score')

        maxScore = max(Scores)
        plt.annotate(f'Max Score ({maxScore})',
             xy=(Scores.index(maxScore), maxScore), xycoords='data',
             xytext=(-100, -30), textcoords='offset points', fontsize=10,
             arrowprops=dict(arrowstyle="->", connectionstyle="arc3, rad=-.2"))

    def __ScoreDiagramWithRandom(self):
        """Compare performance with random play"""
        
        Scores = findTitleGetNumList('Current Score', self.__content)
        
        rounds = len(Scores)

        x = np.linspace(1, rounds, num=rounds, dtype=np.int)

        plt.figure(figsize=(10, 6))        
        plt.plot(x, Scores, label='NN Score')

        # Random
        randomLog = 'random.log'
        if not os.path.isfile(randomLog):
            print(f'Playing {rounds} rounds Random games....')
            Play().random(play_round=rounds, log=randomLog) # Play random
        with open(randomLog, 'r') as log:
            content = log.read()
        randomScore = findTitleGetNumList('Current Score', content)

        plt.plot(x, randomScore, label='Random Score')

        plt.xticks(np.linspace(1, rounds, num=min(rounds, 10)))
        plt.title('Score Diagram with Random')
        plt.xlabel('Rounds')
        plt.ylabel('Score')

        maxScore = max(Scores)
        plt.annotate(f'Max Score ({maxScore})',
             xy=(Scores.index(maxScore), maxScore), xycoords='data',
             xytext=(-100, -30), textcoords='offset points', fontsize=10,
             arrowprops=dict(arrowstyle="->", connectionstyle="arc3, rad=-.2"))

        plt.legend(loc='upper left', frameon=False)

    def report(self, outputFile=None, compare=False):
        MaxTileRateDict = self.__MaxTileSuccessRate()
        if outputFile: # dump to output file
            if not os.path.isdir(REPORT_PATH):
                os.makedirs(REPORT_PATH)
            outputFile = os.path.join(REPORT_PATH, outputFile)

            # Draw diagram and save
            self.__ScoreDiagram()
            plt.savefig(os.path.join(REPORT_PATH, "ScoreDiagram.png"))
            if compare:
                self.__ScoreDiagramWithRandom()
                plt.savefig(os.path.join(REPORT_PATH, "ScoreDiagramWithRandom"))

            # Generate report
            with open(outputFile, 'w') as title:
                title.write('# Report of the 2048 Model #\n\n')
            # Success Rate of Tiles
            with open(outputFile, 'a') as result:
                result.write('## Success Rate of Tiles ##\n\n')
                result.write('Tile|Success Rate\n')
                result.write('----|------------\n')
                for tile, rate in MaxTileRateDict.items():
                    result.write('%s|%.2f%%\n' % (tile, 100*rate))
            # Score Diagrams
            with open(outputFile, 'a') as result:
                result.write('\n## Score Diagram ##\n\n')
                result.write('![Score Diagram](ScoreDiagram.png)\n')
                if compare:
                    result.write('![Score Diagram with Random](ScoreDiagramWithRandom.png)\n')
        else: # stdout
            print('=== Report of the 2048 Model ===\n')
            print('== Success Rate of Tiles ==')
            print(MaxTileRateDict)
            self.__ScoreDiagram()
            if compare:
                self.__ScoreDiagramWithRandom()
            plt.show() # Plot all diagram

if __name__ == "__main__":
    # Performance('training.log').report(compare=False)
    # Performance('training.log').report('StatisticsResult.md', compare=False)
    Performance('training_dqn.log').report(compare=False)
    Performance('training_dqn.log').report('StatisticsResult.md', compare=False)
    