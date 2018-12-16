import re
import math
from collections import defaultdict

class Performance:
    def __init__(self, trainingLog):
        self.__inputLogPath = trainingLog
        with open(trainingLog, 'r') as log:
            self.__content = log.read()
    
    def __MaxTileSuccessRate(self):
        CurrentMaxTileStrings = re.findall(r'Current Max Tile: \d+', self.__content)

        MaxTiles = []

        for CMTString in CurrentMaxTileStrings:
            MaxTiles.append(int(CMTString.replace('Current Max Tile: ', '')))

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
    
    def report(self, outputFile=None):
        MaxTileRateDict = self.__MaxTileSuccessRate()
        if outputFile: # dump to output file
            with open(outputFile, 'w') as title:
                title.write('# Report of the 2048 Model #\n\n')
            with open(outputFile, 'a') as result:
                result.write('## Success Rate of Tiles ##\n\n')
                result.write('Tile|Success Rate\n')
                result.write('----|------------\n')
                for tile, rate in MaxTileRateDict.items():
                    result.write(f'{tile}|{100*rate}%\n')
        else: # stdout
            print('=== Report of the 2048 Model ===\n')
            print('== Success Rate of Tiles ==')
            print(MaxTileRateDict)
            
            
if __name__ == "__main__":
    Performance('training.log').report()
    Performance('training.log').report('StatisticsResult.md')
