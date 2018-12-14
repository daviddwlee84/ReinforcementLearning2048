"""
Play 2048 game
"""

from RL2048.Game.Game import ACTION, Game
from RL2048.Game.StateEval import Metrics
import numpy as np

ACT_DICT = {0: ACTION.LEFT, 1: ACTION.UP, 2: ACTION.RIGHT, 3: ACTION.DOWN}

class Play:
    def keyboard(self):
        import sys
        import os
        import termios
        import contextlib

        @contextlib.contextmanager
        def raw_mode(file):
            old_attrs = termios.tcgetattr(file.fileno())
            new_attrs = old_attrs[:]
            new_attrs[3] = new_attrs[3] & ~(termios.ECHO | termios.ICANON)
            try:
                termios.tcsetattr(file.fileno(), termios.TCSADRAIN, new_attrs)
                yield
            finally:
                termios.tcsetattr(file.fileno(), termios.TCSADRAIN, old_attrs)

        game = Game()
        
        valid = True
        isShift = False
        with raw_mode(sys.stdin):
            try:
                while not game.gameOver:
                    if valid:
                        os.system('clear')
                        grid = game.getCopyGrid()
                        print("Score:", game.getCurrentScore())
                        print("StateEval Score:", Metrics(grid).ThreeEvalValue(10, 2, 7))
                        game.printGrid()

                    ch = sys.stdin.read(1)
                    if not ch or ch == chr(4):
                        break
                    
                    if ch in ('w', 'W'):
                        action = ACTION.UP
                    elif ch in ('s', 'S'):
                        action = ACTION.DOWN
                    elif ch in ('a', 'A'):
                        action = ACTION.LEFT
                    elif ch in ('d', 'D'):
                        action = ACTION.RIGHT
                    else:
                        print('Invalid input (use w, a, s, d)')
                        valid = False
                        continue
                    isShift = game.doAction(action)
                    if not isShift:
                        print("Invalid direction (can't shift the board)")
                    valid = isShift
                else:
                    os.system('clear')
                    print("Score:", game.getCopyGrid().getScore())
                    print("StateEval Score:", Metrics(grid).ThreeEvalValue(10, 2, 7))
                    game.printGrid()
                    print('Game over')
            except (KeyboardInterrupt, EOFError):
                pass
    def random(self, round=1):
        game = Game()
        for i in range(round):
            while not game.gameOver:
                action = ACT_DICT[np.random.randint(0, 4)]
                game.doAction(action)
            else:
                game.dumpLog('random.log')
                print('Round', i+1, 'game over')
                game.newGame()

if __name__ == "__main__":
    seleciton = eval(input('Play mode:\n1. Keyboard (use w, a, s, d, exit with ^C or ^D)\n2. Random\n\n select: '))
    if seleciton == 1:
        Play().keyboard()
    elif seleciton == 2:
        Play().random(100)
