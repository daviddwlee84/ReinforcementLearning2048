import numpy as np
import random
import copy
import pickle

BATCH_SIZE = 40
MEM_PATH = 'memory.pickle'

def get_empty_experience():
    EMPTY_EXPERIENCE = {'state_before': [], 'action': [], 'reward': [], 'state_after': [], 'isAlive': []}
    return copy.deepcopy(EMPTY_EXPERIENCE)

class Memory:
    def __init__(self, memory_size=8000):
        self.__good_memory = get_empty_experience()
        self.__bad_memory = get_empty_experience()
        self.__memory_size = memory_size
        # FIFO cache
        self.__goodUpdateIndex = 0
        self.__badUpdateIndex = 0

    def append(self, state_before, action, reward, state_after, isAlive):
        if isAlive:
            for memory, input_value in zip(self.__good_memory.values(), [state_before, action, reward, state_after, isAlive]):
                if self.__goodUpdateIndex < self.__memory_size:
                    memory.append(input_value)
                else:
                    memory[self.__goodUpdateIndex % self.__memory_size] = input_value
            self.__goodUpdateIndex += 1
        else: # Gameover memory
            for memory, input_value in zip(self.__bad_memory.values(), [state_before, action, reward, state_after, isAlive]):
                if self.__badUpdateIndex < self.__memory_size:
                    memory.append(input_value)
                else:
                    memory[self.__badUpdateIndex % self.__memory_size] = input_value
            self.__badUpdateIndex += 1

    def sample(self, prob=1, batch_size=BATCH_SIZE, goodSample=True):
        """Get a batch of sample with same memory"""
        experience = get_empty_experience()
        index_list = [i for i in range(self.getMemoryNum(good=goodSample))]
        sample_index_list = random.sample(index_list, round(prob*batch_size))
        if goodSample:
            for key, value in self.__good_memory.items():
                experience[key] = [value[i] for i in sample_index_list]
            return experience
        else:
            for key, value in self.__bad_memory.items():
                experience[key] = [value[i] for i in sample_index_list]
            return experience
    
    def getSampleBatch(self, goodProb=0.8, batch_size=BATCH_SIZE):
        """Get a batch of sample contain both good and bad memory"""
        experienceBatch = get_empty_experience()
        good_exp = self.sample(goodProb, batch_size, goodSample=True)
        bad_exp = self.sample(1-goodProb, batch_size, goodSample=False)
        index_list = [i for i in range(batch_size)]
        random.shuffle(index_list)        
        for key, value in experienceBatch.items():
            temp = []
            temp.extend(good_exp[key])
            temp.extend(bad_exp[key])
            value.extend([temp[i] for i in index_list])

        return experienceBatch

    def getMemoryNum(self, good=False):
        """Get current memory number"""
        if good:
            return min(self.__goodUpdateIndex, self.__memory_size)
        else:
            return min(self.__badUpdateIndex, self.__memory_size)

    def getmemory(self):
        return self.__good_memory, self.__bad_memory

def saveMemory(memory, filename=MEM_PATH):
    """Save memory object as pickle format"""
    with open(filename, 'wb') as mem_pickle:
        pickle.dump(memory, mem_pickle)

def loadMemory(filename=MEM_PATH):
    """Load pickle memory object"""
    with open(filename, 'rb') as mem_pickle:
        return pickle.load(mem_pickle)

if __name__ == "__main__":
    # Basic test
    mem = Memory(2)
    mem.append(1, 2, 3, 4, True)
    mem.append(6, 7, 8, 9, False)
    mem.append(1, 3, 5, 7, True)
    print(mem.sample(batch_size=1, goodSample=True))
    print(mem.sample(batch_size=1, goodSample=False))

    print(mem.getSampleBatch(goodProb=0.4, batch_size=2))

    # Test save and load
    saveMemory(mem)
    load_mem = loadMemory()
    print('Loaded memory', load_mem.getSampleBatch(goodProb=1, batch_size=2))


    # Test
    from RL2048.Game.Game import Game, ACTION
    from RL2048.Game.StateEval import Strategy

    memory = Memory(memory_size=1000)
    BAD_SAMPLE_PROB = 0.2
    game = Game()
    reward = 0.0
    penalty = 0.0
    ACT_DICT = {0: ACTION.LEFT, 1: ACTION.UP, 2: ACTION.RIGHT, 3: ACTION.DOWN}
    def move_and_get_reward(last_reward, game_obj, cumulate_penalty, forceAction):
        isShift = game_obj.doAction(forceAction)
        
        reward = -last_reward
        reward += game_obj.getCurrentScore()

        if not isShift:
            penalty = reward+1
        else:
            penalty = 0.0
        return reward-penalty, penalty

    move = 0
    rounds = 0
    while memory.getMemoryNum(good=False) < BATCH_SIZE * BAD_SAMPLE_PROB: # Get enough bad memory
        state_t = game.getCopyGrid().getState()
        action = Strategy(game.getCopyGrid()).RandomValidMove()
        reward, penalty = move_and_get_reward(reward, game, penalty, action)
        state_t_1 = game.getCopyGrid().getState()
        isAlive = not game.gameOver
        memory.append(state_t, action.value, reward, state_t_1, isAlive)
        if not isAlive:
            rounds += 1
            game.printGrid()
            print("Played", rounds, "rounds and", move, "moves.")
            game.newGame()
        else:
            move += 1

    print(memory.getMemoryNum(good=True))
    print(memory.getMemoryNum(good=False))
    print(memory.getSampleBatch())
