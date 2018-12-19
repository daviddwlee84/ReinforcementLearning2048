import numpy as np
import random
from RL2048.Learning.forward import BATCH_NUM
import copy

def get_empty_experience():
    EMPTY_EXPERIENCE = {'state_before': [], 'action': [], 'reward': [], 'state_after': [], 'isEnd': []}
    return copy.deepcopy(EMPTY_EXPERIENCE)

class Memory:
    def __init__(self, memory_size=8000):
        self.__good_memory = get_empty_experience()
        self.__bad_memory = get_empty_experience()
        self.__memory_size = memory_size
        # FIFO cache
        self.__goodUpdateIndex = 0
        self.__badUpdateIndex = 0

    def append(self, state_before, action, reward, state_after, isEnd):
        if not isEnd:
            for memory, input_value in zip(self.__good_memory.values(), [state_before, action, reward, state_after, isEnd]):
                if self.__goodUpdateIndex < self.__memory_size:
                    memory.append(input_value)
                else:
                    print(self.__goodUpdateIndex % self.__memory_size)
                    memory[self.__goodUpdateIndex % self.__memory_size] = input_value
            self.__goodUpdateIndex += 1
        else: # Gameover memory
            for memory, input_value in zip(self.__bad_memory.values(), [state_before, action, reward, state_after, isEnd]):
                if self.__badUpdateIndex < self.__memory_size:
                    memory.append(input_value)
                else:
                    print(self.__badUpdateIndex % self.__memory_size)
                    memory[self.__badUpdateIndex % self.__memory_size] = input_value
            self.__badUpdateIndex += 1

    def sample(self, prob=1, batch_size=BATCH_NUM, goodSample=True):
        experience = get_empty_experience()
        if goodSample:
            for key, value in self.__good_memory.items():
                experience[key] = random.sample(value, round(prob*batch_size))
            return experience
        else:
            for key, value in self.__bad_memory.items():
                experience[key] = random.sample(value, round(prob*batch_size))
            return experience
    
    def getSampleBatch(self, goodProb=0.8, batch_size=BATCH_NUM):
        experienceBatch = get_empty_experience()
        good_exp = self.sample(goodProb, batch_size, goodSample=True)
        bad_exp = self.sample(1-goodProb, batch_size, goodSample=False)
        for key, value in experienceBatch.items():
            value.extend(good_exp[key])
            value.extend(bad_exp[key])
            random.shuffle(value)

        return experienceBatch

if __name__ == "__main__":
    mem = Memory(2)
    mem.append(1, 2, 3, 4, True)
    mem.append(6, 7, 8, 9, False)
    mem.append(1, 3, 5, 7, True)
    print(mem.sample(batch_size=1, goodSample=True))
    print(mem.sample(batch_size=1, goodSample=False))

    print(mem.getSampleBatch(goodProb=0.4, batch_size=2))
