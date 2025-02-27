from collections import deque
import random
import numpy as np

class Memory:
    def __init__(self, batch_size,maxlen):
        self.batch_size = batch_size
        self.memory = deque(maxlen=maxlen)

    def retrieve(self):
        batch = random.sample(self.memory, self.batch_size )
        state, next_state, action, reward, done = map(np.stack, zip(*batch))
        return state, next_state, action, reward, done 
    
    def retrieve_last(self,i):
        state, next_state, action, reward, done = map(np.stack, zip(*list(self.memory)[-i-1:-1]))
        return state, next_state, action, reward, done 

    def store(self, state, next_state, action, reward, done):
        self.memory.append( [state, next_state, action, reward, done] )

    def clear(self):
        self.memory = []