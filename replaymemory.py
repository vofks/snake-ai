from random import sample
from collections import deque
from transition import Transition


class ReplayMemory:
    def __init__(self, capacity):
        self._memory = deque([], maxlen=capacity)

    def push(self, *args):
        self._memory.append(Transition(*args))

    def sample(self, batch_size):
        return sample(self._memory, batch_size)

    def __len__(self):
        return len(self._memory)
