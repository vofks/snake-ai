import torch
import math
import random
import numpy as np
from replaymemory import ReplayMemory
from trainer import Trainer
from torch.cuda import is_available

_DEFAULT_DEVICE = 'cuda' if is_available() else 'cpu'


class Agent:
    def __init__(self, model, env, memory_size, min_replay_size, batch_size, lr, epsilon_start, epsilon_end, epsilon_decay,  device=_DEFAULT_DEVICE):
        print(f'Running on device: {device}')

        self._frame = 0
        self._episode = 0
        self._device = device
        self._env = env

        self._online_model = model().to(device)
        self._target_model = model().to(device)
        self._target_model.eval()

        self._batch_size = batch_size
        self._epsilon_start = epsilon_start
        self._epsilon_end = epsilon_end
        self._epsilon_decay = epsilon_decay

        self._min_replay_size = min_replay_size
        self._replay_memory = ReplayMemory(memory_size)
        self._trainer = Trainer(
            self._online_model, self._target_model, self._device, learning_rate=lr)

    @property
    def frame(self):
        return self._frame

    @frame.setter
    def frame(self, value):
        self._frame = value

    @property
    def model(self):
        return self._online_model

    @property
    def episode(self):
        return self._episode

    @episode.setter
    def episode(self, value):
        self._episode = value

    def train_short(self, *args):
        self._trainer.step(args)

    def optimize(self):
        if len(self._replay_memory) < self._min_replay_size:
            return

        batch = self._replay_memory.sample(self._batch_size)

        self._trainer.step(batch)

    def update_target(self):
        self._target_model.load_state_dict(self._online_model.state_dict())
        self._target_model.save()

    def record_experience(self, *args):
        self._replay_memory.push(*args)

    def predict(self, state):
        ''' 
        Epsilon-Greedy Algorithm
        Plot https://www.wolframalpha.com/input?i=plot%5B0.01+%2B+%280.99+-+0.01%29+*+Exp%5B-x%2F200%5D%2C+%7Bx%2C+0%2C+1000%7D%5D
        '''
        epsilon = self._epsilon_end + (self._epsilon_start - self._epsilon_end) * \
            math.exp(-1 * self._frame / self._epsilon_decay)

        if random.random() <= epsilon:
            return torch.tensor([[self._env.action_space.sample()]], device=self._device, dtype=torch.int64)

        return self.action(state)

    def action(self, state):
        with torch.no_grad():
            prediction = self._online_model(state)
            action = torch.argmax(prediction, dim=1)[0].view(1, 1).detach()

            return action
