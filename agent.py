from this import d
import torch
import math
import random
import numpy as np
from copy import deepcopy
from replaymemory import ReplayMemory
from trainer import Trainer
from torchvision import transforms as T
from torch.cuda import is_available

_DEFAULT_DEVICE = 'cuda' if is_available() else 'cpu'
BATCH_SIZE = 256
EPSILON_START = 1.0
EPSILON_END = 0.01
EPSILON_DECAY = 200
MIN_REPLAY_SIZE = 1000
MEMORY_SIZE = 100_000


class Agent:
    def __init__(self, model, device=_DEFAULT_DEVICE):
        print(f'Device: {device}')

        self._frame = 0
        self._episode = 0
        self._device = device
        self._online_model = model.to(device)
        self._target_model = deepcopy(model)
        self._target_model.eval()
        self._replay_memory = ReplayMemory(MEMORY_SIZE)
        self._trainer = Trainer(
            self._online_model, self._target_model, self._device)

    def preprocess_image(frame):
        preprocess = T.Compose([T.ToPILImage(), T.Resize(
            20, interpolation=T.InterpolationMode.NEAREST), T.Grayscale(), T.ToTensor()])

        frame = frame.transpose((2, 0, 1))
        frame = np.ascontiguousarray(frame, dtype=np.float32) / 255
        frame = torch.from_numpy(frame)

        return preprocess(frame).unsqueeze(0)

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
        if len(self._replay_memory) < MIN_REPLAY_SIZE:
            return

        batch = self._replay_memory.sample(BATCH_SIZE)

        self._trainer.step(batch)

    def update_target(self):
        self._target_model.load_state_dict(self._online_model.state_dict())

    def record_experience(self, *args):
        self._replay_memory.push(*args)

    def predict(self, state):
        ''' 
        Epsilon-Greedy Algorithm
        Plot https://www.wolframalpha.com/input?i=plot%5B0.01+%2B+%280.99+-+0.01%29+*+Exp%5B-x%2F200%5D%2C+%7Bx%2C+0%2C+1000%7D%5D
        '''
        epsilon = EPSILON_END + (EPSILON_START - EPSILON_END) * \
            math.exp(-1 * self._frame / EPSILON_DECAY)

        if random.random() <= epsilon:
            return torch.tensor([[random.randrange(3)]], device=self._device, dtype=torch.int64)

        return self.action(state)

    def action(self, state):
        with torch.no_grad():
            prediction = self._online_model(state)
            action = torch.argmax(prediction, dim=1)[0].view(1, 1).detach()

            return action
