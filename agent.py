import torch
import random
import math
import numpy as np
import datetime
from torchvision import transforms as T
from snake import SnakeEngine
from direction import Direction
from cell import Cell
from constants import CELL_SIZE
from collections import deque
from model import Conv1, Linear1, Linear2, QTrainer
import plot_helper as plt
from logger import ExperimentLog
import time

BUFFER_SIZE = 100_000
BATCH_SIZE = 2000
LEARNING_RATE = 1e-3
EPSILON_START = 0.9
EPSILON_END = 0.005
EPSILON_DECAY = 15


class Agent:
    def __init__(self, model):
        self.episodes = 0
        self.epsilon = 0
        self.gamma = 0.9
        self.replay_memory = deque(maxlen=BUFFER_SIZE)
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model = model.to(self.device)
        self.trainer = QTrainer(
            self.model, LEARNING_RATE, self.gamma, self.device)

    def render(self, env):
        resize = T.Compose([T.ToPILImage(),
                            T.Resize(
                                (30, 40), interpolation=T.InterpolationMode.BICUBIC),
                            T.ToTensor()])

        frame = env.render().transpose((2, 0, 1))
        frame = np.ascontiguousarray(frame, dtype=np.float32)
        frame = torch.from_numpy(frame).to(self.device)
        frame = resize(frame)

        return frame.numpy()

    def get_state(self, env):
        head = env.snake[0]

        left_point = Cell(head.x - CELL_SIZE, head.y)
        top_point = Cell(head.x, head.y - CELL_SIZE)
        right_point = Cell(head.x + CELL_SIZE, head.y)
        bottom_point = Cell(head.x, head.y + CELL_SIZE)

        top_left_point = Cell(head.x - CELL_SIZE, head.y - CELL_SIZE)
        top_right_point = Cell(head.x + CELL_SIZE, head.y - CELL_SIZE)
        bottom_left_point = Cell(head.x - CELL_SIZE, head.y + CELL_SIZE)
        bottom_right_point = Cell(head.x + CELL_SIZE, head.y + CELL_SIZE)

        heading_left = env.direction == Direction.LEFT
        heading_up = env.direction == Direction.UP
        heading_right = env.direction == Direction.RIGHT
        heading_down = env.direction == Direction.DOWN

        state = [
            # front collision
            (heading_left and env.collides(left_point)) or
            (heading_up and env.collides(top_point)) or
            (heading_right and env.collides(right_point)) or
            (heading_down and env.collides(bottom_point)),

            # right collision
            (heading_left and env.collides(top_point)) or
            (heading_up and env.collides(right_point)) or
            (heading_right and env.collides(bottom_point)) or
            (heading_down and env.collides(left_point)),

            # left collision
            (heading_left and env.collides(bottom_point)) or
            (heading_up and env.collides(left_point)) or
            (heading_right and env.collides(top_point)) or
            (heading_down and env.collides(right_point)),

            # front left collision
            (heading_left and env.collides(bottom_left_point)) or
            (heading_up and env.collides(top_left_point)) or
            (heading_right and env.collides(top_right_point)) or
            (heading_down and env.collides(bottom_right_point)),

            # front right collision
            (heading_left and env.collides(top_left_point)) or
            (heading_up and env.collides(top_right_point)) or
            (heading_right and env.collides(bottom_right_point)) or
            (heading_down and env.collides(bottom_left_point)),

            heading_left,
            heading_up,
            heading_right,
            heading_down,

            # food direction
            env.food.x < env.head.x,  # left
            env.food.y < env.head.y,  # up
            env.food.x > env.head.x,  # right
            env.food.y > env.head.y  # down
        ]

        return np.array(state, dtype=np.int32)

    def memorize(self, state, action, reward, next_state, done):
        self.replay_memory.append((state, action, reward, next_state, done))

    def train_long(self):
        if len(self.replay_memory) > BATCH_SIZE:
            sample = random.sample(self.replay_memory, BATCH_SIZE)
        else:
            sample = self.replay_memory

        states, actions, rewards, next_states, dones = zip(*sample)

        self.trainer.train_step(states, actions, rewards, next_states, dones)

    def train_short(self, state, action, reward, next_state, done):
        self.trainer.train_step(state, action, reward, next_state, done)

    def get_action(self, state):
        eps_threshold = EPSILON_END + \
            (EPSILON_START - EPSILON_END) * \
            math.exp(-1. * self.episodes / EPSILON_DECAY)
        action = [0, 0, 0]

        if random.random() < eps_threshold:
            i = random.randint(0, 2)
            action[i] = 1
        else:
            with torch.no_grad():
                state0 = torch.tensor(
                    state, dtype=torch.float, device=self.device)
                prediction = self.model(state0.unsqueeze(0))
                arg = torch.argmax(prediction).item()
                action[arg] = 1

        return action


def train():
    project = 'Conv1'
    timestamp = datetime.datetime.now()

    logger = ExperimentLog(project, timestamp)
    logger.setup()

    scores = []
    total_score = 0
    mean_scores = []
    cumulative_reward = 0
    best_score = 0

    model = Conv1(project)

    agent = Agent(model)
    env = SnakeEngine()

    while True:
        '''
        old_state = agent.get_state(env)
        action = agent.get_action(old_state)
        done, reward, score, frame = env.step(action)
        new_state = agent.get_state(env)
        '''

        old_state = agent.render(env)
        action = agent.get_action(old_state)
        done, reward, score, frame = env.step(action)
        new_state = agent.render(env)

        cumulative_reward += reward

        # train short
        agent.train_short(old_state, action, reward, new_state, done)

        # remember
        agent.memorize(old_state, action, reward, new_state, done)

        if done:
            # replay memory
            env.reset()
            agent.episodes += 1
            agent.train_long()

            if score > best_score:
                best_score = score
                agent.model.save()

            logger.logrow([agent.episodes, cumulative_reward,
                          score, frame, best_score])
            print('Episode:', agent.episodes, 'reward:', cumulative_reward, 'score:',
                  score, 'best:', best_score)

            scores.append(score)
            total_score += score
            mean_score = total_score / agent.episodes
            mean_scores.append(mean_score)
            cumulative_reward = 0

            plt.plot(scores, mean_scores)


if __name__ == "__main__":
    train()
