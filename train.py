import sys
import pygame
import torch
import datetime
import itertools
import numpy as np
from collections import deque
from logger import ExperimentLog
from agent import Agent
from env.engine import GameEngine
from env.constants.agentmode import AgentMode
from model import SingleLinear, DoubleLinear, LinearFlatten, NatureCnn
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms as T

BATCH_SIZE = 256
EPSILON_START = 1.0
EPSILON_END = 0.01
EPSILON_DECAY = 10_000
MEMORY_SIZE = 100_000
MIN_REPLAY_SIZE = 5_000
TARGET_UPDATE_FREQ = 10_000
LR = 5e-5
PROJECT = 'single_512_ram_test'
LOG_DIR = './logs/' + PROJECT
LOG_INTEVAL = 100
IMAGE = False
SAVE_FREQUENCY = 10_000

device = 'cuda' if torch.cuda.is_available() else 'cpu'
image_preprocess = T.Compose([T.ToPILImage(), T.Resize(
    84, interpolation=T.InterpolationMode.NEAREST), T.Grayscale(), T.ToTensor()])


def preprocess_state(state):
    if IMAGE:
        return image_preprocess(state).unsqueeze(0)

    return torch.from_numpy(state).unsqueeze(0)


if __name__ == '__main__':
    try:
        pygame.init()

        summary_writer = SummaryWriter(LOG_DIR)

        episode_scores = deque([], maxlen=100)
        episode_rewards = deque([], maxlen=100)
        episode_length = deque([], maxlen=100)

        max_score = 0
        max_reward = 0
        max_length = 0

        total_score = 0
        cumulative_reward = 0
        best_score = 0

        env = GameEngine(mode=AgentMode.AGENT, cell_size=10,
                         state_type='img' if IMAGE else'vector')

        state = env.reset()

        state = preprocess_state(state)

        # def model(): return NatureCnn(PROJECT, state.cpu())

        def model(): return SingleLinear(PROJECT,
                                         env.observation_space.shape[0], 512, env.action_space.n)

        # def model(): return DoubleLinear(
        #     PROJECT, env.observation_space.shape[0], 512, 256, env.action_space.n)

        agent = Agent(model, env, MEMORY_SIZE, MIN_REPLAY_SIZE, BATCH_SIZE, LR,
                      EPSILON_START, EPSILON_END, EPSILON_DECAY, device=device)

        # Replay buffer initialization
        for _ in range(MIN_REPLAY_SIZE):
            action = torch.tensor(
                [[env.action_space.sample()]], dtype=torch.int64)

            next_state, reward, done, _ = env.step(
                action)

            next_state = preprocess_state(next_state)

            done = torch.tensor([[done]], dtype=torch.int64)
            reward = torch.tensor(
                [[reward]], dtype=torch.float32)

            agent.record_experience(
                state, action, reward, next_state, done)

            state = next_state

            if done:
                state = env.reset()
                state = preprocess_state(state)

        print('Initialization finished')

        # Training process

        log = ExperimentLog(PROJECT, datetime.datetime.now())
        log.setup()

        state = env.reset()

        state = preprocess_state(state)

        for step in itertools.count():
            action = agent.predict(state)

            next_state, reward, done, info = env.step(
                action)

            agent.frame = step

            cumulative_reward += reward

            next_state = preprocess_state(next_state)

            done = torch.tensor([[done]], dtype=torch.int64)
            reward = torch.tensor(
                [[reward]], dtype=torch.float32)

            agent.record_experience(
                state, action, reward, next_state, done)

            state = next_state

            agent.optimize()

            if done:
                state = env.reset()
                state = preprocess_state(state)

                agent.episode += 1

                if info['score'] > max_score:
                    max_score = info['score']

                if info['frame'] > max_length:
                    max_length = info['frame']

                if cumulative_reward > max_reward:
                    max_reward = cumulative_reward

                episode_scores.append(info['score'])
                episode_rewards.append(cumulative_reward)
                episode_length.append(info['frame'])

                average_reward = np.mean(episode_rewards)
                average_length = np.mean(episode_length)
                average_score = np.mean(episode_scores)

                summary_writer.add_scalar(
                    'Avarage reward', average_reward, global_step=agent.episode)
                summary_writer.add_scalar(
                    'Avarage length', average_length, global_step=agent.episode)

                log.logrow(episode=agent.episode, step=step, average_reward=average_reward, average_length=average_length,
                           average_score=average_score, stamp=datetime.datetime.now())

                if agent.episode % 10 == 0:
                    print('Episode: ', agent.episode)
                    print('Step: ', step)
                    print('avg. reward: ', average_reward)
                    print('avg. length: ', average_length)
                    print('avg. score: ', average_score)
                    print()

                cumulative_reward = 0

            if step % TARGET_UPDATE_FREQ == 0 and step != 0:
                agent.update_target()

            if step % SAVE_FREQUENCY == 0 and step != 0:
                agent.save()

    except KeyboardInterrupt:
        print('Ctrl+C pressed.')

        summary_writer.close()

        pygame.quit()
        sys.exit()
