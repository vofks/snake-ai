import sys
import pygame
import torch
import datetime
import itertools
import numpy as np
from logger import ExperimentLog
from agent import Agent
from env.engine import GameEngine
from env.constants.agentmode import AgentMode
from model import SingleLinear, DoubleLinear, LinearFlatten, NatureCnn
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms as T

BATCH_SIZE = 256
EPSILON_START = 1.0
EPSILON_END = 0.1
EPSILON_DECAY = 200
MEMORY_SIZE = 10_000
MIN_REPLAY_SIZE = 1000
TARGET_UPDATE_FREQ = 100
LR = 5e-4
PROJECT = 'refactoring_test_5e4_double'
LOG_DIR = './logs/' + PROJECT
LOG_INTEVAL = 100
IMAGE = False

device = 'cuda' if torch.cuda.is_available() else 'cpu'
image_preprocess = T.Compose([T.ToPILImage(), T.Resize(
    80, interpolation=T.InterpolationMode.NEAREST), T.Grayscale(), T.ToTensor()])


def preprocess_state(state):
    if IMAGE:
        return image_preprocess(state).unsqueeze(0).to(device)

    return torch.from_numpy(state).unsqueeze(0).to(device)


if __name__ == '__main__':
    try:
        pygame.init()

        summary_writer = SummaryWriter(LOG_DIR)

        episode_scores = []
        mean_scores = []
        episode_rewards = []
        episode_length = []

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

        # def model(): return SingleLinear(PROJECT,
        #                                  env.observation_space.shape[0], 256, env.action_space.n)

        def model(): return DoubleLinear(
            PROJECT, env.observation_space.shape[0], 512, 256, env.action_space.n)

        agent = Agent(model, env, MEMORY_SIZE, MIN_REPLAY_SIZE, BATCH_SIZE, LR,
                      EPSILON_START, EPSILON_END, EPSILON_DECAY, device=device)

        # Replay buffer initialization
        for _ in range(MIN_REPLAY_SIZE):
            action = torch.tensor(
                [[env.action_space.sample()]], device=device, dtype=torch.int64)

            next_state, reward, done, _ = env.step(
                action)

            next_state = preprocess_state(next_state)

            done = torch.tensor([[done]], device=device, dtype=torch.int64)
            reward = torch.tensor(
                [[reward]], device=device, dtype=torch.float32)

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

            done = torch.tensor([[done]], device=device, dtype=torch.int64)
            reward = torch.tensor(
                [[reward]], device=device, dtype=torch.float32)

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

                summary_writer.add_scalar(
                    'Avarage reward', average_reward, global_step=agent.episode)
                summary_writer.add_scalar(
                    'Avarage length', average_length, global_step=agent.episode)

                cumulative_reward = 0

            if step % LOG_INTEVAL == 0 and step != 0:
                average_score = np.mean(episode_scores)
                average_reward = np.mean(episode_rewards)
                average_length = np.mean(episode_length)

                print(
                    f'Ep: {agent.episode} Step: {step}\n avg. rew: {average_reward} max. rew: {max_reward}\n vag. len: {average_length} max len: {max_length}\n avg. score: {average_score} max score: {max_score}\n')

                log.logrow(episode=agent.episode, step=step, average_reward=average_reward, max_reward=max_reward, average_length=average_length,
                           max_length=max_length, average_score=average_score, max_score=max_score, stamp=datetime.datetime.now())

            if step % TARGET_UPDATE_FREQ == 0 and step != 0:
                agent.update_target()

    except KeyboardInterrupt:
        print('Ctrl+C pressed.')

        summary_writer.close()

        pygame.quit()
        sys.exit()
