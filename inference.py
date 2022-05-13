import model
import torch
import datetime
from logger import ExperimentLog
from snake import SnakeEngine
from agent import Agent
import plot_helper as plt

MODEL_NAME = 'model/linear2_512_256/linear2_512_256_15-31-53 23-12-2021.pth'
NUM_EPISODES = 100
PROJECT = 'linear2_512_256_inference'
FOLDER = 'inference'


def main():
    env = SnakeEngine()

    dqn = model.Linear2(PROJECT, 512, 256)
    dqn.load_state_dict(torch.load(MODEL_NAME))
    dqn.eval()

    agent = Agent(dqn)

    log = ExperimentLog(PROJECT, datetime.datetime.now(), FOLDER)
    log.setup()

    best_score = 0
    scores = []

    for i in range(NUM_EPISODES):
        cumulative_reward = 0
        score = 0

        env.reset()
        done = False
        env.step([1, 0, 0])

        while not done:
            state = agent.get_state(env)
            action = agent.eval(state)
            done, reward, score, frame = env.step(action)

            cumulative_reward += reward

            if done:
                if score > best_score:
                    best_score = score

                scores.append(score)

                log.logrow([i + 1, cumulative_reward,
                            score, frame, best_score])

                plt.plot(scores)


if __name__ == '__main__':
    main()
