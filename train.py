import sys
import pygame
import torch
import itertools
import numpy as np
from agent import Agent
from env.engine import GameEngine
from env.constants.mode import Mode
from model import Linear1, Linear2, LinearFlatten
from torch.utils.tensorboard import SummaryWriter

TARGET_UPDATE_FREQ = 1000
LOG_DIR = './logs/snake'
LOG_INTEVAL = 1000

if __name__ == '__main__':
    try:
        pygame.init()

        device = 'cuda' if torch.cuda.is_available() else 'cpu'

        summary_writer = SummaryWriter(LOG_DIR)

        scores = []
        mean_scores = []
        rewards = []
        frames = []
        total_score = 0
        cumulative_reward = 0
        best_score = 0

        env = GameEngine(mode=Mode.AGENT, cell_size=10)
        # env.reset()
        #state = env.get_frame()
        #state = Agent.preprocess_image(state).to(device)

        #state_flatten = state.flatten(start_dim=1)
        #input_size = state_flatten.shape[1]

        #model = LinearFlatten('test', input_size, 600, 512)
        model = Linear2('test', 256, 128)
        agent = Agent(model, device=device)

        state = torch.from_numpy(env.reset()).unsqueeze(0).to(device)

        for step in itertools.count():
            action = agent.predict(state)

            done, reward, score, frame, next_state = env.step(
                action.item())

            agent.frame = step

            cumulative_reward += reward

            #next_state = Agent.preprocess_image(env.get_frame()).to(device)

            done_t = torch.tensor([[done]], device=device, dtype=torch.int64)
            reward_t = torch.tensor([[reward]], device=device)
            next_state_t = torch.tensor([np.array(next_state)], device=device)

            agent.record_experience(
                state, action, reward_t, next_state_t, done_t)

            state = next_state_t

            agent.optimize()

            if done:
                env.reset()
                agent.episode += 1
                frames.append(frame)

                if score > best_score:
                    best_score = score

                scores.append(score)
                total_score += score

                mean_score = total_score / agent.episode
                mean_scores.append(mean_score)

                rewards.append(cumulative_reward)
                cumulative_reward = 0

                # TODO define plotting module
                # plt.plot(scores, mean_scores)

            if step % LOG_INTEVAL == 0:
                average_reward = np.mean(rewards)
                average_length = np.mean(frames)

                print(
                    f'Episode: {agent.episode} Step: {step} score: {score} best score: {best_score} avg. reward: {average_reward}')

                summary_writer.add_scalar(
                    'Avarage reward', average_reward, global_step=agent.episode)
                summary_writer.add_scalar(
                    'Avarage length', average_length, global_step=agent.episode)

            if step % TARGET_UPDATE_FREQ == 0:
                agent.update_target()

    except KeyboardInterrupt:
        print('Ctrl+C pressed.')

        summary_writer.close()

        pygame.quit()
        sys.exit()
