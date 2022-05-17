import sys
import pygame
import torch
import numpy as np
from agent import Agent
from env.engine import GameEngine
from env.constants.mode import Mode
from model import Linear1, Linear2

TARGET_UPDATE_FREQ = 200

if __name__ == '__main__':
    try:
        pygame.init()

        device = 'cuda' if torch.cuda.is_available() else 'cpu'

        scores = []
        mean_scores = []
        rewards = []
        total_score = 0
        cumulative_reward = 0
        best_score = 0

        model = Linear2('test', 512, 256)
        agent = Agent(model, device=device)

        env = GameEngine(mode=Mode.AGENT, cell_size=5)

        state = torch.from_numpy(env.reset()).unsqueeze(0).to(device)

        while True:
            action = agent.predict(state)

            done, reward, score, frame, next_state = env.step(
                action.item())

            cumulative_reward += reward

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

                if score > best_score:
                    best_score = score
                    # ????
                    # agent.model.save()

                if agent.episode % 100 == 0:
                    print(
                        f'Episode: {agent.episode} frames: {frame} score: {score} best: {best_score} reward: {np.mean(rewards)}')

                scores.append(score)
                total_score += score

                mean_score = total_score / agent.episode
                mean_scores.append(mean_score)

                rewards.append(cumulative_reward)
                cumulative_reward = 0

                # TODO define plotting module
                # plt.plot(scores, mean_scores)

            if agent.episode % TARGET_UPDATE_FREQ == 0:
                agent.update_target()

    except KeyboardInterrupt:
        print('Ctrl+C pressed.')

        pygame.quit()
        sys.exit()
