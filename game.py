import sys
import pygame
from env.engine import GameEngine

if __name__ == "__main__":
    pygame.init()

    env = GameEngine(speed=8, cell_size=20)

    test = True

    while True:
        done, reward, score, frame, state = env.step()

        if test:
            frame = env.get_frame()
            test = False

        if done:
            print(f'Score: {score}')

            pygame.quit()
            sys.exit()
