import pygame
import random
from direction import Direction
from color import Color
from constants import CELL_SIZE, SPEED, FONT_SIZE, FRAME_FACTOR, INNER_CELL_SIZE
from cell import Cell
from collections import deque


event_direction_mapping = {
    pygame.K_LEFT: Direction.LEFT,
    pygame.K_RIGHT: Direction.RIGHT,
    pygame.K_UP: Direction.UP,
    pygame.K_DOWN: Direction.DOWN
}


class SnakeEngine:
    def __init__(self, w=800, h=600, speed=SPEED):
        pygame.init()

        self.speed = speed
        self.w = w
        self.h = h
        self.display = pygame.display.set_mode((self.w, self.h))
        pygame.display.set_caption('Snake Game')
        self.clock = pygame.time.Clock()

        self.font = pygame.font.SysFont('arial', FONT_SIZE)
        self.moves = {
            Direction.LEFT: lambda: self._change_x(-CELL_SIZE),
            Direction.RIGHT: lambda: self._change_x(CELL_SIZE),
            Direction.UP: lambda: self._change_y(-CELL_SIZE),
            Direction.DOWN: lambda: self._change_y(CELL_SIZE)
        }

        self.reset()

    def reset(self):
        self.direction = Direction.UP
        self.head = Cell(self.w/2, self.h/2)
        self.snake = [self.head, Cell(self.head.x, self.head.y+CELL_SIZE)]
        self.path = deque(self.snake, maxlen=len(self.snake)*2)
        self.score = 0
        self.food = None
        self.frame = 0

        self._place_food()

    def _place_food(self):
        x = random.randint(0, (self.w-CELL_SIZE)//CELL_SIZE)*CELL_SIZE
        y = random.randint(0, (self.h-CELL_SIZE)//CELL_SIZE)*CELL_SIZE

        self.food = Cell(x, y)

        if self.food in self.snake:
            self._place_food()

    def _key_event_handler(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                quit()

            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    pygame.quit()
                    quit()

                self.direction = self._validate_direction(
                    event_direction_mapping[event.key]) or self.direction

    def _handle_action(self, action):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                quit()

        directions = [Direction.LEFT, Direction.UP,
                      Direction.RIGHT, Direction.DOWN]
        i = directions.index(self.direction)

        if(action[1]) == 1:
            self.direction = directions[(i+1) % 4]
        if(action[2]) == 1:
            self.direction = directions[(i-1) % 4]

    def step(self, action=None):
        done = False
        reward = 1
        self.frame += 1

        if(action is None):
            self._key_event_handler()
        else:
            self._handle_action(action)

        self._move_head()
        self.snake.insert(0, self.head)

        if(self.head not in self.path):
            reward = 3

        self.path.append(self.head)

        if self.collides() or self.frame > len(self.snake)*FRAME_FACTOR:
            done = True
            reward = -10
            return done, reward, self.score, self.frame

        if self.head == self.food:
            self.score += 1
            reward = 40
            self._place_food()
            self.path = deque(self.path, maxlen=len(self.snake)*2)
        else:
            self.snake.pop()

        self._render()
        self.clock.tick(self.speed)

        return done, reward, self.score, self.frame

    def render(self):
        self._render()
        return pygame.surfarray.array3d(pygame.display.get_surface())

    def _render(self):
        self.display.fill(Color.GUNMETAL.value)

        for cell in self.snake:
            pygame.draw.rect(self.display, Color.BDAZZLED.value, pygame.Rect(
                cell.x, cell.y, CELL_SIZE, CELL_SIZE))
            pygame.draw.rect(self.display, Color.CERULEAN.value, pygame.Rect(
                cell.x+2, cell.y+2, INNER_CELL_SIZE, INNER_CELL_SIZE))

        pygame.draw.ellipse(self.display, Color.SODA.value, pygame.Rect(
            self.food.x, self.food.y, CELL_SIZE, CELL_SIZE))

        score = self.font.render(
            "Score: " + str(self.score), True, Color.WHITE.value)
        self.display.blit(score, [0, 0])
        pygame.display.flip()

    def _get_direction_from_action(self, action):
        directions = [Direction.LEFT, Direction.UP,
                      Direction.RIGHT, Direction.DOWN]
        direction = self.direction
        i = directions.index(direction)

        if(action[1]) == 1:
            direction = directions[(i+1) % 4]
        if(action[2]) == 1:
            direction = directions[(i-1) % 4]

        return direction

    def _move_head(self):
        self.head = self.moves[self.direction]()

    def _validate_direction(self, direction):
        if(self.direction == Direction.LEFT and direction == Direction.RIGHT):
            return
        elif(self.direction == Direction.RIGHT and direction == Direction.LEFT):
            return
        elif(self.direction == Direction.DOWN and direction == Direction.UP):
            return
        elif(self.direction == Direction.UP and direction == Direction.DOWN):
            return

        return direction

    def _change_x(self, delta):
        head = self.head
        return Cell(head.x + delta, head.y)

    def _change_y(self, delta):
        head = self.head
        return Cell(head.x, head.y + delta)

    def collides(self, point=None):
        if(point is None):
            point = self.head

        if point.x > self.w - CELL_SIZE or point.x < 0 or point.y > self.h - CELL_SIZE or point.y < 0:
            return True

        if point in self.snake[1:]:
            return True

        return False


if __name__ == "__main__":
    engine = SnakeEngine(speed=20)

    while True:
        res = engine.step()
        print(res[1])

        if res[0] == True:
            print(f'Score: {res[2]}')
            break

    pygame.quit()
