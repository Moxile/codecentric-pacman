from . import BaseAgent
import random

class Ghost(BaseAgent.BaseAgent):
    color = "red"
    
    def sense_pacman(self, pacman_pos):
        ghost_x = self.position.x
        ghost_y = self.position.y
        pacman_x = pacman_pos.x
        pacman_y = pacman_pos.y

        if abs(ghost_x - pacman_x) == 0 and abs(ghost_y - pacman_y) < 3*20:
            if ghost_y < pacman_y and self.valid(self.position + self.UP):
                return self.UP
            elif self.valid(self.position + self.DOWN):
                return self.DOWN
        elif abs(ghost_y - pacman_y) == 0 and abs(ghost_x - pacman_x) < 3*20:
            if ghost_x < pacman_x and self.valid(self.position + self.RIGHT):
                return self.RIGHT
            elif self.valid(self.position + self.LEFT):
                return self.LEFT
        return None

    def step(self, game_state):
        if self.sense_pacman(game_state["pacman"]):
            self.course = self.sense_pacman(game_state["pacman"])
            self._move(self.course)
        elif self.course and self.valid(self.position + self.course):
            self._move(self.course)
        else:
            options = [
                self.DOWN,
                self.UP,
                self.RIGHT,
                self.LEFT,
            ]
            self.course = random.choice(options)
