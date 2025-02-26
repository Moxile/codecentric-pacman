from . import BaseAgent
import random

class Ghost(BaseAgent.BaseAgent):

    def step(self, game_state):
        if self.course and self.valid(self.position + self.course):
            self._move(self.course)
        else:
            options = [
                self.DOWN,
                self.UP,
                self.RIGHT,
                self.LEFT,
            ]
            self.course = random.choice(options)
