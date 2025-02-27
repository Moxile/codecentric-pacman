from . import BaseAgent
from turtle import onkey

class HumanPacman(BaseAgent.BaseAgent):
    color = "yellow"
    next_in_queue = None
    timer = 0

    def down(self):
        self.next_in_queue = self.DOWN
        self.timer = 10

    def up(self):
        self.next_in_queue = self.UP
        self.timer = 10

    def left(self):
        self.next_in_queue = self.LEFT
        self.timer = 10

    def right(self):
        self.next_in_queue = self.RIGHT
        self.timer = 10

    def step(self, game_state):
        # check whether any new direction in queue
        if self.timer > 0:
            self.timer -= 1
            if self.next_in_queue and self.valid(self.position + self.next_in_queue):
                self.course = self.next_in_queue
                timer = 0
                self.next_in_queue = None

        onkey(lambda: self.right(), "Right")
        onkey(lambda: self.left(), "Left")
        onkey(lambda: self.up(), "Up")
        onkey(lambda: self.down(), "Down")
        if self.course:
            self.move(self.course)