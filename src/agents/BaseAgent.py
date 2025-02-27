from freegames import vector

class BaseAgent:

    
    DOWN = vector(0, -5)
    UP = vector(0, 5)
    LEFT = vector(-5, 0)
    RIGHT = vector(5, 0)

    course = None

    def __init__(self, position, valid_function) -> None:
        self.position = position
        self.valid = valid_function

    def move(self, delta):
        """Change the agent's position if valid."""
        if self.valid(self.position + delta):
            self.position.move(delta)
            return True
        return False

    def step(self, game_state):
        pass
