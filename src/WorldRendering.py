from turtle import *
from agents.Ghost import Ghost

TILE_SIZE = 20

class WorldRendering:
    WRITER = Turtle(visible=False)         # For maze drawing
    SCORE_WRITER = Turtle(visible=False)   # For score display
    HIGHSCORE_WRITER = Turtle(visible=False)  # For highscore display
    AGENT_WRITER = Turtle(visible=False)   # For agent drawing

    def __init__(self, maze, font=("Verdana", 16, "bold")):
        self.maze = maze
        self.font = font
        self.AGENT_WRITER.speed(0)  # Fastest speed for agent rendering

    def draw_square(self, x, y):
        """Draw a square at the given coordinates."""
        self.WRITER.up()
        self.WRITER.goto(x, y)
        self.WRITER.down()
        self.WRITER.begin_fill()
        for _ in range(4):
            self.WRITER.forward(20)
            self.WRITER.left(90)
        self.WRITER.end_fill()

    def world(self):
        """Draw the entire maze."""
        bgcolor("black")
        self.WRITER.color("blue")
        for index in range(len(self.maze)):
            tile = self.maze[index]
            if tile > 0:
                x = (index % 20) * 20 - 200
                y = 180 - (index // 20) * 20
                self.draw_square(x, y)
                if tile == 1:  # Small dot
                    self.WRITER.up()
                    self.WRITER.goto(x + 10, y + 10)
                    self.WRITER.dot(2, "white")
                elif tile == 3:  # Power pellet
                    self.WRITER.up()
                    self.WRITER.goto(x + 10, y + 10)
                    self.WRITER.dot(10, "white")
                elif tile == 4:  # Special item
                    self.WRITER.up()
                    self.WRITER.goto(x + 10, y + 10)
                    self.WRITER.dot(12, "#FFFD5F")

    def render_agent(self, agent):
        """Render an agent (Pacman or Ghost) at its position."""
        self.AGENT_WRITER.up()
        x = agent.position.x + 10
        y = agent.position.y + 10
        self.AGENT_WRITER.goto(x, y)
        if isinstance(agent, Ghost) and agent.kill_timer > 0:
            self.AGENT_WRITER.dot(15, "violet")  # Smaller size for visibility
        else:
            self.AGENT_WRITER.dot(15, agent.color)  # Smaller size for visibility

    def render_empty_tile(self, index):
        """Redraw a tile as empty when eaten."""
        x = (index % 20) * 20 - 200
        y = 180 - (index // 20) * 20
        self.draw_square(x, y)

    def render_score(self, score):
        """Display the current score."""
        self.SCORE_WRITER.undo()
        self.SCORE_WRITER.goto(100, 160)
        self.SCORE_WRITER.color("white")
        self.SCORE_WRITER.write("SCORE: " + str(score), font=self.font)

    def render_end_game(self, message, tcolor):
        """Display an end game message."""
        self.WRITER.penup()
        self.WRITER.goto(0, 180)
        self.WRITER.color(tcolor)
        self.WRITER.pendown()
        self.WRITER.write(message, align="center", font=self.font)

    def clear_end_game(self):
        """Clear the end game message."""
        self.WRITER.clear()

    def render_highscore(self, score):
        """Display the highscore."""
        self.HIGHSCORE_WRITER.undo()
        self.HIGHSCORE_WRITER.goto(-160, 160)
        self.HIGHSCORE_WRITER.color("white")
        self.HIGHSCORE_WRITER.write("HIGHSCORE: " + str(score), font=self.font)