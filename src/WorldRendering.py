from turtle import *
from agents.Ghost import Ghost

TILE_SIZE = 20

class WorldRendering:
    WRITER = Turtle(visible=False)
    SCORE_WRITER = Turtle(visible=False)
    HIGHSCORE_WRITER = Turtle(visible=False)

    def __init__(self, maze, font=("Verdana", 16, "bold")):
        self.maze = maze
        self.font = font

    # Draw square using path at (x, y).
    def draw_square(self, x, y):
        self.WRITER.up()
        self.WRITER.goto(x, y)
        self.WRITER.down()
        self.WRITER.begin_fill()

        for count in range(4):
            self.WRITER.forward(20)
            self.WRITER.left(90)

        self.WRITER.end_fill()

    # Draw world using path.
    def world(self):
        bgcolor("black")
        self.WRITER.color("blue")

        for index in range(len(self.maze)):
            tile = self.maze[index]

            if tile > 0:
                x = (index % 20) * 20 - 200
                y = 180 - (index // 20) * 20
                self.draw_square(x, y)

                if tile == 1:
                    self.WRITER.up()
                    self.WRITER.goto(x + 10, y + 10)
                    self.WRITER.dot(2, "white")
                if tile == 3:
                    self.WRITER.up()
                    self.WRITER.goto(x + 10, y + 10)
                    self.WRITER.dot(10, "white")
                if tile == 4:
                    self.WRITER.up()
                    self.WRITER.goto(x + 10, y + 10)
                    self.WRITER.dot(12,"#FFFD5F")


    def render_agent(self, agent):
        up()
        x = agent.position.x + 10
        y = agent.position.y + 10
        goto(x, y)
        if isinstance(agent, Ghost) and agent.kill_timer > 0:
            dot(TILE_SIZE, "violet")
            return
           

        dot(TILE_SIZE, agent.color)

    def render_empty_tile(self, index):
        x = (index % 20) * 20 - 200
        y = 180 - (index // 20) * 20
        self.draw_square(x, y)

    def render_score(self, score):
        self.SCORE_WRITER.undo()
        self.SCORE_WRITER.goto(100, 160)
        self.SCORE_WRITER.color("white")
        self.SCORE_WRITER.write("SCORE: " + str(score), font=self.font)

    def render_end_game(self, message, tcolor):
        self.WRITER.penup()
        self.WRITER.goto(0, 180)
        self.WRITER.color(tcolor)
        self.WRITER.pendown()
        self.WRITER.write(message, align="center", font=self.font)

    def clear_end_game(self):
        """Clears the end game message from the screen."""
        self.WRITER.clear()

    def render_highscore(self, score):
        self.HIGHSCORE_WRITER.undo()
        self.HIGHSCORE_WRITER.goto(-160, 160)
        self.HIGHSCORE_WRITER.color("white")
        self.HIGHSCORE_WRITER.write("HIGHSCORE: " + str(score), font=self.font)
