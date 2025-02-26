# Pacman, classic arcade game.

from freegames import floor, vector
from agents.HumanPacman import HumanPacman
from agents.Ghost import Ghost
from WorldRendering import *
from Mazes import *
from agents.customghost import Blinky, Pinky, Inky, Clyde

WRITER = Turtle(visible=False)

MAZE = Mazes.level_1
MAX_SCORE = Mazes.level_1_max_score
WORLD = WorldRendering(MAZE)

state = {"score": 0}

def offset(point):
    """Return offset of point in tiles."""
    x = (floor(point.x, 20) + 200) / 20
    y = (180 - floor(point.y, 20)) / 20
    index = int(x + y * 20)
    return index

def valid(position):
    """Return True if the agent position is valid."""
    index = offset(position)
    if MAZE[index] == TILE_WALL:
        return False

    index = offset(position + 19)
    if MAZE[index] == TILE_WALL:
        return False
    
    is_in_column = position.y % TILE_SIZE == 0
    is_in_row = position.x % TILE_SIZE == 0
    return is_in_row or is_in_column

def update_world():
    """Updates the world repeatedly until the game finishes. 
    - Moves pacman and all ghosts.
    - Checks if game is lost/won.
    """
    clear()
    index = offset(pacman.position)
    if MAZE[index] == TILE_DOT:
        MAZE[index] = TILE_EMPTY
        state["score"] += 1
        WORLD.render_empty_tile(index)
    WORLD.render_score(state["score"])

    if MAZE[index] == 3:
        MAZE[index] = TILE_EMPTY
        state["score"] += 100
        WORLD.render_empty_tile(index)
    WORLD.render_score(state["score"])

    # move all agents
    for ghost in ghosts:
        ghost.step(get_agent_game_state(ghost))
        WORLD.render_agent(ghost)
    pacman.step(get_agent_game_state(pacman))
    WORLD.render_agent(pacman)
    update()

    # check for game end
    if state["score"] == MAX_SCORE:
        WORLD.render_end_game("You won!", "yellow")
        return
    for ghost in ghosts:
        if abs(pacman.position - ghost.position) < 20:
            WORLD.render_end_game("You lost!", "red")
            return

    ontimer(update_world, 100)

def get_agent_game_state(agent):
    """Returns the part of the world that the given agent can see.
    Currently, each agent has a complete view of the world.
    """
    agent_state = {}
    agent_state["score"] = state["score"]
    agent_state["max_score"] = MAX_SCORE
    agent_state["surrounding"] = MAZE
    agent_state["pacman"] = pacman.position
    agent_state["ghosts"] = [ghost.position for ghost in ghosts]
    return agent_state


pacman = HumanPacman(vector(-40, -80), valid)
ghosts = [
    Blinky(vector(-120, -100), valid),
    Pinky(vector(-40, 100), valid),
    Inky(vector(100, 100), valid),
    Clyde(vector(100, -100), valid),
]

setup(420, 420, 370, 0) # window
hideturtle()
tracer(False)
listen()
WORLD.world()
update_world()
done()
