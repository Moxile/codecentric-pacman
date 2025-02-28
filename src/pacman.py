# Pacman, classic arcade game.

from freegames import floor, vector
from agents.HumanPacman import HumanPacman
from agents.Ghost import Ghost
from WorldRendering import *
from Mazes import *
from agents.customghost import Blinky, Pinky, Inky, Clyde
from agents.DQNPacman import DQNPacman
import os
from copy import copy
import pickle

# Global turtle for general drawing (not used for agents)
WRITER = Turtle(visible=False)

# Initial game setup
MAZE = copy(Mazes.level_1)
MAX_SCORE = Mazes.level_1_max_score
WORLD = WorldRendering(MAZE)

pause = False
end = True
state = {"score": 0}

# Highscore file handling
HIGHSCORE_FILE = "highscore.pkl"

def save_highscore(highscore):
    """Save the highscore to a file."""
    with open(HIGHSCORE_FILE, 'wb') as f:
        pickle.dump(highscore, f)

def load_highscore():
    """Load the highscore from a file."""
    try:
        with open(HIGHSCORE_FILE, 'rb') as f:
            return pickle.load(f)
    except (FileNotFoundError, EOFError):
        return 0

highscore = load_highscore()

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

def reset_game():
    """Resets the game state to start a new game."""
    global pacman, ghosts, state, MAZE, pause, end
    if pause or not end:
        return
    end = False
    WORLD.clear_end_game()  # Clear any end game message
    state["score"] = 0
    MAZE = copy(Mazes.level_1)
    WORLD.maze = MAZE
    pacman = DQNPacman(vector(-40, -60), valid, model_path="dqn_pacman_model.pth")
    ghosts = [
        Blinky(vector(-120, -100), valid),
        Pinky(vector(-40, 100), valid),
        Inky(vector(100, 100), valid),
        Clyde(vector(100, -100), valid),
    ]
    WORLD.world()  # Draw the maze once at reset
    update_world()

def update_world():
    """Updates the world repeatedly until the game finishes."""
    WORLD.AGENT_WRITER.clear()  # Clear only agent drawings

    if pause:
        ontimer(update_world, 100)
        return

    index = offset(pacman.position)
    if MAZE[index] == TILE_DOT:
        MAZE[index] = TILE_EMPTY
        state["score"] += 1
        WORLD.render_empty_tile(index)
    elif MAZE[index] == TILE_COINT:
        MAZE[index] = TILE_EMPTY
        state["score"] += 100
        WORLD.render_empty_tile(index)
    elif MAZE[index] == TILE_DEAD:
        MAZE[index] = TILE_EMPTY
        pacman.kill_points += 1
        WORLD.render_empty_tile(index)

    WORLD.render_score(state["score"])

    # Move all agents
    for ghost in ghosts:
        if ghost.kill_timer > 0:
            ghost.kill_timer -= 1
            WORLD.render_agent(ghost)
            continue
        ghost.step(get_agent_game_state(ghost))
        WORLD.render_agent(ghost)
    pacman.step(get_agent_game_state(pacman))
    WORLD.render_agent(pacman)
    update()

    # Update highscore if necessary
    global highscore
    if state["score"] > highscore:
        highscore = state["score"]
        save_highscore(highscore)
        WORLD.render_highscore(state["score"])

    # Train DQN Pacman if applicable
    if isinstance(pacman, DQNPacman):
        pacman.replay()

    # Check for game end conditions
    global end
    if state["score"] == MAX_SCORE:
        WORLD.render_end_game("You won!", "yellow")
        end = True
        return
    for ghost in ghosts:
        if abs(pacman.position - ghost.position) < 20 and ghost.kill_timer <= 0:
            if pacman.kill_points > 0:
                pacman.kill_points -= 1
                ghost.kill_timer = 100
                continue
            WORLD.render_end_game("You lost!", "red")
            end = True
            return
        
    ontimer(update_world, 100)

def get_agent_game_state(agent):
    """Returns the agent's view of the game state."""
    agent_state = {
        "score": state["score"],
        "max_score": MAX_SCORE,
        "surrounding": MAZE,
        "pacman": pacman.position,
        "ghosts": [ghost.position for ghost in ghosts]
    }
    return agent_state

def train_ai(episodes):
    """Train the DQN Pacman agent for a given number of episodes."""
    for episode in range(episodes):
        reset_game()
        while not end:
            update_world()
        if episode % 10 == 0:
            print(f"Episode {episode} completed.")
            pacman.save_model("dqn_pacman_model.pth")
    pacman.save_model("dqn_pacman_model.pth")

def toggle_pause():
    """Toggle the pause state of the game."""
    global pause
    pause = not pause
    if not pause:
        WORLD.clear_end_game()
        WORLD.maze = MAZE
        WORLD.world()
    else:
        WORLD.render_end_game("Paused", "white")

def main():
    """Setup and run the game."""
    setup(420, 420, 370, 0)  # Window setup
    hideturtle()
    tracer(False)
    listen()
    WORLD.render_highscore(highscore)
    reset_game()

    # Train the AI
    #train_ai(50000)

    # Run the game to observe AI performance
    reset_game()
    onkey(toggle_pause, 'Escape')
    onkey(reset_game, 'Return')
    done()

if __name__ == "__main__":
    main()

'''
# Initial window setup for immediate game start
setup(820, 420, 370, 0)
hideturtle()
tracer(False)
listen()
WORLD.render_highscore(highscore)
reset_game()
onkey(toggle_pause, 'Escape')
onkey(reset_game, 'Return')
done()
'''