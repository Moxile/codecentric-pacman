from .BaseAgent import BaseAgent
import torch
import numpy as np
from collections import deque
import random
import os
from freegames import floor

def offset(point):
    """Return offset of point in tiles."""
    x = (floor(point.x, 20) + 200) / 20
    y = (180 - floor(point.y, 20)) / 20
    index = int(x + y * 20)
    return index

class DQNPacman(BaseAgent):
    color = "yellow"
    BATCH_SIZE = 64
    MEMORY_SIZE = 10000
    GAMMA = 0.99
    EPS_START = 1.0
    EPS_END = 0.01
    EPS_DECAY = 0.995

    def __init__(self, position, valid_function, model_path=None):
        super().__init__(position, valid_function)
        self.action_space = [self.UP, self.DOWN, self.LEFT, self.RIGHT]
        self.memory = deque(maxlen=self.MEMORY_SIZE)
        self.epsilon = self.EPS_START
        
        # Check if GPU is available and set device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Neural Network (input: maze state + positions, output: 4 actions)
        self.model = torch.nn.Sequential(
            torch.nn.Linear(370, 128),  # Changed from self._get_state_size() to 370
            torch.nn.ReLU(),
            torch.nn.Linear(128, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, 4)
        ).to(self.device)  # Move model to GPU if available

        self.target_model = torch.nn.Sequential(
            torch.nn.Linear(370, 128),  # Same fix here
            torch.nn.ReLU(),
            torch.nn.Linear(128, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, 4)
        ).to(self.device)  # Move target model to GPU if available

        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)

        # Load model if path is provided
        if model_path and os.path.exists(model_path):
            self.load_model(model_path)

    def _get_state_size(self):
        # Corrected: Maze (360) + Pacman (2) + Ghosts (8)
        return 360 + 2 + 8  # Total 370

    def _preprocess_state(self, game_state):
        """Convert game state to neural network input tensor"""
        # Flatten maze
        maze = np.array(game_state["surrounding"], dtype=np.float32).flatten()
        # Positions (normalized coordinates)
        pacman = [game_state["pacman"].x/200, game_state["pacman"].y/180]
        ghosts = [g.x/200 for g in game_state["ghosts"]] + [g.y/180 for g in game_state["ghosts"]]
        state_tensor = torch.FloatTensor(np.concatenate([maze, pacman, ghosts])).to(self.device)  # Move tensor to GPU if available
        #print(f"State tensor is on device: {state_tensor.device}")
        return state_tensor

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        if random.random() < self.epsilon:
            return random.choice(range(4))
        with torch.no_grad():
            q_values = self.model(state.unsqueeze(0))  # Add batch dimension
            return torch.argmax(q_values).item()

    def replay(self):
        if len(self.memory) < self.BATCH_SIZE:
            return
        
        batch = random.sample(self.memory, self.BATCH_SIZE)
        states, actions, rewards, next_states, dones = zip(*batch)
        
        # Convert to tensors and move to GPU if available
        states = torch.stack(states).to(self.device)
        next_states = torch.stack(next_states).to(self.device)
        #print(f"States tensor is on device: {states.device}")
        #print(f"Next states tensor is on device: {next_states.device}")
        
        # Q-values prediction
        current_q = self.model(states).gather(1, torch.LongTensor(actions).unsqueeze(1).to(self.device))
        
        # Target Q-values
        next_q = self.target_model(next_states).max(1)[0].detach()
        target_q = torch.FloatTensor(rewards).to(self.device) + self.GAMMA * next_q * (1 - torch.FloatTensor(dones).to(self.device))
        
        # Compute loss
        loss = torch.nn.MSELoss()(current_q.squeeze(), target_q)
        
        # Backpropagation
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        # Decay epsilon
        self.epsilon = max(self.EPS_END, self.epsilon * self.EPS_DECAY)

    def step(self, game_state):
        state = self._preprocess_state(game_state)
        
        if hasattr(self, 'last_state'):
            reward = -1  # Small penalty for each step
            index = offset(game_state["pacman"])

            if game_state["surrounding"][index] == 1:
                reward += 10  # Reward for taking a coin
            if game_state["surrounding"][index] == 3:
                reward += 100  # Reward for taking a pellet
            if any(abs(self.position - ghost) < 20 for ghost in game_state["ghosts"]):
                reward -= 10000  # Penalty for dying
            
            self.remember(self.last_state, self.last_action, reward, state, False)
        
        action = self.act(state)
        self.course = self.action_space[action]
        
        if self.course and self.valid(self.position + self.course):
            self._move(self.course)
        
        self.last_state = state
        self.last_action = action

    def save_model(self, filepath):
        """Save the model to the specified filepath"""
        torch.save(self.model.state_dict(), filepath)

    def load_model(self, filepath):
        """Load the model from the specified filepath"""
        self.model.load_state_dict(torch.load(filepath))
        self.model.to(self.device)