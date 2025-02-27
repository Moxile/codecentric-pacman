from .BaseAgent import BaseAgent
import torch
import torch.nn as nn
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
    """A Deep Q-Network (DQN) agent optimized for Pacman, leveraging RTX 4080 GPU power."""
    color = "yellow"
    
    # Hyperparameters tuned for RTX 4080
    BATCH_SIZE = 128       # Larger batch size for GPU parallelization
    MEMORY_SIZE = 50000    # Increased memory for more experiences
    GAMMA = 0.99           # Discount factor
    EPS_START = 1.0        # Initial epsilon for exploration
    EPS_END = 0.01         # Minimum epsilon
    EPS_DECAY = 0.999      # Slower decay for extended exploration
    TARGET_UPDATE = 1000   # Update target network every 1000 steps
    kill_points = 0

    def __init__(self, position, valid_function, model_path=None):
        """Initialize the DQN Pacman agent."""
        super().__init__(position, valid_function)
        self.action_space = [self.UP, self.DOWN, self.LEFT, self.RIGHT]
        self.memory = deque(maxlen=self.MEMORY_SIZE)
        self.epsilon = self.EPS_START
        self.step_counter = 0  # For target network updates

        # Use GPU if available (RTX 4080 should be detected)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # CNN-based Q-network: deeper architecture for RTX 4080
        self.model = nn.Sequential(
            nn.Conv2d(8, 32, kernel_size=3, stride=1, padding=1),  # Input: 8 channels
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(128 * 18 * 20, 256),  # 18x20 grid after convolutions
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 4)  # Output: Q-values for 4 actions
        ).to(self.device)

        # Target network (same architecture)
        self.target_model = nn.Sequential(
            nn.Conv2d(8, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(128 * 18 * 20, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 4)
        ).to(self.device)

        # Sync target model with main model
        self.target_model.load_state_dict(self.model.state_dict())
        self.target_model.eval()  # Set to evaluation mode

        # Optimizer with a lower learning rate for the deeper network
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.0001)

        # Load pre-trained model if provided
        if model_path and os.path.exists(model_path):
            self.load_model(model_path)

    def _preprocess_state(self, game_state):
        """Convert game state to an 8-channel 18x20 tensor for CNN input."""
        # Reshape maze to 18x20 grid
        maze = np.array(game_state["surrounding"]).reshape(18, 20)
        
        # Channel 1: Walls
        walls = (maze == 0).astype(np.float32)
        # Channel 2: Dots
        dots = (maze == 1).astype(np.float32)
        # Channel 3: Pellets
        pellets = (maze == 3).astype(np.float32)
        # Channel 4: Pacman position
        pacman_pos = np.zeros((18, 20), dtype=np.float32)
        pacman_index = offset(game_state["pacman"])
        pacman_pos[pacman_index // 20, pacman_index % 20] = 1.0
        # Channels 5-8: Ghost positions
        ghost_pos = [np.zeros((18, 20), dtype=np.float32) for _ in range(4)]
        for i, ghost in enumerate(game_state["ghosts"]):
            ghost_index = offset(ghost)
            ghost_pos[i][ghost_index // 20, ghost_index % 20] = 1.0

        # Stack all channels into an 8x18x20 tensor
        state = np.stack([walls, dots, pellets, pacman_pos] + ghost_pos, axis=0)
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)  # Add batch dimension
        return state_tensor

    def remember(self, state, action, reward, next_state, done):
        """Store experience in replay memory."""
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        """Choose an action using epsilon-greedy policy."""
        if random.random() < self.epsilon:
            return random.choice(range(4))  # Random action
        with torch.no_grad():
            q_values = self.model(state)
            return torch.argmax(q_values).item()  # Best action

    def replay(self):
        """Train the model using a batch of experiences (Double DQN)."""
        if len(self.memory) < self.BATCH_SIZE:
            return

        # Sample a batch
        batch = random.sample(self.memory, self.BATCH_SIZE)
        states, actions, rewards, next_states, dones = zip(*batch)

        # Convert to tensors
        states = torch.cat(states).to(self.device)
        next_states = torch.cat(next_states).to(self.device)
        actions = torch.LongTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        dones = torch.FloatTensor(dones).to(self.device)

        # Double DQN: Main model selects actions, target model evaluates Q-values
        current_q = self.model(states).gather(1, actions.unsqueeze(1)).squeeze(1)
        next_actions = self.model(next_states).argmax(1).unsqueeze(1)
        next_q = self.target_model(next_states).gather(1, next_actions).squeeze(1)
        target_q = rewards + self.GAMMA * next_q * (1 - dones)

        # Compute loss and optimize
        loss = nn.MSELoss()(current_q, target_q)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Decay epsilon for exploration
        self.epsilon = max(self.EPS_END, self.epsilon * self.EPS_DECAY)

        # Periodically update target network
        self.step_counter += 1
        if self.step_counter % self.TARGET_UPDATE == 0:
            self.target_model.load_state_dict(self.model.state_dict())

    def step(self, game_state):
        """Perform one step: observe, act, and learn."""
        # Preprocess current state
        state = self._preprocess_state(game_state)

        # If we have a previous state, calculate reward and store experience
        if hasattr(self, 'last_state'):
            reward = -0.1  # Small penalty per step
            index = offset(game_state["pacman"])
            if game_state["surrounding"][index] == 1:  # Dot
                reward += 1
            if game_state["surrounding"][index] == 3:  # Pellet
                reward += 100
            if any(abs(self.position - ghost) < 20 for ghost in game_state["ghosts"]):  # Collision with ghost
                reward -= 50
            done = False  # Game end handled externally
            self.remember(self.last_state, self.last_action, reward, state, done)
            self.replay()  # Train on the experience

        # Choose and perform action
        action = self.act(state)
        self.course = self.action_space[action]
        if self.course and self.valid(self.position + self.course):
            self.move(self.course)

        # Store state and action for next step
        self.last_state = state
        self.last_action = action

    def save_model(self, filepath):
        """Save the model weights to a file."""
        torch.save(self.model.state_dict(), filepath)

    def load_model(self, filepath):
        """Load model weights from a file."""
        self.model.load_state_dict(torch.load(filepath))
        self.model.to(self.device)