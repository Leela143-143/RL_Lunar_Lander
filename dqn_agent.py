import gymnasium as gym
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
import collections

# --- Hyperparameters ---
LEARNING_RATE = 0.0003     # Changed from 0.0005 (Slower, more stable learning)
GAMMA = 0.99
BUFFER_SIZE = 50000
BATCH_SIZE = 64
EPSILON_START = 1.0
EPSILON_END = 0.01
EPSILON_DECAY = 0.998      # Changed from 0.995 (Explore for longer)
TARGET_UPDATE = 10

# --- 1. The Neural Network (Q-Network) ---
class QNetwork(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(QNetwork, self).__init__()
        # Changed 64 to 128 for more capacity
        self.fc1 = nn.Linear(state_dim, 128)  
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, action_dim)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)

# --- 2. Replay Buffer (Memory) ---
class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = collections.deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = zip(*batch)
        return state, action, reward, next_state, done

    def __len__(self):
        return len(self.buffer)

# --- 3. The DQN Agent ---
class DQNAgent:
    def __init__(self, state_dim, action_dim):
        self.state_dim = state_dim
        self.action_dim = action_dim
        
        # Policy Network (The one we train)
        self.policy_net = QNetwork(state_dim, action_dim)
        # Target Network (The stable reference)
        self.target_net = QNetwork(state_dim, action_dim)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval() # Target net is not trained directly

        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=LEARNING_RATE)
        self.memory = ReplayBuffer(BUFFER_SIZE)
        self.epsilon = EPSILON_START

    def select_action(self, state):
        # Epsilon-Greedy Strategy
        if random.random() < self.epsilon:
            return random.randint(0, self.action_dim - 1) # Explore
        else:
            with torch.no_grad():
                state_tensor = torch.FloatTensor(state).unsqueeze(0)
                q_values = self.policy_net(state_tensor)
                return q_values.argmax().item() # Exploit

    def train_step(self):
        if len(self.memory) < BATCH_SIZE:
            return

        # Sample a batch from memory
        states, actions, rewards, next_states, dones = self.memory.sample(BATCH_SIZE)

        # Convert to PyTorch tensors
        states = torch.FloatTensor(np.array(states))
        actions = torch.LongTensor(actions).unsqueeze(1)
        rewards = torch.FloatTensor(rewards).unsqueeze(1)
        next_states = torch.FloatTensor(np.array(next_states))
        dones = torch.FloatTensor(dones).unsqueeze(1)

        # Calculate Q(s, a)
        current_q = self.policy_net(states).gather(1, actions)

        # Calculate Target Q using Target Network: R + gamma * max(Q(s', a'))
        with torch.no_grad():
            next_q = self.target_net(next_states).max(1)[0].unsqueeze(1)
            target_q = rewards + (GAMMA * next_q * (1 - dones))

        # Loss Calculation (MSE)
        loss = nn.MSELoss()(current_q, target_q)

        # Optimization Step (Backpropagation)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def update_epsilon(self):
        self.epsilon = max(EPSILON_END, self.epsilon * EPSILON_DECAY)

    def update_target_network(self):
        self.target_net.load_state_dict(self.policy_net.state_dict())

# --- 4. Main Training Loop ---
if __name__ == "__main__":
    env = gym.make("LunarLander-v3") # No video during training for speed
    
    # Get dimensions
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    
    agent = DQNAgent(state_dim, action_dim)
    
    num_episodes = 1000
    scores = []

    print(f"Training DQN on {env.spec.id} for {num_episodes} episodes...")

    for episode in range(num_episodes):
        state, info = env.reset()
        total_reward = 0
        done = False
        truncated = False

        while not (done or truncated):
            action = agent.select_action(state)
            next_state, reward, done, truncated, info = env.step(action)
            
            # Store experience
            agent.memory.push(state, action, reward, next_state, done or truncated)
            
            # Train the network
            agent.train_step()
            
            state = next_state
            total_reward += reward

        # Update hyperparameters
        agent.update_epsilon()
        if episode % TARGET_UPDATE == 0:
            agent.update_target_network()

        scores.append(total_reward)
        avg_score = np.mean(scores[-100:])

        print(f"Episode {episode+1}: Score = {total_reward:.2f}, Avg Score (last 100) = {avg_score:.2f}, Epsilon = {agent.epsilon:.2f}")

        # Stop early if solved
        if avg_score >= 200:
            print(f"\nSolved in {episode+1} episodes!")
            torch.save(agent.policy_net.state_dict(), "dqn_lunar_lander.pth")
            break

    env.close()
    
    # Save the trained model if not already saved
    if avg_score < 200:
        torch.save(agent.policy_net.state_dict(), "dqn_lunar_lander.pth")
        print("Model saved as dqn_lunar_lander.pth")

    # --- 5. Test Run (Video Recording) ---
    print("\nStarting Test Run with Video Recording...")
    test_env = gym.make("LunarLander-v3", render_mode="rgb_array")
    test_env = gym.wrappers.RecordVideo(test_env, video_folder="./video_dqn", name_prefix="dqn_agent")
    
    state, info = test_env.reset()
    done = False
    truncated = False
    
    while not (done or truncated):
        # Force Exploitation (No Randomness)
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        action = agent.policy_net(state_tensor).argmax().item()
        state, reward, done, truncated, info = test_env.step(action)
        
    test_env.close()
    print("Test run complete. Check ./video_dqn for the video.")