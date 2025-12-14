import gymnasium as gym
import torch
import torch.nn as nn
import numpy as np

# --- 1. Define the Neural Network ---
class QNetwork(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(QNetwork, self).__init__()
        # Ensure this matches your trained model (64 or 128)
        self.fc1 = nn.Linear(state_dim, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, action_dim)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)

def save_agent_video():
    # --- 2. Setup Environment for Recording ---
    # We use "rgb_array" so the computer can 'see' the pixels to save them
    env = gym.make("LunarLander-v3", render_mode="rgb_array")
    
    # Wrap the environment to record the video
    env = gym.wrappers.RecordVideo(
        env, 
        video_folder="./video_final_result", 
        name_prefix="final_agent_gameplay",
        episode_trigger=lambda x: True # Record every episode in this loop
    )
    
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    # --- 3. Load the Trained Model ---
    model = QNetwork(state_dim, action_dim)
    
    try:
        # Load weights (CPU compatible)
        model.load_state_dict(torch.load("dqn_lunar_lander.pth", map_location=torch.device('cpu')))
        model.eval()
        print("Model loaded. Starting recording...")
    except FileNotFoundError:
        print("Error: 'dqn_lunar_lander.pth' not found.")
        return

    # --- 4. Run Loop (Record 3 Episodes) ---
    num_episodes = 3
    
    for episode in range(1, num_episodes + 1):
        state, info = env.reset()
        done = False
        truncated = False
        total_reward = 0
        
        while not (done or truncated):
            # Select Best Action
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            with torch.no_grad():
                action = model(state_tensor).argmax().item()
            
            # Step environment
            state, reward, done, truncated, info = env.step(action)
            total_reward += reward

        print(f"Episode {episode} recorded. Score: {total_reward:.2f}")

    env.close()
    print("\nDone! Videos saved in './video_final_result' folder.")

if __name__ == "__main__":
    save_agent_video()