import gymnasium as gym
import numpy as np

def run_simulation():
    # Setup the LunarLander environment
    # render_mode="rgb_array" allows us to capture frames for video
    env = gym.make("LunarLander-v3", render_mode="rgb_array")
    
    # Wrap the env to record video (Proves the simulator works)
    env = gym.wrappers.RecordVideo(
        env, 
        video_folder="./video_output", 
        episode_trigger=lambda x: True, 
        name_prefix="task2_random_run"
    )

    print("--- Task 2: Simulator Initialization ---")
    print(f"Observation Space: {env.observation_space}")
    print(f"Action Space: {env.action_space}")

    # Run 3 Episodes with a Random Policy
    for episode in range(1, 4):
        state, info = env.reset()
        done = False
        truncated = False
        total_reward = 0
        steps = 0
        
        while not (done or truncated):
            # Random Action (The baseline policy)
            action = env.action_space.sample() 
            next_state, reward, done, truncated, info = env.step(action)
            total_reward += reward
            steps += 1
            
        print(f"Episode {episode}: Steps={steps}, Reward={total_reward:.2f}")

    env.close()
    print("Simulation Complete. Video saved in 'video_output' folder.")

if __name__ == "__main__":
    run_simulation()