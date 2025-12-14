# RL_Lunar_Lander
## 🧠 Model Architecture & Methodology

To solve the LunarLander-v3 environment, we implemented a **Deep Q-Network (DQN)**, a model-free, off-policy Reinforcement Learning algorithm. This approach approximates the optimal Action-Value function $Q^*(s, a)$ using a neural network.

### 1. Neural Network Architecture
The agent uses a Multi-Layer Perceptron (MLP) implemented in PyTorch:
* **Input Layer (8 Neurons):** Accepts the state vector representing the lander's physics:
    * Coordinates $(x, y)$
    * Velocities $(v_x, v_y)$
    * Angle and Angular Velocity $(\theta, \omega)$
    * Leg contact sensors (Left, Right)
* **Hidden Layers:** Two fully connected layers with **64 neurons** each.
    * **Activation:** Rectified Linear Unit (ReLU) for non-linearity.
* **Output Layer (4 Neurons):** Outputs the Q-values for the discrete action space:
    * 0: Do Nothing
    * 1: Fire Left Orientation Engine
    * 2: Fire Main Engine
    * 3: Fire Right Orientation Engine

### 2. Algorithm Enhancements
To ensure stable training (overcoming the instability of naive Q-Learning), we incorporated:
* **Experience Replay Buffer:** Stores transitions $(s, a, r, s', done)$ in a cyclic buffer (Size: 50,000). The agent trains on random mini-batches, breaking the correlation between consecutive time steps.
* **Target Network:** A clone of the policy network that is frozen and updated only every `C` steps (C=10 episodes). This provides a stable target for the loss calculation, preventing the "moving target" problem.

### 3. Hyperparameters
The following hyperparameters were tuned for convergence:

| Parameter | Value | Description |
| :--- | :--- | :--- |
| **Learning Rate** | `0.0003` | Adam Optimizer step size |
| **Gamma ($\gamma$)** | `0.99` | Discount factor for future rewards |
| **Batch Size** | `64` | Samples per training step |
| **Buffer Size** | `50,000` | Max stored experiences |
| **Epsilon Start** | `1.0` | Initial exploration rate (100% random) |
| **Epsilon Decay** | `0.995` | Decay rate per episode |
| **Epsilon Min** | `0.01` | Minimum exploration (1%) |