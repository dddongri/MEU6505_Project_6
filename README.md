# MEU6505 Project 6: Hierarchical LLM-guided Symmetry-aware Bimanual Manipulation

[![IsaacSim](https://img.shields.io/badge/IsaacSim-5.0.0-silver.svg)](https://docs.omniverse.nvidia.com/isaacsim/latest/overview.html)
[![Isaac Lab](https://img.shields.io/badge/IsaacLab-2.2.0-silver)](https://isaac-sim.github.io/IsaacLab)
[![Python](https://img.shields.io/badge/python-3.10+-blue.svg)](https://docs.python.org/3/whatsnew/3.10.html)
[![Linux platform](https://img.shields.io/badge/platform-linux--64-orange.svg)](https://releases.ubuntu.com/22.04/)
[![pre-commit](https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit&logoColor=white)](https://pre-commit.com/)
[![License](https://img.shields.io/badge/license-Apache%202.0-yellow.svg)](https://opensource.org/license/apache-2-0)

> Yonsei University MEU6505 - Optimal Control and Reinforcement Learning Course - Project 6

<p align="center">
  <img src="https://github.com/user-attachments/assets/88dfa8d7-1f10-4c78-999b-fc86a6b95b13" alt="GR1T2 Bimanual Manipulation" width="800"/>
</p>

## 📖 Overview

**Groot** is a hierarchical framework that uses Large Language Models (LLMs) to generate symmetry-aware subgoals and RL controllers to execute them for efficient bimanual manipulation tasks in simulation. Built on Isaac Lab, it targets long-horizon, high-dimensional robotic tasks.

### 🎯 Key Features

- **Hierarchical Control**: LLM-based high-level planning + RL-based low-level execution
- **Symmetry-aware**: Efficient learning leveraging symmetry in bimanual manipulation
- **Isaac Lab Integration**: Built on NVIDIA Isaac Sim's high-performance simulation environment
- **Flexible and Extensible**: Applicable to various robots and tasks

### 🤖 Motivation and Background

- Long-horizon bimanual manipulation tasks (e.g., cloth folding, jar opening) are challenging for standard reinforcement learning due to large state-action spaces
- Existing LLM-based planners (such as LABOR, LLM+MAP) typically rely on scripted skills, limiting flexibility and generalization
- This project develops a hierarchical framework where an LLM generates symmetry-aware subgoals that are executed by RL controllers

### ✨ Contributions

1. **Hierarchical Framework Integrating LLMs and RL**
   We propose a hierarchical control framework that addresses the challenges of long-horizon bimanual tasks and unstable training in end-to-end RL. By leveraging Large Language Models (LLMs) to generate symmetry-aware subgoals and skill sequences, the framework effectively guides low-level RL policies (Pick, Handover, Place), significantly improving task success rates in complex scenarios. To overcome the failure of pure RL in precise bimanual coordination, we adopted a DeepMimic-style approach for the handover subtask.

2. **Efficiency via Symmetry-Based Policy Mirroring (Flipping)**
   We maximize sample efficiency by exploiting the structural symmetry of the humanoid robot. The learned policy is designed to be directly reusable for the opposite arm through state/action flipping, allowing for bi-directional execution without the need for redundant training on both sides.

3. **Optimized State-Based Observation for Robust Generalization**
   To ensure computational efficiency and meet tight schedules, we optimized the observation space to focus on object poses and robot states rather than heavy vision-based inputs (RGB/Point-clouds). This approach enables sample-efficient learning and ensures robust generalization across diverse initial poses and object layouts in the simulation environment.

### 📊 Dataset and Environment

- Utilizes Isaac Lab's simulated bimanual manipulation task environments
- **Input**: Humanoid joint configurations and object position information
- **Output**: Subgoals for RL controllers to execute

## 🏗️ System Architecture & Task Description

This framework integrates a Large Language Model (LLM) with the Isaac Lab simulation environment using the **Model Context Protocol (MCP)** server.

### 🔌 MCP Integration Framework

The system utilizes the Model Context Protocol to establish a standardized connection between the AI assistant and the simulation environment.

- **LLM Provider**: **Claude** (via Anthropic API) acts as the intelligent agent.
- **MCP Server**: Acts as the bridge, exposing simulation state and control interfaces as "tools" to Claude.
- **Workflow**:
    1.  **Observation**: Claude requests the current state of the robot and object via MCP tools.
    2.  **Reasoning**: Based on the user's command (e.g., "Move the beaker"), Claude plans the sequence of skills (Pick -> Transfer -> Place).
    3.  **Execution**: Claude calls the appropriate MCP tool to trigger the RL policy for the chosen skill in Isaac Lab.

### 🧪 Target Task

The primary task demonstrates bimanual manipulation capabilities: **"Grasp a beaker and place it at a desired target location."**

### 🦾 Learned Skills (Primitives)

The robot executes the high-level plan using three core reinforcement learning-trained skills:

1.  **Pick**: Grasping the target object (beaker) securely from the surface.
2.  **Place**: Accurately positioning and releasing the object at the specified target coordinates.
3.  **Hand-to-Hand Transfer**: Passing the object from one hand to the other, enabling the robot to manipulate objects across a wider workspace and leverage symmetry.

## 🧠 Algorithm

This project utilizes **Proximal Policy Optimization (PPO)**, a state-of-the-art on-policy reinforcement learning algorithm, implemented via the `rsl_rl` library. PPO is chosen for its stability, ease of tuning, and sample efficiency compared to other policy gradient methods.

### Key Characteristics

- **Clipped Objective Function**: PPO prevents large policy updates that could destabilize training by clipping the probability ratio between the new and old policies.
- **On-Policy Learning**: It learns from data collected by the current policy, ensuring that the updates are relevant to the agent's current behavior.
- **Actor-Critic Architecture**: Uses two networks: an *Actor* that decides which action to take, and a *Critic* that estimates the value of the state.

### Mathematical Formulation

The core objective function of PPO is defined as:

$$
L^{CLIP}(\theta) = \hat{\mathbb{E}}_t \left[ \min(r_t(\theta)\hat{A}_t, \text{clip}(r_t(\theta), 1-\epsilon, 1+\epsilon)\hat{A}_t) \right]
$$

Where:
- $\theta$ is the policy parameter.
- $r_t(\theta) = \frac{\pi_\theta(a_t|s_t)}{\pi_{\theta_{old}}(a_t|s_t)}$ is the probability ratio.
- $\hat{A}_t$ is the estimated advantage at time $t$.
- $\epsilon$ is a hyperparameter (usually 0.1 or 0.2) that defines the clipping range.

This objective encourages the new policy to improve upon the old one while staying within a "trust region" to prevent performance collapse.

### 🎭 Motion Imitation (Mimic)

To achieve natural and physically plausible motions, we employ a **DeepMimic-style** motion tracking approach. The goal is to train the policy to reproduce a reference motion (e.g., human motion capture data) as closely as possible in the physics simulation.

#### Reward Structure

The total reward $r_t$ is a weighted sum of individual reward terms designed to encourage tracking accuracy:

$$
r_t = w_{p} r_t^p + w_{q} r_t^q + w_{v} r_t^v + w_{\omega} r_t^{\omega}
$$

Each term is typically formulated as an exponential kernel of the error:

$$
r_t^x = \exp(-k_x \| x_{student} - x_{teacher} \|^2)
$$

Where:
- $r_t^p$: **Joint Position Reward** (penalizes deviation in joint angles)
- $r_t^q$: **Joint Orientation Reward** (penalizes deviation in body/link orientations)
- $r_t^v$: **Linear Velocity Reward** (penalizes deviation in end-effector/body velocities)
- $r_t^{\omega}$: **Angular Velocity Reward** (penalizes deviation in angular velocities)

This formulation ensures that the reward is maximized (close to 1) when the error is zero and decays smoothly as the error increases, providing dense and stable feedback for the RL agent.



## 🚀 Quick Start

### Prerequisites

- Ubuntu 22.04 LTS
- Python 3.10 or higher
- NVIDIA GPU (RTX series recommended)
- CUDA 11.8 or higher

### Installation

**Step 1.** Install Isaac Lab

Follow the official Isaac Lab installation guide: [Installation Guide](https://isaac-sim.github.io/IsaacLab/source/setup/installation/index.html)

**Step 2.** Clone the repository

```bash
git clone --recursive https://github.com/dddongri/MEU6505_Project_6.git
cd MEU6505_Project_6
```

**Step 3.** Install Groot library

Run the following command in the Python environment where Isaac Lab is installed:

```bash
cd source/groot
pip install -e .
```

**Step 4.** (Optional) Install pre-commit hooks

To automate code formatting and linting:

```bash
pip install pre-commit
pre-commit install
```

---

## 💻 Usage

### List Available Environments

Check all available environments:

```bash
python scripts/list_envs.py
```

**Available Tasks:**
- `Template-Isaac-Velocity-Flat-Anymal-D-v0` - Anymal-D robot flat terrain locomotion
- `Template-Isaac-Velocity-Rough-Anymal-D-v0` - Anymal-D robot rough terrain locomotion
- **`GR1T2-Basic`** - GR1T2 humanoid basic bimanual manipulation
- **`GR1T2-PickPlace`** - GR1T2 pick-and-place task
- **`GR1T2-HandToHand`** - GR1T2 hand-to-hand object transfer task
- **`GROOT-Mimic`** - GR1T2 motion tracking (mimic) task

### Training

Start reinforcement learning training for a specific task:

```bash
# Basic training
python scripts/rsl_rl/train.py --task TASK_NAME

# Headless mode (without GUI)
python scripts/rsl_rl/train.py --task TASK_NAME --headless

# Example: Train GR1T2 pick-and-place task
python scripts/rsl_rl/train.py --task GR1T2-PickPlace --headless
```

**Key Training Options:**
- `--task`: Name of the task to train
- `--headless`: Train without GUI (recommended for server environments)
- `--num_envs`: Number of parallel environments (default: varies by task)

> **💡 Tip:** Groot is compatible with `rsl_rl`, `Stable Baselines3`, and custom reinforcement learning algorithms.

#### Motion Tracking (Mimic) Training

For the motion tracking task (`GROOT-Mimic`), use the dedicated scripts:

```bash
# Train Mimic Policy
python scripts/rsl_rl/train_mimic.py --task GROOT-Mimic --headless
```

### Evaluation

Evaluate trained models:

```bash
# Automatically load the latest log directory
python scripts/rsl_rl/play.py --task TASK_NAME

# Specify a particular log directory
python scripts/rsl_rl/play.py --task TASK_NAME --log_dir PATH_TO_LOG --num_envs NUM_ENVS

# Example
python scripts/rsl_rl/play.py --task GR1T2-PickPlace --num_envs 16
```

#### Motion Tracking (Mimic) Evaluation

To visualize the trained mimic policy:

```bash
python scripts/rsl_rl/play_mimic.py --task GROOT-Mimic-Play
```

### Motion Data Processing

Scripts for processing motion data are located in `scripts/animation_motions/`:

- `csv_to_npz.py`: Converts motion data from CSV to NPZ format.
- `replay_motion.py`: Replays the processed motion data.

> **💡 Tip:** If `--log_dir` is not specified, the latest log is automatically loaded!

### Training Monitoring

Monitor training logs in real-time via TensorBoard:

```bash
tensorboard --logdir logs/rsl_rl/TASK_NAME/
```

Access `http://localhost:6006` in your browser to view training progress.

---

## 📁 Project Structure

```
MEU6505_Project_6/
├── source/groot/              # Groot library main code
│   ├── groot/
│   │   └── tasks/            # Task definitions
│   │       ├── locomotion/   # Locomotion tasks (Anymal-D)
│   │       └── manipulation/ # Manipulation tasks (GR1T2)
│   │           ├── pick_place/      # Pick-and-place
│   │           └── hand_to_hand/    # Hand-to-hand transfer
│   ├── config/               # Extension configuration
│   └── setup.py              # Installation script
├── scripts/
│   ├── rsl_rl/              # RSL-RL training/evaluation scripts
│   │   ├── train.py         # Training script
│   │   └── play.py          # Evaluation script
│   ├── list_envs.py         # List environments
│   └── rename_template.py   # Rename template
├── docs/                     # Documentation and images
├── logs/                     # Training logs (generated)
├── README.md                 # This file
└── README_setup.md          # Detailed setup guide
```

---

## 🛠️ Code Formatting

This project uses pre-commit to automatically manage code style.

### Installing and Using Pre-commit

```bash
# Install pre-commit
pip install pre-commit

# Install pre-commit hooks
pre-commit install

# Manually run on all files
pre-commit run --all-files
```

Code formatting and linting will be performed automatically on commit.

---

## Results
https://github.com/user-attachments/assets/21962bb4-2b13-44d9-8068-15ecebc05828

https://github.com/user-attachments/assets/499807a0-a2ce-4b77-b3db-b387dd2a092b

---


## 🗣️ Discussion

### ⚖️ Comparative Analysis: Pure RL vs. Motion Imitation

| Feature | Pure RL (End-to-End) | Motion Imitation (DeepMimic) |
| :--- | :--- | :--- |
| **Motion Quality** | Often unnatural, jittery, or exploits physics quirks | Natural, smooth, and physically plausible (human-like) |
| **Exploration** | Hard to explore complex coordination (sparse reward problem) | Guided exploration via reference motion (dense reward) |
| **Sample Efficiency** | Low (requires millions of steps to find solution) | High (reference motion narrows search space) |
| **Robustness** | Can be brittle to dynamics changes | Generally more robust due to structured motion priors |
| **Task Applicability** | Simple, single-objective tasks | Complex, coordinated tasks (e.g., bimanual handover) |

### 📉 Why Pure RL Failed in Bimanual Handover?

In our experiments, training the **Hand-to-Hand Transfer** skill using pure RL (without motion priors) proved to be extremely challenging. The key reasons for this failure include:

1.  **High-Dimensional Coordination**: Bimanual manipulation requires precise synchronization between two 7-DOF arms and dexterous hands. The joint state space is too vast for random exploration to effectively traverse.
2.  **Sparse Reward Landscape**: The successful handover of an object is a "sparse" event. Without intermediate guidance (like a reference motion), the agent rarely stumbles upon the exact coordination needed to pass the object without dropping it.
3.  **Unnatural Postures**: Even when pure RL manages to transfer the object, it often adopts awkward or physically infeasible postures that are unstable and difficult to transition to subsequent tasks (like placing).
4.  **Local Optima**: The agent tends to get stuck in local optima, such as holding the object with one hand and refusing to attempt the risky transfer action to avoid the penalty of dropping it.

By adopting the **DeepMimic** approach, we provide the agent with a "template" of how a successful handover looks, transforming the problem from *exploration* to *tracking*, which significantly improves learning stability and success rates.

### 🔄 Impact of Symmetry on Training Efficiency

Leveraging the bilateral symmetry of the humanoid robot proved to be a crucial factor in efficient learning.

- **Data Augmentation**: By mirroring states and actions, we effectively double the amount of experience gathered from each episode. A successful trajectory for the left arm provides a valid training signal for the right arm (and vice versa).
- **Reduced Exploration Space**: The agent learns a unified policy that generalizes across both sides, rather than learning separate policies for each arm. This significantly reduces the dimensionality of the effective search space.
- **Result**: We observed faster convergence rates and more consistent behavior between the left and right arms compared to non-symmetry-aware baselines.

### 🧠 LLM Integration Analysis: Role and Limitations

The integration of a Large Language Model (LLM) as a high-level planner brings both significant advantages and unique challenges to the robotic control pipeline.

**Advantages:**
- **Flexibility & Natural Language Understanding**: The LLM can interpret vague or complex user commands (e.g., "Move the beaker to the far right") and translate them into a structured sequence of skills without requiring hard-coded rules for every possible scenario.
- **Context Awareness**: It can maintain context over a long horizon, understanding that a "Place" action must be preceded by a "Pick" action.

**Limitations & Challenges:**
- **Hallucination**: LLMs can sometimes generate plausible but incorrect plans, such as inventing non-existent skills or assuming the robot can reach physically impossible locations.
- **Physical Grounding**: The LLM lacks an inherent understanding of physics (e.g., collision, gravity), which can lead to plans that are logically sound but physically infeasible.

**Role of MCP (Model Context Protocol):**
To mitigate these issues, the **MCP server** acts as a crucial grounding layer. By exposing the simulation state and available skills as strictly defined "tools," MCP constrains the LLM's output space.
- **State Verification**: Before planning, the LLM is forced to query the actual robot/object state via MCP, reducing hallucinations based on incorrect assumptions.
- **Structured Execution**: Instead of generating free-form text, the LLM must call specific MCP tools (e.g., `execute_pick_skill`), ensuring that only valid, pre-trained skills are triggered.

### 🔮 Future Work: Sim-to-Real Transfer

While our framework demonstrates robust performance in the Isaac Lab simulation, deploying it to the physical GR1T2 robot presents several challenges:

- **Dynamics Mismatch**: Discrepancies in friction, mass distribution, and actuator dynamics between the simulation and the real world.
- **Sensor Noise**: Real-world sensors (joint encoders, IMUs) are noisy, and perfect object pose estimation (assumed in simulation) is difficult to achieve with vision systems.

**Planned Mitigation Strategies:**
1.  **Domain Randomization**: Randomizing physical parameters (mass, friction, damping) during training to make the policy robust to variations.
2.  **System Identification**: Fine-tuning simulation parameters to better match the real robot's behavior.
3.  **Vision-Based Policy**: Transitioning from state-based observations to direct visual inputs (RGB-D) to reduce reliance on precise object pose estimation.


## 📚 References

### Related Papers

1. T. Z. Zhao, et al., "Learning to Acquire Novel Bimanual Object Manipulation through Large Language Models," *arXiv preprint arXiv:2404.02018*, 2024.
2. Y. Shao and C. Xiao, "Bimanual grasp synthesis for dexterous robot hands," *IEEE Robotics and Automation Letters*, 2024.
3. B. Zhou, H. Yuan, Y. Fu, and Z. Lu, "Learning diverse bimanual dexterous manipulation skills from human demonstrations," *arXiv preprint arXiv:2410.02477*, 2024.
4. B. Huang, Y. Chen, T. Wang, Y. Qin, Y. Yang, N. Atanasov, and X. Wang, "Dynamic handover: Throw and catch with bimanual hands," *arXiv preprint arXiv:2309.05655*, 2023.
5. S. Wang, L. Sun, F. Zha, W. Guo, and P. Wang, "Learning adaptive reaching and pushing skills using contact information," *Frontiers in Neurorobotics*, vol. 17, p. 1271607, 2023.
6. H. Zhou and X. Lin, "Intelligent redundant manipulation for long-horizon operations with multiple goal-conditioned hierarchical learning," *Advanced Robotics*, vol. 39, no. 6, pp. 291–304, 2025.

### Technologies Used
- [NVIDIA Isaac Sim](https://docs.omniverse.nvidia.com/isaacsim/latest/overview.html) - High-performance robot simulation
- [Isaac Lab](https://isaac-sim.github.io/IsaacLab) - Reinforcement learning environment framework
- [RSL-RL](https://github.com/leggedrobotics/rsl_rl) - Reinforcement learning library
    - [PPO (Proximal Policy Optimization)](https://arxiv.org/abs/1707.06347) - The core reinforcement learning algorithm used for training skills
- [Model Context Protocol (MCP)](https://modelcontextprotocol.io/) - Standardized protocol for connecting AI models to external systems
- [Claude](https://www.anthropic.com/claude) - Advanced AI assistant by Anthropic

---

## 👥 Contributors

- **Author**: [Donghyun Lee](https://github.com/dddongri), [Sol Choi](https://github.com/S-CHOI-S), [Seungyeon Lee](https://github.com/LEEcat01081), [Kyungjae Bang](https://github.com/SalmonHan)
- **Course**: MEU6505 - Optimal Control and Reinforcement Learning
- **Institution**: Yonsei University
---
