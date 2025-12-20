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

---

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
https://github.com/user-attachments/assets/499807a0-a2ce-4b77-b3db-b387dd2a092b

---


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
