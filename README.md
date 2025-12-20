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

### 📊 Dataset and Environment

- Utilizes Isaac Lab's simulated bimanual manipulation task environments
- **Input**: Humanoid joint configurations and object position information
- **Output**: Subgoals for RL controllers to execute

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
python -m pip install -e .
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

## 📚 References

### Related Papers
- **LABOR**: [Learning to Acquire Novel Bimanual Object Manipulation through Large Language Models](https://arxiv.org/pdf/2404.02018)
- **Isaac Lab**: [Documentation](https://isaac-sim.github.io/IsaacLab/main/source/overview/environments.html)

### Technologies Used
- [NVIDIA Isaac Sim](https://docs.omniverse.nvidia.com/isaacsim/latest/overview.html) - High-performance robot simulation
- [Isaac Lab](https://isaac-sim.github.io/IsaacLab) - Reinforcement learning environment framework
- [RSL-RL](https://github.com/leggedrobotics/rsl_rl) - Reinforcement learning library

---

## 👥 Contributors

- **Author**: [Sol Choi](https://github.com/S-CHOI-S)
- **Course**: MEU6505 - Optimal Control and Reinforcement Learning
- **Institution**: Yonsei University, Department of Mechanical Engineering

---

## 📄 License

This project is distributed under the Apache License 2.0. See the [LICENSE](LICENCE) file for details.

```
Copyright 2024 The Isaac Lab Project Developers

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
```

---

## 🙏 Acknowledgments

This project is built upon:
- [Isaac Lab Extension Template](https://github.com/isaac-sim/IsaacLabExtensionTemplate)
- NVIDIA Isaac Lab framework
- Yonsei University MEU6505 course

---

## 📞 Contact

For questions or issues about the project:
- GitHub Issues: [Create an issue](https://github.com/dddongri/MEU6505_Project_6/issues)
- Author: [@S-CHOI-S](https://github.com/S-CHOI-S)

---

<p align="center">
  Made with ❤️ for MEU6505 Optimal Control and Reinforcement Learning
</p>
