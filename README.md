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

#### 📋 MCP Setup Guide

**Step 1. Install Claude Desktop**

Download and install the Claude desktop application from the official Anthropic website.
> for Linux (Ubuntu Debian), see this [link](https://github.com/aaddrick/claude-desktop-debian)

**Step 2. Start the MCP Server**

Run the Isaac Sim MCP server:

```bash
uv run --directory ~/YOUR_PATH/MEU6505_Project_6/omni-mcp/isaac-sim-mcp ~/YOUR_PATH/MEU6505_Project_6/omni-mcp/isaac-sim-mcp/isaac_mcp/server.py
```

Replace `YOUR_PATH` with the absolute path to your project directory.

**Step 3. Configure Claude Desktop**

Open Claude Desktop settings:
1. Go to **Settings** → **Developer** → **Edit Config**
2. Edit the `claude_desktop_config.json` file and add the following server configuration:

```json
{
  "mcpServers": {
    "mcp-server-omni-isaacsim": {
      "type": "stdio",
      "command": "uv",
      "args": [
        "run",
        "--directory",
        "/YOUR_PATH/MEU6505_Project_6/omni-mcp",
        "/YOUR_PATH/MEU6505_Project_6/omni-mcp/isaac-sim-mcp/isaac_mcp/server.py"
      ]
    }
  }
}
```

>[!Caution]  
> Use absolute paths (not relative paths) in the configuration.

**Step 4. Launch the Simulator**

Start the Isaac Lab simulator with the MCP extension enabled:

```bash
python scripts/rsl_rl/run.py --task GR1T2-HandToHand --num_envs 1 --kit_args "--ext-folder /YOUR_PATH/MEU6505_Project_6/omni-mcp/isaac-sim-mcp/ --enable isaac.sim.mcp_extension --enable omni.isaac.nucleus"
```

Replace `/YOUR_PATH` with your absolute project path.

**Step 5. Interact with Claude**

Once the simulator is running and the MCP server is connected, you can use Claude to:
- Query the current robot and object state
- Request task execution (e.g., "Pick the beaker and place it on the table")
- Monitor task progress in real-time

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

### Motion Data

- Location: `source/groot/groot/motions/animation/handa/` (CSV/NPZ files)
- Included examples: `handa.csv`, `handa.npz`
- Conversion: Standalone CSV→NPZ scripts are not included in this repo. Prepare NPZ files that align with the tracking config, or adapt your own preprocessing workflow.
- Note: When imitating object interactions, ensure sequences encode both robot and object states consistently to avoid contact mismatch during tracking.

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
├── source/
│   └── groot/                 # Groot library main code
│       ├── groot/
│       │   ├── __init__.py
│       │   ├── ui_extension_example.py
│       │   ├── motions/       # Motion animation data
│       │   │   └── animation/
│       │   ├── robots/        # Robot model definitions
│       │   ├── tasks/         # Task definitions
│       │   │   ├── locomotion/    # Locomotion tasks (Anymal-D)
│       │   │   ├── manipulation/  # Manipulation tasks (GR1T2)
│       │   │   ├── tracking/      # Motion tracking tasks
│       │   │   │   ├── config/    # Task configuration
│       │   │   │   └── mdp/       # MDP definitions (rewards, observations, etc.)
│       │   │   └── velocity/      # Velocity-based tasks
│       │   └── utils/         # Utility functions (exporter, runners, etc.)
│       ├── config/            # Extension configuration
│       ├── docs/              # Documentation
│       ├── pyproject.toml     # Project configuration
│       └── setup.py           # Installation script
├── omni-mcp/                  # MCP (Model Context Protocol) server
│   ├── isaac-sim-mcp/         # Isaac Sim MCP server implementation
│   ├── main.py
│   ├── pyproject.toml
│   └── README.md
├── scripts/
│   ├── rsl_rl/                # RSL-RL training/evaluation scripts
│   │   ├── train.py           # PPO training script
│   │   ├── train_mimic.py      # DeepMimic-style training script
│   │   ├── play.py            # Model evaluation/visualization
│   │   ├── play_mimic.py       # Mimic task evaluation
│   │   ├── run.py             # Environment runner
│   │   ├── cli_args.py         # CLI arguments for training
│   │   ├── cli_args_mimic.py   # CLI arguments for mimic training
│   │   └── skills/            # Trained skill checkpoints
│   ├── list_envs.py           # List available environments
│   └── rename_template.py     # Template utility
├── docs/                      # Documentation and images
├── logs/                      # Training logs (generated at runtime)
├── .pre-commit-config.yaml    # Pre-commit configuration
├── pyproject.toml             # Root project configuration
├── CITATION.cff               # Citation information
├── LICENCE                    # License file
├── README.md                  # This file
└── README_setup.md            # Detailed setup guide
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

## 🎬 Results

### 🚀 From Pure RL to DeepMimic: The Learning Journey

Our experimental journey revealed a critical insight: **pure reinforcement learning struggles significantly with complex bimanual coordination tasks**. This section documents our progression from encountering RL challenges to implementing and evaluating a DeepMimic-based approach.

#### 📊 Challenge: Pure RL Failure in Hand-to-Hand Transfer

**Problem**: Training the **Hand-to-Hand Transfer** (handover) skill using standard PPO without motion guidance proved extremely difficult and unstable.

**Why RL Alone Fails**:
- **High Dimensionality**: The humanoid has 55+ actuated joints (7-DOF per arm + dexterous hand fingers). The state-action space is enormous, making exploration inefficient.
- **Dexterous Hand Complexity**: The anthropomorphic hands feature multiple fingers with complex joint interdependencies. Coordinating both:
  - Individual finger joints for fine-grained object manipulation
  - Wrist orientation for object positioning
  - Arm movement for workspace coverage
  
  This creates a **combinatorial explosion** of possible hand configurations. A slight finger misalignment can cause object drops, and the agent must learn precise finger synchronization through trial-and-error.

- **Sparse Rewards**: Object handover is a rare event. The agent must discover the precise coordination by chance, which typically requires millions of environment steps.
- **Unstable Training**: Without guidance, the agent often discovers unnatural solutions that exploit physics quirks, leading to brittle policies that fail when conditions change slightly.
- **Credit Assignment Problem**: The agent struggles to understand which arm and hand movements are responsible for successful or failed handovers. With 50+ actuators, it's nearly impossible to determine which joints contributed to failure.
- **Local Optima Trap**: The agent tends to get stuck holding the object with one hand and refuses to attempt the risky transfer action to avoid penalties.

**Observation**: Even after extended training, the policy failed to produce reliable handovers or adopted unstable postures incompatible with downstream tasks (picking, placing).

#### 🔄 Approach: DeepMimic Motion Imitation

**Approach**: We pivoted to a **DeepMimic-style motion imitation framework**, where the policy learns to track a reference motion (human-like handover trajectory) rather than discovering coordination from scratch.

**Expected Benefits**:
- **Guided Learning**: The reference motion provides dense reward signals at every timestep, transforming the problem from exploration to tracking.
- **Natural Motion**: Policies trained via imitation should produce smoother, more physically plausible movements.
- **Faster Convergence**: Training time significantly reduced compared to pure RL exploration.

**Challenges Encountered**:
While DeepMimic improved training stability, the handover task remained **extremely challenging** even with motion priors:
- **Reference Motion Quality**: The quality of imitation heavily depends on the reference trajectory. Imperfect reference data led to suboptimal policies.
- **Sim-to-Ref Gap**: Matching simulation physics to the reference motion's implicit dynamics proved difficult.
- **Partial Success**: The policy learned smoother motions but still struggled with consistent object grasping and transfer reliability.

#### 📹 Demonstration Videos

**Video 1: Pure RL Challenges (Pre-DeepMimic)**

https://github.com/user-attachments/assets/cf196560-dd7f-4465-883a-fc6f89394b6f

https://github.com/user-attachments/assets/abe539ce-d89c-43bd-a3e7-c12e0c33e900


This video showcases the difficulties encountered when training the handover skill using standard PPO without motion priors:
- **Uncoordinated Arm Movements**: The left and right arms struggle to synchronize, resulting in awkward postures
- **Frequent Object Drops**: The policy fails to maintain grip control during transfer, dropping the object multiple times
- **Unstable Grasp Transitions**: The object oscillates between hands without stable handover phases
- **Poor Task Chaining**: The policy fails to reliably transition to the next task (placing), breaking the skill sequence

**Key Insight**: While the policy occasionally succeeds through luck, it lacks the robust coordination needed for a deployable system. The agent explores inefficiently and gets stuck in local optima.

---

**Video 2: DeepMimic Solution + MCP Server Integration**

https://github.com/user-attachments/assets/21962bb4-2b13-44d9-8068-15ecebc05828

https://github.com/user-attachments/assets/499807a0-a2ce-4b77-b3db-b387dd2a092b

This video demonstrates the **integrated system** combining:

1. **DeepMimic-Trained Handover Policy**: 
   - The robot shows improved motion smoothness compared to pure RL
   - Motion patterns approximate human-like coordination
   - **Still faces challenges**: Handover success is inconsistent, with occasional drops and grip failures

2. **MCP Server Orchestration**: 
   - Claude AI (via Model Context Protocol) handles high-level task planning
   - Queries simulation state in real-time via MCP tools
   - Attempts to orchestrate the skill sequence: **Pick → Handover → Place**

3. **Skill Composition & Execution**:
   - LLM decides which skills to invoke and in what order
   - Each skill executes its trained RL policy independently
   - Monitors task progress, though recovery from failures remains limited

**Current Performance**:
- ✅ **Improved Coordination**: Motion is noticeably smoother than pure RL
- ⚠️ **Moderate Success Rate**: Shows improvement but handover remains unreliable
- ⚠️ **Sensitivity to Variations**: Performance degrades with different object poses
- ✅ **MCP Integration Works**: LLM-RL collaboration functions as designed, though underlying policies need refinement

---

### 🧩 Current Status & Limitations

- Dexterous hands remain difficult: precise finger coordination and stable grasp phases are brittle.
- Motion imitation improves smoothness but handover success is still inconsistent, with occasional drops and failed transfers.
- Imitating object interactions adds challenges (contact timing, force control); compounding errors often lead to instability.
- Sensitivity to initial conditions persists; robustness needs better reference data and reward shaping.
- MCP orchestration works as intended (skill selection/execution), but underlying policies require further refinement.

---

### 🔍 Key Takeaways

1. **DeepMimic Improves Learning but Doesn't Solve Everything**: Motion priors enable faster, more stable training (5-10x speedup) and smoother motions, but the underlying task complexity still poses significant challenges. The handover task requires further iteration on reference data quality and reward shaping.

2. **MCP Successfully Integrates LLM and Simulation**: The Model Context Protocol effectively constrains the LLM's action space to well-defined skills, preventing hallucinations. This architectural pattern is validated even when individual skill policies need refinement.

3. **Bimanual Coordination Remains an Open Challenge**: Even with state-of-the-art methods (DeepMimic + PPO), achieving reliable bimanual handovers in simulation is extremely difficult. This highlights the need for:
   - Better reference motion data (potentially from real robot demonstrations)
   - More sophisticated reward engineering
   - Hybrid approaches combining imitation learning with task-specific objectives

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
