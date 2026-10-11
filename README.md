# ColdChain Multi-AGV RL
### Risk-Aware Multi-AGV Path Planning for Cold-Chain Warehouses Using Reinforcement Learning

A simulation-based research project exploring how multiple Automated Guided Vehicles (AGVs) can coordinate warehouse transportation while accounting for dynamic thermal conditions, environmental risk, product urgency, and proactive rescue.

**Core focus:** Reinforcement Learning · Multi-AGV Coordination · Cold-Chain Logistics · Dynamic Risk Modelling · Collision-Aware Control

---

## Overview

Conventional warehouse path-planning systems often emphasize route efficiency, task completion, and collision avoidance. However, transporting temperature-sensitive products introduces an additional challenge: a route that is geometrically efficient may expose products to unsafe thermal conditions.

**ColdChain Multi-AGV RL** extends multi-AGV warehouse navigation with a simulated cold-chain risk model. It combines reinforcement-learning-based decision making with thermal-risk assessment, environmental risk propagation, product urgency, remaining safe exposure time, and proactive rescue mechanisms.

The goal is to study the interaction between transportation efficiency, multi-vehicle coordination, and simulated product protection under changing warehouse conditions.

> **Project type:** Simulation-based AI and reinforcement learning research. This project is not a deployment-ready warehouse control system.

## Problem Statement

Consider a warehouse in which several AGVs transport temperature-sensitive products between pickup and delivery locations.

The system must address several interacting challenges:

- **Multi-AGV coordination:** Multiple vehicles operate in the same warehouse and must avoid conflicting movements.
- **Dynamic thermal conditions:** Temperature and cooling conditions can change during transportation.
- **Environmental risk:** A zone may become unsafe because of heating, cooling degradation, or power failure.
- **Product urgency:** Products with less remaining safe exposure time may require earlier intervention.
- **Proactive rescue:** A product may need to be moved temporarily to a safer location before its simulated exposure limit is reached.

The central objective is to investigate risk-aware transportation decisions rather than optimize geometric distance alone.

## Research Objectives

1. Implement and evaluate a reinforcement-learning baseline for multi-AGV warehouse navigation.
2. Incorporate simulated thermal and environmental conditions into warehouse decision making.
3. Model spatial risk propagation between neighbouring warehouse zones.
4. Represent product urgency and Remaining Safe Exposure Time (RSET).
5. Evaluate proactive rescue behaviour under simulated cold-chain disturbances.
6. Compare Q-learning and DQN-based approaches while measuring task completion, collisions, reward, and operational efficiency.

## System Configuration

| Component | Configuration |
|---|---|
| Warehouse layout | 20 × 20 grid |
| Warehouse cells | 400 |
| AGVs | 3 |
| Thermal zones | 16 |
| Primitive actions per AGV | 5 |
| Joint-action combinations | 125 |
| Final state representation | 24 features |
| Environment | Simulated warehouse |
| Main learning approaches | Q-learning and DQN |

Each AGV has five primitive actions: stay, move right, move left, move up, and move down.

For three AGVs, the centralized joint-action space contains:

`5 × 5 × 5 = 125 possible joint actions`

The final DQN formulation represents the combined warehouse decision state using 24 features, organised as eight features per AGV.

## System Architecture

```text
             Warehouse Configuration
                       |
                       v
             Multi-AGV Environment
                       |
             +---------+---------+
             |                   |
             v                   v
       Task Information     Thermal Model
                                 |
                                 v
                       Local Environmental Risk
                                 |
                                 v
                     DERPG Risk Propagation
                                 |
                                 v
                   Product Urgency and RSET
                                 |
                                 v
                  Combined Decision State
                                 |
                                 v
                       RL Decision Module
                     Q-learning / DQN
                                 |
                                 v
                    Collision-Safety Checks
                                 |
                                 v
                    Execute AGV Movements
                                 |
                                 v
                  Update State and Rewards
                                 |
                                 v
                      Evaluate Outcomes
```

The diagram presents the conceptual workflow. Individual implementation details and execution paths depend on the corresponding source modules.

## Key Components

### 1. Multi-AGV Warehouse Environment

The warehouse is represented as a discrete grid. Three AGVs execute transportation tasks involving pickup and delivery locations.

The environment tracks vehicle positions, task progress, target locations, and interactions between vehicles. Collision-related checks are used to reject unsafe joint movements.

### 2. Thermal and Environmental Risk Model

The simulation represents thermal conditions at the zone level rather than assigning an independent thermal model to every warehouse cell.

Zone conditions include:

- Temperature and temperature trend
- Cooling effectiveness
- Power availability
- External environmental heat
- Local risk
- Dynamic risk

Simulated cooling and power disturbances influence zone temperature and the resulting risk assessment.

The thermal model is intended to demonstrate the behaviour of the control framework under disturbances. It is not a calibrated physical model of a specific warehouse.

### 3. DERPG — Dynamic Environmental Risk Propagation Graph

DERPG represents the influence of neighbouring warehouse zones on environmental risk.

Instead of treating every zone as completely independent, the model incorporates neighbouring-zone information into the dynamic risk assessment.

Conceptually:

```text
Local Zone Conditions
          +
Neighbouring-Zone Risk
          |
          v
Dynamic Risk Assessment
          |
          v
Risk-Aware Decisions
```

The implemented propagation weight is 0.30, balancing local risk and neighbouring-zone influence.

An ablation experiment compares configurations with DERPG enabled and disabled to examine its contribution.

### 4. Product Urgency and RSET

**RSET stands for Remaining Safe Exposure Time.**

It represents the remaining exposure margin in the simulation before a product's permitted safe-exposure limit is exhausted.

The framework distinguishes between environmental risk and product urgency:

- Environmental risk describes the danger associated with a zone.
- Product urgency represents how urgently a particular product may require intervention.

This distinction allows the framework to represent situations in which two products experience similar environmental conditions but have different exposure histories.

### 5. Proactive Product Rescue

The rescue mechanism is intended to move an at-risk product to a safer temporary location before its simulated exposure limit is reached.

The intended workflow is:

1. Assess current thermal risk and product urgency.
2. Identify a suitable safer location.
3. Perform the rescue intervention.
4. Preserve the original delivery objective.
5. Resume delivery when appropriate.

Rescue behaviour is evaluated in simulation. It should not be interpreted as a validated real-world pharmaceutical or food-preservation procedure.

## Reinforcement Learning Methods

### Q-learning

Q-learning is used as a tabular reinforcement-learning baseline. It estimates the value of taking an action in a state using rewards and estimated future returns.

Its limitations become important when the state includes multiple vehicles, target locations, thermal information, and product-risk features.

### Deep Q-Network (DQN)

DQN approximates action values using a neural network rather than storing every state-action value in a table.

The final joint-action formulation uses:

- A 24-feature input state
- Two hidden layers of 128 neurons each
- 125 joint-action outputs
- Experience replay
- A target network
- Huber loss
- The Adam optimizer
- Epsilon-greedy exploration

The DQN architecture estimates values for joint actions involving all three AGVs.

### Collision-Aware Hybrid Control

The project also evaluates a DQN-guided hybrid operational controller with target-directed movement and collision-safety checks.

This distinction matters: the strongest reported operational results belong to the hybrid controller evaluation and should not be attributed to an unconstrained pure DQN policy.

## Reward Design

The reward function is designed to encourage useful transportation behaviour while discouraging undesirable outcomes.

| Behaviour | Reward-design intention |
|---|---|
| Movement | Account for movement cost |
| Staying | Discourage unnecessary waiting |
| Collision | Apply a substantial penalty |
| Invalid movement | Discourage invalid actions |
| Target progress | Encourage progress towards objectives |
| Pickup | Reward successful pickup |
| Delivery | Reward task completion |
| Thermal risk | Discourage unsafe exposure |
| Rescue | Encourage successful protective intervention |

The reward design combines navigation and task objectives with risk-aware behaviour. Individual reward values are implementation-specific and should be read from the corresponding environment code.

## Experimental Evaluation

The project evaluates navigation outcomes alongside simulated cold-chain behaviour.

### Navigation and coordination metrics

- Task completion rate
- Average episode reward
- Average steps per episode
- Collision count
- Completed tasks per episode

### Cold-chain metrics

- Thermal exposure and damage
- Simulated product quality
- Environmental risk
- Product urgency and RSET
- Rescue activation and outcomes

### Reported operational results

The final project evidence records the following results for the DQN-guided hybrid operational controller:

| Metric | Reported result |
|---|---:|
| Evaluation episodes | 100 |
| Average reward | 1082.63 |
| Average steps | 77.12 |
| Average completed tasks | 3 out of 3 |
| Completion rate | 100% |
| Collisions | 0 |

These figures describe the reported hybrid operational evaluation, not a guarantee of performance on arbitrary warehouse layouts or real AGV hardware.

### DERPG ablation

The recorded ablation results are:

| Metric | DERPG enabled | DERPG disabled |
|---|---:|---:|
| Average steps | 80 | 79 |
| Average reward | 1010.982 | 974.726 |

Both configurations completed all three tasks in the recorded experiment and reported zero collisions.

The enabled configuration achieved a higher reported reward but took one additional average step. Therefore, these results do not demonstrate that DERPG always improves travel efficiency. They provide evidence of a reward difference in the tested configuration.

### Interpretation

The reported experiments illustrate why multiple metrics matter. Completion rate and collision count alone do not establish thermal safety, while reward alone does not explain which behaviours improved.

Results should be interpreted within the tested simulation settings, episode counts, seeds, and controller configuration.

## Technology Stack

The project uses a Python-based reinforcement-learning and simulation workflow. The intended core technologies include:

- **Python** — implementation and experimentation
- **PyTorch** — DQN neural-network components
- **NumPy** — numerical computation
- **Pandas** — dataset and tabular-data handling
- **Gymnasium-style API** — environment reset and step interaction
- **Matplotlib** — experimental visualisation
- **Git and GitHub** — version control and project documentation

Refer to the actual source files and dependency manifest for the exact versions and installed packages. Do not assume every listed library is required by every execution path.

## Repository Contents

The repository contains implementation code, algorithm and baseline modules, model checkpoints, and experimental artefacts.

The principal visible areas include:

- `agents/` — agent-related code
- `algorithms/` — algorithm implementations
- `baselines/` — baseline-related code
- `PPT_RESULTS/` — presentation and result artefacts
- Model checkpoints and evaluation outputs

The `models/` directory contains multiple checkpoint versions and experimental outputs. These should be treated as distinct experimental artefacts unless their relationship to the final evaluation has been verified.

## Reproducibility and Setup

The project is simulation-based, but a clean reproduction requires the correct source files, compatible dependencies, datasets, and model assets.

### 1. Clone the repository

```bash
git clone https://github.com/thrishikav9-del/coldchain-multiagv-rl.git
cd coldchain-multiagv-rl
```

### 2. Create a virtual environment

```bash
python -m venv .venv
```

Activate it on Windows:

```bash
.venv\Scripts\activate
```

On Linux or macOS:

```bash
source .venv/bin/activate
```

### 3. Install dependencies

If a verified `requirements.txt` is present:

```bash
pip install -r requirements.txt
```

Otherwise, install the dependencies specified by the actual source code and project documentation. Do not assume the environment can be reproduced from the command above until the dependency manifest is checked.

### 4. Prepare required data and assets

Verify that the dataset and risk-engine modules required by the environment are available in their expected locations. The development notes identify `dataset/ALL_benchmark_W60.parquet` and `dataset/risk_engine.py` as required by one environment execution path.

These assets must be included or obtained through a documented source before that path can be reproduced.

### 5. Run an experiment

Use the project's verified training or evaluation entry point after checking its filename, arguments, dataset paths, and checkpoint requirements. A universal run command is intentionally not specified here because the exact entry point and dependencies must be verified against the current repository contents.

## Research Contribution

The project combines established reinforcement-learning techniques with a cold-chain-oriented simulation framework.

Its intended contributions are:

1. Extending multi-AGV navigation with simulated thermal and environmental risk.
2. Representing spatial risk influence between warehouse zones through DERPG.
3. Incorporating product urgency and remaining safe exposure time into the problem formulation.
4. Exploring proactive rescue for products approaching simulated exposure limits.
5. Evaluating Q-learning and DQN-based approaches alongside collision-aware hybrid control.

The contribution lies in the integration and experimental study of these mechanisms, not in claiming to have invented Q-learning or DQN.

## Limitations

The current project has several important limitations:

- **Simulation scope:** Results are obtained in a simulated environment, not a physical warehouse.
- **Thermal calibration:** The thermal model has not been established here as a validated physical model for a particular facility.
- **Product degradation:** The simulated quality model is not a validated pharmaceutical or food-science spoilage model.
- **Fleet scalability:** The centralized joint-action space grows exponentially with the number of AGVs.
- **Generalisation:** Performance on the tested configuration does not establish performance on arbitrary layouts, fleets, or disturbances.
- **Controller attribution:** Hybrid-controller results must be distinguished from pure DQN results.
- **Reproducibility:** Dataset and risk-engine availability, dependency versions, and the final experiment entry point must be documented to support independent reproduction.

## Future Work

Potential directions include:

- Testing larger warehouses and more AGVs.
- Comparing scalable multi-agent reinforcement-learning architectures.
- Evaluating additional thermal and power-failure scenarios.
- Calibrating thermal dynamics using real sensor measurements.
- Validating product-specific exposure and degradation models.
- Improving experiment tracking, checkpoint selection, and reproducibility.
- Evaluating generalisation across layouts and task distributions.

## Reference

**Foundation paper:** *Deep Reinforcement Learning for Dynamic Order Picking in Warehouse Operations.*

The project uses a related multi-AGV reinforcement-learning direction as a foundation and extends the simulation towards cold-chain risk-aware decision making. Consult the original publication for its exact bibliographic details and cite it appropriately in academic use.

## Author

**Vullasa Thrishika**

B.Tech Artificial Intelligence  
Amrita Vishwa Vidyapeetham, Coimbatore

GitHub: [@thrishikav9-del](https://github.com/thrishikav9-del)

---

## Disclaimer

This repository is intended for academic research, experimentation, and educational use. Its simulation results do not establish safety or performance guarantees for physical warehouse operations or temperature-sensitive products.
