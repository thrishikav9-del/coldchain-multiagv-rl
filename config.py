"""
Global configuration for the Cold-Chain Multi-AGV RL project.

Phase 1:
    Base multi-AGV warehouse and Q-learning system.

Later phases will extend this configuration with:
    - thermal parameters
    - DERPG
    - product urgency
    - DQN
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class WarehouseConfig:

    # --------------------------------------------------
    # Warehouse dimensions
    # --------------------------------------------------

    width: int = 20
    height: int = 20

    # --------------------------------------------------
    # Multi-AGV configuration
    # --------------------------------------------------

    num_agvs: int = 3

    # --------------------------------------------------
    # Movement
    # --------------------------------------------------

    movement_step: int = 1

    # --------------------------------------------------
    # Simulation
    # --------------------------------------------------

    max_steps: int = 500

    # --------------------------------------------------
    # Reproducibility
    # --------------------------------------------------

    seed: int = 42


@dataclass(frozen=True)
class RLConfig:

    # --------------------------------------------------
    # Q-learning
    # --------------------------------------------------

    learning_rate: float = 0.1

    discount_factor: float = 0.95

    epsilon: float = 1.0

    epsilon_min: float = 0.05

    epsilon_decay: float = 0.995

    # --------------------------------------------------
    # Training
    # --------------------------------------------------

    episodes: int = 1000


@dataclass(frozen=True)
class SimulationConfig:

    warehouse: WarehouseConfig = WarehouseConfig()

    rl: RLConfig = RLConfig()


CONFIG = SimulationConfig()
