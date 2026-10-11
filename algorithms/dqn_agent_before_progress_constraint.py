"""
Deep Q-Network Agent for the Multi-AGV Cold-Chain Warehouse.

Architecture
------------
Centralized DQN:

    state (18)
        ↓
    DQN network
        ↓
    125 joint-action Q-values
        ↓
    decode one joint action
        ↓
    3 AGV primitive actions

For 3 AGVs and 5 actions per AGV:

    5 ** 3 = 125 joint actions

Actions per AGV
---------------
    0 = Stay
    1 = Right
    2 = Left
    3 = Up
    4 = Down

Features
--------
- Epsilon-greedy exploration
- Policy network
- Target network
- Experience replay
- Huber loss
- Adam optimizer
- Gradient clipping
- Soft target updates
- Centralized joint-action encoding/decoding
- Collision-free joint-action filtering
- Save/load support
"""

from __future__ import annotations

import random

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from algorithms.dqn_network import DQNNetwork
from algorithms.replay_buffer import ReplayBuffer


class DQNAgent:

    def __init__(
        self,
        state_size: int,
        num_agvs: int,
        action_size: int = 5,
        learning_rate: float = 0.001,
        gamma: float = 0.99,
        tau: float = 0.001,
        epsilon: float = 1.0,
        epsilon_min: float = 0.05,
        epsilon_decay: float = 0.995,
        replay_capacity: int = 200_000,
        batch_size: int = 64,
        seed: int = 42,
    ):
        # --------------------------------------------------
        # Basic configuration
        # --------------------------------------------------

        self.state_size = int(state_size)
        self.num_agvs = int(num_agvs)
        self.action_size = int(action_size)

        if self.state_size <= 0:
            raise ValueError(
                "state_size must be positive."
            )

        if self.num_agvs <= 0:
            raise ValueError(
                "num_agvs must be positive."
            )

        if self.action_size <= 0:
            raise ValueError(
                "action_size must be positive."
            )

        # --------------------------------------------------
        # Centralized joint action space
        # --------------------------------------------------

        self.joint_action_size = (
            self.action_size ** self.num_agvs
        )

        # --------------------------------------------------
        # Hyperparameters
        # --------------------------------------------------

        self.learning_rate = float(
            learning_rate
        )

        self.gamma = float(gamma)
        self.tau = float(tau)

        self.epsilon = float(epsilon)
        self.epsilon_min = float(
            epsilon_min
        )
        self.epsilon_decay = float(
            epsilon_decay
        )

        self.batch_size = int(
            batch_size
        )

        if self.learning_rate <= 0:
            raise ValueError(
                "learning_rate must be positive."
            )

        if not 0.0 <= self.gamma <= 1.0:
            raise ValueError(
                "gamma must be in [0, 1]."
            )

        if not 0.0 < self.tau <= 1.0:
            raise ValueError(
                "tau must be in (0, 1]."
            )

        if not 0.0 <= self.epsilon_min <= 1.0:
            raise ValueError(
                "epsilon_min must be in [0, 1]."
            )

        if not 0.0 < self.epsilon_decay <= 1.0:
            raise ValueError(
                "epsilon_decay must be in (0, 1]."
            )

        if self.batch_size <= 0:
            raise ValueError(
                "batch_size must be positive."
            )

        # --------------------------------------------------
        # Reproducibility
        # --------------------------------------------------

        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

        # --------------------------------------------------
        # Device
        # --------------------------------------------------

        self.device = torch.device(
            "cuda"
            if torch.cuda.is_available()
            else "cpu"
        )

        # --------------------------------------------------
        # Policy network
        # --------------------------------------------------

        self.policy_network = DQNNetwork(
            state_size=self.state_size,
            num_agvs=self.num_agvs,
            action_size=self.action_size,
        ).to(self.device)

        # --------------------------------------------------
        # Target network
        # --------------------------------------------------

        self.target_network = DQNNetwork(
            state_size=self.state_size,
            num_agvs=self.num_agvs,
            action_size=self.action_size,
        ).to(self.device)

        self.target_network.load_state_dict(
            self.policy_network.state_dict()
        )

        self.target_network.eval()

        # --------------------------------------------------
        # Optimizer
        # --------------------------------------------------

        self.optimizer = optim.Adam(
            self.policy_network.parameters(),
            lr=self.learning_rate,
        )

        # --------------------------------------------------
        # Huber loss
        # --------------------------------------------------

        self.loss_function = nn.SmoothL1Loss()

        # --------------------------------------------------
        # Replay buffer
        # --------------------------------------------------

        self.replay_buffer = ReplayBuffer(
            capacity=replay_capacity
        )

        # --------------------------------------------------
        # Runtime statistics
        # --------------------------------------------------

        self.training_steps = 0

        # --------------------------------------------------
        # Environment reference used for joint-action safety
        #
        # train_dqn.py assigns the active environment here.
        # --------------------------------------------------

        self._evaluation_env = None

    # ======================================================
    # JOINT ACTION ENCODING
    # ======================================================

    def encode_joint_action(
        self,
        actions,
    ) -> int:
        """
        Convert individual AGV actions to one joint index.

        Example for 3 AGVs:

            [1, 2, 3]

        becomes one integer in:

            [0, 124]
        """

        actions = np.asarray(
            actions,
            dtype=np.int64,
        ).reshape(-1)

        if actions.size != self.num_agvs:
            raise ValueError(
                "Expected "
                f"{self.num_agvs} actions, "
                f"got {actions.size}."
            )

        index = 0

        for action in actions:

            action = int(action)

            if not 0 <= action < self.action_size:
                raise ValueError(
                    f"Invalid action {action}. "
                    f"Expected 0..{self.action_size - 1}."
                )

            index = (
                index * self.action_size
                + action
            )

        return int(index)

    # ======================================================
    # JOINT ACTION DECODING
    # ======================================================

    def decode_joint_action(
        self,
        index,
    ) -> np.ndarray:
        """
        Convert one joint-action index back to
        individual AGV actions.
        """

        index = int(index)

        if not 0 <= index < self.joint_action_size:
            raise ValueError(
                f"Invalid joint action index {index}. "
                f"Expected "
                f"0..{self.joint_action_size - 1}."
            )

        actions = []

        remaining = index

        for _ in range(
            self.num_agvs
        ):
            actions.append(
                remaining
                % self.action_size
            )

            remaining //= (
                self.action_size
            )

        actions.reverse()

        return np.asarray(
            actions,
            dtype=np.int64,
        )

    # ======================================================
    # JOINT ACTION SAFETY
    # ======================================================

    def check_collision_free(
        self,
        actions,
    ) -> bool:
        """
        Check whether a joint action is safe in the
        currently attached environment.

        The check covers:

            1. Same destination cell
            2. Moving into a stationary AGV
            3. Swap collision

        The environment's own _get_target_position()
        is used so action semantics remain identical
        between the agent and environment.
        """

        if self._evaluation_env is None:
            return True

        env = self._evaluation_env

        if getattr(
            env,
            "system",
            None,
        ) is None:
            return True

        actions = np.asarray(
            actions,
            dtype=np.int64,
        ).reshape(-1)

        if actions.size != self.num_agvs:
            return False

        current_positions = {}
        proposed_positions = {}

        # --------------------------------------------------
        # Calculate current/proposed positions
        # --------------------------------------------------

        for agv_id in range(
            1,
            self.num_agvs + 1,
        ):
            agv = env.system.agvs.get(
                agv_id
            )

            if agv is None:
                return False

            current = tuple(
                map(
                    int,
                    agv.position,
                )
            )

            try:
                proposed = tuple(
                    map(
                        int,
                        env._get_target_position(
                            agv,
                            int(
                                actions[
                                    agv_id - 1
                                ]
                            ),
                        ),
                    )
                )

            except Exception:
                return False

            current_positions[
                agv_id
            ] = current

            proposed_positions[
                agv_id
            ] = proposed

        # --------------------------------------------------
        # Same destination collision
        #
        # Ignore AGVs that stay in place.
        # --------------------------------------------------

        position_to_agvs = {}

        for agv_id in range(
            1,
            self.num_agvs + 1,
        ):
            current = current_positions[
                agv_id
            ]

            proposed = proposed_positions[
                agv_id
            ]

            if proposed == current:
                continue

            position_to_agvs.setdefault(
                proposed,
                [],
            ).append(
                agv_id
            )

        for agv_ids in position_to_agvs.values():

            if len(agv_ids) > 1:
                return False

        # --------------------------------------------------
        # Moving into stationary AGV
        # --------------------------------------------------

        for agv_id in range(
            1,
            self.num_agvs + 1,
        ):
            current = current_positions[
                agv_id
            ]

            proposed = proposed_positions[
                agv_id
            ]

            if proposed == current:
                continue

            for other_id in range(
                1,
                self.num_agvs + 1,
            ):
                if other_id == agv_id:
                    continue

                other_current = (
                    current_positions[
                        other_id
                    ]
                )

                other_proposed = (
                    proposed_positions[
                        other_id
                    ]
                )

                if (
                    proposed
                    == other_current
                    and
                    other_proposed
                    == other_current
                ):
                    return False

        # --------------------------------------------------
        # Swap collision
        # --------------------------------------------------

        for i in range(
            self.num_agvs
        ):
            for j in range(
                i + 1,
                self.num_agvs,
            ):

                agv_a = i + 1
                agv_b = j + 1

                a_current = (
                    current_positions[
                        agv_a
                    ]
                )

                b_current = (
                    current_positions[
                        agv_b
                    ]
                )

                a_proposed = (
                    proposed_positions[
                        agv_a
                    ]
                )

                b_proposed = (
                    proposed_positions[
                        agv_b
                    ]
                )

                if (
                    a_proposed
                    == b_current
                    and
                    b_proposed
                    == a_current
                    and
                    a_proposed
                    != a_current
                    and
                    b_proposed
                    != b_current
                ):
                    return False

        return True

    # ======================================================
    # SAFE JOINT ACTION INDICES
    # ======================================================

    def get_safe_joint_action_indices(
        self,
    ):
        """
        Return all collision-free joint-action indices
        for the currently attached environment.
        """

        safe_indices = []

        for joint_index in range(
            self.joint_action_size
        ):

            if self.check_collision_free(
                self.decode_joint_action(
                    joint_index
                )
            ):
                safe_indices.append(
                    joint_index
                )

        return safe_indices

    # ======================================================
    # JOINT ACTION SAFETY HELPER
    # ======================================================

    def _joint_action_is_safe(
        self,
        joint_index: int,
    ) -> bool:
        """
        Check one joint action index for safety.
        """

        actions = self.decode_joint_action(
            joint_index
        )

        return bool(
            self.check_collision_free(
                actions
            )
        )

    # ======================================================
    # ACTION SELECTION
    # ======================================================

    def select_action(
        self,
        state,
        training: bool = True,
    ) -> np.ndarray:
        """
        Select one joint action for all AGVs.

        During exploration:
            random collision-free joint action
            when an environment is attached.

        During exploitation:
            highest-Q collision-free joint action.

        If no environment is attached, all joint actions
        are considered valid.
        """

        state_array = np.asarray(
            state,
            dtype=np.float32,
        ).reshape(-1)

        if state_array.size != self.state_size:
            raise ValueError(
                f"Expected state size "
                f"{self.state_size}, "
                f"got {state_array.size}."
            )

        # --------------------------------------------------
        # Build valid action set
        # --------------------------------------------------

        if self._evaluation_env is not None:

            safe_indices = (
                self.get_safe_joint_action_indices()
            )

        else:

            safe_indices = list(
                range(
                    self.joint_action_size
                )
            )

        if not safe_indices:
            raise RuntimeError(
                "No valid joint actions are available."
            )

        # --------------------------------------------------
        # Epsilon-greedy exploration
        # --------------------------------------------------

        if (
            training
            and
            random.random() < self.epsilon
        ):
            joint_index = random.choice(
                safe_indices
            )

            return self.decode_joint_action(
                joint_index
            )

        # --------------------------------------------------
        # Greedy exploitation
        # --------------------------------------------------

        state_tensor = torch.as_tensor(
            state_array,
            dtype=torch.float32,
            device=self.device,
        ).unsqueeze(0)

        self.policy_network.eval()

        with torch.no_grad():

            q_values = (
                self.policy_network.action_values(
                    state_tensor
                )
            )

        if q_values.ndim != 2:
            raise RuntimeError(
                "Invalid DQN output shape: "
                f"{q_values.shape}."
            )

        if q_values.shape != (
            1,
            self.joint_action_size,
        ):
            raise RuntimeError(
                "Invalid DQN output size: "
                f"{q_values.shape}. "
                f"Expected "
                f"(1, {self.joint_action_size})."
            )

        safe_indices_tensor = torch.as_tensor(
            safe_indices,
            dtype=torch.long,
            device=self.device,
        )

        safe_q_values = q_values[
            0,
            safe_indices_tensor,
        ]

        best_position = int(
            torch.argmax(
                safe_q_values
            ).item()
        )

        joint_index = safe_indices[
            best_position
        ]

        return self.decode_joint_action(
            joint_index
        )

    # ======================================================
    # LEGACY JOINT ACTION METHOD
    # ======================================================

    def select_joint_action(
        self,
        state,
    ) -> np.ndarray:
        """
        Compatibility wrapper.

        Uses the same centralized action-selection
        logic as select_action().
        """

        return self.select_action(
            state,
            training=True,
        )

    # ======================================================
    # STORE EXPERIENCE
    # ======================================================

    def remember(
        self,
        state,
        action,
        reward,
        next_state,
        done,
    ):
        """
        Store one centralized joint transition.
        """

        action_array = np.asarray(
            action,
            dtype=np.int64,
        ).reshape(-1)

        if action_array.size != self.num_agvs:
            raise ValueError(
                f"Expected "
                f"{self.num_agvs} action values, "
                f"got {action_array.size}."
            )

        # Validate each action before storing.
        for action_value in action_array:

            if not 0 <= int(action_value) < self.action_size:
                raise ValueError(
                    f"Invalid action "
                    f"{action_value}."
                )

        self.replay_buffer.add(
            state,
            action_array,
            reward,
            next_state,
            done,
        )

    # ======================================================
    # TRAINING STEP
    # ======================================================

    def train_step(self):
        """
        Perform one DQN update.

        Returns:
            float loss
            or None if replay buffer is too small.
        """

        if len(
            self.replay_buffer
        ) < self.batch_size:

            return None

        (
            states,
            actions,
            rewards,
            next_states,
            dones,
        ) = self.replay_buffer.sample(
            self.batch_size
        )

        # --------------------------------------------------
        # Convert to tensors
        # --------------------------------------------------

        states_tensor = torch.as_tensor(
            states,
            dtype=torch.float32,
            device=self.device,
        )

        actions_tensor = torch.as_tensor(
            actions,
            dtype=torch.long,
            device=self.device,
        )

        rewards_tensor = torch.as_tensor(
            rewards,
            dtype=torch.float32,
            device=self.device,
        )

        next_states_tensor = torch.as_tensor(
            next_states,
            dtype=torch.float32,
            device=self.device,
        )

        dones_tensor = torch.as_tensor(
            dones,
            dtype=torch.float32,
            device=self.device,
        )

        # --------------------------------------------------
        # Validate state shape
        # --------------------------------------------------

        if states_tensor.ndim != 2:
            raise RuntimeError(
                "Invalid state batch shape: "
                f"{states_tensor.shape}."
            )

        if states_tensor.shape != (
            self.batch_size,
            self.state_size,
        ):
            raise RuntimeError(
                "Invalid state batch shape: "
                f"{states_tensor.shape}. "
                f"Expected "
                f"({self.batch_size}, "
                f"{self.state_size})."
            )

        # --------------------------------------------------
        # Validate action shape
        # --------------------------------------------------

        if actions_tensor.ndim == 1:

            actions_tensor = (
                actions_tensor.reshape(
                    self.batch_size,
                    self.num_agvs,
                )
            )

        if actions_tensor.shape != (
            self.batch_size,
            self.num_agvs,
        ):
            raise RuntimeError(
                "Invalid action tensor shape: "
                f"{actions_tensor.shape}. "
                f"Expected "
                f"({self.batch_size}, "
                f"{self.num_agvs})."
            )

        # --------------------------------------------------
        # Encode each action row into one joint index
        # --------------------------------------------------

        joint_actions = []

        for action_row in actions_tensor:

            joint_index = 0

            for action in action_row:

                action_value = int(
                    action.item()
                )

                if not 0 <= action_value < self.action_size:
                    raise RuntimeError(
                        f"Invalid action value "
                        f"{action_value} in replay batch."
                    )

                joint_index = (
                    joint_index
                    * self.action_size
                    + action_value
                )

            joint_actions.append(
                joint_index
            )

        joint_actions_tensor = torch.as_tensor(
            joint_actions,
            dtype=torch.long,
            device=self.device,
        )

        # --------------------------------------------------
        # Current Q-values
        # --------------------------------------------------

        self.policy_network.train()

        current_q = (
            self.policy_network.action_values(
                states_tensor
            )
        )

        if current_q.shape != (
            self.batch_size,
            self.joint_action_size,
        ):
            raise RuntimeError(
                "Invalid current Q shape: "
                f"{current_q.shape}. "
                f"Expected "
                f"({self.batch_size}, "
                f"{self.joint_action_size})."
            )

        # --------------------------------------------------
        # Q-value for actions actually taken
        # --------------------------------------------------

        chosen_q = (
            current_q
            .gather(
                1,
                joint_actions_tensor.unsqueeze(1),
            )
            .squeeze(1)
        )

        # --------------------------------------------------
        # Target Q-values
        # --------------------------------------------------

        with torch.no_grad():

            next_q = (
                self.target_network.action_values(
                    next_states_tensor
                )
            )

            if next_q.shape != (
                self.batch_size,
                self.joint_action_size,
            ):
                raise RuntimeError(
                    "Invalid next Q shape: "
                    f"{next_q.shape}. "
                    f"Expected "
                    f"({self.batch_size}, "
                    f"{self.joint_action_size})."
                )

            max_next_q = (
                next_q
                .max(
                    dim=1
                )
                .values
            )

            target_q = (
                rewards_tensor
                +
                self.gamma
                * max_next_q
                *
                (
                    1.0
                    -
                    dones_tensor
                )
            )

        # --------------------------------------------------
        # Huber loss
        # --------------------------------------------------

        loss = self.loss_function(
            chosen_q,
            target_q,
        )

        if not torch.isfinite(loss):
            raise RuntimeError(
                f"Non-finite DQN loss: {loss.item()}"
            )

        # --------------------------------------------------
        # Backpropagation
        # --------------------------------------------------

        self.optimizer.zero_grad()

        loss.backward()

        # --------------------------------------------------
        # Gradient clipping
        # --------------------------------------------------

        torch.nn.utils.clip_grad_norm_(
            self.policy_network.parameters(),
            max_norm=1.0,
        )

        self.optimizer.step()

        self.training_steps += 1

        # --------------------------------------------------
        # Soft target update
        # --------------------------------------------------

        self.soft_update()

        return float(
            loss.item()
        )

    # ======================================================
    # SOFT TARGET UPDATE
    # ======================================================

    def soft_update(self):
        """
        Soft-update target network parameters.
        """

        with torch.no_grad():

            for (
                target_param,
                policy_param,
            ) in zip(
                self.target_network.parameters(),
                self.policy_network.parameters(),
            ):

                target_param.data.mul_(
                    1.0 - self.tau
                )

                target_param.data.add_(
                    self.tau
                    * policy_param.data
                )

    # ======================================================
    # EPSILON DECAY
    # ======================================================

    def decay_epsilon(self):
        """
        Decay epsilon once per episode.
        """

        self.epsilon = max(
            self.epsilon_min,
            self.epsilon
            * self.epsilon_decay,
        )

    # ======================================================
    # EVALUATION MODE
    # ======================================================

    def evaluation_mode(self):
        """
        Switch to greedy evaluation.
        """

        self._evaluation_epsilon = (
            self.epsilon
        )

        self.policy_network.eval()
        self.target_network.eval()

        self.epsilon = 0.0

    # ======================================================
    # TRAINING MODE
    # ======================================================

    def training_mode(self):
        """
        Switch back to training mode and restore
        the previous epsilon value.
        """

        self.policy_network.train()
        self.target_network.eval()

        if hasattr(
            self,
            "_evaluation_epsilon",
        ):

            self.epsilon = (
                self._evaluation_epsilon
            )

            del self._evaluation_epsilon

    # ======================================================
    # SAVE
    # ======================================================

    def save(
        self,
        filepath,
    ):
        """
        Save model, optimizer and agent state.
        """

        torch.save(
            {
                "policy_network":
                    self.policy_network.state_dict(),

                "target_network":
                    self.target_network.state_dict(),

                "optimizer":
                    self.optimizer.state_dict(),

                "epsilon":
                    self.epsilon,

                "training_steps":
                    self.training_steps,

                "state_size":
                    self.state_size,

                "num_agvs":
                    self.num_agvs,

                "action_size":
                    self.action_size,

                "joint_action_size":
                    self.joint_action_size,
            },
            filepath,
        )

    # ======================================================
    # LOAD
    # ======================================================

    def load(
        self,
        filepath,
    ):
        """
        Load model, optimizer and agent state.
        """

        checkpoint = torch.load(
            filepath,
            map_location=self.device,
        )

        # --------------------------------------------------
        # Compatibility checks
        # --------------------------------------------------

        checkpoint_state_size = checkpoint.get(
            "state_size",
            self.state_size,
        )

        checkpoint_num_agvs = checkpoint.get(
            "num_agvs",
            self.num_agvs,
        )

        checkpoint_action_size = checkpoint.get(
            "action_size",
            self.action_size,
        )

        if int(checkpoint_state_size) != self.state_size:
            raise ValueError(
                "Checkpoint state size mismatch: "
                f"checkpoint={checkpoint_state_size}, "
                f"agent={self.state_size}."
            )

        if int(checkpoint_num_agvs) != self.num_agvs:
            raise ValueError(
                "Checkpoint AGV count mismatch: "
                f"checkpoint={checkpoint_num_agvs}, "
                f"agent={self.num_agvs}."
            )

        if int(checkpoint_action_size) != self.action_size:
            raise ValueError(
                "Checkpoint action size mismatch: "
                f"checkpoint={checkpoint_action_size}, "
                f"agent={self.action_size}."
            )

        # --------------------------------------------------
        # Networks
        # --------------------------------------------------

        self.policy_network.load_state_dict(
            checkpoint[
                "policy_network"
            ]
        )

        self.target_network.load_state_dict(
            checkpoint[
                "target_network"
            ]
        )

        # --------------------------------------------------
        # Optimizer
        # --------------------------------------------------

        if "optimizer" in checkpoint:

            try:

                self.optimizer.load_state_dict(
                    checkpoint[
                        "optimizer"
                    ]
                )

            except Exception:
                # Model weights are still usable even when
                # optimizer state cannot be restored.
                pass

        # --------------------------------------------------
        # Scalar state
        # --------------------------------------------------

        self.epsilon = float(
            checkpoint.get(
                "epsilon",
                self.epsilon,
            )
        )

        self.training_steps = int(
            checkpoint.get(
                "training_steps",
                0,
            )
        )

        self.policy_network.to(
            self.device
        )

        self.target_network.to(
            self.device
        )

        self.target_network.eval()


__all__ = [
    "DQNAgent",
]
