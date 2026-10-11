"""
Centralized Deep Q-Network Agent.

Supports:

- Multiple AGVs
- Joint-action DQN
- Epsilon-greedy exploration
- Policy network
- Target network
- Experience replay
- Double DQN
- Huber loss
- Adam optimizer
- Soft target updates

For N AGVs and A actions:

    joint_action_count = A ** N

Example:

    3 AGVs
    5 actions

    5 ** 3 = 125 joint actions
"""

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
        learning_rate: float = 0.0001,
        gamma: float = 0.99,
        tau: float = 0.001,
        epsilon: float = 1.0,
        epsilon_min: float = 0.05,
        epsilon_decay: float = 0.995,
        replay_capacity: int = 200_000,
        batch_size: int = 64,
        seed: int = 42,
    ):

        self.state_size = int(state_size)
        self.num_agvs = int(num_agvs)
        self.action_size = int(action_size)

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

        # --------------------------------------------------
        # Random seeds
        # --------------------------------------------------

        random.seed(seed)

        np.random.seed(seed)

        torch.manual_seed(seed)

        # --------------------------------------------------
        # Device
        # --------------------------------------------------

        self.device = torch.device(
            "cuda"
            if torch.cuda.is_available()
            else "cpu"
        )

        # --------------------------------------------------
        # Joint action count
        # --------------------------------------------------

        self.joint_action_size = (
            self.action_size ** self.num_agvs
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
        # Loss
        # --------------------------------------------------

        self.loss_function = nn.SmoothL1Loss()

        # --------------------------------------------------
        # Replay buffer
        # --------------------------------------------------

        self.replay_buffer = ReplayBuffer(
            capacity=replay_capacity
        )

        self.training_steps = 0

    # ======================================================
    # JOINT ACTION ENCODING
    # ======================================================

        # ======================================================
    # JOINT ACTION ENCODING
    # ======================================================

    def encode_joint_action(self, actions):
        """
        Encode one action per AGV into a single integer.

        For 3 AGVs and 5 actions:
            [0,0,0] -> 0
            [1,0,0] -> 25
            [0,1,0] -> 5
            [3,1,4] -> 84
            [4,4,4] -> 124

        This helper is NOT used by the current DQN training loop.
        The environment still receives the joint action vector.
        """

        actions = np.asarray(
            actions,
            dtype=np.int64,
        ).reshape(-1)

        if actions.size != self.num_agvs:
            raise ValueError(
                f"Expected {self.num_agvs} actions, "
                f"got {actions.size}."
            )

        if np.any(actions < 0) or np.any(
            actions >= self.action_size
        ):
            raise ValueError(
                f"Actions must be in "
                f"[0, {self.action_size - 1}]. "
                f"Got {actions.tolist()}."
            )

        encoded = 0

        for action in actions:
            encoded = (
                encoded * self.action_size
                + int(action)
            )

        return int(encoded)

    # ======================================================
    # JOINT ACTION DECODING
    # ======================================================

    def decode_joint_action(self, encoded_action):
        """
        Decode one integer into one action per AGV.

        Example for 3 AGVs and 5 actions:
            0   -> [0,0,0]
            25  -> [1,0,0]
            5   -> [0,1,0]
            84  -> [3,1,4]
            124 -> [4,4,4]
        """

        encoded_action = int(
            encoded_action
        )

        total_actions = (
            self.action_size
            ** self.num_agvs
        )

        if (
            encoded_action < 0
            or encoded_action >= total_actions
        ):
            raise ValueError(
                f"Encoded joint action must be in "
                f"[0, {total_actions - 1}]. "
                f"Got {encoded_action}."
            )

        actions = np.zeros(
            self.num_agvs,
            dtype=np.int64,
        )

        value = encoded_action

        for i in range(
            self.num_agvs - 1,
            -1,
            -1,
        ):
            actions[i] = (
                value % self.action_size
            )

            value //= self.action_size

        return actions
    # ======================================================
    # ACTION SELECTION
    # ======================================================

    def select_action(
        self,
        state,
        training=True,
    ):
        """Select a joint action for all AGVs."""
        state_array = np.asarray(
            state,
            dtype=np.float32,
        ).reshape(-1)
        if state_array.size != self.state_size:
            raise ValueError(
                f"Expected state size {self.state_size}, "
                f"got {state_array.size}."
            )
        # ==================================================
        # TRAINING: NORMAL DQN POLICY
        # ==================================================
        if training and random.random() < self.epsilon:
            return np.random.randint(
                0,
                self.action_size,
                size=self.num_agvs,
                dtype=np.int64,
            )
        state_tensor = torch.as_tensor(
            state_array,
            dtype=torch.float32,
            device=self.device,
        ).reshape(
            1,
            self.state_size,
        )
        with torch.no_grad():
            q_values = self.policy_network.action_values(
                state_tensor
            )
        expected_joint_size = (
            self.action_size ** self.num_agvs
        )
        if q_values.ndim != 2:
            raise RuntimeError(
                f"Expected 2-D Q-values, got {tuple(q_values.shape)}."
            )
        if q_values.shape != (
            1,
            expected_joint_size,
        ):
            raise RuntimeError(
                "DQN joint-action output mismatch. "
                f"Expected (1, {expected_joint_size}), "
                f"got {tuple(q_values.shape)}."
            )
        joint_action_index = int(
            torch.argmax(
                q_values,
                dim=1,
            ).item()
        )
        actions = self.decode_joint_action(
            joint_action_index
        ).astype(
            np.int64
        )
        # ==================================================
        return np.asarray(
            actions,
            dtype=np.int64,
        )    # ======================================================
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

        self.replay_buffer.add(
            state,
            action,
            reward,
            next_state,
            done,
        )

    # ======================================================
    # TRAIN
    # ======================================================

    def train_step(
        self,
    ):

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
        # Tensors
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

        # ==================================================
        # ENCODE JOINT ACTIONS
        # ==================================================

        joint_action_indices = []

        for action in actions_tensor.cpu().numpy():

            joint_action_indices.append(
                self.encode_joint_action(
                    action
                )
            )

        joint_action_indices = torch.as_tensor(
            joint_action_indices,
            dtype=torch.long,
            device=self.device,
        )

        # ==================================================
        # CURRENT Q
        # ==================================================

        current_q = (
            self.policy_network
            .action_values(
                states_tensor
            )
        )

        # Shape:

        # [batch, 125]

        chosen_q = (
            current_q
            .gather(
                1,
                joint_action_indices.unsqueeze(1),
            )
            .squeeze(1)
        )

        # ==================================================
        # DOUBLE DQN TARGET
        # ==================================================

        with torch.no_grad():

            # --------------------------------------------------
            # Policy chooses best NEXT JOINT action
            # --------------------------------------------------

            next_policy_q = (
                self.policy_network
                .action_values(
                    next_states_tensor
                )
            )

            next_joint_actions = (
                next_policy_q.argmax(
                    dim=1
                )
            )

            # --------------------------------------------------
            # Target evaluates selected action
            # --------------------------------------------------

            next_target_q = (
                self.target_network
                .action_values(
                    next_states_tensor
                )
            )

            max_next_q = (
                next_target_q
                .gather(
                    1,
                    next_joint_actions.unsqueeze(1),
                )
                .squeeze(1)
            )

            # --------------------------------------------------
            # Bellman target
            # --------------------------------------------------

            target_q = (
                rewards_tensor
                +
                self.gamma
                *
                max_next_q
                *
                (
                    1.0
                    -
                    dones_tensor
                )
            )

        # ==================================================
        # LOSS
        # ==================================================

        loss = self.loss_function(
            chosen_q,
            target_q,
        )

        # ==================================================
        # BACKPROPAGATION
        # ==================================================

        self.optimizer.zero_grad()

        loss.backward()

        # ==================================================
        # GRADIENT CLIPPING
        # ==================================================

        torch.nn.utils.clip_grad_norm_(
            self.policy_network.parameters(),
            max_norm=1.0,
        )

        self.optimizer.step()

        self.training_steps += 1

        # ==================================================
        # TARGET UPDATE
        # ==================================================

        self.soft_update()

        return float(
            loss.item()
        )

    # ======================================================
    # SOFT TARGET UPDATE
    # ======================================================

    def soft_update(
        self,
    ):

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
                    *
                    policy_param.data
                )

    # ======================================================
    # EPSILON DECAY
    # ======================================================

    def decay_epsilon(
        self,
    ):

        self.epsilon = max(
            self.epsilon_min,
            self.epsilon
            *
            self.epsilon_decay,
        )

    # ======================================================
    # EVALUATION MODE
    # ======================================================

    def evaluation_mode(
        self,
    ):

        self._evaluation_epsilon = (
            self.epsilon
        )

        self.policy_network.eval()

        self.target_network.eval()

        self.epsilon = 0.0

    # ======================================================
    # TRAINING MODE
    # ======================================================

    def training_mode(
        self,
    ):

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

        checkpoint = torch.load(
            filepath,
            map_location=self.device,
        )

        # --------------------------------------------------
        # Architecture validation
        # --------------------------------------------------

        saved_state_size = checkpoint.get(
            "state_size",
            self.state_size,
        )

        saved_num_agvs = checkpoint.get(
            "num_agvs",
            self.num_agvs,
        )

        saved_action_size = checkpoint.get(
            "action_size",
            self.action_size,
        )

        if (
            int(saved_state_size)
            != self.state_size
        ):

            raise ValueError(
                "Checkpoint state_size mismatch. "
                f"Checkpoint={saved_state_size}, "
                f"Agent={self.state_size}"
            )

        if (
            int(saved_num_agvs)
            != self.num_agvs
        ):

            raise ValueError(
                "Checkpoint num_agvs mismatch. "
                f"Checkpoint={saved_num_agvs}, "
                f"Agent={self.num_agvs}"
            )

        if (
            int(saved_action_size)
            != self.action_size
        ):

            raise ValueError(
                "Checkpoint action_size mismatch. "
                f"Checkpoint={saved_action_size}, "
                f"Agent={self.action_size}"
            )

        # --------------------------------------------------
        # Load networks
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

            self.optimizer.load_state_dict(
                checkpoint[
                    "optimizer"
                ]
            )

        # --------------------------------------------------
        # Training state
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


__all__ = [
    "DQNAgent",
]


