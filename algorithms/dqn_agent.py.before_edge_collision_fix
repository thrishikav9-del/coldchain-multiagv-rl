"""
Deep Q-Network Agent.

Phase 3B:
- epsilon-greedy exploration
- policy network
- target network
- experience replay
- Huber loss
- Adam optimizer
- soft target updates
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

        self.state_size = int(state_size)
        self.num_agvs = int(num_agvs)
        self.action_size = int(action_size)
        # ==================================================
        # CENTRALIZED JOINT ACTION SPACE
        #
        # Example:
        # 3 AGVs x 5 actions = 125 joint actions
        # ==================================================
        self.joint_action_size = (
            self.action_size ** self.num_agvs
        )
        # Environment reference used only for
        # evaluation-time action masking
        self._evaluation_env = None
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

        self.training_steps = 0

    # ======================================================
    # ACTION SELECTION
    # ======================================================




    # ======================================================
    # JOINT ACTION ENCODING / DECODING
    # Phase 4.2 Centralized DQN
    # ======================================================
    def encode_joint_action(
        self,
        actions,
    ):
        """
        Convert individual AGV actions into one joint index.
        Example:
            [0,1,4] -> joint index
        """
        index = 0
        for action in actions:
            index = (
                index * self.action_size
                + int(action)
            )
        return int(index)
    def decode_joint_action(
        self,
        index,
    ):
        """
        Convert joint index into AGV actions.
        Example:
            joint index -> [0,1,4]
        """
        actions = []
        value = int(index)
        for _ in range(
            self.num_agvs
        ):
            actions.append(
                value % self.action_size
            )
            value //= self.action_size
        actions.reverse()
        return np.asarray(
            actions,
            dtype=np.int64
        )
    def select_action(
        self,
        state,
        training=True,
    ):
        """
        Phase 4.2 centralized joint action selection.
        Output:
            [action_agv1, action_agv2, action_agv3]
        """
        if (
            training
            and random.random() < self.epsilon
        ):
            joint_index = np.random.randint(
                0,
                self.joint_action_size
            )
            return self.decode_joint_action(
                joint_index
            )
        state_array = np.asarray(
            state,
            dtype=np.float32
        )
        state_tensor = torch.as_tensor(
            state_array,
            dtype=torch.float32,
            device=self.device
        ).unsqueeze(0)
        with torch.no_grad():
            q_values = (
                self.policy_network.action_values(
                    state_tensor
                )
            )
        if q_values.ndim != 2:
            raise RuntimeError(
                f"Expected Q shape [1,125], got {q_values.shape}"
            )
        joint_index = int(
            torch.argmax(
                q_values,
                dim=1
            ).item()
        )
        return self.decode_joint_action(
            joint_index
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

    def train_step(self):

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
        # CURRENT Q VALUES
        # ==================================================

        current_q = (
            self.policy_network.action_values(
                states_tensor
            )
        )

        # Shape:
        #
        # [batch_size, num_agvs, action_size]
        #
        # Select the action actually taken by
        # every AGV.

        chosen_q = (
            current_q
            .gather(
                2,
                actions_tensor.unsqueeze(-1),
            )
            .squeeze(-1)
        )

        # ==================================================
        # TARGET Q VALUES
        # ==================================================

        with torch.no_grad():

            next_q = (
                self.target_network.action_values(
                    next_states_tensor
                )
            )

            # Best next action for every AGV.
            #
            # Shape:
            #
            # [batch_size, num_agvs]

            max_next_q = (
                next_q
                .max(
                    dim=2
                )
                .values
            )

            # --------------------------------------------------
            # Distribute the joint environment reward across
            # AGVs instead of averaging Q-values first.
            # --------------------------------------------------

            reward_per_agv = (
                rewards_tensor.unsqueeze(1)
                /
                float(self.num_agvs)
            )

            # --------------------------------------------------
            # Terminal mask
            # --------------------------------------------------

            done_mask = (
                1.0
                -
                dones_tensor.unsqueeze(1)
            )

            # --------------------------------------------------
            # Bellman target
            # --------------------------------------------------

            target_q = (
                reward_per_agv
                +
                self.gamma
                *
                max_next_q
                *
                done_mask
            )

        # ==================================================
        # HUBER LOSS
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

        # --------------------------------------------------
        # Gradient clipping
        # --------------------------------------------------

        torch.nn.utils.clip_grad_norm_(
            self.policy_network.parameters(),
            max_norm=1.0,
        )

        self.optimizer.step()

        self.training_steps += 1

        # ==================================================
        # SOFT TARGET UPDATE
        # ==================================================

        self.soft_update()

        return float(
            loss.item()
        )

    # ======================================================
    # SOFT UPDATE
    # ======================================================

    def soft_update(self):

        with torch.no_grad():

            for target_param, policy_param in zip(
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

    def decay_epsilon(self):

        self.epsilon = max(
            self.epsilon_min,
            self.epsilon
            *
            self.epsilon_decay,
        )

    # ======================================================
    # EVALUATION MODE
    # ======================================================

    def evaluation_mode(self):
    
     self._evaluation_epsilon = self.epsilon

     self.policy_network.eval()
     self.target_network.eval()

     self.epsilon = 0.0

    # ======================================================
    # TRAINING MODE
    # ======================================================

    def training_mode(self):
    

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

        if "optimizer" in checkpoint:

            self.optimizer.load_state_dict(
                checkpoint[
                    "optimizer"
                ]
            )

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

