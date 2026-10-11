"""
Centralized Deep Q-Network for Multi-AGV warehouse.

For N AGVs and 5 actions per AGV:

    joint_action_size = 5 ** N

Example:

    3 AGVs
    5 actions per AGV
    5 ** 3 = 125 joint actions

Architecture:

    state_size
        ↓
    128 neurons
        ↓
    128 neurons
        ↓
    joint_action_size

For the current environment:

    18 → 128 → 128 → 125
"""

from __future__ import annotations

import torch
import torch.nn as nn


class DQNNetwork(nn.Module):

    def __init__(
        self,
        state_size: int,
        num_agvs: int,
        action_size: int = 5,
    ):
        super().__init__()

        self.state_size = int(state_size)
        self.num_agvs = int(num_agvs)
        self.action_size = int(action_size)

        # --------------------------------------------------
        # VALIDATION
        # --------------------------------------------------

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
        # JOINT ACTION SPACE
        # --------------------------------------------------

        self.joint_action_size = (
            self.action_size ** self.num_agvs
        )

        # --------------------------------------------------
        # CENTRALIZED DQN
        # --------------------------------------------------
        #
        # Current paper configuration:
        #
        # 18 → 128 → 128 → 125
        #
        # --------------------------------------------------

        self.network = nn.Sequential(

            nn.Linear(
                self.state_size,
                128,
            ),

            nn.ReLU(),

            nn.Linear(
                128,
                128,
            ),

            nn.ReLU(),

            nn.Linear(
                128,
                self.joint_action_size,
            ),
        )

    # ======================================================
    # FORWARD
    # ======================================================

    def forward(
        self,
        state: torch.Tensor,
    ) -> torch.Tensor:

        if not isinstance(
            state,
            torch.Tensor,
        ):
            state = torch.as_tensor(
                state,
                dtype=torch.float32,
            )

        if state.ndim == 1:
            state = state.unsqueeze(0)

        if state.ndim != 2:
            raise ValueError(
                "State must have shape "
                "[batch, state_size]. "
                f"Got {tuple(state.shape)}."
            )

        if state.shape[-1] != self.state_size:
            raise ValueError(
                f"Expected state size "
                f"{self.state_size}, "
                f"got {state.shape[-1]}."
            )

        return self.network(state)

    # ======================================================
    # ACTION VALUES
    # ======================================================

    def action_values(
        self,
        state: torch.Tensor,
    ) -> torch.Tensor:

        """
        Return Q-values for every joint action.

        For 3 AGVs and 5 actions:

            [batch, 125]
        """

        return self.forward(state)


__all__ = [
    "DQNNetwork",
]