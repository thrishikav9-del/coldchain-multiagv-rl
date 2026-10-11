"""
Deep Q-Network for the Multi-AGV warehouse.

Phase 3:
- 4 features per AGV
- supports variable number of AGVs
- 5 movement actions per AGV
"""

import torch
import torch.nn as nn


class DQNNetwork(nn.Module):
    """
    Neural network used by the DQN agent.

    Input:
        flattened environment observation

    Output:
        Q-value for every AGV/action combination
    """

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

        output_size = (
            self.num_agvs
            * self.action_size
        )

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
                64,
            ),
            nn.ReLU(),

            nn.Linear(
                64,
                output_size,
            ),
        )

    def forward(
        self,
        state: torch.Tensor,
    ) -> torch.Tensor:

        if state.ndim == 1:
            state = state.unsqueeze(0)

        if state.shape[-1] != self.state_size:
            raise ValueError(
                f"Expected state size "
                f"{self.state_size}, "
                f"got {state.shape[-1]}."
            )

        return self.network(state)

    def action_values(
        self,
        state: torch.Tensor,
    ) -> torch.Tensor:
        """
        Return Q-values with shape:

            [batch, num_agvs, action_size]
        """

        q_values = self.forward(
            state
        )

        return q_values.view(
            -1,
            self.num_agvs,
            self.action_size,
        )


__all__ = [
    "DQNNetwork",
]
