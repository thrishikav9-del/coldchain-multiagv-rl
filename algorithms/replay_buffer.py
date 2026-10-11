"""
Experience Replay Buffer for DQN.
"""

import random
from collections import deque

import numpy as np


class ReplayBuffer:

    def __init__(
        self,
        capacity: int = 200_000,
    ):
        if capacity <= 0:
            raise ValueError(
                "capacity must be positive."
            )

        self.capacity = int(
            capacity
        )

        self.buffer = deque(
            maxlen=self.capacity
        )

    def add(
        self,
        state,
        action,
        reward,
        next_state,
        done,
    ):

        self.buffer.append(
            (
                np.asarray(
                    state,
                    dtype=np.float32,
                ),
                np.asarray(
                    action,
                    dtype=np.int64,
                ),
                float(reward),
                np.asarray(
                    next_state,
                    dtype=np.float32,
                ),
                bool(done),
            )
        )

    def sample(
        self,
        batch_size: int,
    ):

        if len(self.buffer) < batch_size:
            raise ValueError(
                f"Not enough experiences. "
                f"Have {len(self.buffer)}, "
                f"need {batch_size}."
            )

        batch = random.sample(
            self.buffer,
            batch_size,
        )

        states, actions, rewards, next_states, dones = zip(
            *batch
        )

        return (
            np.asarray(
                states,
                dtype=np.float32,
            ),
            np.asarray(
                actions,
                dtype=np.int64,
            ),
            np.asarray(
                rewards,
                dtype=np.float32,
            ),
            np.asarray(
                next_states,
                dtype=np.float32,
            ),
            np.asarray(
                dones,
                dtype=np.float32,
            ),
        )

    def __len__(self):
        return len(self.buffer)

    def clear(self):
        self.buffer.clear()


__all__ = [
    "ReplayBuffer",
]
