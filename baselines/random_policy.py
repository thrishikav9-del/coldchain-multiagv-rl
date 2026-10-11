"""
Random-policy baseline for the multi-AGV warehouse.
"""

from __future__ import annotations

import random


class RandomPolicy:

    def __init__(
        self,
        action_count=5,
        seed=42,
    ):

        self.action_count = int(
            action_count
        )

        self.random = random.Random(
            seed
        )

    def select_action(self):

        return self.random.randrange(
            self.action_count
        )


__all__ = [
    "RandomPolicy",
]
