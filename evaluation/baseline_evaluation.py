"""
Evaluate a random multi-AGV policy.
"""

from __future__ import annotations

import numpy as np

from baselines.random_policy import (
    RandomPolicy,
)


def evaluate_random_policy(
    trainer,
    episodes=20,
):

    rewards = []
    steps = []
    completions = []

    for episode in range(
        episodes
    ):

        states, _ = trainer.env.reset(
            seed=(
                trainer.seed
                +
                20000
                +
                episode
            )
        )

        policies = [
            RandomPolicy(
                action_count=5,
                seed=(
                    trainer.seed
                    +
                    episode
                    +
                    agv_id
                ),
            )
            for agv_id in range(
                trainer.num_agvs
            )
        ]

        total_reward = 0.0
        completed = 0

        for step in range(
            trainer.max_steps
        ):

            actions = np.array(
                [
                    policy.select_action()
                    for policy in policies
                ],
                dtype=np.int64,
            )

            (
                states,
                reward,
                terminated,
                truncated,
                info,
            ) = trainer.env.step(
                actions
            )

            total_reward += float(
                reward
            )

            completed = int(
                info.get(
                    "completed_tasks",
                    0,
                )
            )

            if terminated or truncated:
                break

        rewards.append(
            total_reward
        )

        steps.append(
            step + 1
        )

        completions.append(
            completed
        )

    return {
        "episodes": episodes,

        "average_reward": float(
            np.mean(rewards)
        ),

        "best_reward": float(
            np.max(rewards)
        ),

        "average_steps": float(
            np.mean(steps)
        ),

        "average_completed": float(
            np.mean(completions)
        ),

        "rewards": rewards,

        "steps": steps,

        "completions": completions,
    }
