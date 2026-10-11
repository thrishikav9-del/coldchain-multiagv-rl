"""
Evaluation utilities for the base multi-AGV Q-learning system.
"""

from __future__ import annotations

import numpy as np


def evaluate_agent(
    trainer,
    episodes=20,
):
    """
    Evaluate trained Q-learning agents without exploration.
    """

    rewards = []
    steps = []
    completions = []

    for episode in range(
        episodes
    ):

        states, _ = trainer.env.reset(
            seed=trainer.seed + 10000 + episode
        )

        total_reward = 0.0
        completed = 0

        for step in range(
            trainer.max_steps
        ):

            actions = np.array(
                [
                    agent.select_action(
                        trainer._get_agv_state(
                            states,
                            agv_id,
                        ),
                        training=False,
                    )
                    for agv_id, agent
                    in enumerate(
                        trainer.agents
                    )
                ],
                dtype=np.int64,
            )

            (
                next_states,
                reward,
                terminated,
                truncated,
                info,
            ) = trainer.env.step(
                actions
            )

            states = next_states

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
