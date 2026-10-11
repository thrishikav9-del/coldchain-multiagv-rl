"""
Training loop for the base multi-AGV Q-learning system.
"""

from __future__ import annotations

import numpy as np

from algorithms.q_learning import QLearningAgent
from environment.multi_agv_env import MultiAGVEnv


class QLearningTrainer:

    def __init__(
        self,
        episodes: int = 100,
        max_steps: int = 500,
        num_agvs: int = 3,
        seed: int = 42,
    ):

        self.episodes = int(
            episodes
        )

        self.max_steps = int(
            max_steps
        )

        self.num_agvs = int(
            num_agvs
        )

        self.seed = int(
            seed
        )

        # --------------------------------------------------
        # Environment
        # --------------------------------------------------

        self.env = MultiAGVEnv(
            width=20,
            height=20,
            num_agvs=self.num_agvs,
            max_steps=self.max_steps,
            risk_aware_assignment=True,
        )

        # --------------------------------------------------
        # One Q-learning agent per AGV
        # --------------------------------------------------

        self.agents = [
            QLearningAgent(
                action_count=5,
                learning_rate=0.1,
                discount_factor=0.95,
                epsilon=1.0,
                epsilon_min=0.05,
                epsilon_decay=0.995,
                seed=self.seed + agv_id,
            )
            for agv_id in range(
                self.num_agvs
            )
        ]

        self.episode_rewards = []

        self.episode_lengths = []

        self.episode_completions = []

    # ==================================================
    # TRAIN ONE EPISODE
    # ==================================================

    def train_episode(
        self,
        episode: int,
    ):

        states, _ = self.env.reset(
            seed=self.seed + episode
        )

        total_reward = 0.0

        completed_tasks = 0

        for step in range(
            self.max_steps
        ):

            # --------------------------------------------------
            # Select action independently for every AGV
            # --------------------------------------------------

            actions = np.array(
                [
                    agent.select_action(
                        self._get_agv_state(
                            states,
                            agv_id,
                        ),
                        training=True,
                    )
                    for agv_id, agent
                    in enumerate(
                        self.agents
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
            ) = self.env.step(
                actions
            )

            # --------------------------------------------------
            # Update each Q learner
            # --------------------------------------------------

            for agv_id, agent in enumerate(
                self.agents
            ):

                state = self._get_agv_state(
                    states,
                    agv_id,
                )

                next_state = self._get_agv_state(
                    next_states,
                    agv_id,
                )

                agent.update(
                    state=state,
                    action=int(
                        actions[agv_id]
                    ),
                    reward=reward,
                    next_state=next_state,
                    terminated=(
                        terminated
                        or truncated
                    ),
                )

            states = next_states

            total_reward += float(
                reward
            )

            completed_tasks = int(
                info.get(
                    "completed_tasks",
                    0,
                )
            )

            if terminated or truncated:

                break

        # --------------------------------------------------
        # Exploration decay
        # --------------------------------------------------

        for agent in self.agents:

            agent.decay_epsilon()

        self.episode_rewards.append(
            total_reward
        )

        self.episode_lengths.append(
            step + 1
        )

        self.episode_completions.append(
            completed_tasks
        )

        return {
            "reward":
                total_reward,

            "steps":
                step + 1,

            "completed":
                completed_tasks,

            "epsilon":
                self.agents[0].epsilon,
        }

    # ==================================================
    # TRAIN
    # ==================================================

    def train(
        self,
    ):

        for episode in range(
            1,
            self.episodes + 1,
        ):

            result = self.train_episode(
                episode
            )

            if (
                episode == 1
                or
                episode % 10 == 0
                or
                episode == self.episodes
            ):

                print(
                    f"Episode "
                    f"{episode:04d}/"
                    f"{self.episodes:04d} | "
                    f"Reward="
                    f"{result['reward']:.2f} | "
                    f"Steps="
                    f"{result['steps']} | "
                    f"Completed="
                    f"{result['completed']} | "
                    f"Epsilon="
                    f"{result['epsilon']:.4f}"
                )

        return {
            "rewards":
                self.episode_rewards,

            "steps":
                self.episode_lengths,

            "completions":
                self.episode_completions,
        }

    # ==================================================
    # AGV STATE
    # ==================================================

        # ==================================================
    # MULTI-AGV STATE
    # ==================================================

    def _get_agv_state(
        self,
        state,
        agv_id,
    ):

        state = np.asarray(
            state,
            dtype=np.float32,
        )

        # --------------------------------------------------
        # Own AGV state
        # --------------------------------------------------

        own_start = (
            agv_id * 8
        )

        own_end = (
            own_start + 8
        )

        own_state = state[
            own_start:own_end
        ]

        # --------------------------------------------------
        # Other AGV relative states
        # --------------------------------------------------

        other_features = []

        for other_id in range(
            self.num_agvs
        ):

            if other_id == agv_id:

                continue

            other_start = (
                other_id * 8
            )

            other_end = (
                other_start + 8
            )

            other_state = state[
                other_start:other_end
            ]

            relative_state = (
                other_state
                -
                own_state
            )

            other_features.extend(
                relative_state.tolist()
            )

        # --------------------------------------------------
        # Combined state
        # --------------------------------------------------

        combined_state = np.concatenate(
            [
                own_state,
                np.asarray(
                    other_features,
                    dtype=np.float32,
                ),
            ]
        )

        return tuple(
            np.round(
                combined_state,
                6,
            )
        )

__all__ = [
    "QLearningTrainer",
]


