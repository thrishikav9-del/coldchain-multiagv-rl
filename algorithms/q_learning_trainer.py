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

        # PHASE 23-24 risk-aware learning hooks.
        self.risk_aware_state_builder = (
            self.agents[0].build_risk_aware_state
        )
        self.risk_aware_reward_builder = (
            self.agents[0].risk_aware_reward_from_transition
        )

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

            risk_states = [
                self.risk_aware_state_builder(
                    self._get_agv_state(states, agv_id)
                )
                for agv_id in range(self.num_agvs)
            ]

            actions = np.array(
                [
                    agent.select_action(
                        risk_states[agv_id],
                        training=True,
                    )
                    for agv_id, agent
                    in enumerate(self.agents)
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

                state = risk_states[agv_id]

                next_state = self.risk_aware_state_builder(
                    self._get_agv_state(next_states, agv_id)
                )

                risk_reward = self.risk_aware_reward_builder(
                    operational_reward=float(reward),
                    previous_risk=float(info.get('risk_score', 0.0)),
                    current_risk=float(info.get('risk_score', 0.0)),
                )

                agent.update(
                    state=state,
                    action=int(
                        actions[agv_id]
                    ),
                    reward=risk_reward,
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
    # EVALUATE
    # ==================================================
    def evaluate(
        self,
        episodes=20,
        max_steps=None,
        seed=None,
    ):
        max_steps = int(max_steps or self.max_steps)
        base_seed = self.seed if seed is None else int(seed)
        rewards = []
        lengths = []
        completions = []
        collisions = []

        for episode in range(int(episodes)):
            states, _ = self.env.reset(seed=base_seed + episode)
            total_reward = 0.0
            completed = 0
            collision_count = 0

            for step in range(max_steps):
                actions = np.array([
                    agent.select_action(
                        self._get_agv_state(states, agv_id),
                        training=False,
                    )
                    for agv_id, agent in enumerate(self.agents)
                ], dtype=np.int64)

                next_states, reward, terminated, truncated, info = self.env.step(actions)
                total_reward += float(reward)
                completed = int(info.get('completed_tasks', completed))
                collision_count = int(info.get('episode_collision_count', info.get('collision_count', collision_count)))
                states = next_states

                if terminated or truncated:
                    break

            rewards.append(total_reward)
            lengths.append(step + 1)
            completions.append(completed)
            collisions.append(collision_count)

        return {
            'episodes': int(episodes),
            'max_steps': max_steps,
            'average_reward': float(np.mean(rewards)) if rewards else 0.0,
            'average_steps': float(np.mean(lengths)) if lengths else 0.0,
            'average_completed': float(np.mean(completions)) if completions else 0.0,
            'completion_rate': float(100.0 * np.mean(np.asarray(completions) > 0)) if completions else 0.0,
            'average_collisions': float(np.mean(collisions)) if collisions else 0.0,
            'rewards': rewards,
            'steps': lengths,
            'completions': completions,
            'collisions': collisions,
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


