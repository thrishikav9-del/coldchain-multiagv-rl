"""
DQN Trainer for the Multi-AGV warehouse.

Features:
- Gymnasium environment integration
- Multi-AGV DQN training
- Experience replay
- Periodic checkpoints
- Resume training
- Optional GPU / CPU fallback
- Reduced console output
- Episode statistics
"""

from __future__ import annotations

import os
import random
from typing import Optional

import numpy as np
import torch

from environment.multi_agv_env import MultiAGVEnv
from algorithms.dqn_agent import DQNAgent


class DQNTrainer:

    def __init__(
        self,
        episodes: int = 5,
        max_steps: int = 500,
        num_agvs: int = 1,
        width: int = 20,
        height: int = 20,
        seed: int = 42,

        # --------------------------------------------------
        # DQN hyperparameters
        # --------------------------------------------------

        learning_rate: float = 0.0001,
        gamma: float = 0.99,
        tau: float = 0.001,

        epsilon: float = 1.0,
        epsilon_min: float = 0.05,
        epsilon_decay: float = 0.995,

        replay_capacity: int = 200_000,
        batch_size: int = 64,

        # --------------------------------------------------
        # Model
        # --------------------------------------------------

        model_path: str = "models/dqn_model.pt",

        # --------------------------------------------------
        # Checkpoint / resume
        # --------------------------------------------------

        checkpoint_interval: int = 100,
        checkpoint_dir: str = "models/checkpoints",
        resume: bool = False,
        resume_from: Optional[str] = None,

        # --------------------------------------------------
        # Console
        # --------------------------------------------------

        verbose: bool = True,
        print_every: int = 50,

        # --------------------------------------------------
        # Device
        # --------------------------------------------------

        device: Optional[str] = None,
    ):

        self.episodes = int(episodes)
        self.max_steps = int(max_steps)
        self.num_agvs = int(num_agvs)

        self.width = int(width)
        self.height = int(height)

        self.seed = int(seed)

        self.model_path = str(model_path)

        # --------------------------------------------------
        # Checkpoint settings
        # --------------------------------------------------

        self.checkpoint_interval = int(
            checkpoint_interval
        )

        self.checkpoint_dir = str(
            checkpoint_dir
        )

        self.resume = bool(resume)

        self.resume_from = (
            str(resume_from)
            if resume_from is not None
            else None
        )

        # --------------------------------------------------
        # Console settings
        # --------------------------------------------------

        self.verbose = bool(verbose)

        self.print_every = int(
            print_every
        )

        # --------------------------------------------------
        # Validation
        # --------------------------------------------------

        if self.episodes <= 0:
            raise ValueError(
                "episodes must be positive."
            )

        if self.max_steps <= 0:
            raise ValueError(
                "max_steps must be positive."
            )

        if self.num_agvs <= 0:
            raise ValueError(
                "num_agvs must be positive."
            )

        if self.checkpoint_interval < 0:
            raise ValueError(
                "checkpoint_interval cannot be negative."
            )

        if self.print_every < 0:
            raise ValueError(
                "print_every cannot be negative."
            )

        # --------------------------------------------------
        # Reproducibility
        # --------------------------------------------------

        random.seed(self.seed)

        np.random.seed(
            self.seed
        )

        torch.manual_seed(
            self.seed
        )

        if torch.cuda.is_available():

            torch.cuda.manual_seed_all(
                self.seed
            )

        # --------------------------------------------------
        # Device
        #
        # "auto" / None:
        #     CUDA if available, otherwise CPU
        #
        # "cuda":
        #     CUDA if available, otherwise CPU
        #
        # "cpu":
        #     CPU
        # --------------------------------------------------

        self.device = self._resolve_device(
            device
        )

        # --------------------------------------------------
        # Environment
        # --------------------------------------------------

        self.env = MultiAGVEnv(
            width=self.width,
            height=self.height,
            num_agvs=self.num_agvs,
            max_steps=self.max_steps,
        )

        # --------------------------------------------------
        # Determine dimensions
        # --------------------------------------------------

        self.state_size = int(
            self.env.observation_space.shape[0]
        )

        self.action_size = 5

        # --------------------------------------------------
        # DQN Agent
        # --------------------------------------------------

        self.agent = DQNAgent(
            state_size=self.state_size,
            num_agvs=self.num_agvs,
            action_size=self.action_size,

            learning_rate=learning_rate,
            gamma=gamma,
            tau=tau,

            epsilon=epsilon,
            epsilon_min=epsilon_min,
            epsilon_decay=epsilon_decay,

            replay_capacity=replay_capacity,
            batch_size=batch_size,

            seed=self.seed,
        )

        # --------------------------------------------------
        # Move networks to selected device
        #
        # DQNAgent normally chooses its own device.
        # If it exposes policy/target networks, explicitly
        # move them to the trainer's selected device.
        # --------------------------------------------------

        self._configure_agent_device()

        # --------------------------------------------------
        # Training history
        # --------------------------------------------------

        self.rewards = []
        self.steps = []
        self.completions = []
        self.losses = []
        self.collisions = []

        # --------------------------------------------------
        # Resume state
        # --------------------------------------------------

        self.start_episode = 1

        self.total_episodes_completed = 0

        if self.resume or self.resume_from is not None:

            self._resume_training()

    # ======================================================
    # DEVICE
    # ======================================================

    @staticmethod
    def _resolve_device(
        requested_device: Optional[str],
    ) -> torch.device:

        if requested_device is None:

            requested_device = "auto"

        requested_device = str(
            requested_device
        ).lower().strip()

        # --------------------------------------------------
        # Automatic selection
        # --------------------------------------------------

        if requested_device == "auto":

            if torch.cuda.is_available():

                return torch.device(
                    "cuda"
                )

            return torch.device(
                "cpu"
            )

        # --------------------------------------------------
        # Explicit CPU
        # --------------------------------------------------

        if requested_device == "cpu":

            return torch.device(
                "cpu"
            )

        # --------------------------------------------------
        # Explicit CUDA
        # --------------------------------------------------

        if requested_device in {
            "cuda",
            "gpu",
        }:

            if torch.cuda.is_available():

                return torch.device(
                    "cuda"
                )

            print(
                "WARNING: CUDA requested but "
                "CUDA is unavailable. "
                "Falling back to CPU."
            )

            return torch.device(
                "cpu"
            )

        raise ValueError(
            "device must be one of: "
            "'auto', 'cpu', or 'cuda'."
        )

    # ======================================================
    # AGENT DEVICE
    # ======================================================

    def _configure_agent_device(
        self,
    ):

        # --------------------------------------------------
        # DQNAgent usually has a device attribute.
        # --------------------------------------------------

        if hasattr(
            self.agent,
            "device",
        ):

            self.agent.device = (
                self.device
            )

        # --------------------------------------------------
        # Move policy network
        # --------------------------------------------------

        if hasattr(
            self.agent,
            "policy_network",
        ):

            self.agent.policy_network = (
                self.agent.policy_network.to(
                    self.device
                )
            )

        # --------------------------------------------------
        # Move target network
        # --------------------------------------------------

        if hasattr(
            self.agent,
            "target_network",
        ):

            self.agent.target_network = (
                self.agent.target_network.to(
                    self.device
                )
            )

    # ======================================================
    # CHECKPOINT PATH
    # ======================================================

    def _checkpoint_path(
        self,
        episode: int,
    ) -> str:

        os.makedirs(
            self.checkpoint_dir,
            exist_ok=True,
        )

        return os.path.join(
            self.checkpoint_dir,
            f"dqn_checkpoint_ep_{episode:05d}.pt",
        )

    # ======================================================
    # SAVE CHECKPOINT
    # ======================================================

    def save_checkpoint(
        self,
        episode: int,
    ):

        filepath = (
            self._checkpoint_path(
                episode
            )
        )

        directory = os.path.dirname(
            filepath
        )

        if directory:

            os.makedirs(
                directory,
                exist_ok=True,
            )

        # --------------------------------------------------
        # Prefer a trainer-aware checkpoint.
        # --------------------------------------------------

        checkpoint = {
            "episode":
                int(episode),

            "seed":
                self.seed,

            "episodes":
                self.episodes,

            "max_steps":
                self.max_steps,

            "num_agvs":
                self.num_agvs,

            "width":
                self.width,

            "height":
                self.height,

            "state_size":
                self.state_size,

            "action_size":
                self.action_size,

            "device":
                str(self.device),

            "rewards":
                self.rewards,

            "steps":
                self.steps,

            "completions":
                self.completions,

            "losses":
                self.losses,

            "collisions":
                self.collisions,

            "agent":
                self._agent_checkpoint(),
        }

        torch.save(
            checkpoint,
            filepath,
        )

        return filepath

    # ======================================================
    # AGENT CHECKPOINT
    # ======================================================

    def _agent_checkpoint(
        self,
    ):

        result = {}

        if hasattr(
            self.agent,
            "policy_network",
        ):

            result[
                "policy_network"
            ] = (
                self.agent.policy_network.state_dict()
            )

        if hasattr(
            self.agent,
            "target_network",
        ):

            result[
                "target_network"
            ] = (
                self.agent.target_network.state_dict()
            )

        if hasattr(
            self.agent,
            "optimizer",
        ):

            result[
                "optimizer"
            ] = (
                self.agent.optimizer.state_dict()
            )

        if hasattr(
            self.agent,
            "epsilon",
        ):

            result[
                "epsilon"
            ] = self.agent.epsilon

        if hasattr(
            self.agent,
            "train_steps",
        ):

            result[
                "train_steps"
            ] = self.agent.train_steps

        return result

    # ======================================================
    # LOAD CHECKPOINT
    # ======================================================

    def load_checkpoint(
        self,
        filepath: str,
    ):

        if not os.path.exists(
            filepath
        ):

            raise FileNotFoundError(
                f"Checkpoint not found: "
                f"{filepath}"
            )

        checkpoint = torch.load(
            filepath,
            map_location=self.device,
        )

        # --------------------------------------------------
        # Agent
        # --------------------------------------------------

        agent_data = checkpoint.get(
            "agent"
        )

        if agent_data is not None:

            if (
                "policy_network"
                in agent_data
                and hasattr(
                    self.agent,
                    "policy_network",
                )
            ):

                self.agent.policy_network.load_state_dict(
                    agent_data[
                        "policy_network"
                    ]
                )

            if (
                "target_network"
                in agent_data
                and hasattr(
                    self.agent,
                    "target_network",
                )
            ):

                self.agent.target_network.load_state_dict(
                    agent_data[
                        "target_network"
                    ]
                )

            if (
                "optimizer"
                in agent_data
                and hasattr(
                    self.agent,
                    "optimizer",
                )
            ):

                try:

                    self.agent.optimizer.load_state_dict(
                        agent_data[
                            "optimizer"
                        ]
                    )

                except Exception:

                    # Optimizer device incompatibilities
                    # should not prevent model recovery.
                    pass

            if (
                "epsilon"
                in agent_data
                and hasattr(
                    self.agent,
                    "epsilon",
                )
            ):

                self.agent.epsilon = float(
                    agent_data[
                        "epsilon"
                    ]
                )

            if (
                "train_steps"
                in agent_data
                and hasattr(
                    self.agent,
                    "train_steps",
                )
            ):

                self.agent.train_steps = int(
                    agent_data[
                        "train_steps"
                    ]
                )

        else:

            # --------------------------------------------------
            # Compatibility with existing DQNAgent.save()
            # --------------------------------------------------

            self.agent.load(
                filepath
            )

        # --------------------------------------------------
        # Training history
        # --------------------------------------------------

        self.rewards = list(
            checkpoint.get(
                "rewards",
                self.rewards,
            )
        )

        self.steps = list(
            checkpoint.get(
                "steps",
                self.steps,
            )
        )

        self.completions = list(
            checkpoint.get(
                "completions",
                self.completions,
            )
        )

        self.losses = list(
            checkpoint.get(
                "losses",
                self.losses,
            )
        )

        self.collisions = list(
            checkpoint.get(
                "collisions",
                self.collisions,
            )
        )

        saved_episode = int(
            checkpoint.get(
                "episode",
                0,
            )
        )

        self.total_episodes_completed = (
            saved_episode
        )

        self.start_episode = (
            saved_episode + 1
        )

        self._configure_agent_device()

        return checkpoint

    # ======================================================
    # FIND LATEST CHECKPOINT
    # ======================================================

    def find_latest_checkpoint(
        self,
    ) -> Optional[str]:

        if not os.path.isdir(
            self.checkpoint_dir
        ):

            return None

        candidates = []

        for filename in os.listdir(
            self.checkpoint_dir
        ):

            if not filename.startswith(
                "dqn_checkpoint_ep_"
            ):

                continue

            if not filename.endswith(
                ".pt"
            ):

                continue

            filepath = os.path.join(
                self.checkpoint_dir,
                filename,
            )

            candidates.append(
                filepath
            )

        if not candidates:

            return None

        def episode_number(
            filepath
        ):

            filename = os.path.basename(
                filepath
            )

            try:

                text = filename[
                    len(
                        "dqn_checkpoint_ep_"
                    ):
                    -len(".pt")
                ]

                return int(
                    text
                )

            except Exception:

                return -1

        candidates.sort(
            key=episode_number
        )

        return candidates[-1]

    # ======================================================
    # RESUME
    # ======================================================

    def _resume_training(
        self,
    ):

        filepath = self.resume_from

        if filepath is None:

            filepath = (
                self.find_latest_checkpoint()
            )

        if filepath is None:

            if self.verbose:

                print(
                    "No checkpoint found. "
                    "Training will start from episode 1."
                )

            return

        checkpoint = (
            self.load_checkpoint(
                filepath
            )
        )

        if self.verbose:

            print()
            print(
                f"RESUMED FROM: {filepath}"
            )

            print(
                f"Next episode: "
                f"{self.start_episode}"
            )

    # ======================================================
    # TRAIN ONE EPISODE
    # ======================================================

    def train_episode(
        self,
        episode: int,
    ):

        reset_seed = (
            self.seed + episode - 1
        )

        state, info = (
            self.env.reset(
                seed=reset_seed
            )
        )

        total_reward = 0.0
        total_loss = 0.0
        loss_count = 0

        completed_tasks = 0
        collision_count = 0

        actual_steps = 0

        for step in range(
            self.max_steps
        ):

            actual_steps += 1

            # --------------------------------------------------
            # Select one action for every AGV
            # --------------------------------------------------

            actions = (
                self.agent.select_action(
                    state,
                    training=True,
                )
            )

            # --------------------------------------------------
            # Environment step
            # --------------------------------------------------

            (
                next_state,
                reward,
                terminated,
                truncated,
                info,
            ) = self.env.step(
                actions
            )

            done = (
                terminated
                or
                truncated
            )

            # --------------------------------------------------
            # Replay
            # --------------------------------------------------

            self.agent.remember(
                state,
                actions,
                reward,
                next_state,
                done,
            )

            # --------------------------------------------------
            # DQN update
            # --------------------------------------------------

            loss = (
                self.agent.train_step()
            )

            if loss is not None:

                total_loss += float(
                    loss
                )

                loss_count += 1

            # --------------------------------------------------
            # Statistics
            # --------------------------------------------------

            total_reward += float(
                reward
            )

            completed_tasks = int(
                info.get(
                    "completed_tasks",
                    completed_tasks,
                )
            )

            collision_count += int(
                info.get(
                    "collision_count",
                    0,
                )
            )

            state = next_state

            if done:

                break

        # --------------------------------------------------
        # Decay once per episode
        # --------------------------------------------------

        self.agent.decay_epsilon()

        average_loss = (
            total_loss / loss_count
            if loss_count > 0
            else 0.0
        )

        self.rewards.append(
            total_reward
        )

        self.steps.append(
            actual_steps
        )

        self.completions.append(
            completed_tasks
        )

        self.losses.append(
            average_loss
        )

        self.collisions.append(
            collision_count
        )

        return {
            "episode":
                episode,

            "reward":
                total_reward,

            "steps":
                actual_steps,

            "completed_tasks":
                completed_tasks,

            "collisions":
                collision_count,

            "loss":
                average_loss,

            "epsilon":
                self.agent.epsilon,
        }

    # ======================================================
    # TRAIN
    # ======================================================

    def train(
        self,
        verbose: Optional[bool] = None,
    ):

        if verbose is None:

            verbose = self.verbose

        # --------------------------------------------------
        # If already finished, do not silently train again.
        # --------------------------------------------------

        if self.start_episode > self.episodes:

            if verbose:

                print(
                    "Training already completed."
                )

            self.save_model(
                self.model_path
            )

            return self._training_results()

        if verbose:

            print("=" * 72)
            print("DQN TRAINING")
            print("=" * 72)

            print(
                f"Episodes      : "
                f"{self.episodes}"
            )

            print(
                f"Start episode : "
                f"{self.start_episode}"
            )

            print(
                f"Max steps     : "
                f"{self.max_steps}"
            )

            print(
                f"AGVs          : "
                f"{self.num_agvs}"
            )

            print(
                f"State size    : "
                f"{self.state_size}"
            )

            print(
                f"Action size   : "
                f"{self.action_size}"
            )

            print(
                f"Device        : "
                f"{self.device}"
            )

            print(
                f"Learning rate : "
                f"{self.agent.learning_rate}"
            )

            print(
                f"Gamma         : "
                f"{self.agent.gamma}"
            )

            print(
                f"Tau           : "
                f"{self.agent.tau}"
            )

            print(
                f"Replay        : "
                f"{self.agent.replay_buffer.capacity}"
            )

            print(
                f"Batch         : "
                f"{self.agent.batch_size}"
            )

            print(
                f"Checkpoint    : "
                f"every "
                f"{self.checkpoint_interval} "
                f"episodes"
            )

            print("=" * 72)

        for episode in range(
            self.start_episode,
            self.episodes + 1,
        ):

            result = (
                self.train_episode(
                    episode
                )
            )

            self.total_episodes_completed = (
                episode
            )

            # --------------------------------------------------
            # Reduced console output
            # --------------------------------------------------

            should_print = (
                verbose
                and
                (
                    episode == self.start_episode
                    or
                    episode == self.episodes
                    or
                    (
                        self.print_every > 0
                        and
                        episode % self.print_every == 0
                    )
                )
            )

            if should_print:

                print(
                    f"Episode "
                    f"{episode:04d}/"
                    f"{self.episodes:04d} | "
                    f"Reward="
                    f"{result['reward']:.2f} | "
                    f"Steps="
                    f"{result['steps']} | "
                    f"Completed="
                    f"{result['completed_tasks']} | "
                    f"Collision="
                    f"{result['collisions']} | "
                    f"Loss="
                    f"{result['loss']:.5f} | "
                    f"Epsilon="
                    f"{result['epsilon']:.4f}"
                )

            # --------------------------------------------------
            # Checkpoint
            # --------------------------------------------------

            if (
                self.checkpoint_interval > 0
                and
                (
                    episode
                    %
                    self.checkpoint_interval
                    == 0
                    or
                    episode == self.episodes
                )
            ):

                checkpoint = (
                    self.save_checkpoint(
                        episode
                    )
                )

                if verbose:

                    print(
                        f"CHECKPOINT SAVED: "
                        f"{checkpoint}"
                    )

        # --------------------------------------------------
        # Final model
        # --------------------------------------------------

        self.save_model(
            self.model_path
        )

        if verbose:

            print()
            print(
                f"MODEL SAVED: "
                f"{self.model_path}"
            )

        return self._training_results()

    # ======================================================
    # RESULTS
    # ======================================================

    def _training_results(
        self,
    ):

        return {
            "rewards":
                self.rewards.copy(),

            "steps":
                self.steps.copy(),

            "completions":
                self.completions.copy(),

            "losses":
                self.losses.copy(),

            "collisions":
                self.collisions.copy(),
        }

    # ======================================================
    # SAVE MODEL
    # ======================================================

    def save_model(
        self,
        filepath,
    ):

        directory = os.path.dirname(
            filepath
        )

        if directory:

            os.makedirs(
                directory,
                exist_ok=True,
            )

        self.agent.save(
            filepath
        )

    # ======================================================
    # LOAD MODEL
    # ======================================================

    def load_model(
        self,
        filepath,
    ):

        self.agent.load(
            filepath
        )

        self._configure_agent_device()

    # ======================================================
    # EVALUATION
    # ======================================================

    def evaluate(
        self,
        episodes: int = 10,
        max_steps: int | None = None,
    ):

        episodes = int(
            episodes
        )

        if episodes <= 0:

            raise ValueError(
                "episodes must be positive."
            )

        if max_steps is None:

            max_steps = self.max_steps

        max_steps = int(
            max_steps
        )

        if max_steps <= 0:

            raise ValueError(
                "max_steps must be positive."
            )

        self.agent.evaluation_mode()

        evaluation_rewards = []
        evaluation_completions = []
        evaluation_steps = []
        evaluation_collisions = []

        for episode in range(
            episodes
        ):

            state, info = (
                self.env.reset(
                    seed=(
                        self.seed
                        + 10_000
                        + episode
                    )
                )
            )

            total_reward = 0.0
            completed = 0
            collisions = 0
            steps = 0

            for step in range(
                max_steps
            ):

                actions = (
                    self.agent.select_action(
                        state,
                        training=False,
                    )
                )

                (
                    next_state,
                    reward,
                    terminated,
                    truncated,
                    info,
                ) = self.env.step(
                    actions
                )

                total_reward += float(
                    reward
                )

                completed = int(
                    info.get(
                        "completed_tasks",
                        completed,
                    )
                )

                collisions += int(
                    info.get(
                        "collision_count",
                        0,
                    )
                )

                steps += 1

                state = next_state

                if (
                    terminated
                    or
                    truncated
                ):

                    break

            evaluation_rewards.append(
                total_reward
            )

            evaluation_completions.append(
                completed
            )

            evaluation_steps.append(
                steps
            )

            evaluation_collisions.append(
                collisions
            )

        self.agent.training_mode()

        return {
            "rewards":
                evaluation_rewards,

            "completions":
                evaluation_completions,

            "steps":
                evaluation_steps,

            "collisions":
                evaluation_collisions,

            "average_reward":
                float(
                    np.mean(
                        evaluation_rewards
                    )
                ),

            "average_completion":
                float(
                    np.mean(
                        evaluation_completions
                    )
                ),

            "average_steps":
                float(
                    np.mean(
                        evaluation_steps
                    )
                ),

            "average_collisions":
                float(
                    np.mean(
                        evaluation_collisions
                    )
                ),
        }


__all__ = [
    "DQNTrainer",
]