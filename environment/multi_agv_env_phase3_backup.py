"""
Gymnasium environment for the multi-AGV warehouse.

Phase 1:
    - multiple AGVs
    - dynamic task
    - movement
    - collision prevention
    - reward
    - Gymnasium API

Cold-chain logic will be added later.
"""

from __future__ import annotations

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from environment.agv import AGV
from environment.multi_agv import MultiAGVSystem
from environment.task import Task
from environment.warehouse import Warehouse


class MultiAGVEnv(gym.Env):

    metadata = {
        "render_modes": ["ansi"]
    }

    # ==================================================
    # INITIALIZATION
    # ==================================================

    def __init__(
        self,
        width=20,
        height=20,
        num_agvs=3,
        max_steps=500,
        render_mode=None,
    ):

        super().__init__()

        self.width = int(width)
        self.height = int(height)
        self.num_agvs = int(num_agvs)
        self.max_steps = int(max_steps)

        self.render_mode = render_mode

        if self.num_agvs <= 0:
            raise ValueError(
                "num_agvs must be positive."
            )

        # --------------------------------------------------
        # Warehouse
        # --------------------------------------------------

        self.warehouse = Warehouse(
            width=self.width,
            height=self.height,
        )

        # --------------------------------------------------
        # One shared action space
        #
        # 0 = stay
        # 1 = right
        # 2 = left
        # 3 = up
        # 4 = down
        # --------------------------------------------------

        self.action_space = spaces.MultiDiscrete(
            [5] * self.num_agvs
        )

        # --------------------------------------------------
        # Observation
        #
        # Per AGV:
        #
        # x
        # y
        # goal_x
        # goal_y
        #
        # normalized to [0, 1]
        # --------------------------------------------------

        observation_size = (
            self.num_agvs * 4
        )

        self.observation_space = spaces.Box(
            low=0.0,
            high=1.0,
            shape=(observation_size,),
            dtype=np.float32,
        )

        self.system = None

        self.tasks = []

        self.current_step = 0

    # ==================================================
    # RESET
    # ==================================================

    def reset(
        self,
        *,
        seed=None,
        options=None,
    ):

        super().reset(
            seed=seed
        )

        self.current_step = 0

        # --------------------------------------------------
        # Fresh system
        # --------------------------------------------------

        self.system = MultiAGVSystem(
            warehouse=self.warehouse
        )

        self.tasks = []

        # --------------------------------------------------
        # Create AGVs
        # --------------------------------------------------

        for agv_id in range(
            1,
            self.num_agvs + 1,
        ):

            position = (
                0,
                agv_id - 1,
            )

            self.system.add_agv(
                AGV(
                    agv_id=agv_id,
                    position=position,
                )
            )

        # --------------------------------------------------
        # Create tasks
        # --------------------------------------------------

        for agv_id in range(
            1,
            self.num_agvs + 1,
        ):

            pickup = (
                min(
                    2 + agv_id,
                    self.width - 1,
                ),
                min(
                    2 + agv_id,
                    self.height - 1,
                ),
            )

            delivery = (
                self.width - 1,
                self.height - agv_id,
            )

            task = Task(
                task_id=agv_id,
                pickup=pickup,
                delivery=delivery,
            )

            self.system.add_task(
                task
            )

            self.system.assign_task(
                task_id=agv_id,
                agv_id=agv_id,
            )

            task.start()

            self.tasks.append(
                task
            )

            # After assignment, the AGV is going
            # to the pickup location.

        observation = self._get_observation()

        info = self._get_info()

        return (
            observation,
            info,
        )

    # ==================================================
    # STEP
    # ==================================================

    def step(
        self,
        actions,
    ):

        actions = np.asarray(
            actions,
            dtype=np.int64,
        )

        if actions.shape != (
            self.num_agvs,
        ):

            raise ValueError(
                f"Expected {self.num_agvs} "
                f"actions, got {actions.shape}."
            )

        self.current_step += 1

        total_reward = 0.0

        collision_count = 0

        completed_count = 0

        # --------------------------------------------------
        # Proposed positions
        # --------------------------------------------------

        proposed_positions = {}

        for agv_id in range(
            1,
            self.num_agvs + 1,
        ):

            agv = self.system.agvs[
                agv_id
            ]

            action = int(
                actions[
                    agv_id - 1
                ]
            )

            proposed_positions[
                agv_id
            ] = self._get_target_position(
                agv,
                action,
            )

        # --------------------------------------------------
        # Detect same-cell conflicts
        # --------------------------------------------------

        positions = list(
            proposed_positions.values()
        )

        duplicates = (
            len(positions)
            !=
            len(set(positions))
        )

        if duplicates:

            collision_count += 1

        # --------------------------------------------------
        # Execute movements
        # --------------------------------------------------

        for agv_id in range(
            1,
            self.num_agvs + 1,
        ):

            agv = self.system.agvs[
                agv_id
            ]

            action = int(
                actions[
                    agv_id - 1
                ]
            )

            target = proposed_positions[
                agv_id
            ]

            # --------------------------------------------------
            # Stay
            # --------------------------------------------------

            if action == 0:

                total_reward -= 0.1

                continue

            # --------------------------------------------------
            # Collision
            # --------------------------------------------------

            if duplicates:

                other_targets = [
                    pos
                    for other_id, pos
                    in proposed_positions.items()
                    if other_id != agv_id
                ]

                if target in other_targets:

                    collision_count += 1

                    total_reward -= 5.0

                    continue

            # --------------------------------------------------
            # Move
            # --------------------------------------------------

            moved = self.system.move_agv(
                agv_id,
                target,
            )

            if moved:

                total_reward -= 1.0

            else:

                total_reward -= 2.0

        # --------------------------------------------------
        # Task progression
        # --------------------------------------------------

        for task in self.tasks:

            if task.is_completed:

                continue

            agv_id = task.assigned_agv

            if agv_id is None:

                continue

            agv = self.system.agvs[
                agv_id
            ]

            # Pickup reached
            if (
                task.status == "IN_PROGRESS"
                and
                agv.position == task.pickup
            ):

                agv.destination = (
                    task.delivery
                )

            # Delivery reached
            if (
                agv.position
                ==
                task.delivery
                and
                task.status
                ==
                "IN_PROGRESS"
            ):

                task.complete()

                agv.complete_task()

                completed_count += 1

                total_reward += 20.0

        # --------------------------------------------------
        # Termination
        # --------------------------------------------------

        terminated = (
            completed_count > 0
            and
            self.system.all_tasks_completed()
        )

        truncated = (
            self.current_step
            >=
            self.max_steps
        )

        observation = (
            self._get_observation()
        )

        info = self._get_info()

        info.update(
            {
                "collision_count":
                    collision_count,

                "completed_tasks":
                    completed_count,

                "step":
                    self.current_step,
            }
        )

        return (
            observation,
            float(total_reward),
            terminated,
            truncated,
            info,
        )

    # ==================================================
    # TARGET POSITION
    # ==================================================

    def _get_target_position(
        self,
        agv,
        action,
    ):

        x, y = agv.position

        if action == 0:
            return (x, y)

        if action == 1:
            return (x + 1, y)

        if action == 2:
            return (x - 1, y)

        if action == 3:
            return (x, y + 1)

        if action == 4:
            return (x, y - 1)

        raise ValueError(
            f"Invalid action: {action}"
        )

    # ==================================================
    # OBSERVATION
    # ==================================================

    def _get_observation(self):

        values = []

        for agv_id in range(
            1,
            self.num_agvs + 1,
        ):

            agv = self.system.agvs[
                agv_id
            ]

            x, y = agv.position

            if agv.destination is None:

                gx, gy = x, y

            else:

                gx, gy = (
                    agv.destination
                )

            values.extend(
                [
                    x / max(
                        1,
                        self.width - 1,
                    ),

                    y / max(
                        1,
                        self.height - 1,
                    ),

                    gx / max(
                        1,
                        self.width - 1,
                    ),

                    gy / max(
                        1,
                        self.height - 1,
                    ),
                ]
            )

        return np.asarray(
            values,
            dtype=np.float32,
        )

    # ==================================================
    # INFO
    # ==================================================

    def _get_info(self):

        return {
            "step":
                self.current_step,

            "num_agvs":
                self.num_agvs,

            "completed_tasks":
                sum(
                    task.is_completed
                    for task in self.tasks
                ),

            "total_tasks":
                len(self.tasks),
        }

    # ==================================================
    # RENDER
    # ==================================================

    def render(self):

        positions = {
            agv_id: agv.position
            for agv_id, agv
            in self.system.agvs.items()
        }

        goals = [
            task.delivery
            for task in self.tasks
            if not task.is_completed
        ]

        return self.warehouse.render_ascii(
            agv_positions=positions,
            goals=goals,
        )

    # ==================================================
    # CLOSE
    # ==================================================

    def close(self):

        pass


__all__ = [
    "MultiAGVEnv",
]