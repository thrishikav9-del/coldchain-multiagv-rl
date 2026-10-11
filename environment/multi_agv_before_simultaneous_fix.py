"""
Multi-AGV manager for the base-paper warehouse.

Responsibilities:

    - maintain AGVs
    - assign tasks
    - detect position conflicts
    - prevent two AGVs from occupying the same cell
"""

from __future__ import annotations

from dataclasses import dataclass, field

from environment.agv import AGV
from environment.task import Task
from environment.warehouse import (
    Position,
    Warehouse,
)


@dataclass
class MultiAGVSystem:

    warehouse: Warehouse

    agvs: dict[int, AGV] = field(
        default_factory=dict
    )

    tasks: dict[int, Task] = field(
        default_factory=dict
    )

    # ==================================================
    # AGV MANAGEMENT
    # ==================================================

    def add_agv(
        self,
        agv: AGV,
    ) -> None:

        if agv.agv_id in self.agvs:

            raise ValueError(
                f"AGV {agv.agv_id} "
                "already exists."
            )

        if not self.warehouse.is_free(
            agv.position
        ):

            raise ValueError(
                f"AGV {agv.agv_id} "
                f"cannot start at "
                f"{agv.position}."
            )

        if any(
            other.position == agv.position
            for other in self.agvs.values()
        ):

            raise ValueError(
                f"Another AGV already "
                f"occupies {agv.position}."
            )

        self.agvs[
            agv.agv_id
        ] = agv

    # ==================================================
    # TASK MANAGEMENT
    # ==================================================

    def add_task(
        self,
        task: Task,
    ) -> None:

        if task.task_id in self.tasks:

            raise ValueError(
                f"Task {task.task_id} "
                "already exists."
            )

        if not self.warehouse.is_free(
            task.pickup
        ):

            raise ValueError(
                f"Pickup location "
                f"{task.pickup} is invalid."
            )

        if not self.warehouse.is_free(
            task.delivery
        ):

            raise ValueError(
                f"Delivery location "
                f"{task.delivery} is invalid."
            )

        self.tasks[
            task.task_id
        ] = task

    # ==================================================
    # ASSIGNMENT
    # ==================================================

    def assign_task(
        self,
        task_id: int,
        agv_id: int,
    ) -> None:

        if task_id not in self.tasks:

            raise KeyError(
                f"Unknown task {task_id}."
            )

        if agv_id not in self.agvs:

            raise KeyError(
                f"Unknown AGV {agv_id}."
            )

        task = self.tasks[
            task_id
        ]

        agv = self.agvs[
            agv_id
        ]

        task.assign(
            agv_id
        )

        agv.assign_task(
            task_id,
            task.pickup,
        )

    # ==================================================
    # OCCUPIED POSITIONS
    # ==================================================

    def occupied_positions(
        self,
        exclude_agv: int | None = None,
    ) -> set[Position]:

        return {
            agv.position
            for agv_id, agv
            in self.agvs.items()
            if agv_id != exclude_agv
        }

    # ==================================================
    # COLLISION CHECK
    # ==================================================

    def is_move_safe(
        self,
        agv_id: int,
        new_position: Position,
    ) -> bool:

        if agv_id not in self.agvs:

            raise KeyError(
                f"Unknown AGV {agv_id}."
            )

        if not self.warehouse.is_free(
            new_position
        ):

            return False

        return (
            new_position
            not in
            self.occupied_positions(
                exclude_agv=agv_id
            )
        )

    # ==================================================
    # MOVE AGV
    # ==================================================

    def move_agv(
        self,
        agv_id: int,
        new_position: Position,
    ) -> bool:

        if not self.is_move_safe(
            agv_id,
            new_position,
        ):

            return False

        return self.agvs[
            agv_id
        ].move_to(
            new_position,
            self.warehouse,
        )

    # ==================================================
    # CONFLICT DETECTION
    # ==================================================

    def has_collision(
        self,
    ) -> bool:

        positions = [
            agv.position
            for agv in self.agvs.values()
        ]

        return (
            len(positions)
            !=
            len(set(positions))
        )

    # ==================================================
    # ALL TASKS COMPLETE
    # ==================================================

    def all_tasks_completed(
        self,
    ) -> bool:

        if not self.tasks:

            return False

        return all(
            task.is_completed
            for task in self.tasks.values()
        )


__all__ = [
    "MultiAGVSystem",
]