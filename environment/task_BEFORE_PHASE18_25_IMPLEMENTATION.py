"""
Task model for the Multi-AGV warehouse.

A task represents moving an AGV from a pickup location
to a delivery location.

Task lifecycle:

    PENDING
        ↓
    ASSIGNED
        ↓
    IN_PROGRESS
        ↓
    PICKED_UP
        ↓
    COMPLETED
"""

from __future__ import annotations

from dataclasses import dataclass

from environment.warehouse import Position


@dataclass
class Task:

    task_id: int

    pickup: Position

    delivery: Position

    status: str = "PENDING"

    assigned_agv: int | None = None

    # ==================================================
    # ASSIGN
    # ==================================================

    def assign(
        self,
        agv_id: int,
    ) -> None:

        if self.status != "PENDING":

            raise ValueError(
                f"Task {self.task_id} "
                f"cannot be assigned from "
                f"state {self.status}."
            )

        self.assigned_agv = int(
            agv_id
        )

        self.status = "ASSIGNED"

    # ==================================================
    # START
    # ==================================================

    def start(
        self,
    ) -> None:

        if self.status != "ASSIGNED":

            raise ValueError(
                f"Task {self.task_id} "
                f"must be assigned before starting."
            )

        self.status = "IN_PROGRESS"

    # ==================================================
    # PICKUP
    # ==================================================

    def pickup_completed(
        self,
    ) -> None:
        """
        Mark the pickup phase as completed.

        IN_PROGRESS -> PICKED_UP
        """

        if self.status != "IN_PROGRESS":

            raise ValueError(
                f"Task {self.task_id} "
                f"cannot be picked up from "
                f"state {self.status}."
            )

        self.status = "PICKED_UP"

    # ==================================================
    # COMPLETE
    # ==================================================

    def complete(
        self,
    ) -> None:

        if self.status != "PICKED_UP":

            raise ValueError(
                f"Task {self.task_id} "
                f"cannot be completed from "
                f"state {self.status}."
            )

        self.status = "COMPLETED"

    # ==================================================
    # STATUS HELPERS
    # ==================================================

    @property
    def is_completed(
        self,
    ) -> bool:

        return self.status == "COMPLETED"

    @property
    def is_picked_up(
        self,
    ) -> bool:

        return self.status == "PICKED_UP"

    @property
    def is_in_progress(
        self,
    ) -> bool:

        return self.status == "IN_PROGRESS"


__all__ = [
    "Task",
]