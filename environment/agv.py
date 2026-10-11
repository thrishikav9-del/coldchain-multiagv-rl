"""
AGV model for the base multi-AGV warehouse system.
"""

from __future__ import annotations

from dataclasses import dataclass

from environment.warehouse import (
    Position,
    Warehouse,
)


@dataclass
class AGV:

    agv_id: int

    position: Position

    speed: float = 1.0

    destination: Position | None = None

    current_task: int | None = None

    status: str = "IDLE"

    distance_travelled: float = 0.0

    steps_taken: int = 0

    # ==================================================
    # TASK
    # ==================================================

    def assign_task(
        self,
        task_id: int,
        destination: Position,
    ) -> None:

        self.current_task = task_id

        self.destination = destination

        self.status = "MOVING"

    # ==================================================
    # GOAL
    # ==================================================

    @property
    def has_destination(self) -> bool:

        return self.destination is not None

    def reached_destination(self) -> bool:

        return (
            self.destination is not None
            and
            self.position
            ==
            self.destination
        )

    # ==================================================
    # MOVEMENT
    # ==================================================

    def move_to(
        self,
        new_position: Position,
        warehouse: Warehouse,
    ) -> bool:

        if not warehouse.is_free(
            new_position
        ):

            return False

        old_x, old_y = (
            self.position
        )

        new_x, new_y = (
            new_position
        )

        # Phase 1 uses unit-grid movement.
        if abs(
            new_x - old_x
        ) + abs(
            new_y - old_y
        ) != 1:

            return False

        self.position = new_position

        self.distance_travelled += 1.0

        self.steps_taken += 1

        if self.reached_destination():

            self.status = "IDLE"

        return True

    # ==================================================
    # COMPLETE TASK
    # ==================================================

    def complete_task(self) -> None:

        self.status = "IDLE"

        self.current_task = None

        self.destination = None


__all__ = [
    "AGV",
]
