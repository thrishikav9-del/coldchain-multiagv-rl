"""
Tests for the Phase 1 warehouse and multi-AGV foundation.
"""

import numpy as np

from environment.warehouse import Warehouse
from environment.agv import AGV
from environment.task import Task
from environment.multi_agv import MultiAGVSystem


# ==========================================================
# WAREHOUSE TESTS
# ==========================================================

def test_warehouse_bounds():

    warehouse = Warehouse(
        width=10,
        height=10,
    )

    assert warehouse.is_inside(
        (0, 0)
    )

    assert warehouse.is_inside(
        (9, 9)
    )

    assert not warehouse.is_inside(
        (-1, 0)
    )

    assert not warehouse.is_inside(
        (10, 10)
    )


def test_obstacle():

    warehouse = Warehouse(
        width=10,
        height=10,
    )

    warehouse.add_obstacle(
        (3, 3)
    )

    assert warehouse.is_obstacle(
        (3, 3)
    )

    assert not warehouse.is_free(
        (3, 3)
    )


def test_neighbors():

    warehouse = Warehouse(
        width=10,
        height=10,
    )

    neighbors = warehouse.neighbors(
        (5, 5)
    )

    assert len(
        neighbors
    ) == 4


# ==========================================================
# AGV TEST
# ==========================================================

def test_agv_movement():

    warehouse = Warehouse(
        width=10,
        height=10,
    )

    agv = AGV(
        agv_id=1,
        position=(0, 0),
    )

    moved = agv.move_to(
        (1, 0),
        warehouse,
    )

    assert moved

    assert agv.position == (
        1,
        0,
    )


def test_agv_cannot_move_through_obstacle():

    warehouse = Warehouse(
        width=10,
        height=10,
    )

    warehouse.add_obstacle(
        (1, 0)
    )

    agv = AGV(
        agv_id=1,
        position=(0, 0),
    )

    moved = agv.move_to(
        (1, 0),
        warehouse,
    )

    assert not moved

    assert agv.position == (
        0,
        0,
    )


# ==========================================================
# MULTI-AGV TESTS
# ==========================================================

def test_multiple_agvs():

    warehouse = Warehouse(
        width=10,
        height=10,
    )

    system = MultiAGVSystem(
        warehouse=warehouse
    )

    agv1 = AGV(
        agv_id=1,
        position=(0, 0),
    )

    agv2 = AGV(
        agv_id=2,
        position=(0, 1),
    )

    system.add_agv(
        agv1
    )

    system.add_agv(
        agv2
    )

    assert len(
        system.agvs
    ) == 2


def test_task_assignment():

    warehouse = Warehouse(
        width=10,
        height=10,
    )

    system = MultiAGVSystem(
        warehouse=warehouse
    )

    agv = AGV(
        agv_id=1,
        position=(0, 0),
    )

    system.add_agv(
        agv
    )

    task = Task(
        task_id=1,
        pickup=(1, 0),
        delivery=(5, 5),
    )

    system.add_task(
        task
    )

    system.assign_task(
        task_id=1,
        agv_id=1,
    )

    assert task.status == "ASSIGNED"

    assert task.assigned_agv == 1

    assert agv.current_task == 1

    assert agv.destination == (
        1,
        0,
    )


def test_collision_prevention():

    warehouse = Warehouse(
        width=10,
        height=10,
    )

    system = MultiAGVSystem(
        warehouse=warehouse
    )

    system.add_agv(
        AGV(
            agv_id=1,
            position=(0, 0),
        )
    )

    system.add_agv(
        AGV(
            agv_id=2,
            position=(1, 0),
        )
    )

    # AGV 1 cannot move into AGV 2's cell.
    assert not system.is_move_safe(
        1,
        (1, 0),
    )

    # AGV 2 cannot move into AGV 1's cell.
    assert not system.is_move_safe(
        2,
        (0, 0),
    )


def test_multi_agv_movement():

    warehouse = Warehouse(
        width=10,
        height=10,
    )

    system = MultiAGVSystem(
        warehouse=warehouse
    )

    system.add_agv(
        AGV(
            agv_id=1,
            position=(0, 0),
        )
    )

    system.add_agv(
        AGV(
            agv_id=2,
            position=(5, 5),
        )
    )

    moved_1 = system.move_agv(
        1,
        (1, 0),
    )

    moved_2 = system.move_agv(
        2,
        (5, 6),
    )

    assert moved_1

    assert moved_2

    assert system.agvs[1].position == (
        1,
        0,
    )

    assert system.agvs[2].position == (
        5,
        6,
    )


# ==========================================================
# TASK COMPLETION TEST
# ==========================================================

def test_task_completion():

    task = Task(
        task_id=1,
        pickup=(1, 1),
        delivery=(5, 5),
    )

    task.assign(
        agv_id=1
    )

    task.start()

    assert task.status == "IN_PROGRESS"

    task.pickup_completed()

    assert task.status == "PICKED_UP"

    task.complete()

    assert task.status == "COMPLETED"