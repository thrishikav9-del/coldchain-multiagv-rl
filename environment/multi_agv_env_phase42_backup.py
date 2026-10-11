"""
Gymnasium environment for the Multi-AGV Cold-Chain Warehouse.

Dataset
    ↓
Cold-chain risk information
    ↓
Task urgency / priority
    ↓
Multi-AGV observation
    ↓
DQN decision
    ↓
Pickup
    ↓
Delivery

Observation per AGV:
    x
    y
    goal_x
    goal_y
    risk
    urgency

For 3 AGVs:
    3 * 6 = 18 observations

Actions per AGV:
    0 = Stay
    1 = Right
    2 = Left
    3 = Up
    4 = Down
"""

from __future__ import annotations

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from dataset.risk_engine import ColdChainRiskEngine

from environment.agv import AGV
from environment.multi_agv import MultiAGVSystem
from environment.task import Task
from environment.warehouse import Warehouse


class MultiAGVEnv(gym.Env):

    metadata = {
        "render_modes": ["ansi"]
    }

    # ==========================================================
    # INITIALIZATION
    # ==========================================================

    def __init__(
        self,
        width=20,
        height=20,
        num_agvs=3,
        max_steps=500,
        render_mode=None,
        dataset_path="dataset/cold_chain_data.csv",
    ):
        super().__init__()

        self.width = int(width)
        self.height = int(height)
        self.num_agvs = int(num_agvs)
        self.max_steps = int(max_steps)
        self.render_mode = render_mode

        # ------------------------------------------------------
        # Dataset / Risk Engine
        # ------------------------------------------------------

        self.dataset_path = dataset_path

        self.risk_engine = ColdChainRiskEngine(
            self.dataset_path
        )

        if not hasattr(
            self.risk_engine,
            "data",
        ):
            raise AttributeError(
                "ColdChainRiskEngine must provide a 'data' attribute."
            )

        if len(
            self.risk_engine.data
        ) == 0:
            raise ValueError(
                "Cold-chain dataset is empty."
            )

        # ------------------------------------------------------
        # Validation
        # ------------------------------------------------------

        if self.width <= 0:
            raise ValueError(
                "width must be positive."
            )

        if self.height <= 0:
            raise ValueError(
                "height must be positive."
            )

        if self.num_agvs <= 0:
            raise ValueError(
                "num_agvs must be positive."
            )

        if self.num_agvs > self.height:
            raise ValueError(
                "num_agvs cannot exceed warehouse height."
            )

        if self.max_steps <= 0:
            raise ValueError(
                "max_steps must be positive."
            )

        # ------------------------------------------------------
        # Warehouse
        # ------------------------------------------------------

        self.warehouse = Warehouse(
            width=self.width,
            height=self.height,
        )

        # ------------------------------------------------------
        # Action space
        #
        # One action for each AGV.
        # ------------------------------------------------------

        self.action_space = spaces.MultiDiscrete(
            [5] * self.num_agvs
        )

        # ------------------------------------------------------
        # Observation space
        #
        # 6 values per AGV:
        #
        # x
        # y
        # goal_x
        # goal_y
        # risk
        # urgency
        #
        # For 3 AGVs = 18
        # ------------------------------------------------------

        observation_size = (
            self.num_agvs * 6
        )

        self.observation_space = spaces.Box(
            low=0.0,
            high=1.0,
            shape=(observation_size,),
            dtype=np.float32,
        )

        # ------------------------------------------------------
        # Runtime state
        # ------------------------------------------------------

        self.system = None
        self.tasks = []

        self.current_step = 0

        self.last_collision_count = 0
        self.last_completed_count = 0
        self.last_pickup_count = 0

        self.episode_dataset_start = 0

        self.task_risk = {}
        self.current_risk_records = {}

    # ==========================================================
    # DATASET RECORD
    # ==========================================================

    def _get_dataset_record(
        self,
        index,
    ):
        """
        Retrieve one dataset record through the risk engine.
        """

        data_size = len(
            self.risk_engine.data
        )

        if data_size <= 0:
            raise RuntimeError(
                "Cold-chain dataset has no records."
            )

        index = int(index) % data_size

        record = self.risk_engine.get_record(
            index=index
        )

        if not isinstance(
            record,
            dict,
        ):
            raise TypeError(
                "ColdChainRiskEngine.get_record() "
                "must return a dictionary."
            )

        result = dict(record)

        result["dataset_index"] = index

        # ------------------------------------------------------
        # Temperature
        # ------------------------------------------------------

        result["temperature"] = float(
            result.get(
                "temperature",
                result.get(
                    "iot_temperature",
                    0.0,
                ),
            )
        )

        # ------------------------------------------------------
        # Risk
        # ------------------------------------------------------

        result["risk_score"] = float(
            np.clip(
                result.get(
                    "risk_score",
                    0.0,
                ),
                0.0,
                1.0,
            )
        )

        # ------------------------------------------------------
        # Urgency
        # ------------------------------------------------------

        result["urgency_score"] = float(
            np.clip(
                result.get(
                    "urgency_score",
                    0.0,
                ),
                0.0,
                1.0,
            )
        )

        # ------------------------------------------------------
        # Priority
        # ------------------------------------------------------

        priority = result.get(
            "priority",
            None,
        )

        if priority is None:

            urgency = result[
                "urgency_score"
            ]

            if urgency >= 0.70:
                priority = "HIGH"

            elif urgency >= 0.40:
                priority = "MEDIUM"

            else:
                priority = "LOW"

        result["priority"] = str(
            priority
        ).upper()

        # ------------------------------------------------------
        # Weather
        # ------------------------------------------------------

        result["weather"] = float(
            np.clip(
                result.get(
                    "weather",
                    result.get(
                        "weather_risk",
                        result.get(
                            "weather_condition_severity",
                            0.0,
                        ),
                    ),
                ),
                0.0,
                1.0,
            )
        )

        # ------------------------------------------------------
        # Disruption
        # ------------------------------------------------------

        result["disruption"] = float(
            np.clip(
                result.get(
                    "disruption",
                    result.get(
                        "disruption_risk",
                        result.get(
                            "disruption_probability",
                            result.get(
                                "disruption_likelihood_score",
                                0.0,
                            ),
                        ),
                    ),
                ),
                0.0,
                1.0,
            )
        )

        # ------------------------------------------------------
        # Delay
        # ------------------------------------------------------

        result["delay"] = float(
            np.clip(
                result.get(
                    "delay",
                    result.get(
                        "delay_risk",
                        result.get(
                            "delay_probability",
                            0.0,
                        ),
                    ),
                ),
                0.0,
                1.0,
            )
        )

        # ------------------------------------------------------
        # Route risk
        #
        # Dataset may contain route_risk_level on
        # a 0-10 scale.
        # ------------------------------------------------------

        route_risk = result.get(
            "route_risk",
            result.get(
                "route_risk_level",
                0.0,
            ),
        )

        route_risk = float(
            route_risk
        )

        if route_risk > 1.0:
            route_risk /= 10.0

        result["route_risk"] = float(
            np.clip(
                route_risk,
                0.0,
                1.0,
            )
        )

        return result

    # ==========================================================
    # RESET
    # ==========================================================

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

        self.last_collision_count = 0
        self.last_completed_count = 0
        self.last_pickup_count = 0

        self.current_risk_records = {}
        self.task_risk = {}

        # ------------------------------------------------------
        # Dataset starting position
        # ------------------------------------------------------

        dataset_size = len(
            self.risk_engine.data
        )

        if dataset_size <= 0:
            raise RuntimeError(
                "Cold-chain dataset has no records."
            )

        if seed is not None:

            self.episode_dataset_start = (
                int(seed) % dataset_size
            )

        else:

            self.episode_dataset_start = int(
                self.np_random.integers(
                    0,
                    dataset_size,
                )
            )

        # ------------------------------------------------------
        # New AGV system
        # ------------------------------------------------------

        self.system = MultiAGVSystem(
            warehouse=self.warehouse
        )

        self.tasks = []

        # ======================================================
        # CREATE AGVs
        # ======================================================

        for agv_id in range(
            1,
            self.num_agvs + 1,
        ):

            position = (
                0,
                agv_id - 1,
            )

            agv = AGV(
                agv_id=agv_id,
                position=position,
            )

            self.system.add_agv(
                agv
            )

        # ======================================================
        # CREATE TASKS
        # ======================================================

        for agv_id in range(
            1,
            self.num_agvs + 1,
        ):

            # --------------------------------------------------
            # Pickup location
            # --------------------------------------------------

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

            # --------------------------------------------------
            # Delivery location
            # --------------------------------------------------

            delivery = (
                self.width - 1,
                self.height - agv_id,
            )

            # --------------------------------------------------
            # Ensure pickup and delivery differ
            # --------------------------------------------------

            if pickup == delivery:

                if self.width > 1:

                    delivery = (
                        max(
                            0,
                            self.width - 2,
                        ),
                        delivery[1],
                    )

                elif self.height > 1:

                    delivery = (
                        delivery[0],
                        max(
                            0,
                            self.height - 2,
                        ),
                    )

            # ==================================================
            # DATASET RECORD
            # ==================================================

            dataset_index = (
                self.episode_dataset_start
                + agv_id
                - 1
            ) % dataset_size

            dataset_record = (
                self._get_dataset_record(
                    dataset_index
                )
            )

            self.task_risk[
                agv_id
            ] = dataset_record

            self.current_risk_records[
                agv_id
            ] = dataset_record

            # ==================================================
            # TASK
            # ==================================================

            task = Task(
                task_id=agv_id,
                pickup=pickup,
                delivery=delivery,
            )

            # --------------------------------------------------
            # Attach dataset information
            # --------------------------------------------------

            task.dataset_index = (
                dataset_record[
                    "dataset_index"
                ]
            )

            task.temperature = (
                dataset_record[
                    "temperature"
                ]
            )

            task.risk_score = (
                dataset_record[
                    "risk_score"
                ]
            )

            task.urgency_score = (
                dataset_record[
                    "urgency_score"
                ]
            )

            task.priority = (
                dataset_record[
                    "priority"
                ]
            )

            task.weather_risk = (
                dataset_record[
                    "weather"
                ]
            )

            task.disruption_risk = (
                dataset_record[
                    "disruption"
                ]
            )

            task.delay_risk = (
                dataset_record[
                    "delay"
                ]
            )

            task.route_risk = (
                dataset_record[
                    "route_risk"
                ]
            )

            # --------------------------------------------------
            # Register task
            # --------------------------------------------------

            self.system.add_task(
                task
            )

            self.system.assign_task(
                task_id=agv_id,
                agv_id=agv_id,
            )

            task.start()

            # --------------------------------------------------
            # AGV starts toward pickup
            # --------------------------------------------------

            agv = self.system.agvs[
                agv_id
            ]

            agv.current_task = (
                task.task_id
            )

            agv.destination = (
                task.pickup
            )

            agv.status = "MOVING"

            # --------------------------------------------------
            # Distance memory
            # --------------------------------------------------

            agv._previous_distance = (
                self._distance_to_task_target(
                    agv
                )
            )

            self.tasks.append(
                task
            )

        # ------------------------------------------------------
        # Initial observation
        # ------------------------------------------------------

        observation = (
            self._get_observation()
        )

        info = self._get_info()

        return (
            observation,
            info,
        )

    # ==========================================================
    # STEP
    # ==========================================================

        # ==========================================================
    # STEP
    # ==========================================================

    def step(
        self,
        actions,
    ):

        if self.system is None:
            raise RuntimeError(
                "Environment must be reset before step()."
            )

        actions = np.asarray(
            actions,
            dtype=np.int64,
        )

        # ======================================================
        # SHAPE VALIDATION
        # ======================================================

        if actions.shape != (
            self.num_agvs,
        ):
            raise ValueError(
                f"Expected {self.num_agvs} actions, "
                f"got shape {actions.shape}."
            )

        # ======================================================
        # ACTION VALIDATION
        # ======================================================

        if np.any(
            actions < 0
        ) or np.any(
            actions >= 5
        ):
            raise ValueError(
                "Every action must be in the range 0..4."
            )

        self.current_step += 1

        total_reward = 0.0

        collision_count = 0
        completed_count = 0
        pickup_count = 0

        # ======================================================
        # CURRENT / PROPOSED POSITIONS
        # ======================================================

        current_positions = {}
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

            current_positions[
                agv_id
            ] = tuple(
                agv.position
            )

            proposed_positions[
                agv_id
            ] = tuple(
                self._get_target_position(
                    agv,
                    action,
                )
            )

        # ======================================================
        # COLLISION DETECTION
        # ======================================================

        collision_agvs = set()

        # ======================================================
        # 1. SAME DESTINATION COLLISION
        # ======================================================

        position_to_moving_agvs = {}

        for (
            agv_id,
            proposed_position,
        ) in proposed_positions.items():

            current_position = (
                current_positions[
                    agv_id
                ]
            )

            # AGV is staying.
            if (
                proposed_position
                == current_position
            ):
                continue

            position_to_moving_agvs.setdefault(
                proposed_position,
                [],
            ).append(
                agv_id
            )

        for (
            position,
            moving_agvs,
        ) in position_to_moving_agvs.items():

            if len(moving_agvs) > 1:

                # --------------------------------------------------
                # DEBUG
                # --------------------------------------------------

                print(
                    "SAME DESTINATION COLLISION",
                    moving_agvs,
                )

                # --------------------------------------------------
                # All AGVs targeting the same position
                # are blocked.
                # --------------------------------------------------

                collision_agvs.update(
                    moving_agvs
                )

        # ======================================================
        # 2. MOVING INTO OCCUPIED STATIONARY AGV
        # ======================================================

        for agv_id in range(
            1,
            self.num_agvs + 1,
        ):

            proposed_position = (
                proposed_positions[
                    agv_id
                ]
            )

            current_position = (
                current_positions[
                    agv_id
                ]
            )

            # AGV is staying.
            if (
                proposed_position
                == current_position
            ):
                continue

            for other_id in range(
                1,
                self.num_agvs + 1,
            ):

                if other_id == agv_id:
                    continue

                other_current = (
                    current_positions[
                        other_id
                    ]
                )

                other_proposed = (
                    proposed_positions[
                        other_id
                    ]
                )

                # --------------------------------------------------
                # Our target is occupied by another AGV.
                # --------------------------------------------------

                if (
                    proposed_position
                    == other_current
                ):

                    # --------------------------------------------------
                    # Other AGV is stationary.
                    # --------------------------------------------------

                    if (
                        other_proposed
                        == other_current
                    ):

                        # --------------------------------------------------
                        # DEBUG
                        # --------------------------------------------------

                        print(
                            "STATIONARY COLLISION",
                            agv_id,
                            other_id,
                        )

                        # --------------------------------------------------
                        # Block only the moving AGV.
                        # --------------------------------------------------

                        collision_agvs.add(
                            agv_id
                        )

                        break

        # ======================================================
        # 3. SWAP COLLISION
        #
        # AGV A: X -> Y
        # AGV B: Y -> X
        # ======================================================

        agv_ids = list(
            current_positions.keys()
        )

        for i in range(
            len(agv_ids)
        ):

            agv_a = agv_ids[i]

            for j in range(
                i + 1,
                len(agv_ids),
            ):

                agv_b = agv_ids[j]

                is_swap = (
                    proposed_positions[
                        agv_a
                    ]
                    == current_positions[
                        agv_b
                    ]
                    and
                    proposed_positions[
                        agv_b
                    ]
                    == current_positions[
                        agv_a
                    ]
                    and
                    proposed_positions[
                        agv_a
                    ]
                    != current_positions[
                        agv_a
                    ]
                    and
                    proposed_positions[
                        agv_b
                    ]
                    != current_positions[
                        agv_b
                    ]
                )

                if is_swap:

                    # --------------------------------------------------
                    # DEBUG
                    # --------------------------------------------------

                    print(
                        "SWAP COLLISION",
                        agv_a,
                        agv_b,
                    )

                    # --------------------------------------------------
                    # Block both AGVs.
                    # --------------------------------------------------

                    collision_agvs.add(
                        agv_a
                    )

                    collision_agvs.add(
                        agv_b
                    )

        # ======================================================
        # COLLISION NORMALIZATION
        # ======================================================
        #
        # One timestep reports at most ONE collision event.
        #
        # Multiple AGVs may be involved, but the environment
        # reports collision_count as either 0 or 1.
        # ======================================================

        if collision_agvs:

            collision_count = 1

        else:

            collision_count = 0

        # ======================================================
        # EXECUTE MOVEMENTS
        # ======================================================
        #
        # IMPORTANT:
        #
        # Colliding AGVs are blocked BEFORE move_agv().
        #
        # There is NO manual position reset.
        #
        # Therefore:
        #
        # collision detected
        #        ↓
        # AGV movement skipped
        #        ↓
        # AGV remains at old position naturally
        # ======================================================

        for agv_id in range(
            1,
            self.num_agvs + 1,
        ):

            agv = self.system.agvs[
                agv_id
            ]

            # --------------------------------------------------
            # BLOCK COLLIDING AGV MOVEMENT
            # --------------------------------------------------

            if agv_id in collision_agvs:

                # Collision penalty.
                total_reward -= 20.0

                # Urgency-aware collision penalty.
                risk = self.task_risk.get(
                    agv_id,
                    {
                        "urgency_score": 0.0,
                    },
                )

                total_reward -= (
                    1.0
                    * risk.get(
                        "urgency_score",
                        0.0,
                    )
                )

                # --------------------------------------------------
                # DO NOT CALL move_agv().
                # --------------------------------------------------

                continue

            # --------------------------------------------------
            # Action selected by DQN.
            # --------------------------------------------------

            action = int(
                actions[
                    agv_id - 1
                ]
            )

            # ==================================================
            # COMPLETED AGV RETURNING TO DEPOT
            # ==================================================

            if (
                agv.status == "RETURNING"
                and
                agv.current_task is None
            ):

                depot_position = (
                    0,
                    agv_id - 1,
                )

                # Already at depot.
                if (
                    tuple(agv.position)
                    == tuple(depot_position)
                ):

                    agv.destination = (
                        depot_position
                    )

                    agv.status = "IDLE"

                    continue

                # Automatically move one step toward depot.
                target = (
                    self._get_target_position(
                        agv,
                        1,
                    )
                )

                risk = self.task_risk.get(
                    agv_id,
                    {
                        "urgency_score": 0.0,
                        "priority": "LOW",
                    },
                )

            else:

                # ==================================================
                # NORMAL TASK MOVEMENT
                # ==================================================

                target = (
                    proposed_positions[
                        agv_id
                    ]
                )

                risk = self.task_risk.get(
                    agv_id,
                    {
                        "urgency_score": 0.0,
                        "priority": "LOW",
                    },
                )

            # ==================================================
            # STAY
            # ==================================================

            if (
                action == 0
                and not (
                    agv.status == "RETURNING"
                    and
                    agv.current_task is None
                )
            ):

                total_reward -= 0.1

                continue

            # ==================================================
            # DISTANCE BEFORE MOVEMENT
            # ==================================================

            old_distance = (
                self._distance_to_task_target(
                    agv
                )
            )

            # ==================================================
            # MOVEMENT
            # ==================================================

            moved = self.system.move_agv(
                agv_id,
                target,
            )

            # ==================================================
            # INVALID MOVEMENT
            # ==================================================

            if not moved:

                total_reward -= 2.0

                continue

            # ==================================================
            # MOVEMENT COST
            # ==================================================

            total_reward -= 0.2

            # ==================================================
            # DISTANCE AFTER MOVEMENT
            # ==================================================

            new_distance = (
                self._distance_to_task_target(
                    agv
                )
            )

            distance_improvement = (
                old_distance
                - new_distance
            )

            # ==================================================
            # PROGRESS REWARD
            # ==================================================

            total_reward += float(
                distance_improvement
            )

            # ==================================================
            # URGENCY-AWARE PROGRESS
            # ==================================================

            if (
                distance_improvement > 0
            ):

                total_reward += (
                    0.5
                    * risk.get(
                        "urgency_score",
                        0.0,
                    )
                    * distance_improvement
                )

        # ======================================================
        # TASK PROGRESSION
        # ======================================================

        for task in self.tasks:

            if task.is_completed:
                continue

            agv_id = task.assigned_agv

            if agv_id is None:
                continue

            if agv_id not in self.system.agvs:
                continue

            agv = self.system.agvs[
                agv_id
            ]

            risk = self.task_risk.get(
                agv_id,
                {
                    "risk_score": 0.0,
                    "urgency_score": 0.0,
                    "priority": "LOW",
                },
            )

            # ==================================================
            # DISTANCE-BASED GUIDANCE REWARD
            # ==================================================

            previous_distance = getattr(
                agv,
                "_previous_distance",
                None,
            )

            current_distance = (
                self._distance_to_task_target(
                    agv
                )
            )

            if previous_distance is not None:

                if (
                    current_distance
                    < previous_distance
                ):

                    total_reward += 1.0

                elif (
                    current_distance
                    > previous_distance
                ):

                    total_reward -= 1.0

            agv._previous_distance = (
                current_distance
            )

            # ==================================================
            # PICKUP
            # ==================================================

            if (
                task.status == "IN_PROGRESS"
                and
                tuple(agv.position)
                == tuple(task.pickup)
            ):

                task.pickup_completed()

                agv.destination = (
                    task.delivery
                )

                agv.current_task = (
                    task.task_id
                )

                agv.status = "MOVING"

                pickup_count += 1

                total_reward += 25.0

                total_reward += (
                    5.0
                    * risk.get(
                        "urgency_score",
                        0.0,
                    )
                )

                if (
                    risk.get(
                        "priority",
                        "LOW",
                    )
                    == "HIGH"
                ):

                    total_reward += 3.0

            # ==================================================
            # DELIVERY
            # ==================================================

            if (
                task.status == "PICKED_UP"
                and
                tuple(agv.position)
                == tuple(task.delivery)
            ):

                task.complete()

                try:

                    agv.complete_task()

                except Exception:

                    agv.current_task = None
                    agv.destination = None
                    agv.status = "IDLE"

                completed_count += 1

                total_reward += 50.0

                total_reward += (
                    20.0
                    * risk.get(
                        "urgency_score",
                        0.0,
                    )
                )

                if (
                    risk.get(
                        "priority",
                        "LOW",
                    )
                    == "HIGH"
                ):

                    total_reward += 5.0

                agv.destination = (
                    0,
                    agv_id - 1,
                )

                agv.status = "RETURNING"

        # ======================================================
        # MULTI-AGV TASK LIFECYCLE SYNCHRONIZATION
        # ======================================================

        # Task lifecycle is already handled above.
        # Do not synchronize again here.

        # ======================================================
        # STATISTICS
        # ======================================================

        self.last_collision_count = (
            collision_count
        )

        self.last_completed_count = (
            completed_count
        )

        self.last_pickup_count = (
            pickup_count
        )

        # ======================================================
        # TERMINATION
        # ======================================================

        terminated = bool(
            self.system.all_tasks_completed()
        )

        truncated = bool(
            self.current_step
            >= self.max_steps
        )

        # ======================================================
        # OBSERVATION
        # ======================================================

        observation = (
            self._get_observation()
        )

        # ======================================================
        # INFO
        # ======================================================

        info = self._get_info()

        info.update(
            {
                "collision_count":
                    collision_count,

                "completed_tasks":
                    completed_count,

                "pickup_tasks":
                    pickup_count,

                "step":
                    self.current_step,

                "terminated":
                    terminated,

                "truncated":
                    truncated,

                "dataset_index":
                    self.episode_dataset_start,
            }
        )

        # ======================================================
        # GYMNASIUM RETURN
        # ======================================================

        return (
            observation,
            float(total_reward),
            terminated,
            truncated,
            info,
        )
    # ==========================================================
    # DISTANCE TO CURRENT TARGET
    # ==========================================================

    def _distance_to_task_target(
        self,
        agv,
    ):

        if agv.current_task is None:
            return 0.0

        if agv.current_task not in self.system.tasks:
            return 0.0

        task = self.system.tasks[
            agv.current_task
        ]

        if task.is_completed:
            return 0.0

        if task.status == "IN_PROGRESS":

            target = task.pickup

        elif task.status == "PICKED_UP":

            target = task.delivery

        else:

            target = task.pickup

        x1, y1 = agv.position
        x2, y2 = target

        return float(
            abs(x1 - x2)
            + abs(y1 - y2)
        )

    # ==========================================================
    # TARGET POSITION
    # ==========================================================

    def _get_target_position(
        self,
        agv,
        action,
    ):

        x, y = agv.position

        # ------------------------------------------------------
        # Determine target
        # ------------------------------------------------------

        target = None

        # ------------------------------------------------------
        # Active task target has highest priority.
        # ------------------------------------------------------

        if agv.current_task is not None:

            for task in self.tasks:

                if (
                    task.task_id
                    == agv.current_task
                ):

                    if (
                        task.status
                        == "IN_PROGRESS"
                    ):

                        target = (
                            task.pickup
                        )

                    elif (
                        task.status
                        == "PICKED_UP"
                    ):

                        target = (
                            task.delivery
                        )

                    break

        # ------------------------------------------------------
        # No active task:
        # use AGV destination.
        # ------------------------------------------------------

        if target is None:

            target = getattr(
                agv,
                "destination",
                None,
            )

        # ------------------------------------------------------
        # No target -> stay.
        # ------------------------------------------------------

        if target is None:
            return (
                x,
                y,
            )

        tx, ty = target

        # ------------------------------------------------------
        # Actions
        #
        # 0 = stay
        # 1 = right
        # 2 = left
        # 3 = up
        # 4 = down
        # ------------------------------------------------------

        action = int(
            action
        )

        if action == 0:
            return (
                x,
                y,
            )

        if action == 1:

            if x < tx:
                return (
                    x + 1,
                    y,
                )

            return (
                x,
                y,
            )

        if action == 2:

            if x > tx:
                return (
                    x - 1,
                    y,
                )

            return (
                x,
                y,
            )

        if action == 3:

            if y < ty:
                return (
                    x,
                    y + 1,
                )

            return (
                x,
                y,
            )

        if action == 4:

            if y > ty:
                return (
                    x,
                    y - 1,
                )

            return (
                x,
                y,
            )

        return (
            x,
            y,
        )

    # ==========================================================
    # OBSERVATION
    # ==========================================================

    def _get_observation(
        self,
    ):

        values = []

        max_x = max(
            1,
            self.width - 1,
        )

        max_y = max(
            1,
            self.height - 1,
        )

        for agv_id in range(
            1,
            self.num_agvs + 1,
        ):

            agv = self.system.agvs[
                agv_id
            ]

            x, y = agv.position

            # --------------------------------------------------
            # Current goal
            # --------------------------------------------------

            if agv.destination is None:

                gx, gy = (
                    x,
                    y,
                )

            else:

                gx, gy = (
                    agv.destination
                )

            # --------------------------------------------------
            # Risk / urgency
            # --------------------------------------------------

            risk = 0.0
            urgency = 0.0

            if agv.current_task is not None:

                if (
                    agv.current_task
                    in self.system.tasks
                ):

                    task = self.system.tasks[
                        agv.current_task
                    ]

                    risk = float(
                        getattr(
                            task,
                            "risk_score",
                            0.0,
                        )
                    )

                    urgency = float(
                        getattr(
                            task,
                            "urgency_score",
                            0.0,
                        )
                    )

            # --------------------------------------------------
            # Six features
            # --------------------------------------------------

            values.extend(
                [
                    float(
                        np.clip(
                            x / max_x,
                            0.0,
                            1.0,
                        )
                    ),

                    float(
                        np.clip(
                            y / max_y,
                            0.0,
                            1.0,
                        )
                    ),

                    float(
                        np.clip(
                            gx / max_x,
                            0.0,
                            1.0,
                        )
                    ),

                    float(
                        np.clip(
                            gy / max_y,
                            0.0,
                            1.0,
                        )
                    ),

                    float(
                        np.clip(
                            risk,
                            0.0,
                            1.0,
                        )
                    ),

                    float(
                        np.clip(
                            urgency,
                            0.0,
                            1.0,
                        )
                    ),
                ]
            )

        observation = np.asarray(
            values,
            dtype=np.float32,
        )

        # ------------------------------------------------------
        # Observation safety check
        # ------------------------------------------------------

        if not self.observation_space.contains(
            observation
        ):

            raise RuntimeError(
                "Generated observation is outside "
                "observation_space. "
                f"Expected shape: "
                f"{self.observation_space.shape}, "
                f"actual shape: "
                f"{observation.shape}"
            )

        return observation

    # ==========================================================
    # MULTI-AGV TASK PROGRESSION SAFETY
    # ==========================================================

    def _progress_all_agv_tasks(
        self,
    ):
        """
        Reconcile every AGV with its assigned task.

        This deliberately processes ALL AGVs independently.

        Completing one AGV task must never prevent another
        AGV from progressing through pickup -> delivery.
        """

        completed_count = 0
        pickup_count = 0

        if self.system is None:
            return (
                completed_count,
                pickup_count,
            )

        for task in self.tasks:

            if task.is_completed:
                continue

            agv_id = task.assigned_agv

            if agv_id is None:
                continue

            if agv_id not in self.system.agvs:
                continue

            agv = self.system.agvs[
                agv_id
            ]

            # --------------------------------------------------
            # PICKUP
            # --------------------------------------------------

            if (
                task.status
                == "IN_PROGRESS"
                and
                tuple(agv.position)
                == tuple(task.pickup)
            ):

                task.pickup_completed()

                agv.destination = (
                    task.delivery
                )

                agv.current_task = (
                    task.task_id
                )

                agv.status = "MOVING"

                pickup_count += 1

            # --------------------------------------------------
            # DELIVERY
            # --------------------------------------------------

            if (
                task.status
                == "PICKED_UP"
                and
                tuple(agv.position)
                == tuple(task.delivery)
            ):

                task.complete()

                try:

                    agv.complete_task()

                except Exception:

                    agv.current_task = None
                    agv.destination = None
                    agv.status = "IDLE"

                agv.destination = (
                    0,
                    agv_id - 1,
                )

                agv.status = "RETURNING"

                completed_count += 1

        return (
            completed_count,
            pickup_count,
        )

    # ==========================================================
    # MULTI-AGV TASK PROGRESSION SAFETY SYNC
    # ==========================================================

    def _synchronize_multi_agv_tasks(
        self,
    ):
        """
        Synchronize every assigned AGV with its task lifecycle.

        This is intentionally conservative:

        - never changes an already completed task
        - never reassigns a task
        - only acts on the AGV already assigned to that task
        - pickup occurs only at pickup position
        - delivery occurs only at delivery position
        """

        if self.system is None:
            return 0

        completed_now = 0

        for task in self.tasks:

            if task.is_completed:
                continue

            agv_id = task.assigned_agv

            if agv_id is None:
                continue

            if agv_id not in self.system.agvs:
                continue

            agv = self.system.agvs[
                agv_id
            ]

            # --------------------------------------------------
            # PICKUP
            # --------------------------------------------------

            if (
                task.status
                == "IN_PROGRESS"
                and
                tuple(agv.position)
                == tuple(task.pickup)
            ):

                task.pickup_completed()

                agv.destination = (
                    task.delivery
                )

                agv.current_task = (
                    task.task_id
                )

                agv.status = "MOVING"

                continue

            # --------------------------------------------------
            # DELIVERY
            # --------------------------------------------------

            if (
                task.status
                == "PICKED_UP"
                and
                tuple(agv.position)
                == tuple(task.delivery)
            ):

                task.complete()

                try:

                    agv.complete_task()

                except Exception:

                    agv.current_task = None
                    agv.destination = None

                agv.destination = (
                    0,
                    agv_id - 1,
                )

                agv.status = "RETURNING"

                completed_now += 1

        return completed_now

    # ==========================================================
    # INFO
    # ==========================================================

    def _get_info(
        self,
    ):

        completed = sum(
            1
            for task in self.tasks
            if task.is_completed
        )

        picked_up = sum(
            1
            for task in self.tasks
            if task.is_picked_up
        )

        high_risk_tasks = sum(
            1
            for risk in self.task_risk.values()
            if risk.get(
                "priority",
                "LOW",
            ) == "HIGH"
        )

        average_risk = 0.0

        if self.task_risk:

            average_risk = float(
                np.mean(
                    [
                        risk.get(
                            "risk_score",
                            0.0,
                        )
                        for risk
                        in self.task_risk.values()
                    ]
                )
            )

        average_urgency = 0.0

        if self.task_risk:

            average_urgency = float(
                np.mean(
                    [
                        risk.get(
                            "urgency_score",
                            0.0,
                        )
                        for risk
                        in self.task_risk.values()
                    ]
                )
            )

        return {
            "step":
                self.current_step,

            "num_agvs":
                self.num_agvs,

            "completed_tasks":
                completed,

            "picked_up_tasks":
                picked_up,

            "total_tasks":
                len(self.tasks),

            "collision_count":
                min(
                    self.last_collision_count,
                    1,
                ),

            "high_risk_tasks":
                high_risk_tasks,

            "average_risk":
                average_risk,

            "average_urgency":
                average_urgency,

            "dataset_index":
                self.episode_dataset_start,

            "dataset_tasks": [
                {
                    "task_id":
                        task.task_id,

                    "dataset_index":
                        getattr(
                            task,
                            "dataset_index",
                            None,
                        ),

                    "temperature":
                        getattr(
                            task,
                            "temperature",
                            0.0,
                        ),

                    "risk_score":
                        getattr(
                            task,
                            "risk_score",
                            0.0,
                        ),

                    "urgency_score":
                        getattr(
                            task,
                            "urgency_score",
                            0.0,
                        ),

                    "priority":
                        getattr(
                            task,
                            "priority",
                            "LOW",
                        ),
                }
                for task in self.tasks
            ],
        }

    # ==========================================================
    # RENDER
    # ==========================================================

    def render(
        self,
    ):

        if self.system is None:
            return ""

        positions = {
            agv_id: agv.position
            for agv_id, agv
            in self.system.agvs.items()
        }

        goals = [
            (
                task.delivery
                if task.is_picked_up
                else task.pickup
            )
            for task in self.tasks
            if not task.is_completed
        ]

        return self.warehouse.render_ascii(
            agv_positions=positions,
            goals=goals,
        )

    # ==========================================================
    # CLOSE
    # ==========================================================

    def close(
        self,
    ):
        pass


__all__ = [
    "MultiAGVEnv",
]
