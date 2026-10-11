"""
Gymnasium environment for the Multi-AGV Cold-Chain Warehouse.

Pipeline
--------
Dataset
    ↓
Cold-chain risk information
    ↓
Task urgency / priority
    ↓
Risk-aware task prioritization
    ↓
Multi-AGV observation
    ↓
DQN decision
    ↓
Movement
    ↓
Pickup
    ↓
Delivery
    ↓
Return to depot

Observation per AGV:
    x
    y
    goal_x
    goal_y
    risk
    urgency
    future_risk_60
    temperature
    task_phase

For 3 AGVs:
    3 * 8 = 24 observations

Actions per AGV:
    0 = Stay
    1 = Right
    2 = Left
    3 = Up
    4 = Down

Important:
    The five actions are genuine primitive movement actions.

    The environment does NOT automatically convert an action
    into the direction of the current goal.

    The RL agent must therefore learn which movement action
    should be selected.
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
        dataset_path="ALL_benchmark_W60.parquet",
        risk_aware_assignment=False,
    ):
        super().__init__()

        self.width = int(width)
        self.height = int(height)
        self.num_agvs = int(num_agvs)
        self.max_steps = int(max_steps)
        self.render_mode = render_mode
        self.risk_aware_assignment = bool(
            risk_aware_assignment
        )

        # ------------------------------------------------------
        # Basic validation
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
                "ColdChainRiskEngine must provide "
                "a 'data' attribute."
            )

        if len(
            self.risk_engine.data
        ) == 0:
            raise ValueError(
                "Cold-chain dataset is empty."
            )

        # ------------------------------------------------------
        # Trainable dataset indices
        # ------------------------------------------------------

        self.trainable_indices = (
            self._build_trainable_indices()
        )

        if len(
            self.trainable_indices
        ) == 0:
            raise ValueError(
                "Cold-chain dataset contains "
                "no trainable records."
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
        # ------------------------------------------------------

        self.action_space = spaces.MultiDiscrete(
            [5] * self.num_agvs
        )

        # ------------------------------------------------------
        # Observation space
        #
        # Eight values per AGV:
        #
        # x
        # y
        # goal_x
        # goal_y
        # risk
        # urgency
        # future_risk_60
        # temperature
        # ------------------------------------------------------

        observation_size = (
            self.num_agvs * 8
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

        # ------------------------------------------------------
        # Per-step statistics
        # ------------------------------------------------------

        self.last_collision_count = 0
        self.last_completed_count = 0
        self.last_pickup_count = 0

        # ------------------------------------------------------
        # Cumulative episode statistics
        # ------------------------------------------------------

        self.episode_collision_count = 0
        self.episode_completed_count = 0
        self.episode_pickup_count = 0
        self.episode_total_reward = 0.0

        self.episode_dataset_start = 0

        # ------------------------------------------------------
        # Risk state
        # ------------------------------------------------------

        self.task_risk = {}
        self.current_risk_records = {}

    # ==========================================================
    # BOOLEAN NORMALIZATION
    # ==========================================================

    @staticmethod
    def _as_bool(
        value,
    ):
        """
        Safely convert dataset values into booleans.
        """

        if isinstance(
            value,
            bool,
        ):
            return value

        if value is None:
            return False

        if isinstance(
            value,
            (
                int,
                float,
                np.integer,
                np.floating,
            ),
        ):
            return bool(value)

        text = str(
            value
        ).strip().lower()

        if text in {
            "true",
            "1",
            "yes",
            "y",
            "t",
        }:
            return True

        if text in {
            "false",
            "0",
            "no",
            "n",
            "f",
            "",
            "none",
            "nan",
        }:
            return False

        return bool(value)

    # ==========================================================
    # DATASET INDEX MANAGEMENT
    # ==========================================================

    def _build_trainable_indices(
        self,
    ):
        """
        Build dataset indices suitable for RL training.
        """

        indices = []

        data = self.risk_engine.data

        for index in range(
            len(data)
        ):

            row = data.iloc[index]

            is_trainable = self._as_bool(
                row.get(
                    "is_trainable",
                    False,
                )
            )

            is_incomplete = self._as_bool(
                row.get(
                    "is_incomplete",
                    False,
                )
            )

            is_fail_safe = self._as_bool(
                row.get(
                    "is_fail_safe",
                    False,
                )
            )

            if (
                is_trainable
                and not is_incomplete
                and not is_fail_safe
            ):
                indices.append(
                    index
                )

        return indices

    # ==========================================================
    # DATASET RECORD
    # ==========================================================

    def _get_dataset_record(
        self,
        index,
    ):
        """
        Retrieve and normalize one dataset record.
        """

        data_size = len(
            self.risk_engine.data
        )

        if data_size <= 0:
            raise RuntimeError(
                "Cold-chain dataset has no records."
            )

        index = (
            int(index)
            % data_size
        )

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

        result = dict(
            record
        )

        result["dataset_index"] = index

        # ------------------------------------------------------
        # Trainability flags
        # ------------------------------------------------------

        result["is_trainable"] = (
            self._as_bool(
                result.get(
                    "is_trainable",
                    False,
                )
            )
        )

        result["is_incomplete"] = (
            self._as_bool(
                result.get(
                    "is_incomplete",
                    False,
                )
            )
        )

        result["is_fail_safe"] = (
            self._as_bool(
                result.get(
                    "is_fail_safe",
                    False,
                )
            )
        )

        # ------------------------------------------------------
        # Temperature
        # ------------------------------------------------------

        try:

            temperature = float(
                result.get(
                    "temperature",
                    result.get(
                        "iot_temperature",
                        0.0,
                    ),
                )
            )

        except (
            TypeError,
            ValueError,
        ):

            temperature = 0.0

        result["temperature"] = temperature

        # ------------------------------------------------------
        # Risk
        # ------------------------------------------------------

        try:

            risk_score = float(
                result.get(
                    "risk_score",
                    0.0,
                )
            )

        except (
            TypeError,
            ValueError,
        ):

            risk_score = 0.0

        result["risk_score"] = float(
            np.clip(
                risk_score,
                0.0,
                1.0,
            )
        )

        # ------------------------------------------------------
        # Urgency
        # ------------------------------------------------------

        try:

            urgency_score = float(
                result.get(
                    "urgency_score",
                    0.0,
                )
            )

        except (
            TypeError,
            ValueError,
        ):

            urgency_score = 0.0

        result["urgency_score"] = float(
            np.clip(
                urgency_score,
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

        try:

            weather = float(
                result.get(
                    "weather",
                    result.get(
                        "weather_risk",
                        result.get(
                            "weather_condition_severity",
                            0.0,
                        ),
                    ),
                )
            )

        except (
            TypeError,
            ValueError,
        ):

            weather = 0.0

        result["weather"] = float(
            np.clip(
                weather,
                0.0,
                1.0,
            )
        )

        # ------------------------------------------------------
        # Disruption
        # ------------------------------------------------------

        try:

            disruption = float(
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
                )
            )

        except (
            TypeError,
            ValueError,
        ):

            disruption = 0.0

        result["disruption"] = float(
            np.clip(
                disruption,
                0.0,
                1.0,
            )
        )

        # ------------------------------------------------------
        # Delay
        # ------------------------------------------------------

        try:

            delay = float(
                result.get(
                    "delay",
                    result.get(
                        "delay_risk",
                        result.get(
                            "delay_probability",
                            0.0,
                        ),
                    ),
                )
            )

        except (
            TypeError,
            ValueError,
        ):

            delay = 0.0

        result["delay"] = float(
            np.clip(
                delay,
                0.0,
                1.0,
            )
        )

        # ------------------------------------------------------
        # Route risk
        # ------------------------------------------------------

        route_risk = result.get(
            "route_risk",
            result.get(
                "route_risk_level",
                0.0,
            ),
        )

        try:

            route_risk = float(
                route_risk
            )

        except (
            TypeError,
            ValueError,
        ):

            route_risk = 0.0

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
    # RISK-AWARE TASK PRIORITIZATION
    # ==========================================================

    @staticmethod
    def _priority_rank(
        priority,
    ):
        """
        Convert textual priority into a sortable rank.

        HIGH   -> 3
        MEDIUM -> 2
        LOW    -> 1
        """

        priority = str(
            priority
        ).upper()

        if priority == "HIGH":
            return 3

        if priority == "MEDIUM":
            return 2

        return 1

    def _get_risk_priority_score(
        self,
        task,
    ):
        """
        Generate a sortable risk-aware priority tuple.
        """

        task_id = getattr(
            task,
            "task_id",
            None,
        )

        if task_id is None:
            return (
                0,
                0.0,
                0.0,
                0.0,
                0.0,
            )

        assigned_agv = getattr(
            task,
            "assigned_agv",
            None,
        )

        # Prefer task-level risk mapping.
        record = self.task_risk.get(
            task_id,
            None,
        )

        # Fall back to AGV-level mapping.
        if (
            record is None
            and
            assigned_agv is not None
        ):

            record = self.task_risk.get(
                assigned_agv,
                None,
            )

        if record is None:
            record = {}

        priority = getattr(
            task,
            "priority",
            record.get(
                "priority",
                "LOW",
            ),
        )

        urgency = float(
            getattr(
                task,
                "urgency_score",
                record.get(
                    "urgency_score",
                    0.0,
                ),
            )
        )

        risk = float(
            getattr(
                task,
                "risk_score",
                record.get(
                    "risk_score",
                    0.0,
                ),
            )
        )

        future_risk_60 = float(
            record.get(
                "future_risk_60",
                record.get(
                    "y_next_60_R2",
                    0.0,
                ),
            )
        )

        eta_60 = float(
            record.get(
                "eta_60",
                record.get(
                    "eta_to_R2_60",
                    0.0,
                ),
            )
        )

        # Smaller ETA = higher priority.
        eta_priority = (
            -eta_60
            if eta_60 > 0.0
            else 0.0
        )

        return (
            self._priority_rank(
                priority
            ),
            urgency,
            risk,
            future_risk_60,
            eta_priority,
        )

    def _get_risk_aware_task_order(
        self,
    ):
        """
        Return unfinished tasks ordered from highest
        cold-chain priority to lowest.
        """

        active_tasks = []

        for task in self.tasks:

            if task.is_completed:
                continue

            active_tasks.append(
                task
            )

        active_tasks.sort(
            key=self._get_risk_priority_score,
            reverse=True,
        )

        return active_tasks

    # ==========================================================
    # TASK-DIRECTED ACTION
    # ==========================================================

    def _get_task_directed_action(
        self,
        agv,
    ):
        """
        Return a primitive movement action toward the
        current target.

        Actions:

            0 = Stay
            1 = Right
            2 = Left
            3 = Up
            4 = Down
        """

        x, y = map(
            int,
            agv.position,
        )

        target = None

        task_id = getattr(
            agv,
            "current_task",
            None,
        )

        if (
            task_id is not None
            and
            self.system is not None
            and
            task_id in self.system.tasks
        ):

            task = self.system.tasks[
                task_id
            ]

            if task.status == "IN_PROGRESS":

                target = task.pickup

            elif task.status == "PICKED_UP":

                target = task.delivery

            elif task.status == "COMPLETED":

                target = (
                    agv.agv_id - 1,
                    0,
                )

        if target is None:

            target = getattr(
                agv,
                "destination",
                None,
            )

        if target is None:

            target = (
                agv.agv_id - 1,
                0,
            )

        gx, gy = map(
            int,
            target,
        )

        if x < gx:
            return 1

        if x > gx:
            return 2

        if y < gy:
            return 3

        if y > gy:
            return 4

        return 0

    # ==========================================================
    # DISTANCE UTILITY
    # ==========================================================

    @staticmethod
    def _manhattan_distance(
        position,
        target,
    ):
        """
        Return Manhattan distance between two grid positions.
        """

        return (
            abs(
                int(position[0])
                - int(target[0])
            )
            +
            abs(
                int(position[1])
                - int(target[1])
            )
        )

    # ==========================================================
    # DYNAMIC RISK-AWARE RESPONDER SELECTION
    # ==========================================================

    def _select_best_responder(
        self,
        task,
        available_agv_ids=None,
    ):
        """
        Select the best AGV for a task.

        This method is currently available for the next
        cold-chain implementation phase.

        It is intentionally NOT called from reset() yet,
        preserving the validated baseline assignment.
        """

        if self.system is None:
            return None

        if available_agv_ids is None:

            available_agv_ids = list(
                self.system.agvs.keys()
            )

        if not available_agv_ids:
            return None

        priority = self._priority_rank(
            getattr(
                task,
                "priority",
                "LOW",
            )
        )

        urgency = float(
            getattr(
                task,
                "urgency_score",
                0.0,
            )
        )

        risk = float(
            getattr(
                task,
                "risk_score",
                0.0,
            )
        )

        best_agv = None
        best_score = None

        for agv_id in available_agv_ids:

            agv = self.system.agvs.get(
                agv_id
            )

            if agv is None:
                continue

            distance = (
                self._manhattan_distance(
                    agv.position,
                    task.pickup,
                )
            )

            score = (
                float(distance)
                - 10.0 * urgency
                - 5.0 * risk
                - 2.0 * float(priority)
            )

            if (
                best_score is None
                or score < best_score
            ):

                best_score = score
                best_agv = agv_id

        return best_agv

    # ==========================================================
    # TASK ASSIGNMENT
    # ==========================================================
        # ======================================================

        if self.risk_aware_assignment:

            self._assign_tasks_risk_aware()

        else:

            # Preserve the validated baseline:
            # Task 1 -> AGV 1
            # Task 2 -> AGV 2
            # Task 3 -> AGV 3

            for task in self.tasks:

                agv_id = int(
                    task.task_id
                )

                self.system.assign_task(
                    task_id=task.task_id,
                    agv_id=agv_id,
                )

        # ======================================================
        # START ASSIGNED TASKS
        # ======================================================

        for task in self.tasks:

            if task.assigned_agv is None:
                continue

            task.start()

            agv = self.system.agvs[
                task.assigned_agv
            ]

            agv.current_task = (
                task.task_id
            )

            agv.destination = (
                task.pickup
            )

            agv.status = "MOVING"

            agv._previous_distance = (
                self._distance_to_task_target(
                    agv
                )
            )

        return best_agv

    def _assign_tasks_risk_aware(
        self,
    ):
        """
        Assign pending tasks to AGVs using risk-aware
        priority and responder selection.

        Existing assignments are preserved.

        This is currently an available helper and is not
        automatically invoked by reset().
        """

        if self.system is None:
            return

        available_agvs = set(
            self.system.agvs.keys()
        )

        pending_tasks = []

        for task in self.tasks:

            if task.is_completed:
                continue

            if task.assigned_agv is not None:

                available_agvs.discard(
                    task.assigned_agv
                )

                continue

            pending_tasks.append(
                task
            )

        pending_tasks.sort(
            key=lambda task: (
                self._priority_rank(
                    getattr(
                        task,
                        "priority",
                        "LOW",
                    )
                ),
                float(
                    getattr(
                        task,
                        "urgency_score",
                        0.0,
                    )
                ),
                float(
                    getattr(
                        task,
                        "risk_score",
                        0.0,
                    )
                ),
            ),
            reverse=True,
        )

        for task in pending_tasks:

            if not available_agvs:
                break

            agv_id = (
                self._select_best_responder(
                    task,
                    available_agv_ids=available_agvs,
                )
            )

            if agv_id is None:
                continue

            self.system.assign_task(
                task_id=task.task_id,
                agv_id=agv_id,
            )

            task.assigned_agv = agv_id

            available_agvs.remove(
                agv_id
            )

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

        self.episode_collision_count = 0
        self.episode_completed_count = 0
        self.episode_pickup_count = 0
        self.episode_total_reward = 0.0

        self.current_risk_records = {}
        self.task_risk = {}

        # ------------------------------------------------------
        # Select trainable dataset starting position
        # ------------------------------------------------------

        if not self.trainable_indices:

            raise RuntimeError(
                "No trainable dataset records available."
            )

        if seed is not None:

            trainable_position = (
                int(seed)
                %
                len(
                    self.trainable_indices
                )
            )

        else:

            trainable_position = int(
                self.np_random.integers(
                    0,
                    len(
                        self.trainable_indices
                    ),
                )
            )

        self.episode_dataset_start = (
            self.trainable_indices[
                trainable_position
            ]
        )

        # ------------------------------------------------------
        # New multi-AGV system
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

        trainable_position = (
            self.trainable_indices.index(
                self.episode_dataset_start
            )
        )

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
            # Keep pickup and delivery different
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

            dataset_position = (
                trainable_position
                + agv_id
                - 1
            ) % len(
                self.trainable_indices
            )

            dataset_index = (
                self.trainable_indices[
                    dataset_position
                ]
            )

            dataset_record = (
                self._get_dataset_record(
                    dataset_index
                )
            )

            # --------------------------------------------------
            # AGV -> risk mapping
            # --------------------------------------------------

            self.current_risk_records[
                agv_id
            ] = dataset_record

            self.task_risk[
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
            # Task -> risk mapping
            # --------------------------------------------------

            self.task_risk[
                task.task_id
            ] = dataset_record

            # --------------------------------------------------
            # Dataset information
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
            # Preserve current validated assignment
            # --------------------------------------------------

            task.assigned_agv = agv_id

            # --------------------------------------------------
            # Register task
            # --------------------------------------------------

            self.system.add_task(
                task
            )

            self.system.assign_task(
                task_id=task.task_id,
                agv_id=agv_id,
            )

            task.assigned_agv = agv_id

            task.start()

            # --------------------------------------------------
            # Start AGV toward pickup
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

        info = (
            self._get_info()
        )

        return (
            observation,
            info,
        )

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
        # ACTION SHAPE
        # ======================================================

        if actions.shape != (
            self.num_agvs,
        ):

            raise ValueError(
                f"Expected {self.num_agvs} actions, "
                f"got shape {actions.shape}."
            )

        # ======================================================
        # ACTION RANGE
        # ======================================================

        if (
            np.any(actions < 0)
            or
            np.any(actions >= 5)
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
        # CURRENT POSITIONS
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

        # ------------------------------------------------------
        # 1. Same destination
        # ------------------------------------------------------

        position_to_agvs = {}

        for agv_id, proposed in (
            proposed_positions.items()
        ):

            current = (
                current_positions[
                    agv_id
                ]
            )

            if proposed == current:
                continue

            position_to_agvs.setdefault(
                proposed,
                [],
            ).append(
                agv_id
            )

        for moving_agvs in (
            position_to_agvs.values()
        ):

            if len(moving_agvs) > 1:

                collision_agvs.update(
                    moving_agvs
                )

        # ------------------------------------------------------
        # 2. Moving into stationary AGV
        # ------------------------------------------------------

        for agv_id in range(
            1,
            self.num_agvs + 1,
        ):

            proposed = (
                proposed_positions[
                    agv_id
                ]
            )

            current = (
                current_positions[
                    agv_id
                ]
            )

            if proposed == current:
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

                if (
                    proposed
                    == other_current
                    and
                    other_proposed
                    == other_current
                ):

                    collision_agvs.add(
                        agv_id
                    )

        # ------------------------------------------------------
        # 3. Swap collision
        # ------------------------------------------------------

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

                if (
                    proposed_positions[
                        agv_a
                    ]
                    ==
                    current_positions[
                        agv_b
                    ]
                    and
                    proposed_positions[
                        agv_b
                    ]
                    ==
                    current_positions[
                        agv_a
                    ]
                    and
                    proposed_positions[
                        agv_a
                    ]
                    !=
                    current_positions[
                        agv_a
                    ]
                    and
                    proposed_positions[
                        agv_b
                    ]
                    !=
                    current_positions[
                        agv_b
                    ]
                ):

                    collision_agvs.add(
                        agv_a
                    )

                    collision_agvs.add(
                        agv_b
                    )

        # ------------------------------------------------------
        # One collision event per step
        # ------------------------------------------------------

        collision_count = (
            1
            if collision_agvs
            else 0
        )

        # ======================================================
        # EXECUTE MOVEMENTS
        # ======================================================

        for agv_id in range(
            1,
            self.num_agvs + 1,
        ):

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

            # --------------------------------------------------
            # Collision
            # --------------------------------------------------

            if agv_id in collision_agvs:

                total_reward -= 20.0

                total_reward -= (
                    risk.get(
                        "urgency_score",
                        0.0,
                    )
                )

                continue

            # --------------------------------------------------
            # Action
            # --------------------------------------------------

            action = int(
                actions[
                    agv_id - 1
                ]
            )

            # --------------------------------------------------
            # Stay
            # --------------------------------------------------

            if action == 0:

                total_reward -= 0.1

                continue

            # --------------------------------------------------
            # Distance before movement
            # --------------------------------------------------

            old_distance = (
                self._distance_to_task_target(
                    agv
                )
            )

            # --------------------------------------------------
            # Target position
            # --------------------------------------------------

            target = (
                proposed_positions[
                    agv_id
                ]
            )

            # --------------------------------------------------
            # Movement
            # --------------------------------------------------

            moved = self.system.move_agv(
                agv_id,
                target,
            )

            # --------------------------------------------------
            # Invalid movement
            # --------------------------------------------------

            if not moved:

                total_reward -= 2.0

                continue

            # --------------------------------------------------
            # Movement cost
            # --------------------------------------------------

            total_reward -= 0.2

            # --------------------------------------------------
            # Distance after movement
            # --------------------------------------------------

            new_distance = (
                self._distance_to_task_target(
                    agv
                )
            )

            distance_improvement = (
                old_distance
                - new_distance
            )

            # --------------------------------------------------
            # Progress reward
            # --------------------------------------------------

            if distance_improvement > 0:

                total_reward += (
                    2.0
                    *
                    float(
                        distance_improvement
                    )
                )

                total_reward += (
                    0.5
                    *
                    risk.get(
                        "urgency_score",
                        0.0,
                    )
                    *
                    float(
                        distance_improvement
                    )
                )

            elif distance_improvement < 0:

                total_reward += (
                    2.0
                    *
                    float(
                        distance_improvement
                    )
                )

        # ======================================================
        # TASK LIFECYCLE
        # ======================================================

        for task in self.tasks:

            # --------------------------------------------------
            # Already delivered
            # --------------------------------------------------

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
                task.task_id,
                self.current_risk_records.get(
                    agv_id,
                    {
                        "risk_score": 0.0,
                        "urgency_score": 0.0,
                        "priority": "LOW",
                    },
                ),
            )

            # ==================================================
            # PICKUP
            # ==================================================

            if (
                task.status == "IN_PROGRESS"
                and
                tuple(agv.position)
                ==
                tuple(task.pickup)
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
                    *
                    risk.get(
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
                ==
                tuple(task.delivery)
            ):

                task.complete()

                completed_count += 1

                total_reward += 50.0

                total_reward += (
                    20.0
                    *
                    risk.get(
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

                # --------------------------------------------------
                # Return to depot
                # --------------------------------------------------

                agv.destination = (
                    agv_id - 1,
                    0,
                )

                agv.status = "RETURNING"

                agv.current_task =  None
                

        # ======================================================
        # EPISODE STATISTICS
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

        self.episode_collision_count += (
            collision_count
        )

        self.episode_completed_count += (
            completed_count
        )

        self.episode_pickup_count += (
            pickup_count
        )

        self.episode_total_reward += (
            float(
                total_reward
            )
        )
        
        # ======================================================
        # RETURN-TO-DEPOT COMPLETION
        # ======================================================

        for agv_id, agv in self.system.agvs.items():

            if (
                getattr(agv, "status", None)
                == "RETURNING"
                and
                getattr(agv, "destination", None) is not None
                and
                tuple(agv.position)
                == tuple(agv.destination)
            ):

                # Reward successful return
                total_reward += 20.0

                # AGV is now idle at its depot
                agv.status = "IDLE"

                agv.destination = None

                agv.current_task = None

        # ======================================================
        # TERMINATION
        # ======================================================

        tasks_completed = all(
            task.is_completed
            for task in self.tasks
        )

        agvs_returned = all(
            tuple(
                agv.position
            )
            ==
            tuple(
                (
                    agv_id - 1,
                    0,
                )
            )
            for agv_id, agv
            in self.system.agvs.items()
        )

        terminated = bool(
            tasks_completed
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

        info = (
            self._get_info()
        )

        info.update(
            {
                "step":
                    self.current_step,

                "collision_count":
                    collision_count,

                "completed_tasks_this_step":
                    completed_count,

                "pickup_tasks_this_step":
                    pickup_count,

                "episode_collision_count":
                    self.episode_collision_count,

                "episode_completed_count":
                    self.episode_completed_count,

                "episode_pickup_count":
                    self.episode_pickup_count,

                "episode_total_reward":
                    float(
                        self.episode_total_reward
                    ),

                "tasks_completed":
                    tasks_completed,

                "agvs_returned":
                    agvs_returned,

                "terminated":
                    terminated,

                "truncated":
                    truncated,

                "dataset_index":
                    self.episode_dataset_start,

                "trainable_dataset":
                    True,
            }
        )

        return (
            observation,
            float(
                total_reward
            ),
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
        """
        Manhattan distance from the AGV to its current target.

        Handles:

            IN_PROGRESS -> pickup
            PICKED_UP   -> delivery
            RETURNING   -> depot
            IDLE        -> zero
        """

        # ------------------------------------------------------
        # Returning AGV
        # ------------------------------------------------------

        if (
            getattr(
                agv,
                "status",
                None,
            )
            ==
            "RETURNING"
            and
            getattr(
                agv,
                "destination",
                None,
            )
            is not None
        ):

            target = agv.destination

            x1, y1 = map(
                int,
                agv.position,
            )

            x2, y2 = map(
                int,
                target,
            )

            return float(
                abs(
                    x1 - x2
                )
                +
                abs(
                    y1 - y2
                )
            )

        # ------------------------------------------------------
        # No current task
        # ------------------------------------------------------

        if agv.current_task is None:

            if (
                getattr(
                    agv,
                    "destination",
                    None,
                )
                is not None
                and
                getattr(
                    agv,
                    "status",
                    None,
                )
                ==
                "RETURNING"
            ):

                target = agv.destination

                x1, y1 = map(
                    int,
                    agv.position,
                )

                x2, y2 = map(
                    int,
                    target,
                )

                return float(
                    abs(
                        x1 - x2
                    )
                    +
                    abs(
                        y1 - y2
                    )
                )

            return 0.0

        # ------------------------------------------------------
        # Task lookup
        # ------------------------------------------------------

        if (
            self.system is None
            or
            agv.current_task not in self.system.tasks
        ):
            return 0.0

        task = self.system.tasks[
            agv.current_task
        ]

        # ------------------------------------------------------
        # Completed task
        # ------------------------------------------------------

        if task.is_completed:

            if (
                getattr(
                    agv,
                    "destination",
                    None,
                )
                is not None
                and
                getattr(
                    agv,
                    "status",
                    None,
                )
                ==
                "RETURNING"
            ):

                target = agv.destination

                x1, y1 = map(
                    int,
                    agv.position,
                )

                x2, y2 = map(
                    int,
                    target,
                )

                return float(
                    abs(
                        x1 - x2
                    )
                    +
                    abs(
                        y1 - y2
                    )
                )

            return 0.0

        # ------------------------------------------------------
        # Determine task target
        # ------------------------------------------------------

        if task.status == "IN_PROGRESS":

            target = task.pickup

        elif task.status == "PICKED_UP":

            target = task.delivery

        else:

            target = task.pickup

        x1, y1 = map(
            int,
            agv.position,
        )

        x2, y2 = map(
            int,
            target,
        )

        return float(
            abs(
                x1 - x2
            )
            +
            abs(
                y1 - y2
            )
        )

    # ==========================================================
    # TARGET POSITION
    # ==========================================================

    def _get_target_position(
        self,
        agv,
        action,
    ):
        """
        Return the position produced by one primitive movement.

        Actions:

            0 = Stay
            1 = Right
            2 = Left
            3 = Up
            4 = Down
        """

        x, y = map(
            int,
            agv.position,
        )

        action = int(
            action
        )

        # ------------------------------------------------------
        # STAY
        # ------------------------------------------------------

        if action == 0:

            return (
                x,
                y,
            )

        # ------------------------------------------------------
        # RIGHT
        # ------------------------------------------------------

        if action == 1:

            return (
                min(
                    x + 1,
                    self.width - 1,
                ),
                y,
            )

        # ------------------------------------------------------
        # LEFT
        # ------------------------------------------------------

        if action == 2:

            return (
                max(
                    x - 1,
                    0,
                ),
                y,
            )

        # ------------------------------------------------------
        # UP
        # ------------------------------------------------------

        if action == 3:

            return (
                x,
                min(
                    y + 1,
                    self.height - 1,
                ),
            )

        # ------------------------------------------------------
        # DOWN
        # ------------------------------------------------------

        if action == 4:

            return (
                x,
                max(
                    y - 1,
                    0,
                ),
            )

        raise ValueError(
            f"Invalid action {action}. "
            "Expected action in range 0..4."
        )

    # ==========================================================
    # OBSERVATION
    # ==========================================================

    def _get_observation(
        self,
    ):
        """
        Build normalized observation vector.

        Per AGV:

            x
            y
            goal_x
            goal_y
            risk
            urgency
            future_risk_60
            temperature
        """

        if self.system is None:

            raise RuntimeError(
                "Environment must be reset "
                "before creating observations."
            )

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

            x, y = map(
                int,
                agv.position,
            )

            # --------------------------------------------------
            # Current goal
            # --------------------------------------------------

            if (
                getattr(
                    agv,
                    "destination",
                    None,
                )
                is None
            ):

                gx, gy = (
                    x,
                    y,
                )

            else:

                gx, gy = map(
                    int,
                    agv.destination,
                )

            # --------------------------------------------------
            # Risk and urgency
            # --------------------------------------------------

            risk = 0.0
            urgency = 0.0

            future_risk_60 = 0.0
            temperature = 0.0
            task_phase = 0.0

            task_id = getattr(
                agv,
                "current_task",
                None,
            )

            if task_id is not None:

                task = self.system.tasks.get(
                    task_id
                )

                if task is not None:

                    risk = float(
                        getattr(
                            task,
                            "risk_score",
                            self.task_risk.get(
                                task_id,
                                {},
                            ).get(
                                "risk_score",
                                self.task_risk.get(
                                    getattr(
                                        task,
                                        "assigned_agv",
                                        None,
                                    ),
                                    {},
                                ).get(
                                    "risk_score",
                                    0.0,
                                ),
                            ),
                        )
                    )

                    urgency = float(
                        getattr(
                            task,
                            "urgency_score",
                            self.task_risk.get(
                                task_id,
                                {},
                            ).get(
                                "urgency_score",
                                self.task_risk.get(
                                    getattr(
                                        task,
                                        "assigned_agv",
                                        None,
                                    ),
                                    {},
                                ).get(
                                    "urgency_score",
                                    0.0,
                                ),
                            ),
                        )
                    )

                    future_risk_60 = float(
                        self.task_risk.get(
                            task_id,
                            {},
                        ).get(
                            "future_risk_60",
                            self.task_risk.get(
                                getattr(
                                    task,
                                    "assigned_agv",
                                    None,
                                ),
                                {},
                            ).get(
                                "future_risk_60",
                                0.0,
                            ),
                        )
                    )

                    temperature = float(
                        getattr(
                            task,
                            "temperature",
                            self.task_risk.get(
                                task_id,
                                {},
                            ).get(
                                "temperature",
                                0.0,
                            ),
                        )
                    )
                    
                    if task.status == "IN_PROGRESS":
                        task_phase = 0.0
                    elif task.status == "PICKED_UP":
                        task_phase = 1.0
                    elif task.is_completed:
                        task_phase = 2.0
                    else:
                        task_phase = 0.0
                else:
                    task_phase = 0.0

            # --------------------------------------------------
            # Eight normalized features
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

                    float(
                        np.clip(
                            future_risk_60,
                            0.0,
                            1.0,
                        )
                    ),

                    float(
                        np.clip(
                            temperature,
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
        # Safety check
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
    # INFO
    # ==========================================================

    def _get_info(
        self,
    ):
        """
        Generate environment information.

        Includes:

            - task statistics
            - collision statistics
            - risk statistics
            - dataset information
            - risk-aware task ordering
        """

        # ------------------------------------------------------
        # Task counts
        # ------------------------------------------------------

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

        # ------------------------------------------------------
        # Risk statistics
        # ------------------------------------------------------

        task_records = [
            self.task_risk.get(
                task.task_id,
                self.current_risk_records.get(
                    getattr(
                        task,
                        "assigned_agv",
                        None,
                    ),
                    {},
                ),
            )
            for task in self.tasks
        ]

        high_risk_tasks = sum(
            1
            for risk in task_records
            if str(
                risk.get(
                    "priority",
                    "LOW",
                )
            ).upper()
            == "HIGH"
        )

        average_risk = 0.0

        if task_records:

            average_risk = float(
                np.mean(
                    [
                        float(
                            risk.get(
                                "risk_score",
                                0.0,
                            )
                        )
                        for risk in task_records
                    ]
                )
            )

        average_urgency = 0.0

        if task_records:

            average_urgency = float(
                np.mean(
                    [
                        float(
                            risk.get(
                                "urgency_score",
                                0.0,
                            )
                        )
                        for risk in task_records
                    ]
                )
            )

        # ------------------------------------------------------
        # Risk-aware task ordering
        # ------------------------------------------------------

        risk_order = (
            self._get_risk_aware_task_order()
        )

        risk_aware_task_order = []

        for task in risk_order:

            assigned_agv = getattr(
                task,
                "assigned_agv",
                None,
            )

            record = self.task_risk.get(
                task.task_id,
                None,
            )

            if (
                record is None
                and
                assigned_agv is not None
            ):

                record = self.task_risk.get(
                    assigned_agv,
                    None,
                )

            if record is None:
                record = {}

            risk_aware_task_order.append(
                {
                    "task_id":
                        int(
                            task.task_id
                        ),

                    "agv_id":
                        (
                            int(
                                assigned_agv
                            )
                            if assigned_agv is not None
                            else None
                        ),

                    "status":
                        str(
                            getattr(
                                task,
                                "status",
                                "",
                            )
                        ),

                    "priority":
                        str(
                            getattr(
                                task,
                                "priority",
                                record.get(
                                    "priority",
                                    "LOW",
                                ),
                            )
                        ).upper(),

                    "risk_score":
                        float(
                            getattr(
                                task,
                                "risk_score",
                                record.get(
                                    "risk_score",
                                    0.0,
                                ),
                            )
                        ),

                    "urgency_score":
                        float(
                            getattr(
                                task,
                                "urgency_score",
                                record.get(
                                    "urgency_score",
                                    0.0,
                                ),
                            )
                        ),

                    "future_risk_60":
                        float(
                            record.get(
                                "future_risk_60",
                                record.get(
                                    "y_next_60_R2",
                                    0.0,
                                ),
                            )
                        ),

                    "eta_60":
                        float(
                            record.get(
                                "eta_60",
                                record.get(
                                    "eta_to_R2_60",
                                    0.0,
                                ),
                            )
                        ),
                }
            )

        # ------------------------------------------------------
        # Final information dictionary
        # ------------------------------------------------------

        return {
            # --------------------------------------------------
            # Environment
            # --------------------------------------------------

            "step":
                self.current_step,

            "num_agvs":
                self.num_agvs,

            "total_tasks":
                len(
                    self.tasks
                ),

            # --------------------------------------------------
            # Task statistics
            # --------------------------------------------------

            "completed_tasks":
                completed,

            "picked_up_tasks":
                picked_up,

            # --------------------------------------------------
            # Collision statistics
            # --------------------------------------------------

            "collision_count":
                min(
                    self.last_collision_count,
                    1,
                ),

            "episode_collision_count":
                self.episode_collision_count,

            # --------------------------------------------------
            # Risk statistics
            # --------------------------------------------------

            "high_risk_tasks":
                high_risk_tasks,

            "average_risk":
                average_risk,

            "average_urgency":
                average_urgency,

            # --------------------------------------------------
            # Risk-aware prioritization
            # --------------------------------------------------

            "risk_aware_task_order":
                risk_aware_task_order,

            # --------------------------------------------------
            # Dataset
            # --------------------------------------------------

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

                    "weather":
                        getattr(
                            task,
                            "weather_risk",
                            0.0,
                        ),

                    "disruption":
                        getattr(
                            task,
                            "disruption_risk",
                            0.0,
                        ),

                    "delay":
                        getattr(
                            task,
                            "delay_risk",
                            0.0,
                        ),

                    "route_risk":
                        getattr(
                            task,
                            "route_risk",
                            0.0,
                        ),
                }
                for task in self.tasks
            ],
        }

    # ==========================================================
    # RENDER
    # ==========================================================

        # ==========================================================
    # RENDER
    # ==========================================================

    def render(
        self,
    ):
        """
        Render the current warehouse state.

        Symbols:

            1 / 2 / 3 = AGVs
            P         = pickup
            D         = delivery
            G         = current active goal
            #         = obstacle
            .         = free cell
        """

        if self.system is None:
            return ""

        agv_positions = {
            agv_id: agv.position
            for agv_id, agv
            in self.system.agvs.items()
        }

        pickup_positions = set()
        delivery_positions = set()
        active_goals = set()

        for task in self.tasks:

            if task.is_completed:
                continue

            pickup_positions.add(
                tuple(task.pickup)
            )

            delivery_positions.add(
                tuple(task.delivery)
            )

            # Current operational goal.
            if task.status == "IN_PROGRESS":

                active_goals.add(
                    tuple(task.pickup)
                )

            elif task.status == "PICKED_UP":

                active_goals.add(
                    tuple(task.delivery)
                )

        rows = []

        agv_lookup = {
            tuple(position): agv_id
            for agv_id, position
            in agv_positions.items()
        }

        for y in reversed(
            range(self.height)
        ):

            row = []

            for x in range(
                self.width
            ):

                position = (
                    x,
                    y,
                )

                if position in agv_lookup:

                    row.append(
                        str(
                            agv_lookup[
                                position
                            ]
                        )
                    )

                elif position in active_goals:

                    row.append(
                        "G"
                    )

                elif position in pickup_positions:

                    row.append(
                        "P"
                    )

                elif position in delivery_positions:

                    row.append(
                        "D"
                    )

                elif self.warehouse.is_obstacle(
                    position
                ):

                    row.append(
                        "#"
                    )

                else:

                    row.append(
                        "."
                    )

            rows.append(
                " ".join(row)
            )

        return "\n".join(
            rows
        )

    # ==========================================================
    # CLOSE
    # ==========================================================

    def close(
        self,
    ):
        """
        Release environment resources.
        """

        return None


__all__ = [
    "MultiAGVEnv",
]
