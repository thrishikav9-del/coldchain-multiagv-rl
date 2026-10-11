""" 
Gymnasium environment for the Multi-AGV Cold-Chain Warehouse. 
 
Pipeline 
-------- 
Dataset 
    ? 
Cold-chain risk information 
    ? 
Task urgency / priority 
    ? 
Risk-aware task prioritization 
    ? 
Multi-AGV observation 
    ? 
DQN decision 
    ? 
Primitive movement 
    ? 
Pickup 
    ? 
Risk-aware rescue / delivery 
    ? 
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
 
For 3 AGVs: 
    3 * 8 = 24 observations 
 
Actions per AGV: 
    0 = Stay 
    1 = Right 
    2 = Left 
    3 = Up 
    4 = Down 
 
Important 
--------- 
The five actions are genuine primitive movement actions. 
 
The environment does NOT automatically convert an action 
into movement toward the goal. 
 
The RL agent must learn which primitive movement action 
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
    """ 
    Multi-AGV cold-chain warehouse Gymnasium environment. 
    """ 
 
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
        # Temperature normalization range 
        # ------------------------------------------------------ 
 
        self.temperature_min = 0.0 
        self.temperature_max = 10.0 
 
        self._initialize_temperature_range() 
 
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
        # Episode statistics 
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
    def _as_bool(value): 
        """ 
        Safely convert dataset values into booleans. 
        """ 
 
        if isinstance(value, bool): 
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
    # TEMPERATURE NORMALIZATION 
    # ========================================================== 
 
    def _initialize_temperature_range(self): 
        """ 
        Determine a reasonable temperature normalization range 
        from the dataset. 
 
        Observation space requires temperature in [0, 1]. 
        """ 
 
        data = self.risk_engine.data 
 
        candidate_columns = [ 
            "temperature", 
            "iot_temperature", 
            "T_mean", 
            "T_max", 
            "T_min", 
        ] 
 
        values = [] 
 
        for column in candidate_columns: 
 
            if column not in data.columns: 
                continue 
 
            numeric = ( 
                data[column] 
                .apply( 
                    lambda value: self._safe_float( 
                        value, 
                        default=np.nan, 
                    ) 
                ) 
                .replace( 
                    [np.inf, -np.inf], 
                    np.nan, 
                ) 
                .dropna() 
            ) 
 
            if len(numeric) > 0: 
                values.extend( 
                    numeric.tolist() 
                ) 
 
        if not values: 
            return 
 
        minimum = float( 
            np.min(values) 
        ) 
 
        maximum = float( 
            np.max(values) 
        ) 
 
        if maximum <= minimum: 
            return 
 
        self.temperature_min = minimum 
        self.temperature_max = maximum 
 
    @staticmethod 
    def _safe_float( 
        value, 
        default=0.0, 
    ): 
        """ 
        Safely convert a value to float. 
        """ 
 
        try: 
            return float(value) 
        except ( 
            TypeError, 
            ValueError, 
        ): 
            return default 
 
    def _normalize_temperature( 
        self, 
        temperature, 
    ): 
        """ 
        Normalize temperature to [0, 1]. 
        """ 
 
        temperature = self._safe_float( 
            temperature, 
            default=self.temperature_min, 
        ) 
 
        denominator = ( 
            self.temperature_max 
            - self.temperature_min 
        ) 
 
        if denominator <= 0: 
            return 0.0 
 
        normalized = ( 
            temperature 
            - self.temperature_min 
        ) / denominator 
 
        return float( 
            np.clip( 
                normalized, 
                0.0, 
                1.0, 
            ) 
        ) 
 
    # ========================================================== 
    # DATASET INDEX MANAGEMENT 
    # ========================================================== 
 
    def _build_trainable_indices(self): 
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
 
        result["is_trainable"] = self._as_bool( 
            result.get( 
                "is_trainable", 
                False, 
            ) 
        ) 
 
        result["is_incomplete"] = self._as_bool( 
            result.get( 
                "is_incomplete", 
                False, 
            ) 
        ) 
 
        result["is_fail_safe"] = self._as_bool( 
            result.get( 
                "is_fail_safe", 
                False, 
            ) 
        ) 
 
        # ------------------------------------------------------ 
        # Temperature 
        # ------------------------------------------------------ 
 
        temperature = self._safe_float( 
            result.get( 
                "temperature", 
                result.get( 
                    "iot_temperature", 
                    result.get( 
                        "T_mean", 
                        0.0, 
                    ), 
                ), 
            ), 
            default=0.0, 
        ) 
 
        result["temperature"] = temperature 
 
        result["temperature_normalized"] = ( 
            self._normalize_temperature( 
                temperature 
            ) 
        ) 
 
        # ------------------------------------------------------ 
        # Risk 
        # ------------------------------------------------------ 
 
        risk_score = self._safe_float( 
            result.get( 
                "risk_score", 
                0.0, 
            ), 
            default=0.0, 
        ) 
 
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
 
        urgency_score = self._safe_float( 
            result.get( 
                "urgency_score", 
                0.0, 
            ), 
            default=0.0, 
        ) 
 
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
 
        weather = self._safe_float( 
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
            default=0.0, 
        ) 
 
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
 
        disruption = self._safe_float( 
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
            default=0.0, 
        ) 
 
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
 
        delay = self._safe_float( 
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
            default=0.0, 
        ) 
 
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
 
        route_risk = self._safe_float( 
            result.get( 
                "route_risk", 
                result.get( 
                    "route_risk_level", 
                    0.0, 
                ), 
            ), 
            default=0.0, 
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
 
        # ------------------------------------------------------ 
        # Future risk 
        # ------------------------------------------------------ 
 
        future_risk_60 = self._safe_float( 
            result.get( 
                "future_risk_60", 
                result.get( 
                    "y_next_60_R2", 
                    0.0, 
                ), 
            ), 
            default=0.0, 
        ) 
 
        result["future_risk_60"] = float( 
            np.clip( 
                future_risk_60, 
                0.0, 
                1.0, 
            ) 
        ) 
 
        # ------------------------------------------------------ 
        # ETA 
        # ------------------------------------------------------ 
 
        eta_60 = self._safe_float( 
            result.get( 
                "eta_60", 
                result.get( 
                    "eta_to_R2_60", 
                    0.0, 
                ), 
            ), 
            default=0.0, 
        ) 
 
        result["eta_60"] = max( 
            0.0, 
            eta_60, 
        ) 
 
        # ------------------------------------------------------ 
        # Remaining Safe Exposure Time 
        # ------------------------------------------------------ 
 
        rset_minutes = self._safe_float( 
            result.get( 
                "rset_minutes", 
                60.0, 
            ), 
            default=60.0, 
        ) 
 
        result["rset_minutes"] = float( 
            np.clip( 
                rset_minutes, 
                0.0, 
                60.0, 
            ) 
        ) 
 
        return result 
 
    # ========================================================== 
    # PRIORITY 
    # ========================================================== 
 
    @staticmethod 
    def _priority_rank( 
        priority, 
    ): 
        """ 
        Convert priority into numerical rank. 
 
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
        Generate risk-aware sorting score. 
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
                0.0, 
            ) 
 
        assigned_agv = getattr( 
            task, 
            "assigned_agv", 
            None, 
        ) 
 
        record = self.task_risk.get( 
            task_id, 
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
 
        priority = getattr( 
            task, 
            "priority", 
            record.get( 
                "priority", 
                "LOW", 
            ), 
        ) 
 
        urgency = self._safe_float( 
            getattr( 
                task, 
                "urgency_score", 
                record.get( 
                    "urgency_score", 
                    0.0, 
                ), 
            ) 
        ) 
 
        risk = self._safe_float( 
            getattr( 
                task, 
                "risk_score", 
                record.get( 
                    "risk_score", 
                    0.0, 
                ), 
            ) 
        ) 
 
        future_risk_60 = self._safe_float( 
            record.get( 
                "future_risk_60", 
                record.get( 
                    "y_next_60_R2", 
                    0.0, 
                ), 
            ) 
        ) 
 
        eta_60 = self._safe_float( 
            record.get( 
                "eta_60", 
                record.get( 
                    "eta_to_R2_60", 
                    0.0, 
                ), 
            ) 
        ) 
 
        rset_minutes = self._safe_float( 
            record.get( 
                "rset_minutes", 
                60.0, 
            ) 
        ) 
 
        rset_minutes = float( 
            np.clip( 
                rset_minutes, 
                0.0, 
                60.0, 
            ) 
        ) 
 
        # Smaller RSET = more urgent. 
        rset_priority = -rset_minutes 
 
        # Smaller ETA = better priority. 
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
            rset_priority, 
            eta_priority, 
        ) 
 
    def _get_risk_aware_task_order( 
        self, 
    ): 
        """ 
        Return unfinished tasks ordered from highest 
        cold-chain priority to lowest. 
        """ 
 
        active_tasks = [ 
            task 
            for task in self.tasks 
            if not task.is_completed 
        ] 
 
        active_tasks.sort( 
            key=self._get_risk_priority_score, 
            reverse=True, 
        ) 
 
        return active_tasks 
 
    # ========================================================== 
    # MANHATTAN DISTANCE 
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
    # CURRENT TARGET 
    # ========================================================== 
 
    def _get_current_target( 
        self, 
        agv, 
    ): 
        """ 
        Determine the AGV's current operational target. 
 
        Priority: 
 
            1. Rescue destination 
            2. Pickup 
            3. Delivery 
            4. Return depot 
            5. AGV destination 
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
            == "RETURNING" 
            and 
            getattr( 
                agv, 
                "destination", 
                None, 
            ) 
            is not None 
        ): 
            return tuple( 
                agv.destination 
            ) 
 
        # ------------------------------------------------------ 
        # Task lookup 
        # ------------------------------------------------------ 
 
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
 
            # Rescue destination has highest priority. 
            if ( 
                getattr( 
                    task, 
                    "in_rescue", 
                    False, 
                ) 
                and 
                getattr( 
                    task, 
                    "rescue_destination", 
                    None, 
                ) is not None 
            ): 
                return tuple( 
                    task.rescue_destination 
                ) 
 
            if task.status == "IN_PROGRESS": 
                return tuple( 
                    task.pickup 
                ) 
 
            if task.status == "PICKED_UP": 
                return tuple( 
                    task.delivery 
                ) 
 
        # ------------------------------------------------------ 
        # AGV destination 
        # ------------------------------------------------------ 
 
        destination = getattr( 
            agv, 
            "destination", 
            None, 
        ) 
 
        if destination is not None: 
            return tuple( 
                destination 
            ) 
 
        # ------------------------------------------------------ 
        # Depot fallback 
        # ------------------------------------------------------ 
 
        return ( 
            int(agv.agv_id) - 1, 
            0, 
        ) 
 
    # ========================================================== 
    # TASK-DIRECTED ACTION 
    # ========================================================== 
 
    def _get_task_directed_action(self, agv):
        """
        Return a collision-aware primitive action toward the AGV's
        current operational target.

        Actions:
            0 = Stay
            1 = Right
            2 = Left
            3 = Up
            4 = Down

        This method is used by the deterministic baseline/debug
        controller. The DQN still operates directly on actions.
        """
        x, y = map(int, agv.position)
        target = self._get_current_target(agv)
        gx, gy = map(int, target)

        candidates = []

        if x < gx:
            candidates.append(1)
        elif x > gx:
            candidates.append(2)

        if y < gy:
            candidates.append(3)
        elif y > gy:
            candidates.append(4)

        if y != gy:
            candidates.extend([1, 2])

        if x != gx:
            candidates.extend([3, 4])

        candidates.append(0)

        ordered_candidates = []
        for action in candidates:
            if action not in ordered_candidates:
                ordered_candidates.append(action)

        other_positions = set()
        if self.system is not None:
            for other_id, other_agv in self.system.agvs.items():
                if int(other_id) == int(agv.agv_id):
                    continue
                other_positions.add(tuple(map(int, other_agv.position)))

        current = tuple(map(int, agv.position))

        for action in ordered_candidates:
            proposed = tuple(self._get_target_position(agv, action))

            if action == 0:
                if proposed not in other_positions:
                    return 0
                continue

            if proposed in other_positions:
                continue

            contested = False
            if self.system is not None:
                for other_id, other_agv in self.system.agvs.items():
                    if int(other_id) == int(agv.agv_id):
                        continue

                    other_action = self._basic_task_direction(other_agv)
                    other_proposed = tuple(
                        self._get_target_position(other_agv, other_action)
                    )

                    if (
                        other_proposed == proposed
                        and int(other_id) < int(agv.agv_id)
                    ):
                        contested = True
                        break

            if contested:
                continue

            swap_detected = False
            if self.system is not None:
                for other_id, other_agv in self.system.agvs.items():
                    if int(other_id) == int(agv.agv_id):
                        continue

                    other_current = tuple(map(int, other_agv.position))
                    other_action = self._basic_task_direction(other_agv)
                    other_proposed = tuple(
                        self._get_target_position(other_agv, other_action)
                    )

                    if (
                        proposed == other_current
                        and other_proposed == current
                    ):
                        swap_detected = True
                        break

            if swap_detected:
                continue

            return action

        return 0

    def _basic_task_direction(self, agv):
        """
        Return the direct task-directed action without collision avoidance.
        """
        x, y = map(int, agv.position)
        target = self._get_current_target(agv)
        gx, gy = map(int, target)

        if x < gx:
            return 1
        if x > gx:
            return 2
        if y < gy:
            return 3
        if y > gy:
            return 4
        return 0

    def _select_safe_destination( 
        self, 
        task, 
    ): 
        """ 
        Select a low-risk warehouse-zone center as a temporary 
        rescue destination for a sufficiently risky task. 
 
        The original delivery location is never modified. 
        """ 
 
        if self.warehouse is None: 
            return None 
 
        risk_score = self._safe_float( 
            getattr( 
                task, 
                "risk_score", 
                0.0, 
            ) 
        ) 
 
        # Rescue only for meaningful risk. 
        if risk_score < 0.50: 
            return None 
 
        # ------------------------------------------------------ 
        # Source zone 
        # ------------------------------------------------------ 
 
        source_zone = self.warehouse.get_zone_id( 
            task.pickup 
        ) 
 
        # ------------------------------------------------------ 
        # Zone risk data 
        # ------------------------------------------------------ 
 
        all_risks = ( 
            self.warehouse.get_all_zone_risks() 
        ) 
 
        if not all_risks: 
            return None 
 
        # ------------------------------------------------------ 
        # Existing rescue destinations 
        # ------------------------------------------------------ 
 
        occupied_destinations = set() 
 
        for other_task in self.tasks: 
 
            if ( 
                other_task.task_id 
                == task.task_id 
            ): 
                continue 
 
            if getattr( 
                other_task, 
                "is_completed", 
                False, 
            ): 
                continue 
 
            if getattr( 
                other_task, 
                "in_rescue", 
                False, 
            ): 
 
                destination = getattr( 
                    other_task, 
                    "rescue_destination", 
                    None, 
                ) 
 
                if destination is not None: 
                    occupied_destinations.add( 
                        tuple(destination) 
                    ) 
 
        # ------------------------------------------------------ 
        # Candidate destinations 
        # ------------------------------------------------------ 
 
        candidates = [] 
 
        for zone_id, risk_data in all_risks.items(): 
 
            try: 
                zone_id_int = int( 
                    zone_id 
                ) 
            except ( 
                TypeError, 
                ValueError, 
            ): 
                continue 
 
            # Never rescue to source zone. 
            if zone_id_int == int( 
                source_zone 
            ): 
                continue 
 
            dynamic_risk = self._safe_float( 
                risk_data.get( 
                    "dynamic_risk", 
                    1.0, 
                ), 
                default=1.0, 
            ) 
 
            # Only genuinely safer zones. 
            if dynamic_risk >= 0.30: 
                continue 
 
            try: 
                center = tuple( 
                    self.warehouse.get_zone_center( 
                        zone_id_int 
                    ) 
                ) 
            except Exception: 
                continue 
 
            occupied_penalty = ( 
                1 
                if center 
                in occupied_destinations 
                else 0 
            ) 
 
            distance = self._manhattan_distance( 
                task.pickup, 
                center, 
            ) 
 
            candidates.append( 
                ( 
                    occupied_penalty, 
                    dynamic_risk, 
                    distance, 
                    zone_id_int, 
                    center, 
                ) 
            ) 
 
        if not candidates: 
            return None 
 
        candidates.sort( 
            key=lambda item: ( 
                item[0], 
                item[1], 
                item[2], 
                item[3], 
            ) 
        ) 
 
        return candidates[0][4] 
 
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
 
        Lower score is better. 
 
        Factors: 
 
            distance 
            urgency 
            risk 
            priority 
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
 
        urgency = self._safe_float( 
            getattr( 
                task, 
                "urgency_score", 
                0.0, 
            ) 
        ) 
 
        risk = self._safe_float( 
            getattr( 
                task, 
                "risk_score", 
                0.0, 
            ) 
        ) 
 
        # Live dynamic risk at the task pickup zone
        zone_risk = 0.0
        try:
            pickup_zone_id = self.warehouse.get_zone_id(
                tuple(task.pickup)
            )
            zone_risks = self.warehouse.get_all_zone_risks()
            zone_risk = self._safe_float(
                zone_risks.get(
                    pickup_zone_id,
                    {},
                ).get(
                    "dynamic_risk",
                    0.0,
                )
            )
        except Exception:
            zone_risk = 0.0
        best_agv = None 
        best_score = None 
 
        for agv_id in available_agv_ids: 
 
            agv = self.system.agvs.get( 
                agv_id 
            ) 
 
            if agv is None: 
                continue 
 
            distance = self._manhattan_distance( 
                agv.position, 
                task.pickup, 
            ) 
 
            score = ( 
                float(distance) 
                - 10.0 * urgency 
                - 5.0 * risk 
                - 5.0 * zone_risk 
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
    # RISK-AWARE TASK ASSIGNMENT 
    # ========================================================== 
 
    def _assign_tasks_risk_aware( 
        self, 
    ): 
        """ 
        Assign pending tasks to AGVs using risk-aware 
        priority and responder selection. 
 
        Existing assignments are preserved. 
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
 
            if task.assigned_agv is not None and task.status != "PENDING": 
 
                available_agvs.discard( 
                    task.assigned_agv 
                ) 
 
                continue 
 
            pending_tasks.append( 
                task 
            ) 
 
        # ------------------------------------------------------
        # PROACTIVE PRE-PICKUP RISK ESCALATION
        #
        # Refresh pending-task risk using the current pickup-zone
        # dynamic risk before risk-aware task ordering.
        # ------------------------------------------------------
        for task in pending_tasks:
            try:
                pickup_zone_id = self.warehouse.get_zone_id(
                    tuple(task.pickup)
                )
                zone_data = (
                    self.warehouse
                    .get_all_zone_risks()
                    .get(
                        pickup_zone_id,
                        {},
                    )
                )
                dynamic_risk = self._safe_float(
                    zone_data.get(
                        "dynamic_risk",
                        0.0,
                    )
                )
            except Exception:
                dynamic_risk = 0.0
            existing_risk = self._safe_float(
                getattr(
                    task,
                    "risk_score",
                    0.0,
                )
            )
            task.risk_score = float(
                np.clip(
                    max(
                        existing_risk,
                        dynamic_risk,
                    ),
                    0.0,
                    1.0,
                )
            )
            existing_future_risk = self._safe_float(
                getattr(
                    task,
                    "future_risk_60",
                    0.0,
                )
            )
            task.future_risk_60 = float(
                np.clip(
                    max(
                        existing_future_risk,
                        dynamic_risk,
                    ),
                    0.0,
                    1.0,
                )
            )
            existing_urgency = self._safe_float(
                getattr(
                    task,
                    "urgency_score",
                    0.0,
                )
            )
            task.urgency_score = float(
                np.clip(
                    max(
                        existing_urgency,
                        dynamic_risk,
                    ),
                    0.0,
                    1.0,
                )
            )
            # Escalate only to priority levels already supported
            # by the existing ranking system.
            if task.risk_score >= 0.80:
                task.priority = "HIGH"
        # Use the canonical cold-chain priority score for actual
        # task assignment. This makes RSET and ETA operational
        # rather than merely diagnostic.
        pending_tasks.sort(
            key=self._get_risk_priority_score,
            reverse=True,
        )for task in pending_tasks: 
 
            if not available_agvs: 
                break 
 
            agv_id = self._select_best_responder( 
                task, 
                available_agv_ids=available_agvs, 
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
    # ASSIGN TASKS 
    # ========================================================== 
 
    def _assign_tasks( 
        self, 
    ): 
        """ 
        Assign all tasks. 
 
        Baseline mode: 
            Task 1 -> AGV 1 
            Task 2 -> AGV 2 
            Task 3 -> AGV 3 
 
        Risk-aware mode: 
            Use dynamic risk-aware responder selection. 
        """ 
 
        if self.system is None: 
            return 
 
        if self.risk_aware_assignment: 
 
            self._assign_tasks_risk_aware() 
 
        else: 
 
            for task in self.tasks: 
 
                agv_id = int( 
                    task.task_id 
                ) 
 
                if agv_id not in self.system.agvs: 
                    continue 
 
                self.system.assign_task( 
                    task_id=task.task_id, 
                    agv_id=agv_id, 
                ) 
 
                task.assigned_agv = ( 
                    agv_id 
                ) 
 
    # ========================================================== 
    # START ASSIGNED TASKS 
    # ========================================================== 
 
    def _start_assigned_tasks( 
        self, 
    ): 
        """ 
        Start assigned tasks and move AGVs toward pickup. 
        """ 
 
        if self.system is None: 
            return 
 
        for task in self.tasks: 
 
            if task.assigned_agv is None: 
                continue 
 
            task.start() 
 
            agv = self.system.agvs.get( 
                task.assigned_agv 
            ) 
 
            if agv is None: 
                continue 
 
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
 
    # ========================================================== 
    # RESET 
    # ========================================================== 
 
    def reset( 
        self, 
        *, 
        seed=None, 
        options=None, 
    ): 
        """ 
        Reset the environment. 
        """ 
        # Reset warehouse thermal state for a fresh episode
        self.warehouse.reset_thermal_state()
 
        super().reset( 
            seed=seed 
        ) 
 
        # ------------------------------------------------------
        # LIVE EXTERNAL WEATHER
        # Fetch once per episode and keep the thermal simulation
        # independent from repeated API calls during steps.
        # ------------------------------------------------------
        try:
            self.warehouse.fetch_external_weather(
                latitude=11.0168,
                longitude=76.9558,
            )
        except Exception:
            # Preserve deterministic offline operation.
            self.warehouse.set_external_weather(
                temperature=25.0,
                humidity=60.0,
            )
        self.current_step = 0 
 
        self.last_collision_count = 0 
        self.last_completed_count = 0 
        self.last_pickup_count = 0 
 
        self.episode_collision_count = 0 
        self.episode_completed_count = 0 
        self.episode_pickup_count = 0 
        self.episode_total_reward = 0.0 
        # ------------------------------------------------------
        # Cold-chain thermal exposure metrics
        # ------------------------------------------------------
        self.episode_total_exposure_steps = 0
        self.episode_tasks_with_exposure = 0
        self.episode_tasks_exposure_exceeded = 0

 
        self.current_risk_records = {} 
        self.task_risk = {} 
 
        # ------------------------------------------------------ 
        # Dataset starting position 
        # ------------------------------------------------------ 
 
        if not self.trainable_indices: 
            raise RuntimeError( 
                "No trainable dataset records available." 
            ) 
 
        if seed is not None: 
 
            trainable_position = ( 
                int(seed) 
                % len( 
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
        # Multi-AGV system 
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
            # Pickup 
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
            # Delivery 
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
 
            # -------------------------------------------------- 
            # Dataset record 
            # -------------------------------------------------- 
 
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
            # Risk mapping 
            # -------------------------------------------------- 
 
            self.current_risk_records[ 
                agv_id 
            ] = dataset_record 
 
            self.task_risk[ 
                agv_id 
            ] = dataset_record 
 
            # ================================================== 
            # CREATE TASK 
            # ================================================== 
 
            task = Task( 
                task_id=agv_id, 
                pickup=pickup, 
                delivery=delivery, 
            ) 
 
            # -------------------------------------------------- 
            # Task risk mapping 
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
 
            task.temperature_normalized = ( 
                dataset_record[ 
                    "temperature_normalized" 
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
 
            task.future_risk_60 = ( 
                dataset_record[ 
                    "future_risk_60" 
                ] 
            ) 
 
            task.rset_minutes = ( 
                dataset_record[ 
                    "rset_minutes" 
                ] 
            ) 
 
            # --------------------------------------------------
            # Cold-chain thermal exposure state
            # --------------------------------------------------
            task.thermal_exposure_steps = 0
            task.picked_up_steps = 0
            task.max_exposure_temperature = 0.0
            task.exposure_exceeded = False

            # --------------------------------------------------
            # Thermal product-quality degradation model
            # --------------------------------------------------
            # Simulation assumption: 2-8 C is the safe range.
            # Outside this range, temperature-dependent damage
            # accumulates while the product is being transported.
            # quality_remaining is normalized to [0, 1].
            task.quality_remaining = 1.0
            task.thermal_damage = 0.0
            task.degradation_occurred = False
            task.spoilage_threshold_reached = False

            task.rescue_recovered = False
            # Rescue state 
            # -------------------------------------------------- 
 
            task.in_rescue = False 
            task.rescue_destination = None 
            task.original_delivery = ( 
                task.delivery 
            ) 
 
            # -------------------------------------------------- 
            # Assignment 
            # -------------------------------------------------- 
 
            task.assigned_agv = ( 
                agv_id 
            ) 
 
            # -------------------------------------------------- 
            # Register 
            # -------------------------------------------------- 
 
            self.system.add_task( 
                task 
            ) 
 
            self.tasks.append( 
                task 
            ) 
 
        # ====================================================== 
        # ASSIGN TASKS 
        # ====================================================== 
 
        # ------------------------------------------------------
        # INITIAL WEATHER-DRIVEN THERMAL UPDATE
        # Apply current external weather to every zone before
        # risk-aware task assignment.
        # ------------------------------------------------------
        for zone_id in range(
            1,
            self.warehouse.total_zones + 1,
        ):
            self.warehouse.update_zone_from_weather(
                zone_id
            )
        self._assign_tasks() 
 
        # ====================================================== 
        # START TASKS 
        # ====================================================== 
 
        self._start_assigned_tasks() 
 
        # ====================================================== 
        # INITIAL OBSERVATION 
        # ====================================================== 
 
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
        """ 
        Execute one environment step. 
        """ 
 
        if self.system is None: 
            raise RuntimeError( 
                "Environment must be reset before step()." 
            ) 
 
        actions = np.asarray( 
            actions, 
            dtype=np.int64, 
        ) 
 
        # ------------------------------------------------------ 
        # Action shape 
        # ------------------------------------------------------ 
 
        if actions.shape != ( 
            self.num_agvs, 
        ): 
            raise ValueError( 
                f"Expected {self.num_agvs} actions, " 
                f"got shape {actions.shape}." 
            ) 
 
        # ------------------------------------------------------ 
        # Action range 
        # ------------------------------------------------------ 
 
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
        # LIVE THERMAL / WEATHER UPDATE
        # ======================================================

        for zone_id in range(1, self.warehouse.total_zones + 1):
            self.warehouse.update_zone_from_weather(zone_id)

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
                ): 
 
                    collision_agvs.add( 
                        agv_a 
                    ) 
 
                    collision_agvs.add( 
                        agv_b 
                    ) 
 
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
 
            action = int( 
                actions[ 
                    agv_id - 1 
                ] 
            ) 
 
            # -------------------------------------------------- 
            # Collision 
            # -------------------------------------------------- 
 
            if agv_id in collision_agvs: 
 
                total_reward -= 20.0 
 
                total_reward -= ( 
                    float( 
                        risk.get( 
                            "urgency_score", 
                            0.0, 
                        ) 
                    ) 
                ) 
 
                continue 
 
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
 
            target = ( 
                proposed_positions[ 
                    agv_id 
                ] 
            ) 
 
            current = ( 
                current_positions[ 
                    agv_id 
                ] 
            ) 
 
            # -------------------------------------------------- 
            # Invalid boundary movement 
            # -------------------------------------------------- 
 
            if target == current: 
 
                total_reward -= 2.0 
 
                continue 
 
            # -------------------------------------------------- 
            # Movement 
            # -------------------------------------------------- 
 
            moved = self.system.move_agv( 
                agv_id, 
                target, 
            ) 
 
            if not moved: 
 
                total_reward -= 2.0 
 
                continue 
 
            # -------------------------------------------------- 
            # --------------------------------------------------
            # Live dynamic-risk penalty
            # --------------------------------------------------

            destination_zone_risk = 0.0

            try:
                destination_zone_id = self.warehouse.get_zone_id(
                    tuple(target)
                )

                destination_zone_data = self.warehouse.get_all_zone_risks().get(
                    destination_zone_id,
                    {},
                )

                destination_zone_risk = self._safe_float(
                    destination_zone_data.get(
                        'dynamic_risk',
                        0.0,
                    )
                )

            except Exception:
                destination_zone_risk = 0.0

            total_reward -= (
                2.0 * destination_zone_risk
            )

            # Movement cost 
            # -------------------------------------------------- 
 
            total_reward -= 0.2 
 
            # -------------------------------------------------- 
            # Distance after movement 
            # --------------------------------------------------

            new_distance = (
                self._distance_to_task_target(agv) 
            )

            distance_improvement = (
                old_distance - new_distance
            )

           
 
            # -------------------------------------------------- 
            # Progress reward 
            # -------------------------------------------------- 
 
            if distance_improvement > 0: 
 
                total_reward += ( 
                    2.0 
                    * float( 
                        distance_improvement 
                    ) 
                ) 
 
                total_reward += ( 
                    0.5 
                    * 
                    float( 
                        risk.get( 
                            "urgency_score", 
                            0.0, 
                        ) 
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
# Delivery progress reward
# --------------------------------------------------

            if task.status == "PICKED_UP":

                delivery_target = task.delivery

                delivery_distance = (
                    self._manhattan_distance(
                        tuple(agv.position),
                        tuple(delivery_target)
                    )
                )

                previous_delivery_distance = getattr(
                    agv,
                    "_previous_delivery_distance",
                    delivery_distance,
                )

                improvement = (
        previous_delivery_distance
        -
        delivery_distance
    )

                if improvement > 0:
                    total_reward += (
                    3.0
                    *
                    float(improvement)
                )

                elif improvement < 0:
                    total_reward -= (
                        3.0
                        *
                        abs(float(improvement))
                    )

                agv._previous_delivery_distance = (
                    delivery_distance
                )
 
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
                == tuple(task.pickup) 
            ): 
 
                task.pickup_completed() 
 
                # --------------------------------------------------
                # CONDITIONAL RESCUE
                # --------------------------------------------------
                # Normal-risk tasks go directly to delivery.
                # Rescue is activated only for sufficiently
                # high current task risk.
                # --------------------------------------------------

                current_risk = self._safe_float(getattr(task, "risk_score", risk.get("risk_score", 0.0)))

                if current_risk >= 0.50:
                    safe_destination = (
                        self._select_safe_destination(
                            task
                        )
                    )
                else:
                    safe_destination = None

                if safe_destination is not None:

                    # Preserve original delivery.
                    task.original_delivery = (
                        task.delivery
                    )

                    task.rescue_destination = (
                        tuple(
                            safe_destination
                        )
                    )

                    task.in_rescue = True

                    agv.destination = (
                        task.rescue_destination
                    )

                else:

                    task.in_rescue = False

                    task.rescue_destination = None

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
                    float( 
                        risk.get( 
                            "urgency_score", 
                            0.0, 
                        ) 
                    ) 
                ) 
 
                if ( 
                    str( 
                        risk.get( 
                            "priority", 
                            "LOW", 
                        ) 
                    ).upper() 
                    == "HIGH" 
                ): 
                    total_reward += 3.0 
 
            # ================================================== 
            # ==================================================
            # COLD-CHAIN THERMAL EXPOSURE
            # ==================================================
            # Metric-only measurement. This does not modify
            # reward, rescue, assignment, or DQN behavior.

            if task.status == "PICKED_UP":
                task.picked_up_steps += 1

                exposure_zone_id = self.warehouse.get_zone_id(
                    tuple(agv.position)
                )

                exposure_temperature = float(
                    self.warehouse._thermal_state[
                        exposure_zone_id
                    ]["temperature"]
                )

                was_exposed = (
                    task.thermal_exposure_steps > 0
                )

                task.thermal_exposure_steps += 1

                if not was_exposed:
                    self.episode_tasks_with_exposure += 1

                self.episode_total_exposure_steps += 1

                task.max_exposure_temperature = max(
                    float(task.max_exposure_temperature),
                    exposure_temperature,
                )

                # First-order thermal quality degradation.
                # The coefficient is a documented simulation parameter,
                # not a fitted food-science constant.
                if exposure_temperature < 2.0:
                    thermal_stress = 2.0 - exposure_temperature
                elif exposure_temperature > 8.0:
                    thermal_stress = exposure_temperature - 8.0
                else:
                    thermal_stress = 0.0

                if thermal_stress > 0.0:
                    degradation_rate = 0.0025 * thermal_stress
                    task.thermal_damage += degradation_rate
                    task.quality_remaining = max(
                        0.0,
                        float(task.quality_remaining) *
                        (2.718281828 ** (-degradation_rate))
                    )
                    task.degradation_occurred = True

                    if (
                        task.quality_remaining <= 0.20
                        and not task.spoilage_threshold_reached
                    ):
                        task.spoilage_threshold_reached = True

                if (
                    exposure_temperature < 2.0
                    or exposure_temperature > 8.0
                ):
                    if not task.exposure_exceeded:
                        task.exposure_exceeded = True
                        self.episode_tasks_exposure_exceeded += 1

            # DELIVERY / RESCUE 
            # ================================================== 
 
            target = None 
 
            if getattr( 
                task, 
                "in_rescue", 
                False, 
            ): 
 
                target = getattr( 
                    task, 
                    "rescue_destination", 
                    None, 
                ) 
 
            else: 
 
                target = task.delivery 
 
            if ( 
                task.status == "PICKED_UP" 
                and 
                target is not None 
                and 
                tuple(agv.position) 
                == tuple(target) 
            ): 
 
                # ------------------------------------------------ 
                # Rescue destination reached 
                # ------------------------------------------------ 
 
                if getattr( 
                    task, 
                    "in_rescue", 
                    False, 
                ): 
 
                    task.in_rescue = False 
 
                    task.rescue_destination = None 
 
                    task.rescue_recovered = True

                    agv.destination = (
                        task.original_delivery
                    )
 
                    agv.status = "MOVING" 
 
                    # Continue toward original delivery. 
                    total_reward +=25.0 
 
                    continue 
 
                # ------------------------------------------------ 
                # Final delivery reached 
                # ------------------------------------------------ 
 
                task.complete() 
 
                completed_count += 1 
 
                total_reward += 100.0 
 
                total_reward += ( 
                    20.0 
                    * 
                    float( 
                        risk.get( 
                            "urgency_score", 
                            0.0, 
                        ) 
                    ) 
                ) 
 
                if ( 
                    str( 
                        risk.get( 
                            "priority", 
                            "LOW", 
                        ) 
                    ).upper() 
                    == "HIGH" 
                ): 
                    total_reward += 5.0 
 
                # ------------------------------------------------ 
                # Return to depot 
                # ------------------------------------------------ 
 
                agv.destination = ( 
                    agv_id - 1, 
                    0, 
                ) 
 
                agv.status = "RETURNING" 
 
                agv.current_task = None 
 
        # ====================================================== 
        # RETURN TO DEPOT 
        # ====================================================== 
 
        for agv_id, agv in ( 
            self.system.agvs.items() 
        ): 
 
            if ( 
                getattr( 
                    agv, 
                    "status", 
                    None, 
                ) 
                == "RETURNING" 
                and 
                getattr( 
                    agv, 
                    "destination", 
                    None, 
                ) is not None 
                and 
                tuple( 
                    agv.position 
                ) 
                == tuple( 
                    agv.destination 
                ) 
            ): 
 
                total_reward += 20.0 
 
                agv.status = "IDLE" 
 
                agv.destination = None 
 
                agv.current_task = None 
 
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
 
        # Episode is complete only when: 
        # 
        # 1. All tasks are delivered 
        # 2. All AGVs return to depot 
 
        terminated = bool( 
            tasks_completed 
            and 
            agvs_returned 
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
 
                "episode_total_exposure_steps":
                    self.episode_total_exposure_steps,

                "episode_tasks_with_exposure":
                    self.episode_tasks_with_exposure,

                "episode_tasks_exposure_exceeded":
                    self.episode_tasks_exposure_exceeded,

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
        Manhattan distance from AGV to current operational target. 
        """ 
 
        target = self._get_current_target( 
            agv 
        ) 
 
        if target is None: 
            return 0.0 
 
        return float( 
            self._manhattan_distance( 
                agv.position, 
                target, 
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
        Return position produced by one primitive movement. 
 
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
 
        Total: 
 
            num_agvs * 8 
 
        For 3 AGVs: 
 
            24 
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
 
            target = self._get_current_target( 
                agv 
            ) 
 
            if target is None: 
 
                gx, gy = ( 
                    x, 
                    y, 
                ) 
 
            else: 
 
                gx, gy = map( 
                    int, 
                    target, 
                ) 
 
            # -------------------------------------------------- 
            # Risk information 
            # -------------------------------------------------- 
 
            risk = 0.0 
            urgency = 0.0 
            future_risk_60 = 0.0 
            temperature_normalized = 0.0 
 
            # --------------------------------------------------
            # Live zone dynamic risk
            # --------------------------------------------------
            
            zone_risk = 0.0
            
            try:
                zone_id = self.warehouse.get_zone_id(
                    tuple(agv.position)
                )
                
                all_zone_risks = self.warehouse.get_all_zone_risks()
                zone_data = all_zone_risks.get(
                    zone_id,
                    {}
                )
                
                zone_risk = self._safe_float(
                    zone_data.get(
                        "dynamic_risk",
                        0.0
                    )
                )
            
            except Exception:
                zone_risk = 0.0
            
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
 
                    risk = self._safe_float( 
                        getattr( 
                            task, 
                            "risk_score", 
                            self.task_risk.get( 
                                task_id, 
                                {}, 
                            ).get( 
                                "risk_score", 
                                0.0, 
                            ), 
                        ) 
                    ) 
 
                    urgency = self._safe_float( 
                        getattr( 
                            task, 
                            "urgency_score", 
                            self.task_risk.get( 
                                task_id, 
                                {}, 
                            ).get( 
                                "urgency_score", 
                                0.0, 
                            ), 
                        ) 
                    ) 
 
                    future_risk_60 = zone_risk 
 
                    temperature_normalized = self._safe_float( 
                        getattr( 
                            task, 
                            "temperature_normalized", 
                            self.task_risk.get( 
                                task_id, 
                                {}, 
                            ).get( 
                                "temperature_normalized", 
                                self._normalize_temperature( 
                                    getattr( 
                                        task, 
                                        "temperature", 
                                        0.0, 
                                    ) 
                                ), 
                            ), 
                        ) 
                    ) 
 
            # -------------------------------------------------- 
            # --------------------------------------------------
            # Combine task risk with live environmental risk
            # --------------------------------------------------
            
            risk = max(
                risk,
                zone_risk
            )
            
            # Normalized 8 features 
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
                            temperature_normalized, 
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
        # Shape check 
        # ------------------------------------------------------ 
 
        expected_shape = ( 
            self.num_agvs * 8, 
        ) 
 
        if observation.shape != expected_shape: 
 
            raise RuntimeError( 
                "Generated observation has incorrect shape. " 
                f"Expected {expected_shape}, " 
                f"got {observation.shape}." 
            ) 
 
        # ------------------------------------------------------ 
        # Observation-space check 
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
                f"{observation.shape}, " 
                f"min: {observation.min()}, " 
                f"max: {observation.max()}" 
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
        # Risk records 
        # ------------------------------------------------------ 
 
        task_records = [] 
 
        for task in self.tasks: 
 
            record = self.task_risk.get( 
                task.task_id, 
                None, 
            ) 
 
            if record is None: 
 
                record = self.current_risk_records.get( 
                    getattr( 
                        task, 
                        "assigned_agv", 
                        None, 
                    ), 
                    {}, 
                ) 
 
            task_records.append( 
                record 
            ) 
 
        # ------------------------------------------------------ 
        # High-risk count 
        # ------------------------------------------------------ 
 
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
 
        # ------------------------------------------------------ 
        # Average risk 
        # ------------------------------------------------------ 
 
        average_risk = 0.0 
 
        if task_records: 
 
            average_risk = float( 
                np.mean( 
                    [ 
                        self._safe_float( 
                            risk.get( 
                                "risk_score", 
                                0.0, 
                            ) 
                        ) 
                        for risk in task_records 
                    ] 
                ) 
            ) 
 
        # ------------------------------------------------------ 
        # Average urgency 
        # ------------------------------------------------------ 
 
        average_urgency = 0.0 
 
        if task_records: 
 
            average_urgency = float( 
                np.mean( 
                    [ 
                        self._safe_float( 
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
        # Risk-aware task order 
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
                        self._safe_float( 
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
                        self._safe_float( 
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
                        self._safe_float( 
                            record.get( 
                                "future_risk_60", 
                                record.get( 
                                    "y_next_60_R2", 
                                    0.0, 
                                ), 
                            ) 
                        ), 
 
                    "eta_60": 
                        self._safe_float( 
                            record.get( 
                                "eta_60", 
                                record.get( 
                                    "eta_to_R2_60", 
                                    0.0, 
                                ), 
                            ) 
                        ), 
 
                    "rset_minutes": 
                        self._safe_float( 
                            record.get( 
                                "rset_minutes", 
                                60.0, 
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
            # Risk-aware task order 
            # -------------------------------------------------- 
 
            "risk_aware_task_order": 
                risk_aware_task_order, 
 
            # -------------------------------------------------- 
            # Dataset 
            # -------------------------------------------------- 
 
            "dataset_index": 
                self.episode_dataset_start, 
 
            "dataset_tasks": 
                [ 
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
 
                        "temperature_normalized": 
                            getattr( 
                                task, 
                                "temperature_normalized", 
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
 
                        "future_risk_60": 
                            getattr( 
                                task, 
                                "future_risk_60", 
                                0.0, 
                            ), 
 
                        "rset_minutes": 
                            getattr( 
                                task, 
                                "rset_minutes", 
                                60.0, 
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
        """ 
        Render current warehouse state. 
 
        Symbols: 
 
            1 / 2 / 3 = AGVs 
            P         = pickup 
            D         = delivery 
            G         = current goal 
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
 
            # -------------------------------------------------- 
            # Active operational goal 
            # -------------------------------------------------- 
 
            if ( 
                getattr( 
                    task, 
                    "in_rescue", 
                    False, 
                ) 
                and 
                getattr( 
                    task, 
                    "rescue_destination", 
                    None, 
                ) is not None 
            ): 
 
                active_goals.add( 
                    tuple( 
                        task.rescue_destination 
                    ) 
                ) 
 
            elif task.status == "IN_PROGRESS": 
 
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
 
        output = "\n".join( 
            rows 
        ) 
 
        return output 
 
    # ========================================================== 
    # CLOSE 
    # ========================================================== 
 
    def close( 
        self, 
    ): 
        """ 
        Release environment resources. 
        """ 
 
        self.system = None 
 
        return None 
 
 
__all__ = [ 
    "MultiAGVEnv", 
] 











