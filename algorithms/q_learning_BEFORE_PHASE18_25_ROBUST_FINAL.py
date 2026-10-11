"""
Tabular Q-learning agent.

This is the base reinforcement-learning algorithm.

The agent uses:

    Q(s, a)

with:

    epsilon-greedy exploration
    learning rate
    discount factor
    temporal-difference update
"""

from __future__ import annotations


import pickle
import random
from collections import defaultdict

import numpy as np


class QLearningAgent:

    # ==========================================================
    # PHASE 23: RISK-AWARE Q-LEARNING STATE
    # ==========================================================
    def build_risk_aware_state(
        self,
        agv_state,
        task_risk=0.0,
        product_urgency=0.0,
        thermal_trend=0.0,
        remaining_safe_time=60.0
    ):
        base=list(agv_state)
        return tuple(
            base+[
                float(task_risk),
                float(product_urgency),
                float(thermal_trend),
                float(remaining_safe_time)
            ]
        )
    # ==========================================================
    # PHASE 24: RISK-AWARE REWARD
    # ==========================================================
    def calculate_risk_aware_reward(
        self,
        operational_reward=0.0,
        risk_reduction=0.0,
        product_protection=0.0,
        successful_rescue=0.0,
        thermal_risk=0.0,
        spoilage=0.0,
        late_rescue=0.0,
        unsafe_route=0.0
    ):
        return float(
            operational_reward
            + 2.0*float(risk_reduction)
            + 3.0*float(product_protection)
            + 5.0*float(successful_rescue)
            - 2.0*float(thermal_risk)
            - 8.0*float(spoilage)
            - 5.0*float(late_rescue)
            - 5.0*float(unsafe_route)
        )
    def risk_aware_reward_from_transition(
        self,
        operational_reward,
        previous_risk,
        current_risk,
        task=None,
        unsafe_route=False
    ):
        risk_reduction=max(
            0.0,
            float(previous_risk)-float(current_risk)
        )
        protection=0.0
        rescue=0.0
        spoilage=0.0
        late=0.0
        thermal=0.0
        if task is not None:
            quality=float(
                getattr(task,"quality_remaining",1.0)
            )
            degradation=bool(
                getattr(task,"degradation_occurred",False)
            )
            completed=(
                getattr(task,"status","")=="COMPLETED"
            )
            protection=(
                1.0 if completed and not degradation
                else 0.0
            )
            rescue=(
                1.0 if getattr(
                    task,"rescue_recovered",False
                )
                else 0.0
            )
            spoilage=(
                1.0 if getattr(
                    task,
                    "spoilage_threshold_reached",
                    False
                )
                else 0.0
            )
            late=(
                1.0 if getattr(
                    task,"late_rescue",False
                )
                else 0.0
            )
            if quality<1.0:
                thermal=1.0-quality
        return self.calculate_risk_aware_reward(
            operational_reward=operational_reward,
            risk_reduction=risk_reduction,
            product_protection=protection,
            successful_rescue=rescue,
            thermal_risk=thermal,
            spoilage=spoilage,
            late_rescue=late,
            unsafe_route=1.0 if unsafe_route else 0.0
        )
    def __init__(
        self,
        action_count: int = 5,
        learning_rate: float = 0.1,
        discount_factor: float = 0.95,
        epsilon: float = 1.0,
        epsilon_min: float = 0.05,
        epsilon_decay: float = 0.995,
        seed: int = 42,
        
    ):
        self.state_decimals = 2
        
        if action_count <= 0:
            raise ValueError(
                "action_count must be positive."
            )

        self.action_count = int(
            action_count
        )

        self.learning_rate = float(
            learning_rate
        )

        self.discount_factor = float(
            discount_factor
        )

        self.epsilon = float(
            epsilon
        )

        self.epsilon_min = float(
            epsilon_min
        )

        self.epsilon_decay = float(
            epsilon_decay
        )

        self.random = random.Random(
            seed
        )

        # --------------------------------------------------
        # Q-table
        # --------------------------------------------------

        self.q_table = defaultdict(
            self._new_q_values
        )

        self.training_steps = 0

    # ==================================================
    # DEFAULT Q VALUES
    # ==================================================

    def _new_q_values(self):

        return np.zeros(
            self.action_count,
            dtype=np.float64,
        )

    # ==================================================
    # STATE KEY
    # ==================================================

    def state_key(
        self,
        state,
    ):

        if isinstance(
            state,
            (np.ndarray, list, tuple),
        ):

            values = np.asarray(
                state,
                dtype=np.float64,
            )

            return tuple(
                np.round(
                    values,
                    self.state_decimals,
                ).tolist()
            )

        return state

    # ==================================================
    # ACTION SELECTION
    # ==================================================

    def select_action(
        self,
        state,
        training=True,
    ):

        key = self.state_key(
            state
        )

        # --------------------------------------------------
        # Exploration
        # --------------------------------------------------

        if (
            training
            and
            self.random.random()
            <
            self.epsilon
        ):

            return self.random.randrange(
                self.action_count
            )

        # --------------------------------------------------
        # Exploitation
        # --------------------------------------------------

        q_values = self.q_table[
            key
        ]

        max_value = np.max(
            q_values
        )

        best_actions = np.flatnonzero(
            np.isclose(
                q_values,
                max_value,
            )
        )

        return int(
            self.random.choice(
                best_actions.tolist()
            )
        )

    # ==================================================
    # Q UPDATE
    # ==================================================

    def update(
        self,
        state,
        action,
        reward,
        next_state,
        terminated,
    ):

        state_key = self.state_key(
            state
        )

        next_state_key = self.state_key(
            next_state
        )

        action = int(
            action
        )

        if not (
            0
            <=
            action
            <
            self.action_count
        ):

            raise ValueError(
                f"Invalid action {action}."
            )

        current_q = self.q_table[
            state_key
        ][
            action
        ]

        # --------------------------------------------------
        # Bellman target
        # --------------------------------------------------

        if terminated:

            target = float(
                reward
            )

        else:

            next_max = np.max(
                self.q_table[
                    next_state_key
                ]
            )

            target = (
                float(reward)
                +
                self.discount_factor
                *
                next_max
            )

        # --------------------------------------------------
        # Temporal-difference update
        # --------------------------------------------------

        td_error = (
            target
            -
            current_q
        )

        self.q_table[
            state_key
        ][
            action
        ] += (
            self.learning_rate
            *
            td_error
        )

        self.training_steps += 1

        return float(
            td_error
        )

    # ==================================================
    # EPSILON DECAY
    # ==================================================

    def decay_epsilon(self):

        self.epsilon = max(
            self.epsilon_min,
            self.epsilon
            *
            self.epsilon_decay,
        )

        return self.epsilon

    # ==================================================
    # GREEDY ACTION
    # ==================================================

    def greedy_action(
        self,
        state,
    ):

        return self.select_action(
            state,
            training=False,
        )

    # ==================================================
    # TABLE SIZE
    # ==================================================

    @property
    def state_count(self):

        return len(
            self.q_table
        )

    # ==================================================
    # RESET
    # ==================================================

    def clear(self):

        self.q_table.clear()

        self.training_steps = 0

        # ==================================================
    # SAVE
    # ==================================================

    def save(
        self,
        filepath,
    ):

        data = {
            "q_table": dict(
                self.q_table
            ),
            "action_count":
                self.action_count,
            "learning_rate":
                self.learning_rate,
            "discount_factor":
                self.discount_factor,
            "epsilon":
                self.epsilon,
            "epsilon_min":
                self.epsilon_min,
            "epsilon_decay":
                self.epsilon_decay,
            "training_steps":
                self.training_steps,
        }

        with open(
            filepath,
            "wb",
        ) as file:

            pickle.dump(
                data,
                file,
            )

    # ==================================================
    # LOAD
    # ==================================================

    def load(
        self,
        filepath,
    ):

        with open(
            filepath,
            "rb",
        ) as file:

            data = pickle.load(
                file
            )

        self.action_count = int(
            data["action_count"]
        )

        self.learning_rate = float(
            data["learning_rate"]
        )

        self.discount_factor = float(
            data["discount_factor"]
        )

        self.epsilon = float(
            data["epsilon"]
        )

        self.epsilon_min = float(
            data["epsilon_min"]
        )

        self.epsilon_decay = float(
            data["epsilon_decay"]
        )

        self.training_steps = int(
            data["training_steps"]
        )

        self.q_table.clear()

        for state, values in (
            data["q_table"].items()
        ):

            self.q_table[state] = np.asarray(
                values,
                dtype=np.float64,
            )

    
__all__ = [
    "QLearningAgent",
]
