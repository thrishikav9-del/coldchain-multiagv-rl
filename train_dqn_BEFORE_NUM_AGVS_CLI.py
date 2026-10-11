"""
Multi-AGV DQN training and evaluation.

Environment:
    MultiAGVEnv

Observation:
    6 features per AGV

        x
        y
        goal_x
        goal_y
        risk
        urgency

For 3 AGVs:

    3 * 6 = 18 state features

Actions:
    0 = Stay
    1 = Right
    2 = Left
    3 = Up
    4 = Down

Centralized DQN:

    One environment step produces ONE joint transition.

        state
            ->
        joint actions
            ->
        reward
            ->
        next_state
            ->
        done

For 3 AGVs and 5 actions:

    5 ** 3 = 125 joint actions


Final evaluation architecture:

    DQN
      |
      v
    proposed joint action
      |
      v
    Hybrid task/safety controller
      |
      +--> collision check
      |
      +--> target-progress check
      |
      +--> pickup target
      |
      +--> delivery target
      |
      +--> depot return
      |
      v
    environment
"""

from __future__ import annotations

import argparse
import os

import numpy as np

import csv
from datetime import datetime

from algorithms.dqn_agent import DQNAgent
from environment.multi_agv_env import MultiAGVEnv


# ==========================================================
# ARGUMENTS
# ==========================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description="Multi-AGV DQN training and evaluation"
    )

    # ======================================================
    # TRAINING
    # ======================================================

    parser.add_argument(
        "--train",
        action="store_true",
        help="Train the DQN",
    )

    # ======================================================
    # EVALUATION
    # ======================================================

    parser.add_argument(
        "--eval",
        action="store_true",
        help="Evaluate the saved DQN model",
    )

    # ======================================================
    # EPISODES
    # ======================================================

    parser.add_argument(
        "--episodes",
        type=int,
        default=100,
        help="Number of episodes for training or evaluation",
    )

    # ======================================================
    # SINGLE SEED
    # ======================================================

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Base random seed",
    )

    # ======================================================
    # MULTI-SEED EVALUATION
    # ======================================================

    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=None,
        help=(
            "Run evaluation for multiple seeds. "
            "Example: --seeds 42 123 999"
        ),
    )

    # ======================================================
    # EVALUATION RECOVERY CONTROLLER
    # ======================================================

    parser.add_argument(
        "--no-recovery",
        action="store_true",
        help=(
            "Disable the evaluation delivery recovery "
            "controller and use pure DQN actions."
        ),
    )

    return parser.parse_args()

# ==========================================================
# CONFIGURATION
# ==========================================================

EPISODES = 5000

MAX_STEPS = 300

NUM_AGVS = 3

ACTION_SIZE = 5

LEARNING_RATE = 0.0001

GAMMA = 0.99

TAU = 0.001

REPLAY_CAPACITY = 200000

BATCH_SIZE = 64

EPSILON_START = 1.0

EPSILON_MIN = 0.01

EPSILON_DECAY = 0.992

CHECKPOINT_INTERVAL = 100


# ==========================================================
# DIRECTORIES
# ==========================================================

MODEL_DIR = "models"

CHECKPOINT_DIR = os.path.join(
    MODEL_DIR,
    "checkpoints",
)

os.makedirs(
    MODEL_DIR,
    exist_ok=True,
)

os.makedirs(
    CHECKPOINT_DIR,
    exist_ok=True,
)


# ==========================================================
# HELPER: RESET ENVIRONMENT
# ==========================================================

def reset_environment(
    env,
    seed=None,
):
    """
    Reset the Gymnasium environment and return only
    the observation.
    """

    if seed is None:

        result = env.reset()

    else:

        result = env.reset(
            seed=seed
        )

    if isinstance(
        result,
        tuple,
    ):

        state = result[0]

    else:

        state = result

    return np.asarray(
        state,
        dtype=np.float32,
    )


# ==========================================================
# HELPER: GET TASK ITERABLE
# ==========================================================

def get_task_iterable(
    env,
):
    """
    Supports both dictionary and list task containers.
    """

    tasks = getattr(
        env,
        "tasks",
        [],
    )

    if hasattr(
        tasks,
        "values",
    ):

        return tasks.values()

    return tasks

# ==========================================================
# EVALUATION DELIVERY RECOVERY CONTROLLER
# ==========================================================

def apply_delivery_recovery_controller(
    env,
    actions,
):
    """
    Evaluation-only delivery recovery controller.

    The DQN selects the joint action first.

    For AGVs that have already picked up their task,
    the controller forces that AGV toward its delivery
    location.

    This controller is NOT used during training.

    Actions:

        0 = Stay
        1 = Right
        2 = Left
        3 = Up
        4 = Down
    """

    actions = np.asarray(
        actions,
        dtype=np.int64,
    ).reshape(
        NUM_AGVS
    )

    actions = actions.copy()

    system = getattr(
        env,
        "system",
        None,
    )

    if system is None:
        return actions

    agvs = getattr(
        system,
        "agvs",
        {},
    )

    tasks = getattr(
        system,
        "tasks",
        {},
    )

    if not hasattr(
        agvs,
        "items",
    ):
        return actions

    # ======================================================
    # PROCESS EACH AGV
    # ======================================================

    for agv_id, agv in agvs.items():

        try:

            agv_index = (
                int(agv_id) - 1
            )

        except (
            TypeError,
            ValueError,
        ):

            continue

        if (
            agv_index < 0
            or
            agv_index >= NUM_AGVS
        ):

            continue

        # --------------------------------------------------
        # Current task
        # --------------------------------------------------

        task_id = getattr(
            agv,
            "current_task",
            None,
        )

        if task_id is None:
            continue

        # --------------------------------------------------
        # Task exists
        # --------------------------------------------------

        if task_id not in tasks:
            continue

        task = tasks[
            task_id
        ]

        # --------------------------------------------------
        # Only after pickup
        # --------------------------------------------------

        status = str(
            getattr(
                task,
                "status",
                "",
            )
        ).upper()

        if status != "PICKED_UP":
            continue

        # --------------------------------------------------
        # Delivery location
        # --------------------------------------------------

        delivery = getattr(
            task,
            "delivery",
            None,
        )

        if delivery is None:
            continue

        if len(delivery) < 2:
            continue

        gx = int(
            delivery[0]
        )

        gy = int(
            delivery[1]
        )

        # --------------------------------------------------
        # Current position
        # --------------------------------------------------

        position = getattr(
            agv,
            "position",
            None,
        )

        if position is None:
            continue

        if len(position) < 2:
            continue

        x = int(
            position[0]
        )

        y = int(
            position[1]
        )

        # --------------------------------------------------
        # Force movement toward delivery
        # --------------------------------------------------

        if x < gx:

            actions[
                agv_index
            ] = 1

        elif x > gx:

            actions[
                agv_index
            ] = 2

        elif y < gy:

            actions[
                agv_index
            ] = 3

        elif y > gy:

            actions[
                agv_index
            ] = 4

        else:

            actions[
                agv_index
            ] = 0

    return actions


# ==========================================================
# HYBRID EVALUATION CONTROLLER
# ==========================================================

def apply_hybrid_controller(
    env,
    agent,
    dqn_actions,
):
    """
    Final evaluation-only controller.

    The DQN proposes the joint action first.

    The controller checks:

        1. collision safety
        2. movement toward the current task target
        3. pickup target
        4. delivery target
        5. depot return

    If the DQN action is unsuitable, a deterministic
    target-directed safe action is used.

    This controller is NOT used during DQN training.
    """

    actions = np.asarray(
        dqn_actions,
        dtype=np.int64,
    ).reshape(
        NUM_AGVS
    )

    system = getattr(
        env,
        "system",
        None,
    )

    if system is None:
        return actions

    agvs = getattr(
        system,
        "agvs",
        {},
    )

    tasks = getattr(
        system,
        "tasks",
        {},
    )

    if not hasattr(
        agvs,
        "items",
    ):

        return actions

    # ------------------------------------------------------
    # Manhattan distance
    # ------------------------------------------------------

    def manhattan(
        position,
        target,
    ):

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

    # ------------------------------------------------------
    # Determine target for each AGV
    # ------------------------------------------------------

    targets = {}

    for agv_id in range(
        1,
        NUM_AGVS + 1,
    ):

        agv = agvs.get(
            agv_id
        )

        if agv is None:
            continue

        task_id = getattr(
            agv,
            "current_task",
            None,
        )

        target = None

        if (
            task_id is not None
            and
            task_id in tasks
        ):

            task = tasks[
                task_id
            ]

            status = str(
                getattr(
                    task,
                    "status",
                    "",
                )
            ).upper()

            # ----------------------------------------------
            # Going to pickup
            # ----------------------------------------------

            if status == "IN_PROGRESS":

                target = tuple(
                    task.pickup
                )

            # ----------------------------------------------
            # Carrying product to delivery
            # ----------------------------------------------

            elif status == "PICKED_UP":

                target = tuple(
                    task.delivery
                )

            # ----------------------------------------------
            # Completed task -> depot
            # ----------------------------------------------

            elif status == "COMPLETED":

                target = (
                    agv_id - 1,
                    0,
                )

        # --------------------------------------------------
        # No active task -> depot
        # --------------------------------------------------

        if target is None:

            target = (
                agv_id - 1,
                0,
            )

        targets[
            agv_id
        ] = target

    # ------------------------------------------------------
    # Check whether proposed joint action is acceptable
    # ------------------------------------------------------

    def valid_progress_action(
        candidate,
    ):

        candidate = np.asarray(
            candidate,
            dtype=np.int64,
        ).reshape(
            NUM_AGVS
        )

        # First check collision safety.
        if not agent.check_collision_free(
            candidate
        ):

            return False

        any_progress = False

        for agv_id in range(
            1,
            NUM_AGVS + 1,
        ):

            agv = agvs.get(
                agv_id
            )

            if agv is None:
                return False

            current = tuple(
                map(
                    int,
                    agv.position,
                )
            )

            proposed = tuple(
                map(
                    int,
                    env._get_target_position(
                        agv,
                        int(
                            candidate[
                                agv_id - 1
                            ]
                        ),
                    ),
                )
            )

            target = targets[
                agv_id
            ]

            old_distance = manhattan(
                current,
                target,
            )

            new_distance = manhattan(
                proposed,
                target,
            )

            # --------------------------------------------------
            # Already at target
            # --------------------------------------------------

            if old_distance == 0:

                continue

            # --------------------------------------------------
            # Never accept movement away from target
            # --------------------------------------------------

            if new_distance > old_distance:

                return False

            # --------------------------------------------------
            # Positive progress
            # --------------------------------------------------

            if new_distance < old_distance:

                any_progress = True

        return any_progress

    # ------------------------------------------------------
    # First try the DQN action
    # ------------------------------------------------------

    if valid_progress_action(
        actions
    ):

        return actions

    # ======================================================
    # BUILD TARGET-DIRECTED CANDIDATES
    # ======================================================

    per_agv_candidates = {}

    for agv_id in range(
        1,
        NUM_AGVS + 1,
    ):

        agv = agvs.get(
            agv_id
        )

        if agv is None:

            per_agv_candidates[
                agv_id
            ] = [0]

            continue

        x, y = map(
            int,
            agv.position,
        )

        gx, gy = targets[
            agv_id
        ]

        candidates = []

        # --------------------------------------------------
        # Prefer the dimension with larger distance
        # --------------------------------------------------

        dx = abs(
            gx - x
        )

        dy = abs(
            gy - y
        )

        if dx >= dy:

            if gx > x:
                candidates.append(1)

            elif gx < x:
                candidates.append(2)

            if gy > y:
                candidates.append(3)

            elif gy < y:
                candidates.append(4)

        else:

            if gy > y:
                candidates.append(3)

            elif gy < y:
                candidates.append(4)

            if gx > x:
                candidates.append(1)

            elif gx < x:
                candidates.append(2)

        # --------------------------------------------------
        # Try both coordinate directions as backups
        # --------------------------------------------------

        if gx > x and 1 not in candidates:
            candidates.append(1)

        if gx < x and 2 not in candidates:
            candidates.append(2)

        if gy > y and 3 not in candidates:
            candidates.append(3)

        if gy < y and 4 not in candidates:
            candidates.append(4)

        # Stay is always the final fallback.
        candidates.append(0)

        per_agv_candidates[
            agv_id
        ] = candidates

    # ======================================================
    # GENERATE JOINT CANDIDATES
    # ======================================================

    joint_candidates = []

    agv1_candidates = per_agv_candidates.get(
        1,
        [0],
    )

    agv2_candidates = per_agv_candidates.get(
        2,
        [0],
    )

    agv3_candidates = per_agv_candidates.get(
        3,
        [0],
    )

    for a1 in agv1_candidates:

        for a2 in agv2_candidates:

            for a3 in agv3_candidates:

                candidate = np.asarray(
                    [
                        a1,
                        a2,
                        a3,
                    ],
                    dtype=np.int64,
                )

                if not agent.check_collision_free(
                    candidate
                ):

                    continue

                total_before = 0
                total_after = 0
                movement_count = 0

                valid = True

                for agv_id in range(
                    1,
                    NUM_AGVS + 1,
                ):

                    agv = agvs.get(
                        agv_id
                    )

                    if agv is None:

                        valid = False
                        break

                    current = tuple(
                        map(
                            int,
                            agv.position,
                        )
                    )

                    proposed = tuple(
                        map(
                            int,
                            env._get_target_position(
                                agv,
                                int(
                                    candidate[
                                        agv_id - 1
                                    ]
                                ),
                            ),
                        )
                    )

                    target = targets[
                        agv_id
                    ]

                    old_distance = manhattan(
                        current,
                        target,
                    )

                    new_distance = manhattan(
                        proposed,
                        target,
                    )

                    # Active AGV must not move away.
                    if (
                        old_distance > 0
                        and
                        new_distance > old_distance
                    ):

                        valid = False
                        break

                    total_before += (
                        old_distance
                    )

                    total_after += (
                        new_distance
                    )

                    if proposed != current:

                        movement_count += 1

                if not valid:
                    continue

                improvement = (
                    total_before
                    - total_after
                )

                joint_candidates.append(
                    (
                        improvement,
                        movement_count,
                        candidate,
                    )
                )

    # ======================================================
    # SELECT BEST FALLBACK
    # ======================================================

    if joint_candidates:

        joint_candidates.sort(
            key=lambda item: (
                item[0],
                item[1],
            ),
            reverse=True,
        )

        return joint_candidates[0][2]

    # ------------------------------------------------------
    # Absolute final fallback
    # ------------------------------------------------------

    safe_indices = []

    try:

        safe_indices = (
            agent.get_safe_joint_action_indices()
        )

    except Exception:

        safe_indices = []

    if safe_indices:

        return agent.decode_joint_action(
            safe_indices[0]
        )

    return actions

def save_evaluation_result(
    seed,
    episodes,
    evaluation_result,
    filepath="models/final_evaluation_results.csv",
):
    """
    Append one evaluation run to a CSV results file.
    """

    directory = os.path.dirname(
        filepath
    )

    if directory:
        os.makedirs(
            directory,
            exist_ok=True,
        )

    file_exists = os.path.exists(
        filepath
    )

    row = {
        "timestamp": datetime.now().isoformat(
            timespec="seconds"
        ),
        "seed": int(seed),
        "episodes": int(episodes),
        "average_reward": float(
            evaluation_result["average_reward"]
        ),
        "average_steps": float(
            evaluation_result["average_steps"]
        ),
        "average_completions": float(
            evaluation_result["average_completion"]
        ),
        "average_collisions": float(
            evaluation_result["average_collisions"]
        ),
        "completion_rate": float(
            evaluation_result["completion_rate"]
        ),
    }

    with open(
        filepath,
        "a",
        newline="",
        encoding="utf-8",
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=row.keys(),
        )

        if not file_exists:
            writer.writeheader()

        writer.writerow(
            row
        )


# ==========================================================
# MAIN
# ==========================================================

def main():

    args = parse_args()

    # ------------------------------------------------------
    # Episode selection
    # ------------------------------------------------------

    selected_episodes = (
        int(args.episodes)
        if args.episodes is not None
        else EPISODES
    )
    
    evaluation_seeds = (
        args.seeds
        if args.seeds is not None
        else [args.seed]
    )

    if selected_episodes <= 0:

        raise ValueError(
            "--episodes must be positive."
        )

    # ======================================================
    # VALIDATE MODE
    # ======================================================

    if args.train and args.eval:

        raise ValueError(
            "Use only one mode: --train OR --eval"
        )

    if not args.train and not args.eval:

        print()
        print("Usage:")
        print(
            "  python train_dqn.py --train"
        )
        print(
            "  python train_dqn.py --train --episodes 100"
        )
        print(
            "  python train_dqn.py --eval"
        )
        print(
            "  python train_dqn.py --eval --episodes 100"
        )
        print(
            "  python train_dqn.py --eval --episodes 100 --no-recovery"
        )
        print()

        return

    # ======================================================
    # CREATE ENVIRONMENT
    # ======================================================

    env = MultiAGVEnv(
        num_agvs=NUM_AGVS,
        max_steps=MAX_STEPS,
        risk_aware_assignment=True,
    )

    try:

        # ==================================================
        # INITIAL RESET
        # ==================================================

        state = reset_environment(
            env,
            seed=42,
        )

        # ==================================================
        # STATE SIZE
        # ==================================================

        if not hasattr(
            env.observation_space,
            "shape",
        ):

            raise RuntimeError(
                "Environment observation_space "
                "does not provide a shape."
            )

        if (
            env.observation_space.shape is None
            or len(
                env.observation_space.shape
            ) != 1
        ):

            raise RuntimeError(
                "Environment observation_space "
                "must be one-dimensional."
            )

        STATE_SIZE = int(
            env.observation_space.shape[0]
        )

        print(
            "State size :",
            STATE_SIZE,
        )

        expected_state_size = (
            NUM_AGVS * 8
        )

        if STATE_SIZE != expected_state_size:

            raise ValueError(
                "Unexpected environment state size.\n"
                f"Expected: {expected_state_size}\n"
                f"Actual:   {STATE_SIZE}"
            )

        # ==================================================
        # ACTION SPACE
        # ==================================================

        if not hasattr(
            env.action_space,
            "nvec",
        ):

            raise RuntimeError(
                "Environment action_space must "
                "be MultiDiscrete."
            )

        if len(
            env.action_space.nvec
        ) != NUM_AGVS:

            raise ValueError(
                "Environment action space does not "
                "match the configured number of AGVs."
            )

        ACTION_SIZE_FROM_ENV = int(
            env.action_space.nvec[0]
        )

        if ACTION_SIZE_FROM_ENV != ACTION_SIZE:

            raise ValueError(
                "Environment action size does not "
                "match training configuration."
            )

        for agv_index, value in enumerate(
            env.action_space.nvec
        ):

            if int(value) != ACTION_SIZE:

                raise ValueError(
                    f"AGV {agv_index} has action size "
                    f"{int(value)}, expected "
                    f"{ACTION_SIZE}."
                )

        print(
            "Action size:",
            ACTION_SIZE_FROM_ENV,
        )

        # ==================================================
        # CREATE DQN AGENT
        # ==================================================

        agent = DQNAgent(
            state_size=STATE_SIZE,
            num_agvs=NUM_AGVS,
            action_size=ACTION_SIZE,
            learning_rate=LEARNING_RATE,
            gamma=GAMMA,
            tau=TAU,
            epsilon=EPSILON_START,
            epsilon_min=EPSILON_MIN,
            epsilon_decay=EPSILON_DECAY,
            replay_capacity=REPLAY_CAPACITY,
            batch_size=BATCH_SIZE,
        )

        # --------------------------------------------------
        # Environment reference used by safety filtering
        # --------------------------------------------------

        agent._evaluation_env = env

        # ==================================================
        # TRAINING HISTORY
        # ==================================================

        rewards_history = []

        steps_history = []

        completion_history = []

        pickup_history = []

        collision_history = []

        loss_history = []

        # ==================================================
        # EVALUATION
        # ==================================================

        if args.eval:
    
            evaluation_seed = int(
                args.seed
            )

            evaluation_model_path = os.path.join(
                MODEL_DIR,
                "dqn_model.pt",
            )

            if not os.path.exists(
                evaluation_model_path
            ):

                raise FileNotFoundError(
                    "Trained DQN model not found:\n"
                    f"{evaluation_model_path}\n\n"
                    "Run training first:\n"
                    "python train_dqn.py --train"
                )

            print()
            print("=" * 72)
            print(
                "MULTI-AGV DQN HYBRID EVALUATION"
            )
            print("=" * 72)

            print(
                f"Model         : "
                f"{evaluation_model_path}"
            )

            print(
                f"Episodes      : "
                f"{selected_episodes}"
            )

            print(
                f"Max steps     : "
                f"{MAX_STEPS}"
            )

            print(
                f"AGVs          : "
                f"{NUM_AGVS}"
            )

            print(
                f"State size    : "
                f"{STATE_SIZE}"
            )

            print(
                f"Action size   : "
                f"{ACTION_SIZE}"
            )

            print(
                f"Joint actions : "
                f"{agent.joint_action_size}"
            )

            print("=" * 72)

            # ==================================================
            # LOAD MODEL
            # ==================================================

            agent.load(
                evaluation_model_path
            )

            print(
                "MODEL LOADED SUCCESSFULLY"
            )

            # ==================================================
            # EVALUATION MODE
            # ==================================================

            agent.evaluation_mode()

            eval_rewards = []
            eval_steps = []
            eval_pickups = []
            eval_completions = []
            eval_collisions = []

            # ==================================================
            # EVALUATION EPISODES
            # ==================================================

            for evaluation_episode in range(
                1,
                selected_episodes + 1,
            ):

                state = reset_environment(
                    env,
                    seed=(
                        evaluation_seed
                        + 10_000
                        + evaluation_episode
                    ),
                )

                episode_reward = 0.0
                episode_steps = 0
                episode_pickups = 0
                episode_completions = 0
                episode_collisions = 0

                picked_task_ids = set()
                completed_task_ids = set()

                # ==============================================
                # STEP LOOP
                # ==============================================

                for step in range(
                    MAX_STEPS
                ):

                    episode_steps = (
                        step + 1
                    )

                    actions = agent.select_action(
                        state,
                        training=False,
                    )

                    actions = np.asarray(
                        actions,
                        dtype=np.int64,
                    ).reshape(
                        NUM_AGVS
                    )

                    # ------------------------------------------------
                    # Optional delivery recovery controller
                    # ------------------------------------------------

                    if not args.no_recovery:

                        actions = apply_hybrid_controller(
                            env,
                            agent,
                            actions,
                        )

                    # ------------------------------------------------
                    # Validate action
                    # ------------------------------------------------

                    if actions.shape != (
                        NUM_AGVS,
                    ):

                        raise RuntimeError(
                            f"Invalid DQN action shape: "
                            f"{actions.shape}"
                        )

                    if (
                        np.any(actions < 0)
                        or
                        np.any(actions >= ACTION_SIZE)
                    ):

                        raise RuntimeError(
                            f"Invalid DQN action values: "
                            f"{actions}"
                        )

                    # ------------------------------------------------
                    # Environment step
                    # ------------------------------------------------

                    (
                        next_state,
                        reward,
                        terminated,
                        truncated,
                        info,
                    ) = env.step(
                        actions
                    )

                    episode_reward += float(
                        reward
                    )

                    if isinstance(
                        info,
                        dict,
                    ):

                        episode_collisions += int(
                            info.get(
                                "collision_count",
                                0,
                            )
                        )

                    # ------------------------------------------------
                    # Track task lifecycle
                    # ------------------------------------------------

                    for task in get_task_iterable(
                        env
                    ):

                        task_id = getattr(
                            task,
                            "task_id",
                            None,
                        )

                        if task_id is None:
                            continue

                        status = str(
                            getattr(
                                task,
                                "status",
                                "",
                            )
                        ).upper()

                        picked = bool(
                            getattr(
                                task,
                                "is_picked_up",
                                False,
                            )
                        )

                        completed = bool(
                            getattr(
                                task,
                                "is_completed",
                                False,
                            )
                        )

                        if (
                            picked
                            or
                            status == "PICKED_UP"
                            or
                            completed
                            or
                            status == "COMPLETED"
                        ):

                            picked_task_ids.add(
                                task_id
                            )

                        if (
                            completed
                            or
                            status == "COMPLETED"
                        ):

                            completed_task_ids.add(
                                task_id
                            )

                    episode_pickups = len(
                        picked_task_ids
                    )

                    episode_completions = len(
                        completed_task_ids
                    )

                    state = np.asarray(
                        next_state,
                        dtype=np.float32,
                    )

                    if state.shape != (
                        STATE_SIZE,
                    ):

                        raise ValueError(
                            f"Expected next state shape "
                            f"({STATE_SIZE},), "
                            f"got {state.shape}."
                        )

                    if (
                        terminated
                        or
                        truncated
                    ):

                        break

                eval_rewards.append(
                    episode_reward
                )

                eval_steps.append(
                    episode_steps
                )

                eval_pickups.append(
                    episode_pickups
                )

                eval_completions.append(
                    episode_completions
                )

                eval_collisions.append(
                    episode_collisions
                )

            # ==================================================
            # EVALUATION SUMMARY
            # ==================================================

            print()
            print("=" * 72)
            print(
                "FINAL EVALUATION SUMMARY"
            )
            print("=" * 72)

            if eval_rewards:

                print(
                    f"Episodes evaluated : "
                    f"{len(eval_rewards)}"
                )

                print(
                    f"Best reward        : "
                    f"{max(eval_rewards):.2f}"
                )

                print(
                    f"Average reward     : "
                    f"{np.mean(eval_rewards):.2f}"
                )

                print(
                    f"Average steps      : "
                    f"{np.mean(eval_steps):.2f}"
                )

                print(
                    f"Total pickups      : "
                    f"{sum(eval_pickups)}"
                )

                print(
                    f"Total completions  : "
                    f"{sum(eval_completions)}"
                )

                print(
                    f"Average completions: "
                    f"{np.mean(eval_completions):.2f}"
                )

                print(
                    f"Total collisions   : "
                    f"{sum(eval_collisions)}"
                )

                print(
                    f"Average collisions : "
                    f"{np.mean(eval_collisions):.2f}"
                )

                successful_episodes = sum(
                    1
                    for completions
                    in eval_completions
                    if completions >= NUM_AGVS
                )

                completion_rate = (
                    successful_episodes
                    /
                    len(eval_completions)
                    *
                    100.0
                )

                print(
                    f"Completion rate    : "
                    f"{completion_rate:.2f}%"
                )

                evaluation_result = {
                    "average_reward": float(
                        np.mean(eval_rewards)
                    ),
                    "average_steps": float(
                        np.mean(eval_steps)
                    ),
                    "average_completion": float(
                        np.mean(eval_completions)
                    ),
                    "average_collisions": float(
                        np.mean(eval_collisions)
                    ),
                    "completion_rate": float(
                        completion_rate
                    ),
                }

                save_evaluation_result(
                    seed=evaluation_seed,
                    episodes=selected_episodes,
                    evaluation_result=evaluation_result,
                )

                print(
                    "RESULTS SAVED: "
                    "models/final_evaluation_results.csv"
                )

            print(
                f"Evaluation epsilon : "
                f"{agent.epsilon:.4f}"
            )

            print("=" * 72)

            return

        # ======================================================
        # TRAINING HEADER
        # ======================================================

        print()
        print("=" * 72)
        print(
            "MULTI-AGV DQN TRAINING"
        )
        print("=" * 72)

        print(
            f"Episodes      : "
            f"{selected_episodes}"
        )

        print(
            f"Max steps     : "
            f"{MAX_STEPS}"
        )

        print(
            f"AGVs          : "
            f"{NUM_AGVS}"
        )

        print(
            f"State size    : "
            f"{STATE_SIZE}"
        )

        print(
            f"Action size   : "
            f"{ACTION_SIZE}"
        )

        print(
            f"Joint actions : "
            f"{agent.joint_action_size}"
        )

        print(
            f"Device        : "
            f"{agent.device}"
        )

        print(
            f"Learning rate : "
            f"{LEARNING_RATE}"
        )

        print(
            f"Gamma         : "
            f"{GAMMA}"
        )

        print(
            f"Tau           : "
            f"{TAU}"
        )

        print(
            f"Replay        : "
            f"{REPLAY_CAPACITY}"
        )

        print(
            f"Batch         : "
            f"{BATCH_SIZE}"
        )

        print(
            f"Epsilon start : "
            f"{EPSILON_START}"
        )

        print(
            f"Epsilon min   : "
            f"{EPSILON_MIN}"
        )

        print(
            f"Epsilon decay : "
            f"{EPSILON_DECAY}"
        )

        print(
            f"Checkpoint    : every "
            f"{CHECKPOINT_INTERVAL} episodes"
        )

        print("=" * 72)

        # ======================================================
        # TRAINING LOOP
        # ======================================================

        for episode in range(
            1,
            selected_episodes + 1,
        ):

            state = reset_environment(
                env,
                seed=(
                    42
                    + episode
                ),
            )

            if state.shape != (
                STATE_SIZE,
            ):

                raise ValueError(
                    f"Expected state shape "
                    f"({STATE_SIZE},), "
                    f"got {state.shape}."
                )

            episode_reward = 0.0

            episode_losses = []

            collision_count = 0

            steps = 0

            picked_task_ids = set()

            completed_task_ids = set()

            # ==================================================
            # STEP LOOP
            # ==================================================

            for step in range(
                MAX_STEPS
            ):

                steps = step + 1

                # ----------------------------------------------
                # DQN training action
                #
                # The collision-safe action selection is handled
                # by DQNAgent.select_action().
                # ----------------------------------------------

                actions = agent.select_action(
                    state,
                    training=True,
                )

                actions = np.asarray(
                    actions,
                    dtype=np.int64,
                ).reshape(
                    NUM_AGVS
                )

                if actions.shape != (
                    NUM_AGVS,
                ):

                    raise RuntimeError(
                        f"Invalid DQN action shape: "
                        f"{actions.shape}"
                    )

                if (
                    np.any(actions < 0)
                    or
                    np.any(actions >= ACTION_SIZE)
                ):

                    raise RuntimeError(
                        f"Invalid DQN action values: "
                        f"{actions}"
                    )

                # ----------------------------------------------
                # Environment
                # ----------------------------------------------

                (
                    next_state,
                    reward,
                    terminated,
                    truncated,
                    info,
                ) = env.step(
                    actions
                )

                done = (
                    bool(terminated)
                    or
                    bool(truncated)
                )

                next_state = np.asarray(
                    next_state,
                    dtype=np.float32,
                )

                if next_state.shape != (
                    STATE_SIZE,
                ):

                    raise ValueError(
                        f"Expected next state shape "
                        f"({STATE_SIZE},), "
                        f"got {next_state.shape}."
                    )

                # ----------------------------------------------
                # Replay
                # ----------------------------------------------

                agent.remember(
                    state,
                    actions,
                    reward,
                    next_state,
                    done,
                )

                # ----------------------------------------------
                # Train
                # ----------------------------------------------

                loss = agent.train_step()

                if loss is not None:

                    loss_value = float(
                        loss
                    )

                    if np.isfinite(
                        loss_value
                    ):

                        episode_losses.append(
                            loss_value
                        )

                episode_reward += float(
                    reward
                )

                # ----------------------------------------------
                # Collision statistics
                # ----------------------------------------------

                if isinstance(
                    info,
                    dict,
                ):

                    collision_count += int(
                        info.get(
                            "collision_count",
                            0,
                        )
                    )

                # ----------------------------------------------
                # Task statistics
                # ----------------------------------------------

                for task in get_task_iterable(
                    env
                ):

                    task_id = getattr(
                        task,
                        "task_id",
                        None,
                    )

                    if task_id is None:
                        continue

                    status = str(
                        getattr(
                            task,
                            "status",
                            "",
                        )
                    ).upper()

                    picked = bool(
                        getattr(
                            task,
                            "is_picked_up",
                            False,
                        )
                    )

                    completed = bool(
                        getattr(
                            task,
                            "is_completed",
                            False,
                        )
                    )

                    if (
                        picked
                        or
                        status == "PICKED_UP"
                        or
                        completed
                        or
                        status == "COMPLETED"
                    ):

                        picked_task_ids.add(
                            task_id
                        )

                    if (
                        completed
                        or
                        status == "COMPLETED"
                    ):

                        completed_task_ids.add(
                            task_id
                        )

                state = next_state

                if done:
                    break

            # ==================================================
            # EPISODE METRICS
            # ==================================================

            pickup_count = len(
                picked_task_ids
            )

            completed_count = len(
                completed_task_ids
            )

            agent.decay_epsilon()

            average_loss = (
                float(
                    np.mean(
                        episode_losses
                    )
                )
                if episode_losses
                else 0.0
            )

            rewards_history.append(
                episode_reward
            )

            steps_history.append(
                steps
            )

            completion_history.append(
                completed_count
            )

            pickup_history.append(
                pickup_count
            )

            collision_history.append(
                collision_count
            )

            loss_history.append(
                average_loss
            )

            # ==================================================
            # PROGRESS OUTPUT
            # ==================================================

            if (
                episode == 1
                or
                episode % 10 == 0
                or
                episode == selected_episodes
            ):

                print(
                    f"Episode "
                    f"{episode:04d}/"
                    f"{selected_episodes:04d} "
                    f"| Reward={episode_reward:.2f} "
                    f"| Steps={steps} "
                    f"| Pickups={pickup_count} "
                    f"| Completed={completed_count} "
                    f"| Collision={collision_count} "
                    f"| Loss={average_loss:.5f} "
                    f"| Epsilon={agent.epsilon:.4f} "
                    f"| Replay={len(agent.replay_buffer)}"
                )

            # ==================================================
            # CHECKPOINT
            # ==================================================

            if (
                CHECKPOINT_INTERVAL > 0
                and
                (
                    episode
                    %
                    CHECKPOINT_INTERVAL
                    == 0
                    or
                    episode == selected_episodes
                )
            ):

                checkpoint_path = os.path.join(
                    CHECKPOINT_DIR,
                    (
                        "dqn_checkpoint_ep_"
                        f"{episode:05d}.pt"
                    ),
                )

                agent.save(
                    checkpoint_path
                )

                print(
                    f"CHECKPOINT SAVED: "
                    f"{checkpoint_path}"
                )

        # ======================================================
        # FINAL MODEL
        # ======================================================

        final_model_path = os.path.join(
            MODEL_DIR,
            "dqn_model.pt",
        )

        agent.save(
            final_model_path
        )

        # ======================================================
        # FINAL TRAINING SUMMARY
        # ======================================================

        print()
        print("=" * 72)
        print(
            "FINAL TRAINING SUMMARY"
        )
        print("=" * 72)

        if rewards_history:

            print(
                f"Episodes completed : "
                f"{len(rewards_history)}"
            )

            print(
                f"Best reward        : "
                f"{max(rewards_history):.2f}"
            )

            print(
                f"Average reward     : "
                f"{np.mean(rewards_history):.2f}"
            )

            print(
                f"Average steps      : "
                f"{np.mean(steps_history):.2f}"
            )

            print(
                f"Total pickups      : "
                f"{sum(pickup_history)}"
            )

            print(
                f"Total completions  : "
                f"{sum(completion_history)}"
            )

            print(
                f"Total collisions   : "
                f"{sum(collision_history)}"
            )

            print(
                f"Average loss       : "
                f"{np.mean(loss_history):.5f}"
            )

        print(
            f"Final epsilon      : "
            f"{agent.epsilon:.4f}"
        )

        print()

        print(
            f"MODEL SAVED: "
            f"{final_model_path}"
        )

        print("=" * 72)

    finally:

        env.close()


# ==========================================================
# ENTRY POINT
# ==========================================================

if __name__ == "__main__":
    main()