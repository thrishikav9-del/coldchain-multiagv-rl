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
"""

from __future__ import annotations

import argparse
import os

import numpy as np

from algorithms.dqn_agent import DQNAgent
from environment.multi_agv_env import MultiAGVEnv


# ==========================================================
# ARGUMENTS
# ==========================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description="Multi-AGV DQN training and evaluation"
    )

    parser.add_argument(
        "--train",
        action="store_true",
        help="Train the DQN",
    )

    parser.add_argument(
        "--eval",
        action="store_true",
        help="Evaluate the saved DQN model",
    )

    parser.add_argument(
        "--episodes",
        type=int,
        default=100,
        help="Number of episodes for training or evaluation",
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

EPSILON_DECAY = 0.999

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

def reset_environment(env, seed=None):
    """
    Reset the Gymnasium environment and return only
    the observation.
    """

    if seed is None:
        result = env.reset()
    else:
        result = env.reset(seed=seed)

    if isinstance(result, tuple):
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

def get_task_iterable(env):
    """
    Supports both dictionary and list task containers.
    """

    tasks = getattr(
        env,
        "tasks",
        [],
    )

    if hasattr(tasks, "values"):
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

    The DQN chooses the normal joint action first.

    For AGVs that have already picked up their task,
    the controller forces the AGV toward the delivery
    location.

    This controller is NOT used during training.

    Action mapping:

        0 = Stay
        1 = Right
        2 = Left
        3 = Up
        4 = Down
    """

    actions = np.asarray(
        actions,
        dtype=np.int64,
    ).reshape(NUM_AGVS)

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
            agv_index = int(agv_id) - 1
        except (
            TypeError,
            ValueError,
        ):
            continue

        if (
            agv_index < 0
            or agv_index >= NUM_AGVS
        ):
            continue

        # --------------------------------------------------
        # CURRENT TASK
        # --------------------------------------------------

        task_id = getattr(
            agv,
            "current_task",
            None,
        )

        if task_id is None:
            continue

        # --------------------------------------------------
        # TASK EXISTS
        # --------------------------------------------------

        if task_id not in tasks:
            continue

        task = tasks[task_id]

        # --------------------------------------------------
        # ONLY AFTER PICKUP
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
        # DELIVERY LOCATION
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

        gx = delivery[0]
        gy = delivery[1]

        # --------------------------------------------------
        # CURRENT POSITION
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

        x = position[0]
        y = position[1]

        # --------------------------------------------------
        # MOVE TOWARD DELIVERY
        # --------------------------------------------------

        if x < gx:

            actions[agv_index] = 1

        elif x > gx:

            actions[agv_index] = 2

        elif y < gy:

            actions[agv_index] = 3

        elif y > gy:

            actions[agv_index] = 4

        else:

            actions[agv_index] = 0

    return actions


# ==========================================================
# MAIN
# ==========================================================

def main():

    args = parse_args()
    
    training_episodes = (
        EPISODES
        if args.episodes is None
        else args.episodes
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
            "  python train_dqn.py --eval"
        )
        print(
            "  python train_dqn.py --eval --episodes 100"
        )
        print()

        return

    # ======================================================
    # CREATE ENVIRONMENT
    # ======================================================

    env = MultiAGVEnv(
        num_agvs=NUM_AGVS,
        max_steps=MAX_STEPS,
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
            or len(env.observation_space.shape) != 1
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

        # ==================================================
        # EXPECTED STATE SIZE
        # ==================================================

        expected_state_size = (
            NUM_AGVS * 6
        )

        if STATE_SIZE != expected_state_size:

            raise ValueError(
                "Unexpected environment state size.\n"
                f"Expected: {expected_state_size}\n"
                f"Actual:   {STATE_SIZE}\n\n"
                "The environment should provide "
                "6 features per AGV."
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
                "be a MultiDiscrete action space "
                "with nvec."
            )

        if len(
            env.action_space.nvec
        ) != NUM_AGVS:

            raise ValueError(
                "Environment action space does not "
                "match the configured number of AGVs.\n"
                f"Expected AGVs: {NUM_AGVS}\n"
                f"Environment nvec: "
                f"{env.action_space.nvec}"
            )

        ACTION_SIZE_FROM_ENV = int(
            env.action_space.nvec[0]
        )

        if ACTION_SIZE_FROM_ENV != ACTION_SIZE:

            raise ValueError(
                "Environment action size does not "
                "match training configuration.\n"
                f"Environment={ACTION_SIZE_FROM_ENV}\n"
                f"Configuration={ACTION_SIZE}"
            )

        # ==================================================
        # VALIDATE EACH AGV ACTION SPACE
        # ==================================================

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

        # ==================================================
        # GIVE ENVIRONMENT TO AGENT
        # ==================================================
        #
        # Required by the valid-joint-action logic
        # implemented in DQNAgent.
        #
        # ==================================================

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
            print("MULTI-AGV DQN EVALUATION")
            print("=" * 72)

            print(
                f"Model         : "
                f"{evaluation_model_path}"
            )

            print(
                f"Episodes      : "
                f"{args.episodes}"
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

            print(
                "Network parameters:",
                sum(
                    p.numel()
                    for p
                    in agent.policy_network.parameters()
                ),
            )

            # ==================================================
            # GREEDY EVALUATION
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
                args.episodes + 1,
            ):

                state = reset_environment(
                    env
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

                episode_pickups = 0

                episode_completions = 0

                episode_collisions = 0

                episode_steps = 0

                picked_task_ids = set()

                completed_task_ids = set()

                # ==================================================
                # EVALUATION STEP LOOP
                # ==================================================

                for step in range(
                    MAX_STEPS
                ):

                    episode_steps = step + 1

                    # ==============================================
                    # PURE DQN ACTION
                    # ==============================================
                    #
                    # The trained DQN chooses the joint action.
                    #
                    # ==============================================

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

                    # ==============================================
                    # DELIVERY RECOVERY CONTROLLER
                    # ==============================================
                    #
                    # Evaluation only.
                    #
                    # After pickup, force the AGV toward delivery.
                    #
                    # ==============================================

                    actions = apply_delivery_recovery_controller(
                        env,
                        actions,
                    )

                    # ==============================================
                    # FINAL ACTION VALIDATION
                    # ==============================================

                    if actions.shape != (
                        NUM_AGVS,
                    ):

                        raise RuntimeError(
                            f"Invalid DQN action shape: "
                            f"{actions.shape}"
                        )

                    if (
                        np.any(actions < 0)
                        or np.any(actions >= ACTION_SIZE)
                    ):

                        raise RuntimeError(
                            f"Invalid DQN action values: "
                            f"{actions}"
                        )

                    # ==============================================
                    # ENVIRONMENT STEP
                    # ==============================================

                    result = env.step(
                        actions
                    )

                    if (
                        not isinstance(
                            result,
                            tuple,
                        )
                        or len(result) != 5
                    ):

                        raise RuntimeError(
                            "Environment step() must return "
                            "(observation, reward, terminated, "
                            "truncated, info)."
                        )

                    (
                        next_state,
                        reward,
                        terminated,
                        truncated,
                        info,
                    ) = result

                    done = (
                        bool(terminated)
                        or bool(truncated)
                    )

                    # ==============================================
                    # REWARD
                    # ==============================================

                    episode_reward += float(
                        reward
                    )

                    # ==============================================
                    # COLLISIONS
                    # ==============================================

                    if isinstance(
                        info,
                        dict,
                    ):

                        episode_collisions += int(
                            info.get(
                                "collision_count",
                                info.get(
                                    "collisions",
                                    0,
                                ),
                            )
                        )

                    # ==============================================
                    # TASK LIFECYCLE
                    # ==============================================

                    for task in get_task_iterable(env):

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

                        completed = bool(
                            getattr(
                                task,
                                "is_completed",
                                False,
                            )
                        )

                        picked = bool(
                            getattr(
                                task,
                                "is_picked_up",
                                False,
                            )
                        )

                        if (
                            picked
                            or status == "PICKED_UP"
                            or completed
                            or status == "COMPLETED"
                        ):

                            picked_task_ids.add(
                                task_id
                            )

                        if (
                            completed
                            or status == "COMPLETED"
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

                    # ==============================================
                    # NEXT STATE
                    # ==============================================

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

                    # ==============================================
                    # TERMINATION
                    # ==============================================

                    if done:
                        break

                # ==================================================
                # STORE EVALUATION RESULTS
                # ==================================================

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
                # EPISODE OUTPUT
                # ==================================================

                print(
                    f"Evaluation "
                    f"{evaluation_episode:04d}/"
                    f"{args.episodes:04d} "
                    f"| Reward={episode_reward:.2f} "
                    f"| Steps={episode_steps} "
                    f"| Pickups={episode_pickups} "
                    f"| Completed={episode_completions} "
                    f"| Collision={episode_collisions}"
                )

            # ======================================================
            # EVALUATION SUMMARY
            # ======================================================

            print()
            print("=" * 72)
            print("FINAL EVALUATION SUMMARY")
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
                    if completions >= len(env.tasks)
                )

                completion_rate = (
                    successful_episodes
                    / len(eval_completions)
                    * 100.0
                )

                print(
                    f"Completion rate    : "
                    f"{completion_rate:.2f}%"
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
        print("MULTI-AGV DQN TRAINING")
        print("=" * 72)

        print(
            f"Episodes      : {training_episodes}"
        )

        print(
            f"Max steps     : {MAX_STEPS}"
        )

        print(
            f"AGVs          : {NUM_AGVS}"
        )

        print(
            f"State size    : {STATE_SIZE}"
        )

        print(
            f"Action size   : {ACTION_SIZE}"
        )

        print(
            f"Joint actions : "
            f"{agent.joint_action_size}"
        )

        print(
            f"Device        : {agent.device}"
        )

        print(
            f"Learning rate : {LEARNING_RATE}"
        )

        print(
            f"Gamma         : {GAMMA}"
        )

        print(
            f"Tau           : {TAU}"
        )

        print(
            f"Replay        : {REPLAY_CAPACITY}"
        )

        print(
            f"Batch         : {BATCH_SIZE}"
        )

        print(
            f"Epsilon start : {EPSILON_START}"
        )

        print(
            f"Epsilon min   : {EPSILON_MIN}"
        )

        print(
            f"Epsilon decay : {EPSILON_DECAY}"
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
            training_episodes + 1,
        ):

            # ==================================================
            # RESET
            # ==================================================

            state = reset_environment(
                env
            )

            if state.shape != (
                STATE_SIZE,
            ):

                raise ValueError(
                    f"Expected state shape "
                    f"({STATE_SIZE},), "
                    f"got {state.shape}."
                )

            # ==================================================
            # EPISODE VARIABLES
            # ==================================================

            episode_reward = 0.0

            episode_losses = []

            collision_count = 0

            completed_count = 0

            pickup_count = 0

            steps = 0

            picked_task_ids = set()

            completed_task_ids = set()

            # ==================================================
            # TRAINING STEP LOOP
            # ==================================================

            for step in range(
                MAX_STEPS
            ):

                steps = step + 1

                # ==============================================
                # DQN TRAINING ACTION
                # ==============================================
                #
                # training=True enables epsilon exploration.
                #
                # IMPORTANT:
                #
                # NO delivery recovery controller is used here.
                #
                # ==============================================

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
                    or np.any(actions >= ACTION_SIZE)
                ):

                    raise RuntimeError(
                        f"Invalid DQN action values: "
                        f"{actions}"
                    )

                # ==============================================
                # ENVIRONMENT STEP
                # ==============================================

                result = env.step(
                    actions
                )

                if (
                    not isinstance(
                        result,
                        tuple,
                    )
                    or len(result) != 5
                ):

                    raise RuntimeError(
                        "Environment step() must return "
                        "(observation, reward, terminated, "
                        "truncated, info)."
                    )

                (
                    next_state,
                    reward,
                    terminated,
                    truncated,
                    info,
                ) = result

                done = (
                    bool(terminated)
                    or bool(truncated)
                )

                # ==============================================
                # NEXT STATE
                # ==============================================

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

                # ==============================================
                # ONE JOINT REPLAY TRANSITION
                # ==============================================

                agent.remember(
                    state,
                    actions,
                    reward,
                    next_state,
                    done,
                )

                # ==============================================
                # TRAIN DQN
                # ==============================================

                loss = agent.train_step()

                if loss is not None:

                    episode_losses.append(
                        float(loss)
                    )

                # ==============================================
                # REWARD
                # ==============================================

                episode_reward += float(
                    reward
                )

                # ==============================================
                # INFO / COLLISIONS
                # ==============================================

                if isinstance(
                    info,
                    dict,
                ):

                    collision_count += int(
                        info.get(
                            "collision_count",
                            info.get(
                                "collisions",
                                0,
                            ),
                        )
                    )

                # ==============================================
                # TASK LIFECYCLE
                # ==============================================

                for task in get_task_iterable(env):

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

                    completed = bool(
                        getattr(
                            task,
                            "is_completed",
                            False,
                        )
                    )

                    picked = bool(
                        getattr(
                            task,
                            "is_picked_up",
                            False,
                        )
                    )

                    if (
                        picked
                        or status == "PICKED_UP"
                        or completed
                        or status == "COMPLETED"
                    ):

                        picked_task_ids.add(
                            task_id
                        )

                    if (
                        completed
                        or status == "COMPLETED"
                    ):

                        completed_task_ids.add(
                            task_id
                        )

                pickup_count = len(
                    picked_task_ids
                )

                completed_count = len(
                    completed_task_ids
                )

                # ==============================================
                # UPDATE STATE
                # ==============================================

                state = next_state

                # ==============================================
                # TERMINATION
                # ==============================================

                if done:
                    break

            # ==================================================
            # EPSILON DECAY
            # ==================================================

            agent.decay_epsilon()

            # ==================================================
            # AVERAGE LOSS
            # ==================================================

            if episode_losses:

                average_loss = float(
                    np.mean(
                        episode_losses
                    )
                )

            else:

                average_loss = 0.0

            # ==================================================
            # HISTORY
            # ==================================================

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
            # PROGRESS
            # ==================================================

            if (
                episode == 1
                or episode % 10 == 0
            ):

                print(
                    f"Episode "
                    f"{episode:04d}/"
                    f"{EPISODES:04d} "
                    f"| Reward={episode_reward:.2f} "
                    f"| Steps={steps} "
                    f"| Pickups={pickup_count} "
                    f"| Completed={completed_count} "
                    f"| Collision={collision_count} "
                    f"| Loss={average_loss:.5f} "
                    f"| Epsilon={agent.epsilon:.4f}"
                )

            # ==================================================
            # CHECKPOINT
            # ==================================================

            if (
                CHECKPOINT_INTERVAL > 0
                and episode % CHECKPOINT_INTERVAL == 0
            ):

                checkpoint_path = os.path.join(
                    CHECKPOINT_DIR,
                    f"dqn_checkpoint_ep_{episode:05d}.pt",
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
        # FINAL SUMMARY
        # ======================================================

        print()
        print("=" * 72)
        print("FINAL TRAINING SUMMARY")
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

        # ======================================================
        # CLEANUP
        # ======================================================

        env.close()


# ==========================================================
# ENTRY POINT
# ==========================================================

if __name__ == "__main__":
    main()