"""
Multi-AGV DQN training.

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

One environment step produces one JOINT transition:

    state
        -> joint actions
        -> reward
        -> next_state
        -> done
"""

from __future__ import annotations

import os

import numpy as np

from algorithms.dqn_agent import DQNAgent
from environment.multi_agv_env import MultiAGVEnv


# ==========================================================
# CONFIGURATION
# ==========================================================

EPISODES = 5
MAX_STEPS = 200

NUM_AGVS = 3
ACTION_SIZE = 5

LEARNING_RATE = 0.0001
GAMMA = 0.99
TAU = 0.001

REPLAY_CAPACITY = 200000
BATCH_SIZE = 64

EPSILON_START = 1.0
EPSILON_MIN = 0.05
EPSILON_DECAY = 0.995

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
# ENVIRONMENT
# ==========================================================

env = MultiAGVEnv(
    num_agvs=NUM_AGVS,
    max_steps=MAX_STEPS,
)


# ==========================================================
# INITIAL RESET
# ==========================================================

state, info = env.reset(
    seed=42
)


state = np.asarray(
    state,
    dtype=np.float32,
)


# ==========================================================
# STATE SIZE
# ==========================================================
#
# IMPORTANT:
#
# Do NOT hard-code 18.
#
# The environment now defines the state size.
#
# 3 AGVs * 6 features = 18
#
# ==========================================================

STATE_SIZE = int(
    env.observation_space.shape[0]
)


# ==========================================================
# ACTION SIZE
# ==========================================================

ACTION_SIZE_FROM_ENV = int(
    env.action_space.nvec[0]
)


if ACTION_SIZE_FROM_ENV != ACTION_SIZE:

    raise ValueError(
        "Environment action size does not match "
        f"training configuration. "
        f"Environment={ACTION_SIZE_FROM_ENV}, "
        f"Configuration={ACTION_SIZE}."
    )


print(
    "State size :",
    STATE_SIZE,
)

print(
    "Action size:",
    ACTION_SIZE_FROM_ENV,
)


# ==========================================================
# STATE VALIDATION
# ==========================================================

expected_state_size = (
    NUM_AGVS * 6
)

if STATE_SIZE != expected_state_size:

    raise ValueError(
        "Unexpected environment state size.\n"
        f"Expected: {expected_state_size}\n"
        f"Actual:   {STATE_SIZE}\n\n"
        "The environment should provide 6 "
        "features per AGV."
    )


# ==========================================================
# DQN AGENT
# ==========================================================
#
# IMPORTANT:
#
# state_size comes directly from the environment.
#
# Therefore the DQN network must become:
#
#     STATE_SIZE -> hidden -> hidden -> actions
#
# For this environment:
#
#     18 -> 128 -> 128 -> 5
#
# Do NOT use DQN(12, 5).
#
# ==========================================================

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


# ==========================================================
# TRAINING HISTORY
# ==========================================================

rewards_history = []

steps_history = []

completion_history = []

pickup_history = []

collision_history = []

loss_history = []


# ==========================================================
# HEADER
# ==========================================================

print()

print(
    "=" * 72
)

print(
    "MULTI-AGV DQN TRAINING"
)

print(
    "=" * 72
)

print(
    f"Episodes      : {EPISODES}"
)

print(
    "Start episode : 1"
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

print(
    "=" * 72
)


# ==========================================================
# TRAINING LOOP
# ==========================================================

for episode in range(
    1,
    EPISODES + 1,
):

    # ======================================================
    # RESET
    # ======================================================

    reset_result = env.reset()

    if isinstance(
        reset_result,
        tuple,
    ):

        state = reset_result[0]

    else:

        state = reset_result


    state = np.asarray(
        state,
        dtype=np.float32,
    )


    # ======================================================
    # VALIDATE INITIAL STATE
    # ======================================================

    if state.shape != (
        STATE_SIZE,
    ):

        raise ValueError(
            f"Expected state shape "
            f"({STATE_SIZE},), "
            f"got {state.shape}."
        )


    # ======================================================
    # EPISODE VARIABLES
    # ======================================================

    episode_reward = 0.0

    episode_losses = []

    collision_count = 0

    completed_count = 0

    pickup_count = 0

    steps = 0

    done = False


    # ======================================================
    # EPISODE LOOP
    # ======================================================

    for step in range(
        MAX_STEPS
    ):

        steps = step + 1


        # ==================================================
        # SELECT JOINT ACTION
        # ==================================================
        #
        # Example for 3 AGVs:
        #
        # [3, 4, 1]
        #
        # AGV 1 -> Up
        # AGV 2 -> Down
        # AGV 3 -> Right
        #
        # ==================================================

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


        # ==================================================
        # ACTION VALIDATION
        # ==================================================

        if actions.shape != (
            NUM_AGVS,
        ):

            raise ValueError(
                f"Expected joint action "
                f"shape ({NUM_AGVS},), "
                f"got {actions.shape}."
            )


        if np.any(
            actions < 0
        ) or np.any(
            actions >= ACTION_SIZE
        ):

            raise ValueError(
                "Invalid action detected. "
                f"Actions={actions}"
            )


        # ==================================================
        # ENVIRONMENT STEP
        # ==================================================

        result = env.step(
            actions
        )


        # ==================================================
        # GYMNASIUM API
        # ==================================================

        if not isinstance(
            result,
            tuple,
        ) or len(result) != 5:

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


        # ==================================================
        # DONE
        # ==================================================

        done = (
            bool(terminated)
            or bool(truncated)
        )


        # ==================================================
        # NEXT STATE
        # ==================================================

        next_state = np.asarray(
            next_state,
            dtype=np.float32,
        )


        if next_state.shape != (
            STATE_SIZE,
        ):

            raise ValueError(
                f"Expected next state "
                f"shape ({STATE_SIZE},), "
                f"got {next_state.shape}."
            )


        # ==================================================
        # JOINT REPLAY TRANSITION
        # ==================================================
        #
        # IMPORTANT:
        #
        # Store ONE transition for all AGVs.
        #
        # state
        # actions
        # reward
        # next_state
        # done
        #
        # Do NOT store each AGV separately.
        #
        # ==================================================

        agent.remember(
            state,
            actions,
            reward,
            next_state,
            done,
        )


        # ==================================================
        # TRAIN
        # ==================================================

        loss = agent.train_step()


        if loss is not None:

            episode_losses.append(
                float(loss)
            )


        # ==================================================
        # REWARD
        # ==================================================

        episode_reward += float(
            reward
        )


        # ==================================================
        # INFO
        # ==================================================

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


            completed_count += int(
                info.get(
                    "completed_tasks",
                    info.get(
                        "completed",
                        0,
                    ),
                )
            )


            pickup_count += int(
                info.get(
                    "pickup_tasks",
                    info.get(
                        "pickups",
                        0,
                    ),
                )
            )


        # ==================================================
        # UPDATE STATE
        # ==================================================

        state = next_state


        # ==================================================
        # TERMINATION
        # ==================================================

        if done:

            break


    # ======================================================
    # EPSILON DECAY
    # ======================================================

    agent.decay_epsilon()


    # ======================================================
    # AVERAGE LOSS
    # ======================================================

    if episode_losses:

        average_loss = float(
            np.mean(
                episode_losses
            )
        )

    else:

        average_loss = 0.0


    # ======================================================
    # HISTORY
    # ======================================================

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


    # ======================================================
    # PROGRESS
    # ======================================================

    if (
        episode == 1
        or episode % 10 == 0
    ):

        print(
            f"Episode "
            f"{episode:04d}/{EPISODES:04d} "
            f"| Reward={episode_reward:.2f} "
            f"| Steps={steps} "
            f"| Pickups={pickup_count} "
            f"| Completed={completed_count} "
            f"| Collision={collision_count} "
            f"| Loss={average_loss:.5f} "
            f"| Epsilon={agent.epsilon:.4f}"
        )


    # ======================================================
    # CHECKPOINT
    # ======================================================

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


# ==========================================================
# FINAL MODEL
# ==========================================================

final_model_path = os.path.join(
    MODEL_DIR,
    "dqn_model.pt",
)


agent.save(
    final_model_path
)


# ==========================================================
# FINAL SUMMARY
# ==========================================================

print()

print(
    "=" * 72
)

print(
    "FINAL TRAINING SUMMARY"
)

print(
    "=" * 72
)

print(
    f"Episodes completed : {EPISODES}"
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

print(
    "=" * 72
)


# ==========================================================
# CLEANUP
# ==========================================================

env.close()
