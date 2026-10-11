import numpy as np
import torch

from algorithms.dqn_agent import DQNAgent
from environment.multi_agv_env import MultiAGVEnv


# ==========================================================
# CONFIGURATION
# ==========================================================

EPISODES = 20
MAX_STEPS = 200

NUM_AGVS = 3
ACTION_SIZE = 5

MODEL_PATH = "models/dqn_model.pt"


# ==========================================================
# ENVIRONMENT
# ==========================================================

env = MultiAGVEnv(
    num_agvs=NUM_AGVS,
    max_steps=MAX_STEPS,
)


# ==========================================================
# STATE SIZE
# ==========================================================

state, info = env.reset(
    seed=42
)

STATE_SIZE = int(
    env.observation_space.shape[0]
)


# ==========================================================
# AGENT
# ==========================================================

agent = DQNAgent(
    state_size=STATE_SIZE,
    num_agvs=NUM_AGVS,
    action_size=ACTION_SIZE,

    learning_rate=0.0001,
    gamma=0.99,
    tau=0.001,

    epsilon=0.0,
    epsilon_min=0.0,
    epsilon_decay=1.0,

    batch_size=64,

# Attach evaluation environment for collision-safe DQN action selection
)


# ==========================================================
# LOAD MODEL
# ==========================================================

checkpoint = torch.load(
    MODEL_PATH,
    map_location=agent.device,
)


# ==========================================================
# CHECKPOINT CONFIGURATION
# ==========================================================

saved_state_size = int(
    checkpoint.get(
        "state_size",
        -1,
    )
)

saved_num_agvs = int(
    checkpoint.get(
        "num_agvs",
        -1,
    )
)

saved_action_size = int(
    checkpoint.get(
        "action_size",
        -1,
    )
)

saved_joint_action_size = int(
    checkpoint.get(
        "joint_action_size",
        saved_action_size ** saved_num_agvs,
    )
)


print("=" * 70)
print("CHECKPOINT CONFIGURATION")
print("=" * 70)

print(
    "Checkpoint state size       :",
    saved_state_size,
)

print(
    "Checkpoint AGVs             :",
    saved_num_agvs,
)

print(
    "Checkpoint action size     :",
    saved_action_size,
)

print(
    "Checkpoint joint actions   :",
    saved_joint_action_size,
)

print(
    "Current state size         :",
    STATE_SIZE,
)

print(
    "Current AGVs               :",
    NUM_AGVS,
)

print(
    "Current action size        :",
    ACTION_SIZE,
)

print("=" * 70)


# ==========================================================
# VALIDATE CHECKPOINT
# ==========================================================

if saved_state_size != STATE_SIZE:

    raise RuntimeError(
        f"STATE SIZE MISMATCH: "
        f"checkpoint={saved_state_size}, "
        f"environment={STATE_SIZE}"
    )


if saved_num_agvs != NUM_AGVS:

    raise RuntimeError(
        f"AGV COUNT MISMATCH: "
        f"checkpoint={saved_num_agvs}, "
        f"evaluation={NUM_AGVS}"
    )


if saved_action_size != ACTION_SIZE:

    raise RuntimeError(
        f"ACTION SIZE MISMATCH: "
        f"checkpoint={saved_action_size}, "
        f"evaluation={ACTION_SIZE}"
    )


expected_joint_actions = (
    ACTION_SIZE ** NUM_AGVS
)


if saved_joint_action_size != expected_joint_actions:

    raise RuntimeError(
        f"JOINT ACTION SIZE MISMATCH: "
        f"checkpoint={saved_joint_action_size}, "
        f"expected={expected_joint_actions}"
    )


# ==========================================================
# LOAD NETWORKS
# ==========================================================

agent.policy_network.load_state_dict(
    checkpoint[
        "policy_network"
    ]
)

agent.target_network.load_state_dict(
    checkpoint[
        "target_network"
    ]
)


# ==========================================================
# EVALUATION MODE
# ==========================================================

agent.evaluation_mode()


# ==========================================================
# HISTORY
# ==========================================================

rewards = []
steps_list = []
pickups_list = []
completed_list = []
collision_list = []


# ==========================================================
# HEADER
# ==========================================================

print()
print("=" * 70)
print("CENTRALIZED MULTI-AGV DQN EVALUATION")
print("=" * 70)

print(
    "Episodes        :",
    EPISODES,
)

print(
    "Model           :",
    MODEL_PATH,
)

print(
    "State size      :",
    STATE_SIZE,
)

print(
    "AGVs            :",
    NUM_AGVS,
)

print(
    "Actions / AGV   :",
    ACTION_SIZE,
)

print(
    "Joint actions   :",
    agent.joint_action_size,
)

print(
    "Epsilon         :",
    agent.epsilon,
)

print(
    "Device          :",
    agent.device,
)

print("=" * 70)


# ==========================================================
# EVALUATION
# ==========================================================

for episode in range(
    1,
    EPISODES + 1,
):

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

    total_reward = 0.0

    completed = 0
    pickups = 0
    collisions = 0

    step = 0

    for step in range(
        1,
        MAX_STEPS + 1,
    ):

        # --------------------------------------------------
        # SELECT JOINT ACTION
        # --------------------------------------------------

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

        # --------------------------------------------------
        # ENVIRONMENT STEP
        # --------------------------------------------------

        (
            next_state,
            reward,
            terminated,
            truncated,
            info,
        ) = env.step(
            actions
        )

        # --------------------------------------------------
        # REWARD
        # --------------------------------------------------

        total_reward += float(
            reward
        )

        # --------------------------------------------------
        # INFO
        # --------------------------------------------------

        if isinstance(
            info,
            dict,
        ):

            completed = max(
                completed,
                int(
                    info.get(
                        "completed_tasks",
                        0,
                    )
                ),
            )

            pickups = max(
                pickups,
                int(
                    info.get(
                        "pickup_tasks",
                        0,
                    )
                ),
            )

            collisions += int(
                info.get(
                    "collision_count",
                    0,
                )
            )

        # --------------------------------------------------
        # STATE
        # --------------------------------------------------

        state = np.asarray(
            next_state,
            dtype=np.float32,
        )

        # --------------------------------------------------
        # TERMINATION
        # --------------------------------------------------

        if (
            terminated
            or
            truncated
        ):

            break

    # ------------------------------------------------------
    # STORE RESULTS
    # ------------------------------------------------------

    rewards.append(
        total_reward
    )

    steps_list.append(
        step
    )

    pickups_list.append(
        pickups
    )

    completed_list.append(
        completed
    )

    collision_list.append(
        collisions
    )

    print(
        f"Episode {episode:02d}/{EPISODES} | "
        f"Reward={total_reward:7.2f} | "
        f"Steps={step:3d} | "
        f"Pickups={pickups} | "
        f"Completed={completed} | "
        f"Collision={collisions}"
    )


# ==========================================================
# SUMMARY
# ==========================================================

print("=" * 70)
print("EVALUATION SUMMARY")
print("=" * 70)

print(
    f"Average reward     : "
    f"{np.mean(rewards):.2f}"
)

print(
    f"Best reward        : "
    f"{np.max(rewards):.2f}"
)

print(
    f"Average steps      : "
    f"{np.mean(steps_list):.2f}"
)

print(
    f"Average pickups    : "
    f"{np.mean(pickups_list):.2f}"
)

print(
    f"Average completed  : "
    f"{np.mean(completed_list):.2f}"
)

print(
    f"Completion rate    : "
    f"{np.mean(np.array(completed_list) == NUM_AGVS) * 100:.1f}%"
)

print(
    f"Average collisions : "
    f"{np.mean(collision_list):.2f}"
)

print("=" * 70)
