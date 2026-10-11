import numpy as np
import torch

from algorithms.dqn_agent import DQNAgent
from environment.multi_agv_env import MultiAGVEnv


# ==========================================================
# CONFIGURATION
# ==========================================================

EPISODES = 100
MAX_STEPS = 300

NUM_AGVS = 3
ACTION_SIZE = 5

MODEL_PATH = "models\dqn_ep2000_test.pt"

# Toggle for controlled baseline comparison
RISK_AWARE_ASSIGNMENT = True


# ==========================================================
# ENVIRONMENT
# ==========================================================

env = MultiAGVEnv(
    num_agvs=NUM_AGVS,
    max_steps=MAX_STEPS,
    risk_aware_assignment=RISK_AWARE_ASSIGNMENT,
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

# Attach active evaluation environment for collision-safe action selection
agent._evaluation_env = env


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
        # Joint action must be collision free.
        if not agent.check_collision_free(
            candidate
        ):
            return False
        active_agvs = 0
        for agv_id in range(
            1,
            NUM_AGVS + 1,
        ):
            agv = agvs.get(
                agv_id
            )
            if agv is None:
                continue
            current = tuple(
                map(
                    int,
                    agv.position,
                )
            )
            target = targets[
                agv_id
            ]
            old_distance = manhattan(
                current,
                target,
            )
            # AGV already at its operational target.
            if old_distance == 0:
                continue
            active_agvs += 1
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
            new_distance = manhattan(
                proposed,
                target,
            )
            # Every active AGV must make strict progress.
            if new_distance >= old_distance:
                return False
        # If no AGV needs to move, let the fallback logic decide.
        if active_agvs == 0:
            return True
        return True
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

        dqn_actions = agent.select_action(
            state,
            training=False,
        )
        actions = apply_hybrid_controller(
            env,
            agent,
            dqn_actions,
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
                        "picked_up_tasks",
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
    f"{np.sum(completed_list) / (EPISODES * NUM_AGVS) * 100:.1f}%"
)

print(
    f"Average collisions : "
    f"{np.mean(collision_list):.2f}"
)

print("=" * 70)
