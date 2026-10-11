import csv
from pathlib import Path

import numpy as np
import torch

from algorithms.dqn_agent import DQNAgent
from environment.multi_agv_env import MultiAGVEnv


# ==========================================================
# CONFIGURATION
# ==========================================================

EPISODES = 20
MAX_STEPS = 300

NUM_AGVS = 3
ACTION_SIZE = 5

MODEL_PATH = "models/dqn_model.pt"

# Controlled experiment switch:
# True  = risk-aware assignment
# False = baseline assignment
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
    seed=1000,
)

STATE_SIZE = int(
    env.observation_space.shape[0]
)


# ==========================================================
# DQN AGENT
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
)

# Required for collision-safe and goal-progress action filtering.
agent._evaluation_env = env


# ==========================================================
# LOAD MODEL
# ==========================================================

checkpoint = torch.load(
    MODEL_PATH,
    map_location=agent.device,
)


# ==========================================================
# CHECKPOINT VALIDATION
# ==========================================================

checkpoint_state_size = int(
    checkpoint.get(
        "state_size",
        -1,
    )
)

checkpoint_num_agvs = int(
    checkpoint.get(
        "num_agvs",
        -1,
    )
)

checkpoint_action_size = int(
    checkpoint.get(
        "action_size",
        -1,
    )
)

checkpoint_joint_action_size = int(
    checkpoint.get(
        "joint_action_size",
        checkpoint_action_size ** checkpoint_num_agvs,
    )
)

if checkpoint_state_size != STATE_SIZE:
    raise ValueError(
        f"Checkpoint state_size={checkpoint_state_size}, "
        f"environment state_size={STATE_SIZE}."
    )

if checkpoint_num_agvs != NUM_AGVS:
    raise ValueError(
        f"Checkpoint num_agvs={checkpoint_num_agvs}, "
        f"evaluator num_agvs={NUM_AGVS}."
    )

if checkpoint_action_size != ACTION_SIZE:
    raise ValueError(
        f"Checkpoint action_size={checkpoint_action_size}, "
        f"evaluator action_size={ACTION_SIZE}."
    )

expected_joint_action_size = (
    ACTION_SIZE ** NUM_AGVS
)

if checkpoint_joint_action_size != expected_joint_action_size:
    raise ValueError(
        f"Checkpoint joint_action_size="
        f"{checkpoint_joint_action_size}, "
        f"expected {expected_joint_action_size}."
    )


# ==========================================================
# LOAD NETWORKS
# ==========================================================

agent.policy_network.load_state_dict(
    checkpoint["policy_network"]
)

agent.target_network.load_state_dict(
    checkpoint["target_network"]
)

agent.evaluation_mode()

agent._evaluation_env = env


# ==========================================================
# HISTORY
# ==========================================================

rewards = []
steps_list = []
pickups_list = []
completed_list = []
collision_list = []

risk_history = []
urgency_history = []
min_rset_history = []
rescue_history = []
max_temperature_history = []


# ==========================================================
# HEADER
# ==========================================================

print()
print("=" * 70)
print("FINAL COLD-CHAIN DQN EVALUATION")
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
    "Risk-aware      :",
    RISK_AWARE_ASSIGNMENT,
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
    "Max steps       :",
    MAX_STEPS,
)

print(
    "Epsilon         :",
    agent.epsilon,
)

print("=" * 70)


# ==========================================================
# EVALUATION
# ==========================================================

for episode in range(
    1,
    EPISODES + 1,
):

    reset_result = env.reset(
        seed=1000 + episode,
    )

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

    episode_risks = []
    episode_urgencies = []
    episode_rsets = []
    episode_temperatures = []

    rescued_task_ids = set()

    step = 0

    # ------------------------------------------------------
    # EPISODE LOOP
    # ------------------------------------------------------

    for step in range(
        1,
        MAX_STEPS + 1,
    ):

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

        (
            next_state,
            reward,
            terminated,
            truncated,
            info,
        ) = env.step(
            actions
        )

        total_reward += float(
            reward
        )

        # --------------------------------------------------
        # ENVIRONMENT METRICS
        # --------------------------------------------------

        if isinstance(
            info,
            dict,
        ):

            completed = max(
                completed,
                int(
                    info.get(
                        "episode_completed_count",
                        info.get(
                            "completed_tasks",
                            0,
                        ),
                    )
                ),
            )

            pickups = max(
                pickups,
                int(
                    info.get(
                        "episode_pickup_count",
                        info.get(
                            "picked_up_tasks",
                            0,
                        ),
                    )
                ),
            )

            collisions += int(
                info.get(
                    "collision_count",
                    0,
                )
            )

            episode_risks.append(
                float(
                    info.get(
                        "average_risk",
                        0.0,
                    )
                )
            )

            episode_urgencies.append(
                float(
                    info.get(
                        "average_urgency",
                        0.0,
                    )
                )
            )

        # --------------------------------------------------
        # TASK-LEVEL COLD-CHAIN METRICS
        # --------------------------------------------------

        for task in getattr(
            env,
            "tasks",
            [],
        ):

            task_id = getattr(
                task,
                "task_id",
                None,
            )

            if (
                task_id is not None
                and
                bool(
                    getattr(
                        task,
                        "in_rescue",
                        False,
                    )
                )
            ):
                rescued_task_ids.add(
                    int(task_id)
                )

            try:
                episode_rsets.append(
                    float(
                        getattr(
                            task,
                            "rset_minutes",
                        )
                    )
                )
            except (
                TypeError,
                ValueError,
                AttributeError,
            ):
                pass

            try:
                episode_temperatures.append(
                    float(
                        getattr(
                            task,
                            "temperature",
                        )
                    )
                )
            except (
                TypeError,
                ValueError,
                AttributeError,
            ):
                pass

        state = np.asarray(
            next_state,
            dtype=np.float32,
        )

        if (
            bool(terminated)
            or
            bool(truncated)
        ):
            break

    # ------------------------------------------------------
    # EPISODE METRICS
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

    risk_history.append(
        float(
            np.mean(
                episode_risks
            )
        )
        if episode_risks
        else 0.0
    )

    urgency_history.append(
        float(
            np.mean(
                episode_urgencies
            )
        )
        if episode_urgencies
        else 0.0
    )

    min_rset_history.append(
        float(
            np.min(
                episode_rsets
            )
        )
        if episode_rsets
        else 60.0
    )

    rescue_history.append(
        len(
            rescued_task_ids
        )
    )

    max_temperature_history.append(
        float(
            np.max(
                episode_temperatures
            )
        )
        if episode_temperatures
        else 0.0
    )

    print(
        f"Episode {episode:02d}/{EPISODES} | "
        f"Reward={total_reward:8.2f} | "
        f"Steps={step:3d} | "
        f"Pickups={pickups} | "
        f"Completed={completed} | "
        f"Collision={collisions}"
    )


# ==========================================================
# SUMMARY
# ==========================================================

task_completion_rate = (
    np.sum(
        completed_list
    )
    /
    (
        EPISODES
        *
        NUM_AGVS
    )
    *
    100.0
)

full_episode_rate = (
    np.mean(
        np.asarray(
            completed_list
        )
        ==
        NUM_AGVS
    )
    *
    100.0
)

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
    f"Task completion    : "
    f"{task_completion_rate:.1f}%"
)

print(
    f"Full-episode rate  : "
    f"{full_episode_rate:.1f}%"
)

print(
    f"Average collisions : "
    f"{np.mean(collision_list):.2f}"
)

print(
    f"Average risk       : "
    f"{np.mean(risk_history):.4f}"
)

print(
    f"Average urgency    : "
    f"{np.mean(urgency_history):.4f}"
)

print(
    f"Average min RSET   : "
    f"{np.mean(min_rset_history):.2f} min"
)

print(
    f"Average rescues    : "
    f"{np.mean(rescue_history):.2f}"
)

print(
    f"Average max temp   : "
    f"{np.mean(max_temperature_history):.2f}"
)

print("=" * 70)


# ==========================================================
# SAVE EPISODE RESULTS
# ==========================================================

results_dir = Path(
    "results"
)

results_dir.mkdir(
    parents=True,
    exist_ok=True,
)

if RISK_AWARE_ASSIGNMENT:

    results_path = (
        results_dir
        /
        "final_dqn_metrics_riskaware.csv"
    )

else:

    results_path = (
        results_dir
        /
        "final_dqn_metrics_baseline.csv"
    )


with results_path.open(
    "w",
    newline="",
    encoding="utf-8",
) as handle:

    writer = csv.writer(
        handle
    )

    writer.writerow(
        [
            "episode",
            "reward",
            "steps",
            "pickups",
            "completed",
            "task_completion_rate",
            "collisions",
            "average_risk",
            "average_urgency",
            "minimum_rset_minutes",
            "rescue_activations",
            "maximum_temperature",
        ]
    )

    for index in range(
        EPISODES
    ):

        writer.writerow(
            [
                index + 1,
                rewards[index],
                steps_list[index],
                pickups_list[index],
                completed_list[index],
                (
                    completed_list[index]
                    /
                    NUM_AGVS
                    *
                    100.0
                ),
                collision_list[index],
                risk_history[index],
                urgency_history[index],
                min_rset_history[index],
                rescue_history[index],
                max_temperature_history[index],
            ]
        )


print(
    "RESULTS SAVED:",
    results_path,
)
