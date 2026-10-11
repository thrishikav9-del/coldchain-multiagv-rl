
import itertools
import numpy as np
from environment.multi_agv_env import MultiAGVEnv
EPISODES = 100
MAX_STEPS = 300
NUM_AGVS = 3
env = MultiAGVEnv(
    num_agvs=NUM_AGVS,
    max_steps=MAX_STEPS,
    risk_aware_assignment=False,
)
def reactive_actions(env):
    """Current-state, target-directed, collision-aware baseline."""
    agvs = env.system.agvs
    candidate_lists = []
    for agv_id in range(1, NUM_AGVS + 1):
        agv = agvs.get(agv_id)
        if agv is None:
            candidate_lists.append([0])
            continue
        target = env._get_current_target(agv)
        if target is None:
            candidate_lists.append([0])
            continue
        x, y = map(int, agv.position)
        gx, gy = map(int, target)
        options = [0]
        if x < gx:
            options.insert(0, 1)
        elif x > gx:
            options.insert(0, 2)
        if y < gy:
            options.append(3)
        elif y > gy:
            options.append(4)
        candidate_lists.append(
            list(dict.fromkeys(options))
        )
    best = None
    best_score = None
    positions = {}
    for agv_id in range(1, NUM_AGVS + 1):
        agv = agvs.get(agv_id)
        if agv is not None:
            positions[agv_id] = tuple(
                map(int, agv.position)
            )
    for joint in itertools.product(*candidate_lists):
        candidate = np.asarray(
            joint,
            dtype=np.int64,
        )
        proposed = {}
        valid = True
        for agv_id in range(1, NUM_AGVS + 1):
            agv = agvs.get(agv_id)
            if agv is None:
                valid = False
                break
            proposed[agv_id] = tuple(
                map(
                    int,
                    env._get_target_position(
                        agv,
                        int(candidate[agv_id - 1]),
                    ),
                )
            )
        if not valid:
            continue
        # Reject same-cell conflicts.
        if len(proposed.values()) != len(set(proposed.values())):
            continue
        # Reject pairwise swaps.
        agv_ids = list(positions.keys())
        for i in range(len(agv_ids)):
            a = agv_ids[i]
            for j in range(i + 1, len(agv_ids)):
                b = agv_ids[j]
                if (
                    proposed[a] == positions[b]
                    and
                    proposed[b] == positions[a]
                    and
                    proposed[a] != positions[a]
                ):
                    valid = False
                    break
            if not valid:
                break
        if not valid:
            continue
        total_distance = 0
        movement_count = 0
        for agv_id in range(1, NUM_AGVS + 1):
            agv = agvs.get(agv_id)
            if agv is None:
                continue
            target = env._get_current_target(agv)
            if target is None:
                continue
            px, py = proposed[agv_id]
            gx, gy = map(int, target)
            total_distance += (
                abs(px - gx)
                +
                abs(py - gy)
            )
            if proposed[agv_id] != positions[agv_id]:
                movement_count += 1
        score = (
            total_distance,
            -movement_count,
        )
        if best_score is None or score < best_score:
            best_score = score
            best = candidate
    if best is not None:
        return best
    return np.zeros(
        NUM_AGVS,
        dtype=np.int64,
    )
rewards = []
steps_history = []
pickups_history = []
completed_history = []
collision_history = []
for episode in range(1, EPISODES + 1):
    reset = env.reset(
        seed=10000 + episode
    )
    state = (
        reset[0]
        if isinstance(reset, tuple)
        else reset
    )
    total_reward = 0.0
    collisions = 0
    episode_pickups = 0
    final_step = MAX_STEPS
    for step in range(1, MAX_STEPS + 1):
        actions = reactive_actions(env)
        (
            next_state,
            reward,
            terminated,
            truncated,
            info,
        ) = env.step(actions)
        total_reward += float(reward)
        if isinstance(info, dict):
            collisions += int(
                info.get(
                    "collision_count",
                    0,
                )
            )
            episode_pickups = max(
                episode_pickups,
                int(
                    info.get(
                        "episode_pickup_count",
                        0,
                    )
                ),
            )
        state = next_state
        if terminated or truncated:
            final_step = step
            break
    picked = episode_pickups
    completed = sum(
        1
        for task in env.tasks
        if task.is_completed
    )
    rewards.append(total_reward)
    steps_history.append(final_step)
    pickups_history.append(picked)
    completed_history.append(completed)
    collision_history.append(collisions)
print("=" * 70)
print("FAIR REACTIVE BASELINE EVALUATION")
print("=" * 70)
print(
    f"Average reward     : "
    f"{np.mean(rewards):.2f}"
)
print(
    f"Average steps      : "
    f"{np.mean(steps_history):.2f}"
)
print(
    f"Average pickups    : "
    f"{np.mean(pickups_history):.2f}"
)
print(
    f"Average completed  : "
    f"{np.mean(completed_history):.2f}"
)
print(
    f"Completion rate    : "
    f"{np.mean(completed_history) / NUM_AGVS * 100:.1f}%"
)
print(
    f"Average collisions : "
    f"{np.mean(collision_history):.2f}"
)
print("=" * 70)
