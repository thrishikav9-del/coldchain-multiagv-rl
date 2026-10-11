from environment.multi_agv_env import MultiAGVEnv
from algorithms.dqn_agent import DQNAgent
import numpy as np
EPISODES = 10
MAX_STEPS = 200
agent = DQNAgent(
    state_size=18,
    num_agvs=3,
    action_size=5,
    batch_size=64,
)
agent.load("models/dqn_model.pt")
agent.epsilon = 0.0
rewards = []
pickups = []
completed = []
collisions = []
print("=" * 72)
print("PHASE 4.2 MULTI-EPISODE TRAINED POLICY TEST")
print("=" * 72)
for episode in range(1, EPISODES + 1):
    env = MultiAGVEnv(
        num_agvs=3,
        max_steps=MAX_STEPS,
    )
    state, _ = env.reset()
    total_reward = 0.0
    ep_pickups = 0
    ep_completed = 0
    ep_collisions = 0
    for step in range(MAX_STEPS):
        action = agent.select_action(
            state,
            training=False,
        )
        next_state, reward, terminated, truncated, info = env.step(
            np.asarray(action, dtype=np.int64)
        )
        total_reward += float(reward)
        ep_pickups += int(
            info.get("pickup_tasks", 0)
        )
        ep_completed += int(
            info.get("completed_tasks", 0)
        )
        ep_collisions += int(
            info.get("collision_count", 0)
        )
        state = next_state
        if terminated or truncated:
            break
    rewards.append(total_reward)
    pickups.append(ep_pickups)
    completed.append(ep_completed)
    collisions.append(ep_collisions)
    print(
        f"EPISODE={episode:02d} "
        f"REWARD={total_reward:8.3f} "
        f"PICKUPS={ep_pickups} "
        f"COMPLETED={ep_completed} "
        f"COLLISIONS={ep_collisions} "
        f"STEPS={step + 1}"
    )
    env.close()
print("-" * 72)
print("AVERAGE_REWARD =", round(float(np.mean(rewards)), 3))
print("TOTAL_PICKUPS  =", sum(pickups))
print("TOTAL_COMPLETED=", sum(completed))
print("TOTAL_COLLISIONS=", sum(collisions))
print("BEST_REWARD    =", round(float(np.max(rewards)), 3))
print("=" * 72)
