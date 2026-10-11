from environment.multi_agv_env import MultiAGVEnv
from algorithms.dqn_agent import DQNAgent
import numpy as np

NUM_AGVS = 3
STATE_SIZE = 18
ACTION_SIZE = 5

env = MultiAGVEnv(
    num_agvs=NUM_AGVS,
    max_steps=30,
)

state, _ = env.reset(seed=42)

agent = DQNAgent(
    state_size=STATE_SIZE,
    num_agvs=NUM_AGVS,
    action_size=ACTION_SIZE,
)

agent.load("models/dqn_model.pt")
agent.evaluation_mode()

print("=" * 90)
print("3-AGV DQN DIAGNOSTIC")
print("=" * 90)

print("Initial state:")
print(state.reshape(NUM_AGVS, 6))

print()
print("Initial positions:")
print([
    env.system.agvs[i].position
    for i in range(1, NUM_AGVS + 1)
])

print()
print("Tasks:")
for task in env.tasks:
    print(
        task.task_id,
        "pickup =", task.pickup,
        "delivery =", task.delivery,
        "status =", task.status,
    )

print()
print("=" * 90)

for step in range(1, 31):

    actions = agent.select_action(
        state,
        training=False,
    )

    actions = np.asarray(
        actions,
        dtype=np.int64,
    )

    next_state, reward, terminated, truncated, info = env.step(
        actions
    )

    positions = [
        env.system.agvs[i].position
        for i in range(1, NUM_AGVS + 1)
    ]

    print(
        f"STEP {step:02d} | "
        f"ACTIONS {actions.tolist()} | "
        f"POS {positions} | "
        f"REWARD {reward:7.2f} | "
        f"PICKUP {info.get('pickup_tasks', 0)} | "
        f"COMPLETE {info.get('completed_tasks', 0)} | "
        f"COLLISION {info.get('collision_count', 0)}"
    )

    state = next_state

    if terminated or truncated:
        print()
        print(
            "TERMINATED:",
            terminated,
            "TRUNCATED:",
            truncated,
        )
        break