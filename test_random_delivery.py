import numpy as np

from environment.multi_agv_env import MultiAGVEnv


env = MultiAGVEnv(
    num_agvs=3
)

state, info = env.reset()

print("Initial reset successful")

for step in range(500):

    # Random actions for 3 AGVs
    action = np.random.randint(
        0,
        5,
        size=3
    )

    state, reward, terminated, truncated, info = env.step(action)

    if step % 50 == 0:
        print(
            f"Step={step} | "
            f"Reward={reward:.2f} | "
            f"Pickup={info.get('pickup_tasks',0)} | "
            f"Completed={info.get('completed_tasks',0)} | "
            f"Collision={info.get('collision_count',0)}"
        )

    if terminated:
        print("Episode terminated")
        break


print("\n========== FINAL ==========")

print(
    "Steps:",
     step
)

print(
    "Pickups:",
    info.get(
        "picked_up_tasks",
        0
    )
)

print(
    "Completed:",
    info.get(
        "completed_tasks",
        0
    )
)

print(
    "Collisions:",
    info.get(
        "collision_count",
        0
    )
)

env.close()
