from environment.multi_agv_env import MultiAGVEnv
env = MultiAGVEnv()
obs, info = env.reset(seed=0)
total_reward = 0.0
print("START")
print("Initial AGVs:")
for agv_id, agv in env.system.agvs.items():
    print(
        f"AGV {agv_id}: "
        f"position={agv.position} "
        f"destination={agv.destination}"
    )
print()
terminated = False
truncated = False
for step in range(500):
    actions = []
    for agv_id in range(1, env.num_agvs + 1):
        agv = env.system.agvs[agv_id]
        x, y = agv.position
        tx, ty = agv.destination
        if x < tx:
            action = 1       # Right
        elif x > tx:
            action = 2       # Left
        elif y < ty:
            action = 3       # Up
        elif y > ty:
            action = 4       # Down
        else:
            action = 0       # Stay
        actions.append(action)
    obs, reward, terminated, truncated, info = env.step(actions)
    total_reward += reward
    pickup_count = info.get("pickup_tasks", 0)
    completed_count = info.get("completed_tasks", 0)
    collision_count = info.get("collision_count", 0)
    if pickup_count or completed_count or collision_count:
        print(
            f"STEP {step + 1}: "
            f"actions={actions}, "
            f"pickup={pickup_count}, "
            f"completed={completed_count}, "
            f"collision={collision_count}, "
            f"reward={reward:.3f}"
        )
    if terminated or truncated:
        break
print()
print("=== FINAL RESULT ===")
print("Steps:", env.current_step)
print("Total reward:", total_reward)
print("Completed:", info.get("completed_tasks", 0))
print("Picked up:", info.get("picked_up_tasks", 0))
print("Terminated:", terminated)
print("Truncated:", truncated)
print()
print("Final AGV states:")
for agv_id, agv in env.system.agvs.items():
    print(
        f"AGV {agv_id}: "
        f"position={agv.position}, "
        f"destination={agv.destination}, "
        f"status={agv.status}"
    )
