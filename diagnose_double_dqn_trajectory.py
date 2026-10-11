from algorithms.dqn_trainer import DQNTrainer
t = DQNTrainer(
    episodes=1,
    max_steps=300,
    num_agvs=3,
    seed=42,
    model_path="models/double_dqn_2000ep.pt",
    resume=False,
    verbose=False,
)
t.agent.evaluation_mode()
env = t.env
state, _ = env.reset(seed=42)
print("INITIAL_TASKS=")
for task in env.tasks:
    print(
        task.task_id,
        task.status,
        "pickup=", tuple(task.pickup),
        "delivery=", tuple(task.delivery),
    )
print()
print("--- GREEDY TRAJECTORY ---")
for step in range(30):
    positions_before = [
        tuple(map(int, env.system.agvs[i].position))
        for i in range(1, 4)
    ]
    targets = [
        env._get_current_target(env.system.agvs[i])
        for i in range(1, 4)
    ]
    action = t.agent.select_action(
        state,
        training=False,
    )
    next_state, reward, terminated, truncated, info = env.step(action)
    positions_after = [
        tuple(map(int, env.system.agvs[i].position))
        for i in range(1, 4)
    ]
    completed = info.get(
        "completed_tasks",
        info.get("completed", 0),
    )
    print(
        f"STEP={step+1} "
        f"ACTION={action.tolist()} "
        f"BEFORE={positions_before} "
        f"TARGETS={targets} "
        f"AFTER={positions_after} "
        f"REWARD={float(reward):.4f} "
        f"COMPLETED={completed}"
    )
    state = next_state
    if terminated or truncated:
        print("EPISODE_ENDED=", step + 1)
        break
