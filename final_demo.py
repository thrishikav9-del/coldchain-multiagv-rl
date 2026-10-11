from environment.multi_agv_env import MultiAGVEnv
env = MultiAGVEnv(
    num_agvs=3,
    max_steps=300,
    risk_aware_assignment=True,
)
state, info = env.reset(seed=42)
total_reward = 0.0
total_pickups = 0
total_completions = 0
total_collisions = 0
print("=" * 72)
print("COLD-CHAIN MULTI-AGV FINAL DEMO")
print("=" * 72)
print("\nINITIAL TASKS")
for task in env.tasks:
    print(
        f"Task {task.task_id} | "
        f"AGV={task.assigned_agv} | "
        f"Pickup={task.pickup} | "
        f"Delivery={task.delivery} | "
        f"Priority={task.priority} | "
        f"Risk={task.risk_score:.3f} | "
        f"Urgency={task.urgency_score:.3f}"
    )
print("\nINITIAL WAREHOUSE")
print(env.render())
terminated = False
truncated = False
for step in range(env.max_steps):
    actions = [
        env._get_task_directed_action(agv)
        for agv in env.system.agvs.values()
    ]
    (
        state,
        reward,
        terminated,
        truncated,
        info,
    ) = env.step(actions)
    total_reward += float(reward)
    step_pickups = int(
        info.get(
            "pickup_tasks_this_step",
            0,
        )
    )
    step_completions = int(
        info.get(
            "completed_tasks_this_step",
            0,
        )
    )
    step_collisions = int(
        info.get(
            "collision_count",
            0,
        )
    )
    total_pickups += step_pickups
    total_completions += step_completions
    total_collisions += step_collisions
    if (
        step_pickups > 0
        or step_completions > 0
        or step_collisions > 0
        or step == 0
        or terminated
        or truncated
    ):
        print()
        print(
            f"STEP {step + 1}"
        )
        print(
            f"Actions      : {actions}"
        )
        print(
            f"Reward       : {reward:.3f}"
        )
        print(
            f"Pickups      : {step_pickups}"
        )
        print(
            f"Completions  : {step_completions}"
        )
        print(
            f"Collisions   : {step_collisions}"
        )
        print(
            "AGV positions:",
            [
                (
                    agv_id,
                    agv.position,
                    agv.status,
                    agv.destination,
                )
                for agv_id, agv
                in env.system.agvs.items()
            ],
        )
        print(
            "Task states:",
            [
                (
                    task.task_id,
                    task.status,
                    task.assigned_agv,
                )
                for task in env.tasks
            ],
        )
    if terminated or truncated:
        break
print()
print("=" * 72)
print("FINAL DEMO SUMMARY")
print("=" * 72)
print(
    f"Steps         : {step + 1}"
)
print(
    f"Total reward  : {total_reward:.3f}"
)
print(
    f"Pickups       : {total_pickups}"
)
print(
    f"Completions   : {total_completions}"
)
print(
    f"Collisions    : {total_collisions}"
)
print(
    f"Terminated    : {terminated}"
)
print(
    f"Truncated     : {truncated}"
)
print(
    "Final tasks   :",
    [
        (
            task.task_id,
            task.status,
            task.assigned_agv,
        )
        for task in env.tasks
    ],
)
print("=" * 72)
env.close()
