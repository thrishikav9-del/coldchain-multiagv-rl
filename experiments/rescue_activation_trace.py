from environment.multi_agv_env import MultiAGVEnv
SEEDS = [1, 5, 7, 21, 42, 100]
print("=" * 90)
print("RESCUE ACTIVATION AND DESTINATION SWITCH EXPERIMENT")
print("=" * 90)
for seed in SEEDS:
    env = MultiAGVEnv(
        num_agvs=3,
        max_steps=100,
        risk_aware_assignment=True,
    )
    env.reset(seed=seed)
    # Controlled task risk profile.
    # Task 1 = LOW
    # Task 2 = MEDIUM
    # Task 3 = HIGH
    risk_values = [0.10, 0.50, 0.95]
    urgency_values = [0.10, 0.50, 0.95]
    rset_values = [60.0, 30.0, 5.0]
    for task, risk, urgency, rset in zip(
        env.tasks,
        risk_values,
        urgency_values,
        rset_values,
    ):
        task.risk_score = risk
        task.urgency_score = urgency
        task.future_risk_60 = risk
        task.rset_minutes = rset
        task.assigned_agv = None
        task.status = "PENDING"
        task.in_rescue = False
        task.rescue_destination = None
        task.original_delivery = task.delivery
    for agv in env.system.agvs.values():
        agv.current_task = None
        agv.status = "IDLE"
    # Start the controlled tasks.
    env._assign_tasks()
    env._start_assigned_tasks()
    print()
    print("-" * 90)
    print(f"SEED={seed}")
    print("-" * 90)
    previous_state = {}
    for step in range(1, 101):
        actions = []
        for agv_id in sorted(env.system.agvs):
            agv = env.system.agvs[agv_id]
            actions.append(
                env._get_task_directed_action(agv)
            )
        state, reward, terminated, truncated, info = env.step(
            actions
        )
        for task in env.tasks:
            agv_id = getattr(
                task,
                "assigned_agv",
                None,
            )
            if agv_id is None:
                continue
            agv = env.system.agvs.get(
                agv_id
            )
            if agv is None:
                continue
            current_state = (
                task.status,
                getattr(
                    task,
                    "in_rescue",
                    False,
                ),
                getattr(
                    task,
                    "rescue_destination",
                    None,
                ),
                tuple(agv.destination)
                if agv.destination is not None
                else None,
            )
            if (
                task.task_id not in previous_state
                or current_state != previous_state[task.task_id]
            ):
                print(
                    f"STEP={env.current_step:>3} "
                    f"TASK={task.task_id} "
                    f"AGV={agv_id} "
                    f"STATUS={task.status} "
                    f"RISK={task.risk_score:.2f} "
                    f"IN_RESCUE={getattr(task, 'in_rescue', False)} "
                    f"RESCUE_DEST={getattr(task, 'rescue_destination', None)} "
                    f"AGV_DEST={agv.destination}"
                )
                previous_state[
                    task.task_id
                ] = current_state
        if terminated or truncated:
            break
    print()
    print(
        f"FINAL SEED={seed} "
        f"STEPS={env.current_step} "
        f"COMPLETED={info.get('completed_tasks', 0)} "
        f"EXPOSURE={env.episode_total_exposure_steps} "
        f"EXCEEDED={env.episode_tasks_exposure_exceeded}"
    )
print()
print("=" * 90)
print("EXPERIMENT COMPLETE")
print("=" * 90)
