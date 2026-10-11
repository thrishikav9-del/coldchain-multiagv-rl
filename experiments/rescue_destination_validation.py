from environment.multi_agv_env import MultiAGVEnv
SEEDS = [1, 5, 7, 21, 42, 100]
print("=" * 90)
print("RESCUE DESTINATION VALIDATION EXPERIMENT")
print("=" * 90)
for seed in SEEDS:
    env = MultiAGVEnv(
        num_agvs=3,
        max_steps=20,
        risk_aware_assignment=True,
    )
    env.reset(seed=seed)
    # Controlled risk values.
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
    print()
    print("-" * 90)
    print(f"SEED={seed}")
    print("-" * 90)
    # Force assignment.
    env._assign_tasks()
    # Start tasks.
    env._start_assigned_tasks()
    for task in env.tasks:
        agv = env.system.agvs.get(
            task.assigned_agv
        )
        if agv is None:
            continue
        print()
        print(
            f"TASK {task.task_id}: "
            f"RISK={task.risk_score:.2f} "
            f"URGENCY={task.urgency_score:.2f} "
            f"RSET={task.rset_minutes:.1f}"
        )
        print(
            f"  ASSIGNED_AGV={task.assigned_agv}"
        )
        print(
            f"  DELIVERY={task.delivery}"
        )
        print(
            f"  CURRENT_AGV_DESTINATION="
            f"{agv.destination}"
        )
        print(
            f"  IN_RESCUE="
            f"{getattr(task, 'in_rescue', False)}"
        )
        print(
            f"  RESCUE_DESTINATION="
            f"{getattr(task, 'rescue_destination', None)}"
        )
        # Directly test the rescue selector.
        safe_destination = (
            env._select_safe_destination(task)
        )
        print(
            f"  SELECTOR_RESULT="
            f"{safe_destination}"
        )
        if safe_destination is not None:
            if tuple(safe_destination) != tuple(task.delivery):
                print(
                    "  RESULT=VALID_RESCUE_DESTINATION"
                )
            else:
                print(
                    "  RESULT=WARNING_RESCUE_EQUALS_DELIVERY"
                )
        else:
            print(
                "  RESULT=NO_RESCUE_DESTINATION"
            )
    print()
    print(
        "=" * 90
    )
print()
print("VALIDATION COMPLETE")
print("=" * 90)
