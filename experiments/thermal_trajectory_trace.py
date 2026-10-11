from environment.multi_agv_env import MultiAGVEnv
SEEDS = [1, 42]
for seed in SEEDS:
    env = MultiAGVEnv(
        num_agvs=3,
        max_steps=100,
        risk_aware_assignment=True,
    )
    env.reset(seed=seed)
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
    # Sustained cooling failure in Zone 1.
    env.warehouse.trigger_cooling_failure(1)
    env._assign_tasks()
    env._start_assigned_tasks()
    print("=" * 100)
    print(f"THERMAL TRAJECTORY TRACE - SEED={seed}")
    print("=" * 100)
    print()
    print(
        "INITIAL TASK STATE"
    )
    for task in env.tasks:
        print(
            f"TASK={task.task_id} "
            f"AGV={task.assigned_agv} "
            f"RISK={task.risk_score:.2f} "
            f"URGENCY={task.urgency_score:.2f} "
            f"RSET={task.rset_minutes:.1f} "
            f"PICKUP={task.pickup} "
            f"DELIVERY={task.delivery}"
        )
    print()
    print(
        "-" * 100
    )
    previous = {}
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
            agv = env.system.agvs.get(agv_id)
            if agv is None:
                continue
            try:
                zone_id = env.warehouse.get_zone_id(
                    tuple(agv.position)
                )
            except Exception:
                zone_id = None
            try:
                temperature = float(
                    env.warehouse._thermal_state[
                        zone_id
                    ]["temperature"]
                )
            except Exception:
                temperature = None
            current = (
                task.status,
                getattr(task, "in_rescue", False),
                getattr(task, "rescue_destination", None),
                tuple(agv.position),
                tuple(agv.destination)
                if agv.destination is not None
                else None,
                zone_id,
                temperature,
                getattr(task, "exposure_time", None),
                getattr(task, "max_exposure_temperature", None),
            )
            if (
                task.task_id not in previous
                or current != previous[task.task_id]
            ):
                print(
                    f"STEP={env.current_step:>3} "
                    f"TASK={task.task_id} "
                    f"AGV={agv_id} "
                    f"STATUS={task.status:<12} "
                    f"RESCUE={getattr(task, 'in_rescue', False)!s:<5} "
                    f"POS={tuple(agv.position)!s:<12} "
                    f"DEST={agv.destination!s:<12} "
                    f"ZONE={zone_id!s:<3} "
                    f"TEMP={temperature if temperature is not None else 'NA'} "
                    f"EXPOSURE={getattr(task, 'exposure_time', None)} "
                    f"MAX_TEMP={getattr(task, 'max_exposure_temperature', None)}"
                )
                previous[task.task_id] = current
        if terminated or truncated:
            break
    print()
    print(
        "-" * 100
    )
    print(
        f"FINAL "
        f"STEPS={env.current_step} "
        f"COMPLETED={info.get('completed_tasks', 0)} "
        f"EXPOSURE={env.episode_total_exposure_steps} "
        f"EXPOSED={env.episode_tasks_with_exposure} "
        f"EXCEEDED={env.episode_tasks_exposure_exceeded} "
        f"REWARD={env.episode_total_reward:.3f}"
    )
    print("=" * 100)
    print()
print("THERMAL TRAJECTORY TRACE COMPLETE")
