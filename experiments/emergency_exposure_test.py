from environment.multi_agv_env import MultiAGVEnv
SEEDS = [1, 5, 7, 21, 42, 100]
def prepare(env, seed):
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
    for task in env.tasks:
        task.assigned_agv = None
        task.status = "PENDING"
    for agv in env.system.agvs.values():
        agv.current_task = None
        agv.status = "IDLE"
    env.warehouse.trigger_cooling_failure(1)
    env._assign_tasks()
    env._start_assigned_tasks()
    return env
def run_case(seed, emergency_on):
    env = MultiAGVEnv(
        num_agvs=3,
        max_steps=120,
        risk_aware_assignment=True,
    )
    original_selector = env._select_safe_destination
    if not emergency_on:
        env._select_safe_destination = lambda task: None
    try:
        prepare(env, seed)
        print()
        print(
            f"--- {('ON' if emergency_on else 'OFF')} "
            f"SEED={seed} INITIAL STATE ---"
        )
        for task in env.tasks:
            print(
                f"TASK {task.task_id}: "
                f"AGV={task.assigned_agv} "
                f"STATUS={task.status} "
                f"RISK={task.risk_score:.2f} "
                f"URGENCY={task.urgency_score:.2f} "
                f"RSET={task.rset_minutes:.1f} "
                f"IN_RESCUE={getattr(task, 'in_rescue', False)} "
                f"RESCUE_DEST={getattr(task, 'rescue_destination', None)} "
                f"DELIVERY={task.delivery}"
            )
        previous = {}
        for step in range(1, 121):
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
                agv_id = getattr(task, "assigned_agv", None)
                if agv_id is None:
                    continue
                agv = env.system.agvs.get(agv_id)
                if agv is None:
                    continue
                try:
                    zone_id = env.warehouse.get_zone_id(
                        tuple(agv.position)
                    )
                    temperature = float(
                        env.warehouse._thermal_state[
                            zone_id
                        ]["temperature"]
                    )
                except Exception:
                    zone_id = None
                    temperature = None
                current = (
                    task.status,
                    getattr(task, "in_rescue", False),
                    getattr(task, "rescue_destination", None),
                    zone_id,
                )
                if (
                    task.task_id not in previous
                    or current != previous[task.task_id]
                ):
                    print(
                        f"{'ON' if emergency_on else 'OFF'} "
                        f"SEED={seed:>3} "
                        f"STEP={env.current_step:>3} "
                        f"TASK={task.task_id} "
                        f"AGV={agv_id} "
                        f"STATUS={task.status} "
                        f"RESCUE={getattr(task, 'in_rescue', False)} "
                        f"ZONE={zone_id} "
                        f"TEMP="
                        f"{temperature:.3f}"
                        if temperature is not None
                        else
                        f"{'ON' if emergency_on else 'OFF'} "
                        f"SEED={seed:>3} "
                        f"STEP={env.current_step:>3} "
                        f"TASK={task.task_id} "
                        f"AGV={agv_id} "
                        f"STATUS={task.status} "
                        f"RESCUE={getattr(task, 'in_rescue', False)} "
                        f"ZONE={zone_id}"
                    )
                    previous[task.task_id] = current
            if terminated or truncated:
                break
        print(
            f"{'ON' if emergency_on else 'OFF'} "
            f"SEED={seed:>3} FINAL "
            f"STEPS={env.current_step} "
            f"COMPLETED={info.get('completed_tasks', 0)} "
            f"EXPOSURE={env.episode_total_exposure_steps} "
            f"EXPOSED={env.episode_tasks_with_exposure} "
            f"EXCEEDED={env.episode_tasks_exposure_exceeded}"
        )
    finally:
        env._select_safe_destination = original_selector
print("=" * 90)
print("EMERGENCY RESCUE PATH / THERMAL TRACE")
print("=" * 90)
for emergency_on in [True, False]:
    for seed in SEEDS:
        run_case(seed, emergency_on)
print("=" * 90)
print("TRACE COMPLETE")
print("=" * 90)
