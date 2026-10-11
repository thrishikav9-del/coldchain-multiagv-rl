from environment.multi_agv_env import MultiAGVEnv
SEEDS = [1, 5, 7, 21, 42, 100]
def configure(env, seed):
    env.reset(seed=seed)
    # Keep the same warehouse emergency.
    env.warehouse.trigger_cooling_failure(1)
    # Make only Task 3 the critical shipment.
    for task in env.tasks:
        task.assigned_agv = None
        task.status = "PENDING"
        task.in_rescue = False
        task.rescue_destination = None
        task.original_delivery = task.delivery
    risk_values = [0.10, 0.10, 0.95]
    urgency_values = [0.10, 0.10, 0.95]
    rset_values = [60.0, 60.0, 5.0]
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
    for agv in env.system.agvs.values():
        agv.current_task = None
        agv.status = "IDLE"
def run_case(seed, rescue_on):
    env = MultiAGVEnv(
        num_agvs=3,
        max_steps=100,
        risk_aware_assignment=True,
    )
    original_selector = env._select_safe_destination
    if not rescue_on:
        env._select_safe_destination = lambda task: None
    try:
        configure(env, seed)
        env._assign_tasks()
        env._start_assigned_tasks()
        target = env.tasks[2]
        print()
        print(
            f"{'RESCUE ON' if rescue_on else 'RESCUE OFF'} "
            f"SEED={seed}"
        )
        print("-" * 90)
        print(
            f"INITIAL TASK 3: "
            f"AGV={target.assigned_agv} "
            f"RISK={target.risk_score:.2f} "
            f"URGENCY={target.urgency_score:.2f} "
            f"RSET={target.rset_minutes:.1f} "
            f"PICKUP={target.pickup} "
            f"DELIVERY={target.delivery}"
        )
        previous_exposure = env.episode_total_exposure_steps
        activation_step = None
        rescue_destination = None
        rescue_reached_step = None
        exposure_at_activation = None
        exposure_at_rescue = None
        max_temperature = 0.0
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
            exposure_now = env.episode_total_exposure_steps
            # Detect rescue activation.
            if (
                rescue_on
                and activation_step is None
                and getattr(target, "in_rescue", False)
            ):
                activation_step = env.current_step
                rescue_destination = getattr(
                    target,
                    "rescue_destination",
                    None,
                )
                exposure_at_activation = exposure_now
                print(
                    f"RESCUE_ACTIVATED "
                    f"STEP={env.current_step} "
                    f"AGV={target.assigned_agv} "
                    f"DEST={rescue_destination} "
                    f"EXPOSURE={exposure_now}"
                )
            # Detect arrival at rescue destination.
            if (
                rescue_on
                and activation_step is not None
                and rescue_reached_step is None
                and not getattr(target, "in_rescue", False)
                and rescue_destination is not None
                and tuple(target.delivery) != tuple(rescue_destination)
            ):
                rescue_reached_step = env.current_step
                exposure_at_rescue = exposure_now
                print(
                    f"RESCUE_REACHED "
                    f"STEP={env.current_step} "
                    f"DEST={rescue_destination} "
                    f"EXPOSURE={exposure_now}"
                )
            # Track actual environment exposure.
            if exposure_now != previous_exposure:
                print(
                    f"EXPOSURE_CHANGE "
                    f"STEP={env.current_step} "
                    f"EXPOSURE={exposure_now} "
                    f"DELTA={exposure_now - previous_exposure}"
                )
                previous_exposure = exposure_now
            # Track maximum thermal exposure temperature.
            task_max_temperature = float(
                getattr(
                    target,
                    "max_exposure_temperature",
                    0.0,
                )
            )
            max_temperature = max(
                max_temperature,
                task_max_temperature,
            )
            if terminated or truncated:
                break
        final_exposure = env.episode_total_exposure_steps
        print()
        print(
            f"FINAL "
            f"STEPS={env.current_step} "
            f"COMPLETED={info.get('completed_tasks', 0)} "
            f"EXPOSURE={final_exposure} "
            f"EXPOSED={env.episode_tasks_with_exposure} "
            f"EXCEEDED={env.episode_tasks_exposure_exceeded} "
            f"MAX_TEMP={max_temperature:.3f}"
        )
        print(
            f"RESCUE_ACTIVATION_STEP={activation_step} "
            f"RESCUE_DESTINATION={rescue_destination} "
            f"RESCUE_REACHED_STEP={rescue_reached_step}"
        )
        print(
            f"EXPOSURE_AT_ACTIVATION={exposure_at_activation} "
            f"EXPOSURE_AT_RESCUE={exposure_at_rescue}"
        )
        return {
            "seed": seed,
            "rescue_on": rescue_on,
            "steps": env.current_step,
            "completed": info.get("completed_tasks", 0),
            "exposure": final_exposure,
            "exposed": env.episode_tasks_with_exposure,
            "exceeded": env.episode_tasks_exposure_exceeded,
            "max_temperature": max_temperature,
            "activation_step": activation_step,
            "rescue_destination": rescue_destination,
            "rescue_reached_step": rescue_reached_step,
            "exposure_at_activation": exposure_at_activation,
            "exposure_at_rescue": exposure_at_rescue,
            "reward": env.episode_total_reward,
        }
    finally:
        env._select_safe_destination = original_selector
print("=" * 90)
print("RESCUE INTERVENTION EFFECTIVENESS - SINGLE CRITICAL SHIPMENT")
print("=" * 90)
results = []
for rescue_on in [True, False]:
    for seed in SEEDS:
        result = run_case(
            seed,
            rescue_on,
        )
        results.append(result)
print()
print("=" * 90)
print("SUMMARY")
print("=" * 90)
for result in results:
    print(
        f"{'ON' if result['rescue_on'] else 'OFF':>3} "
        f"SEED={result['seed']:>3} "
        f"STEPS={result['steps']:>3} "
        f"COMPLETED={result['completed']} "
        f"EXPOSURE={result['exposure']} "
        f"EXPOSED={result['exposed']} "
        f"EXCEEDED={result['exceeded']} "
        f"MAX_TEMP={result['max_temperature']:.3f} "
        f"ACT={result['activation_step']} "
        f"REACHED={result['rescue_reached_step']} "
        f"REWARD={result['reward']:.3f}"
    )
print("=" * 90)
print("EXPERIMENT COMPLETE")
print("=" * 90)
