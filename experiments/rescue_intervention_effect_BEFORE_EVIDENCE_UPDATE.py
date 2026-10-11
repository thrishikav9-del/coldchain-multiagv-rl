from environment.multi_agv_env import MultiAGVEnv
SEEDS = [1, 5, 7, 21, 42, 100]
AMBIENT_TEMPERATURE = 35.0
AMBIENT_HUMIDITY = 80.0
EMERGENCY_ZONE = 6
def configure(env, seed):
    env.reset(seed=seed)
    # Controlled thermal emergency:
    # Task 3 pickup is in Zone 6, so the failed zone directly
    # exposes the critical shipment after pickup.
    env.warehouse.set_external_weather(
        AMBIENT_TEMPERATURE,
        AMBIENT_HUMIDITY,
    )
    env.warehouse.trigger_cooling_failure(
        EMERGENCY_ZONE
    )
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
        activation_step = None
        rescue_destination = None
        rescue_reached_step = None
        pickup_step = None
        product_temperature_at_pickup = None
        product_temperature_at_activation = None
        product_temperature_at_rescue = None
        max_temperature = 0.0
        exposure_at_activation = None
        exposure_at_rescue = None
        previous_in_rescue = False
        previous_status = target.status
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
            current_temperature = float(
                getattr(
                    target,
                    "product_temperature",
                    4.0,
                )
            )
            max_temperature = max(
                max_temperature,
                float(
                    getattr(
                        target,
                        "max_exposure_temperature",
                        0.0,
                    )
                ),
                current_temperature,
            )
            # Detect actual pickup transition.
            if (
                pickup_step is None
                and target.status == "PICKED_UP"
            ):
                pickup_step = env.current_step
                product_temperature_at_pickup = current_temperature
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
                product_temperature_at_activation = (
                    current_temperature
                )
                exposure_at_activation = (
                    env.episode_total_exposure_steps
                )
            # Detect arrival at rescue destination.
            if (
                rescue_on
                and activation_step is not None
                and rescue_reached_step is None
                and previous_in_rescue
                and not getattr(target, "in_rescue", False)
                and rescue_destination is not None
            ):
                rescue_reached_step = env.current_step
                product_temperature_at_rescue = (
                    current_temperature
                )
                exposure_at_rescue = (
                    env.episode_total_exposure_steps
                )
            previous_in_rescue = getattr(
                target,
                "in_rescue",
                False,
            )
            previous_status = target.status
            if terminated or truncated:
                break
        final_exposure = env.episode_total_exposure_steps
        return {
            "seed": seed,
            "rescue_on": rescue_on,
            "steps": env.current_step,
            "completed": info.get("completed_tasks", 0),
            "exposure": final_exposure,
            "exposed": env.episode_tasks_with_exposure,
            "exceeded": env.episode_tasks_exposure_exceeded,
            "pickup_step": pickup_step,
            "pickup_temperature": product_temperature_at_pickup,
            "activation_step": activation_step,
            "activation_temperature": product_temperature_at_activation,
            "rescue_destination": rescue_destination,
            "rescue_reached_step": rescue_reached_step,
            "rescue_temperature": product_temperature_at_rescue,
            "max_temperature": max_temperature,
            "exposure_at_activation": exposure_at_activation,
            "exposure_at_rescue": exposure_at_rescue,
            "reward": env.episode_total_reward,
        }
    finally:
        env._select_safe_destination = original_selector
print("=" * 100)
print("CONTROLLED RESCUE INTERVENTION EFFECTIVENESS")
print(
    f"Emergency Zone={EMERGENCY_ZONE} | "
    f"Ambient={AMBIENT_TEMPERATURE}C | "
    f"Seeds={SEEDS}"
)
print("=" * 100)
results = []
for rescue_on in [True, False]:
    for seed in SEEDS:
        result = run_case(seed, rescue_on)
        results.append(result)
        print(
            f"{'ON' if rescue_on else 'OFF':>3} "
            f"SEED={seed:>3} "
            f"PICKUP={str(result['pickup_step']):>4} "
            f"PICK_T={str(None if result['pickup_temperature'] is None else round(result['pickup_temperature'], 3)):>7} "
            f"ACT={str(result['activation_step']):>4} "
            f"ACT_T={str(None if result['activation_temperature'] is None else round(result['activation_temperature'], 3)):>7} "
            f"REACHED={str(result['rescue_reached_step']):>7} "
            f"RESCUE_T={str(None if result['rescue_temperature'] is None else round(result['rescue_temperature'], 3)):>8} "
            f"MAX_T={result['max_temperature']:.3f} "
            f"EXPOSURE={result['exposure']:>4} "
            f"COMPLETED={result['completed']}"
        )
print("=" * 100)
print("SUMMARY BY CONFIGURATION")
print("=" * 100)
for rescue_on in [True, False]:
    subset = [
        r for r in results
        if r["rescue_on"] == rescue_on
    ]
    label = "RESCUE ON" if rescue_on else "RESCUE OFF"
    avg_max_temp = sum(
        r["max_temperature"] for r in subset
    ) / len(subset)
    avg_exposure = sum(
        r["exposure"] for r in subset
    ) / len(subset)
    avg_completed = sum(
        r["completed"] for r in subset
    ) / len(subset)
    reached = [
        r["rescue_reached_step"]
        for r in subset
        if r["rescue_reached_step"] is not None
    ]
    print(
        f"{label}: "
        f"AVG_MAX_TEMP={avg_max_temp:.3f} "
        f"AVG_EXPOSURE={avg_exposure:.2f} "
        f"AVG_COMPLETED={avg_completed:.2f} "
        f"RESCUE_REACHED={len(reached)}/{len(subset)}"
    )
print("=" * 100)
print("EXPERIMENT COMPLETE")
print("=" * 100)
