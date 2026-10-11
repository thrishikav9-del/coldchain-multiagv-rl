from environment.multi_agv_env import MultiAGVEnv
SEEDS = [1, 5, 7, 21, 42, 100]
def configure(env, seed):
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
    # Controlled thermal experiment:
    # Zone 1 fails completely.
    env.warehouse.trigger_cooling_failure(1)
    # Healthy zones have degraded cooling.
    for zone_id in range(2, env.warehouse.total_zones + 1):
        env.warehouse.set_zone_power(
            zone_id,
            power_available=True,
        )
        env.warehouse.set_zone_cooling(
            zone_id,
            effectiveness=0.5,
        )
    # Zone 6 is the deliberately maintained cold-safe zone.
    env.warehouse.set_zone_power(
        6,
        power_available=True,
    )
    env.warehouse.set_zone_cooling(
        6,
        effectiveness=1.0,
    )
    return env
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
        rescue_activations = {
            task.task_id: 0
            for task in env.tasks
        }
        rescue_destinations = {}
        for _ in range(100):
            actions = [
                env._get_task_directed_action(
                    env.system.agvs[agv_id]
                )
                for agv_id in sorted(env.system.agvs)
            ]
            state, reward, terminated, truncated, info = env.step(
                actions
            )
            for task in env.tasks:
                if getattr(task, "in_rescue", False):
                    rescue_activations[task.task_id] += 1
                    destination = getattr(
                        task,
                        "rescue_destination",
                        None,
                    )
                    if destination is not None:
                        rescue_destinations[
                            task.task_id
                        ] = destination
            if terminated or truncated:
                break
        task_metrics = {}
        for task in env.tasks:
            task_metrics[task.task_id] = {
                "exposure_steps": int(
                    getattr(
                        task,
                        "thermal_exposure_steps",
                        0,
                    )
                ),
                "picked_up_steps": int(
                    getattr(
                        task,
                        "picked_up_steps",
                        0,
                    )
                ),
                "max_temperature": float(
                    getattr(
                        task,
                        "max_exposure_temperature",
                        0.0,
                    )
                ),
                "final_product_temperature": float(
                    getattr(
                        task,
                        "product_temperature",
                        4.0,
                    )
                ),
                "quality_remaining": float(
                    getattr(
                        task,
                        "quality_remaining",
                        1.0,
                    )
                ),
                "thermal_damage": float(
                    getattr(
                        task,
                        "thermal_damage",
                        0.0,
                    )
                ),
                "exposure_exceeded": bool(
                    getattr(
                        task,
                        "exposure_exceeded",
                        False,
                    )
                ),
                "degradation_occurred": bool(
                    getattr(
                        task,
                        "degradation_occurred",
                        False,
                    )
                ),
                "completed": task.status == "COMPLETED",
            }
        return {
            "seed": seed,
            "rescue": "ON" if rescue_on else "OFF",
            "steps": env.current_step,
            "completed": info.get(
                "completed_tasks",
                0,
            ),
            "collisions": env.episode_collision_count,
            "reward": env.episode_total_reward,
            "exposure_steps": env.episode_total_exposure_steps,
            "exposed_tasks": env.episode_tasks_with_exposure,
            "exceeded_tasks": env.episode_tasks_exposure_exceeded,
            "rescue_activations": rescue_activations,
            "rescue_destinations": rescue_destinations,
            "task_metrics": task_metrics,
        }
    finally:
        env._select_safe_destination = original_selector
print("=" * 110)
print("CONTROLLED RESCUE THERMAL EFFECTIVENESS EXPERIMENT")
print("Zone 1 = FAILED | Other healthy zones = 0.5 cooling | Zone 6 = 1.0 cooling")
print("=" * 110)
results = []
for rescue_on in [True, False]:
    for seed in SEEDS:
        result = run_case(
            seed,
            rescue_on,
        )
        results.append(result)
        print(
            f"{result['rescue']:>3} "
            f"SEED={result['seed']:>3} "
            f"STEPS={result['steps']:>3} "
            f"COMPLETED={result['completed']} "
            f"COLLISIONS={result['collisions']} "
            f"REWARD={result['reward']:.3f} "
            f"EXPOSURE={result['exposure_steps']} "
            f"EXCEEDED={result['exceeded_tasks']} "
            f"RESCUE={result['rescue_destinations']}"
        )
        for task_id, metrics in result["task_metrics"].items():
            print(
                f"  TASK={task_id} "
                f"EXP={metrics['exposure_steps']} "
                f"MAX_TEMP={metrics['max_temperature']:.3f} "
                f"FINAL_TEMP={metrics['final_product_temperature']:.3f} "
                f"QUALITY={metrics['quality_remaining']:.6f} "
                f"DAMAGE={metrics['thermal_damage']:.6f} "
                f"EXCEEDED={metrics['exposure_exceeded']} "
                f"COMPLETED={metrics['completed']}"
            )
print("=" * 110)
print("EXPERIMENT COMPLETE")
print("=" * 110)
