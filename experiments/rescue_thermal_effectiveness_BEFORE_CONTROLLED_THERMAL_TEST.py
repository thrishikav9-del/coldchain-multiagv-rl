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
    # Same sustained cooling failure in Zone 1.
    env.warehouse.trigger_cooling_failure(1)
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
        max_temperature = 0.0
        exposure_by_task = {
            task.task_id: 0
            for task in env.tasks
        }
        rescue_activations = {
            task.task_id: 0
            for task in env.tasks
        }
        rescue_destinations = {}
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
                exposure = getattr(
                    task,
                    "exposure_time",
                    None,
                )
                if exposure is not None:
                    try:
                        exposure_by_task[
                            task.task_id
                        ] = max(
                            exposure_by_task[
                                task.task_id
                            ],
                            int(exposure),
                        )
                    except Exception:
                        pass
                task_max_temperature = float(
                    getattr(
                        task,
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
            "max_temperature": max_temperature,
            "rescue_activations": rescue_activations,
            "rescue_destinations": rescue_destinations,
        }
    finally:
        env._select_safe_destination = original_selector
print("=" * 90)
print("RESCUE EFFECTIVENESS WITH THERMAL EXPOSURE COMPARISON")
print("=" * 90)
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
            f"EXPOSED={result['exposed_tasks']} "
            f"EXCEEDED={result['exceeded_tasks']} "
            f"MAX_TEMP={result['max_temperature']:.3f} "
            f"RESCUE_ACTIVATIONS={result['rescue_activations']} "
            f"RESCUE_DEST={result['rescue_destinations']}"
        )
print("=" * 90)
print("EXPERIMENT COMPLETE")
print("=" * 90)
