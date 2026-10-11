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
    # Sustained emergency in Zone 1.
    env.warehouse.trigger_cooling_failure(1)
    env._assign_tasks()
    env._start_assigned_tasks()
    return env
def get_zone_temperature(env, position):
    try:
        zone_id = env.warehouse.get_zone_id(
            tuple(position)
        )
        temperature = float(
            env.warehouse._thermal_state[
                zone_id
            ]["temperature"]
        )
        return zone_id, temperature
    except Exception:
        return None, None
def run_case(seed, rescue_on):
    env = MultiAGVEnv(
        num_agvs=3,
        max_steps=100,
        risk_aware_assignment=True,
    )
    original_selector = env._select_safe_destination
    if not rescue_on:
        env._select_safe_destination = (
            lambda task: None
        )
    try:
        configure(env, seed)
        activation = {}
        rescue_reached = {}
        exposure_start = {}
        exposure_rescue_end = {}
        exposure_final = {}
        max_temperature = {}
        for task in env.tasks:
            activation[task.task_id] = None
            rescue_reached[task.task_id] = None
            exposure_start[task.task_id] = None
            exposure_rescue_end[task.task_id] = None
            exposure_final[task.task_id] = None
            max_temperature[task.task_id] = 0.0
        previous_rescue = {
            task.task_id: False
            for task in env.tasks
        }
        print()
        print("=" * 90)
        print(
            f"{'RESCUE ON' if rescue_on else 'RESCUE OFF'} "
            f"SEED={seed}"
        )
        print("=" * 90)
        for task in env.tasks:
            print(
                f"TASK {task.task_id} "
                f"RISK={task.risk_score:.2f} "
                f"URGENCY={task.urgency_score:.2f} "
                f"RSET={task.rset_minutes:.1f} "
                f"PICKUP={task.pickup} "
                f"DELIVERY={task.delivery}"
            )
        for step in range(1, 101):
            actions = []
            for agv_id in sorted(env.system.agvs):
                agv = env.system.agvs[agv_id]
                actions.append(
                    env._get_task_directed_action(agv)
                )
            state, reward, terminated, truncated, info = (
                env.step(actions)
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
                zone_id, temperature = (
                    get_zone_temperature(
                        env,
                        agv.position,
                    )
                )
                task_id = task.task_id
                task_max = float(
                    getattr(
                        task,
                        "max_exposure_temperature",
                        0.0,
                    )
                )
                max_temperature[task_id] = max(
                    max_temperature[task_id],
                    task_max,
                )
                current_rescue = bool(
                    getattr(
                        task,
                        "in_rescue",
                        False,
                    )
                )
                current_exposure = int(
                    getattr(
                        task,
                        "exposure_time",
                        0,
                    )
                    or 0
                )
                # Rescue activation.
                if (
                    current_rescue
                    and not previous_rescue[task_id]
                ):
                    activation[task_id] = env.current_step
                    exposure_start[task_id] = (
                        current_exposure
                    )
                    print(
                        f"ACTIVATE "
                        f"STEP={env.current_step:>3} "
                        f"TASK={task_id} "
                        f"AGV={agv_id} "
                        f"ZONE={zone_id} "
                        f"TEMP={temperature} "
                        f"EXPOSURE={current_exposure} "
                        f"RSET={task.rset_minutes:.1f} "
                        f"RESCUE_DEST="
                        f"{task.rescue_destination}"
                    )
                # Rescue destination reached.
                if (
                    previous_rescue[task_id]
                    and not current_rescue
                    and rescue_reached[task_id] is None
                ):
                    rescue_reached[task_id] = (
                        env.current_step
                    )
                    exposure_rescue_end[task_id] = (
                        current_exposure
                    )
                    print(
                        f"RESCUE_REACHED "
                        f"STEP={env.current_step:>3} "
                        f"TASK={task_id} "
                        f"AGV={agv_id} "
                        f"ZONE={zone_id} "
                        f"TEMP={temperature} "
                        f"EXPOSURE={current_exposure}"
                    )
                previous_rescue[task_id] = (
                    current_rescue
                )
                if task.is_completed:
                    exposure_final[task_id] = (
                        current_exposure
                    )
            if terminated or truncated:
                break
        print()
        print("-" * 90)
        print("TASK EXPOSURE SUMMARY")
        print("-" * 90)
        for task in env.tasks:
            task_id = task.task_id
            final_exposure = int(
                getattr(
                    task,
                    "exposure_time",
                    0,
                )
                or 0
            )
            exposure_final[task_id] = final_exposure
            if (
                exposure_start[task_id] is not None
                and exposure_rescue_end[task_id] is not None
            ):
                rescue_exposure = (
                    exposure_rescue_end[task_id]
                    - exposure_start[task_id]
                )
            else:
                rescue_exposure = 0
            if (
                exposure_start[task_id] is not None
            ):
                pre_rescue = exposure_start[task_id]
            else:
                pre_rescue = final_exposure
            post_rescue = max(
                0,
                final_exposure
                - pre_rescue
                - rescue_exposure,
            )
            print(
                f"TASK={task_id} "
                f"RISK={task.risk_score:.2f} "
                f"RSET={task.rset_minutes:.1f} "
                f"ACTIVATION={activation[task_id]} "
                f"RESCUE_REACHED={rescue_reached[task_id]} "
                f"PRE_RESCUE_EXP={pre_rescue} "
                f"RESCUE_EXP={rescue_exposure} "
                f"POST_RESCUE_EXP={post_rescue} "
                f"FINAL_EXP={final_exposure} "
                f"MAX_TEMP={max_temperature[task_id]:.3f} "
                f"EXCEEDED="
                f"{final_exposure > task.rset_minutes}"
            )
        print()
        print(
            f"FINAL "
            f"STEPS={env.current_step} "
            f"COMPLETED="
            f"{info.get('completed_tasks', 0)} "
            f"TOTAL_EXPOSURE="
            f"{env.episode_total_exposure_steps} "
            f"EXPOSED="
            f"{env.episode_tasks_with_exposure} "
            f"EXCEEDED="
            f"{env.episode_tasks_exposure_exceeded}"
        )
    finally:
        env._select_safe_destination = (
            original_selector
        )
print("=" * 90)
print("RESCUE EXPOSURE TIMELINE EXPERIMENT")
print("=" * 90)
for rescue_on in [True, False]:
    for seed in SEEDS:
        run_case(
            seed,
            rescue_on,
        )
print()
print("=" * 90)
print("EXPERIMENT COMPLETE")
print("=" * 90)
