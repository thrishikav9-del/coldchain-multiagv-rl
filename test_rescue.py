from environment.multi_agv_env import MultiAGVEnv
env = MultiAGVEnv(num_agvs=3, max_steps=100)
obs, info = env.reset()
print("START")
for agv_id, agv in env.system.agvs.items():
    print(
        "AGV",
        agv_id,
        "POS=",
        agv.position,
        "TARGET=",
        env._get_current_target(agv)
    )
print("\nRUNNING TASK-DIRECTED POLICY")
print("--------------------------------")
done = False
for step in range(100):
    actions = [
        env._get_task_directed_action(agv)
        for agv in env.system.agvs.values()
    ]
    obs, reward, done, truncated, info = env.step(actions)
    rescue_tasks = [
        (
            t.task_id,
            t.rescue_destination,
            t.status
        )
        for t in env.tasks
        if t.in_rescue
    ]
    if rescue_tasks:
        print(
            "STEP",
            step + 1,
            "RESCUE=",
            rescue_tasks
        )
    if info.get("pickup_tasks_this_step", 0):
        print(
            "STEP",
            step + 1,
            "PICKUPS=",
            info.get("pickup_tasks_this_step")
        )
    if info.get("completed_tasks_this_step", 0):
        print(
            "STEP",
            step + 1,
            "COMPLETED=",
            info.get("completed_tasks_this_step")
        )
    if done or truncated:
        break
print("\nFINAL")
print("--------------------------------")
for t in env.tasks:
    print(
        "TASK",
        t.task_id,
        "STATUS=",
        t.status,
        "RISK=",
        t.risk_score,
        "IN_RESCUE=",
        t.in_rescue,
        "RESCUE=",
        t.rescue_destination,
        "ORIGINAL=",
        t.original_delivery
    )
print("\nAGVS")
for agv_id, agv in env.system.agvs.items():
    print(
        "AGV",
        agv_id,
        "POSITION=",
        agv.position,
        "STATUS=",
        agv.status,
        "TASK=",
        agv.current_task,
        "DESTINATION=",
        agv.destination
    )
