
from environment.multi_agv_env import MultiAGVEnv
def prepare(env):
    env.reset(seed=42)
    # Create a controlled risk gradient at the task level.
    #
    # Task 1 = LOW
    # Task 2 = HIGH
    # Task 3 = LOW
    env.tasks[0].risk_score = 0.10
    env.tasks[0].urgency_score = 0.10
    env.tasks[1].risk_score = 0.95
    env.tasks[1].urgency_score = 0.95
    env.tasks[2].risk_score = 0.10
    env.tasks[2].urgency_score = 0.10
    env.tasks[0].future_risk_60 = 0.10
    env.tasks[1].future_risk_60 = 0.95
    env.tasks[2].future_risk_60 = 0.10
    env.tasks[0].rset_minutes = 60.0
    env.tasks[1].rset_minutes = 5.0
    env.tasks[2].rset_minutes = 60.0
    # Force tasks back to a pending state for controlled
    # assignment comparison.
    for task in env.tasks:
        task.assigned_agv = None
        task.status = "PENDING"
    for agv in env.system.agvs.values():
        agv.current_task = None
        agv.status = "IDLE"
    return env
reactive = prepare(
    MultiAGVEnv(
        num_agvs=3,
        max_steps=100,
        risk_aware_assignment=False,
    )
)
proactive = prepare(
    MultiAGVEnv(
        num_agvs=3,
        max_steps=100,
        risk_aware_assignment=True,
    )
)
# Perform controlled assignment.
reactive._assign_tasks()
proactive._assign_tasks()
print("=" * 70)
print("CONTROLLED DIFFERENTIAL-RISK EXPERIMENT")
print("=" * 70)
for task_id in range(1, 4):
    rt = next(
        t for t in reactive.tasks
        if t.task_id == task_id
    )
    pt = next(
        t for t in proactive.tasks
        if t.task_id == task_id
    )
    print(
        f"TASK {task_id}: "
        f"reactive_AGV={rt.assigned_agv} "
        f"proactive_AGV={pt.assigned_agv} "
        f"risk={pt.risk_score:.2f} "
        f"urgency={pt.urgency_score:.2f} "
        f"RSET={pt.rset_minutes:.1f}"
    )
critical = next(
    t for t in proactive.tasks
    if t.task_id == 2
)
print("=" * 70)
print(
    "CRITICAL TASK = TASK 2"
)
print(
    "PROACTIVE ASSIGNED AGV =",
    critical.assigned_agv,
)
print(
    "CRITICAL RISK =",
    critical.risk_score,
)
print(
    "CRITICAL URGENCY =",
    critical.urgency_score,
)
print(
    "CRITICAL RSET =",
    critical.rset_minutes,
)
print("=" * 70)
