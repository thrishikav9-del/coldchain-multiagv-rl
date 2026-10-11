
from environment.multi_agv_env import MultiAGVEnv
def print_assignments(label, env):
    print("=" * 70)
    print(label)
    print("=" * 70)
    for task in env.tasks:
        print(
            "TASK",
            task.task_id,
            "AGV=",
            task.assigned_agv,
            "RISK=",
            round(float(getattr(task, "risk_score", 0.0)), 4),
            "URGENCY=",
            round(float(getattr(task, "urgency_score", 0.0)), 4),
            "RSET=",
            round(float(getattr(task, "rset_minutes", 0.0)), 2),
        )
def make_env(risk_aware):
    env = MultiAGVEnv(
        num_agvs=3,
        max_steps=100,
        risk_aware_assignment=risk_aware,
    )
    env.reset(seed=42)
    # The current dataset contains task-level risk values already.
    # Report the actual post-reset assignment without trying to
    # reassign IN_PROGRESS tasks.
    return env
reactive = make_env(False)
proactive = make_env(True)
print_assignments(
    "REACTIVE / RISK-AGNOSTIC INITIAL ASSIGNMENT",
    reactive,
)
print_assignments(
    "PROACTIVE / RISK-AWARE INITIAL ASSIGNMENT",
    proactive,
)
print("=" * 70)
print("PRIORITIZATION COMPARISON")
print("=" * 70)
for task_id in range(1, 4):
    reactive_task = next(
        t for t in reactive.tasks
        if t.task_id == task_id
    )
    proactive_task = next(
        t for t in proactive.tasks
        if t.task_id == task_id
    )
    print(
        f"TASK {task_id}: "
        f"reactive_AGV={reactive_task.assigned_agv} "
        f"proactive_AGV={proactive_task.assigned_agv} "
        f"risk={float(getattr(proactive_task, 'risk_score', 0.0)):.4f} "
        f"urgency={float(getattr(proactive_task, 'urgency_score', 0.0)):.4f} "
        f"RSET={float(getattr(proactive_task, 'rset_minutes', 0.0)):.2f}"
    )
print("=" * 70)
print(
    "NOTE: This experiment compares the actual initial assignment "
    "created by each configuration."
)
print(
    "It does not artificially reassign already-started tasks."
)
print("=" * 70)
