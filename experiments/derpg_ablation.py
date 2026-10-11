from environment.multi_agv_env import MultiAGVEnv
from environment.warehouse import Warehouse
import csv
import os
SEEDS = [1, 5, 7, 21, 42, 100]
RESULT_PATH = os.path.join("models", "final_derpg_ablation.csv")
def configure(env, seed):
    env.reset(seed=seed)
    env.warehouse.trigger_cooling_failure(1)
    risks = [0.10, 0.10, 0.95]
    urgencies = [0.10, 0.10, 0.95]
    rsets = [60.0, 60.0, 5.0]
    for i, task in enumerate(env.tasks):
        task.status = "PENDING"
        task.assigned_agv = None
        task.in_rescue = False
        task.rescue_destination = None
        task.rescue_recovered = False
        task.risk_score = risks[i]
        task.future_risk_60 = risks[i]
        task.urgency_score = urgencies[i]
        task.rset_minutes = rsets[i]
        task.priority = "HIGH" if risks[i] >= 0.80 else "NORMAL"
    for agv in env.system.agvs.values():
        agv.current_task = None
        agv.destination = None
        agv.status = "IDLE"
def run_case(seed, propagation_weight):
    original_method = Warehouse.get_all_zone_risks
    def patched_get_all_zone_risks(self, propagation_weight=0.30):
        return original_method(
            self,
            propagation_weight=propagation_weight_override,
        )
    propagation_weight_override = float(propagation_weight)
    Warehouse.get_all_zone_risks = patched_get_all_zone_risks
    try:
        env = MultiAGVEnv(
            num_agvs=3,
            max_steps=100,
            risk_aware_assignment=True,
        )
        configure(env, seed)
        initial_risks = env.warehouse.get_all_zone_risks(
            propagation_weight=propagation_weight
        )
        z1 = initial_risks.get(1, {})
        z2 = initial_risks.get(2, {})
        env._assign_tasks()
        env._start_assigned_tasks()
        total_reward = 0.0
        collisions = 0
        steps = 0
        for _ in range(100):
            agvs = sorted(
                env.system.agvs.values(),
                key=lambda a: int(a.agv_id),
            )
            actions = [
                env._get_task_directed_action(agv)
                for agv in agvs
            ]
            _, reward, terminated, truncated, info = env.step(
                actions
            )
            total_reward += float(reward)
            steps += 1
            collisions += int(
                info.get("collisions", 0)
            )
            if terminated or truncated:
                break
        completed = sum(
            1
            for task in env.tasks
            if task.is_completed
        )
        final_risks = env.warehouse.get_all_zone_risks(
            propagation_weight=propagation_weight
        )
        return {
            "seed": seed,
            "derpg": "ON" if propagation_weight > 0.0 else "OFF",
            "propagation_weight": propagation_weight,
            "initial_z1_local": float(
                z1.get("local_risk", 0.0)
            ),
            "initial_z1_neighbor": float(
                z1.get("neighbor_risk", 0.0)
            ),
            "initial_z1_dynamic": float(
                z1.get("dynamic_risk", 0.0)
            ),
            "initial_z2_dynamic": float(
                z2.get("dynamic_risk", 0.0)
            ),
            "final_z1_dynamic": float(
                final_risks.get(1, {}).get(
                    "dynamic_risk",
                    0.0,
                )
            ),
            "steps": steps,
            "completed": completed,
            "collisions": collisions,
            "reward": total_reward,
        }
    finally:
        Warehouse.get_all_zone_risks = original_method
def main():
    rows = []
    for propagation_weight in (0.30, 0.0):
        for seed in SEEDS:
            result = run_case(
                seed,
                propagation_weight,
            )
            rows.append(result)
            print(
                f"{result['derpg']:>3} "
                f"seed={seed:<4} "
                f"W={result['propagation_weight']:.2f} "
                f"Z1_LOCAL={result['initial_z1_local']:.4f} "
                f"Z1_NEIGHBOR={result['initial_z1_neighbor']:.4f} "
                f"Z1_DYNAMIC={result['initial_z1_dynamic']:.4f} "
                f"Z2_DYNAMIC={result['initial_z2_dynamic']:.4f} "
                f"STEPS={result['steps']} "
                f"COMPLETED={result['completed']} "
                f"COLLISIONS={result['collisions']} "
                f"REWARD={result['reward']:.3f}"
            )
    os.makedirs("models", exist_ok=True)
    with open(
        RESULT_PATH,
        "w",
        newline="",
        encoding="utf-8",
    ) as f:
        writer = csv.DictWriter(
            f,
            fieldnames=list(rows[0].keys()),
        )
        writer.writeheader()
        writer.writerows(rows)
    print()
    print(f"SAVED={RESULT_PATH}")
if __name__ == "__main__":
    main()
