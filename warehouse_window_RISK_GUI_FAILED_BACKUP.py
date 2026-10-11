"""
Cold-Chain Multi-AGV Visual Demonstration
Shows the actual validated environment state:
- 20x20 warehouse
- 16 thermal zones
- 3 AGVs
- pickup/delivery tasks
- live temperature
- dynamic risk
- cooling status
- urgency / RSET
- rescue state
- collision count
- DQN-assisted collision-aware operational controller
"""
import itertools
import tkinter as tk
from tkinter import ttk
import numpy as np
from environment.multi_agv_env import MultiAGVEnv
from algorithms.dqn_agent import DQNAgent
# ==========================================================
# CONFIGURATION
# ==========================================================
NUM_AGVS = 3
ACTION_SIZE = 5
MAX_STEPS = 300
MODEL_PATH = "models\dqn_ep2000_test.pt"
GRID_SIZE = 20
CELL_SIZE = 28
WAREHOUSE_PIXELS = GRID_SIZE * CELL_SIZE
PANEL_WIDTH = 430
WINDOW_WIDTH = WAREHOUSE_PIXELS + PANEL_WIDTH + 30
WINDOW_HEIGHT = WAREHOUSE_PIXELS + 95
UPDATE_DELAY = 120
THERMAL_EMERGENCY_STEP = 8
EMERGENCY_ZONE = 1
# ==========================================================
# COLORS
# ==========================================================
AGV_COLORS = [
    "#1565C0",
    "#2E7D32",
    "#EF6C00",
]
PICKUP_COLOR = "#FFD54F"
DELIVERY_COLOR = "#AB47BC"
COMPLETED_COLOR = "#66BB6A"
RISK_LOW = "#C8E6C9"
RISK_MODERATE = "#FFF59D"
RISK_HIGH = "#FFCC80"
RISK_CRITICAL = "#EF9A9A"
WAREHOUSE_COLOR = "#FAFAFA"
GRID_COLOR = "#BDBDBD"
# ==========================================================
# ENVIRONMENT
# ==========================================================
env = MultiAGVEnv(
    num_agvs=NUM_AGVS,
    max_steps=MAX_STEPS,
    risk_aware_assignment=True,
)
STATE_SIZE = int(env.observation_space.shape[0])
# ==========================================================
# DQN
# ==========================================================
agent = DQNAgent(
    state_size=STATE_SIZE,
    num_agvs=NUM_AGVS,
    action_size=ACTION_SIZE,
)
# Use the same active environment attachment as the validated evaluator.
agent._evaluation_env = env
agent.load(MODEL_PATH)
agent.evaluation_mode()
# ==========================================================
# COLLISION-AWARE TARGET CONTROLLER
# Same operational logic used by final evaluation.
# ==========================================================
def select_hybrid_actions(environment, dqn_actions):
    system = getattr(environment, "system", None)
    if system is None:
        return np.zeros(NUM_AGVS, dtype=np.int64)
    agvs = getattr(system, "agvs", {})
    candidates = []
    for agv_id in range(1, NUM_AGVS + 1):
        agv = agvs.get(agv_id)
        if agv is None:
            candidates.append([0])
            continue
        target = environment._get_current_target(agv)
        if target is None:
            candidates.append([0])
            continue
        x, y = map(int, agv.position)
        gx, gy = map(int, target)
        options = []
        if x < gx:
            options.append(1)
        elif x > gx:
            options.append(2)
        if y < gy:
            options.append(3)
        elif y > gy:
            options.append(4)
        options.append(0)
        candidates.append(list(dict.fromkeys(options)))
    best = None
    best_score = None
    for joint in itertools.product(*candidates):
        candidate = np.asarray(
            joint,
            dtype=np.int64,
        )
        try:
            if not agent.check_collision_free(candidate):
                continue
        except Exception:
            continue
        total_distance = 0
        movement_count = 0
        valid = True
        for agv_id in range(1, NUM_AGVS + 1):
            agv = agvs.get(agv_id)
            if agv is None:
                valid = False
                break
            target = environment._get_current_target(agv)
            if target is None:
                continue
            current = tuple(map(int, agv.position))
            proposed = tuple(
                map(
                    int,
                    environment._get_target_position(
                        agv,
                        int(candidate[agv_id - 1]),
                    ),
                )
            )
            before = (
                abs(current[0] - int(target[0]))
                + abs(current[1] - int(target[1]))
            )
            after = (
                abs(proposed[0] - int(target[0]))
                + abs(proposed[1] - int(target[1]))
            )
            if before > 0 and after > before:
                valid = False
                break
            total_distance += after
            if proposed != current:
                movement_count += 1
        if not valid:
            continue
        score = (total_distance, -movement_count)
        if best_score is None or score < best_score:
            best_score = score
            best = candidate
    if best is not None:
        return best
    return np.zeros(NUM_AGVS, dtype=np.int64)
# ==========================================================
# GUI
# ==========================================================
class WarehouseWindow:
    def __init__(self, root):
        self.root = root
        self.root.title(
            "Risk-Aware Cold-Chain Multi-AGV Warehouse"
        )
        self.root.geometry(
            f"{WINDOW_WIDTH}x{WINDOW_HEIGHT}"
        )
        self.root.resizable(False, False)
        self.step_number = 0
        self.total_reward = 0.0
        self.collisions = 0
        self.completed = 0
        self.pickups = 0
        self.running = False
        self.emergency_triggered = False
        self.reset_environment()
        # --------------------------------------------------
        # MAIN FRAME
        # --------------------------------------------------
        self.main_frame = ttk.Frame(self.root)
        self.main_frame.pack(
            fill="both",
            expand=True,
        )
        # --------------------------------------------------
        # WAREHOUSE CANVAS
        # --------------------------------------------------
        self.canvas = tk.Canvas(
            self.main_frame,
            width=WAREHOUSE_PIXELS,
            height=WAREHOUSE_PIXELS,
            bg=WAREHOUSE_COLOR,
            highlightthickness=1,
        )
        self.canvas.pack(
            side="left",
            padx=12,
            pady=12,
        )
        # --------------------------------------------------
        # RIGHT PANEL
        # --------------------------------------------------
        self.info_frame = ttk.Frame(
            self.main_frame,
            width=PANEL_WIDTH,
        )
        self.info_frame.pack(
            side="right",
            fill="y",
            padx=8,
            pady=12,
        )
        ttk.Label(
            self.info_frame,
            text="RISK-AWARE COLD-CHAIN",
            font=("Arial", 16, "bold"),
        ).pack(pady=(3, 0))
        ttk.Label(
            self.info_frame,
            text="Multi-AGV Visual Demonstration",
            font=("Arial", 10),
        ).pack(pady=(0, 8))
        self.mode_label = ttk.Label(
            self.info_frame,
            text="MODE: NORMAL",
            font=("Arial", 11, "bold"),
        )
        self.mode_label.pack(
            anchor="w",
            pady=3,
        )
        self.quick_controls = ttk.Frame(
            self.info_frame
        )
        self.quick_controls.pack(
            fill="x",
            pady=(2, 6),
        )
        self.quick_start_button = ttk.Button(
            self.quick_controls,
            text="START DEMO",
            command=self.start,
        )
        self.quick_start_button.pack(
            side="left",
            expand=True,
            fill="x",
            padx=(0, 3),
        )
        self.quick_reset_button = ttk.Button(
            self.quick_controls,
            text="RESET",
            command=self.reset,
        )
        self.quick_reset_button.pack(
            side="right",
            expand=True,
            fill="x",
            padx=(3, 0),
        )
        self.start_button = self.quick_start_button
        self.reset_button = self.quick_reset_button
        self.step_label = ttk.Label(
            self.info_frame,
            text="Step: 0",
        )
        self.reward_label = ttk.Label(
            self.info_frame,
            text="Reward: 0.00",
        )
        self.pickup_label = ttk.Label(
            self.info_frame,
            text="Pickups: 0",
        )
        self.completed_label = ttk.Label(
            self.info_frame,
            text="Completed: 0",
        )
        self.collision_label = ttk.Label(
            self.info_frame,
            text="Collisions: 0",
        )
        for label in (
            self.step_label,
            self.reward_label,
            self.pickup_label,
            self.completed_label,
            self.collision_label,
        ):
            label.pack(
                anchor="w",
                pady=2,
            )
        ttk.Separator(
            self.info_frame,
            orient="horizontal",
        ).pack(
            fill="x",
            pady=7,
        )
        # --------------------------------------------------
        # EXTERNAL WEATHER
        # --------------------------------------------------
        ttk.Label(
            self.info_frame,
            text="EXTERNAL ENVIRONMENT",
            font=("Arial", 11, "bold"),
        ).pack(
            anchor="w",
            pady=(0, 3),
        )
        self.weather_label = ttk.Label(
            self.info_frame,
            text="Weather: --",
        )
        self.weather_label.pack(
            anchor="w",
            pady=2,
        )
        self.emergency_label = ttk.Label(
            self.info_frame,
            text="Thermal emergency: NOT TRIGGERED",
            wraplength=400,
        )
        self.emergency_label.pack(
            anchor="w",
            pady=2,
        )
        ttk.Separator(
            self.info_frame,
            orient="horizontal",
        ).pack(
            fill="x",
            pady=7,
        )
        # --------------------------------------------------
        # AGV STATUS
        # --------------------------------------------------
        ttk.Label(
            self.info_frame,
            text="AGV STATUS",
            font=("Arial", 11, "bold"),
        ).pack(
            anchor="w",
            pady=(0, 3),
        )
        self.agv_labels = []
        for i in range(NUM_AGVS):
            label = ttk.Label(
                self.info_frame,
                text=f"AGV {i + 1}: --",
                wraplength=400,
            )
            label.pack(
                anchor="w",
                pady=2,
            )
            self.agv_labels.append(label)
        ttk.Separator(
            self.info_frame,
            orient="horizontal",
        ).pack(
            fill="x",
            pady=7,
        )
        # --------------------------------------------------
        # CRITICAL TASK / RISK
        # --------------------------------------------------
        ttk.Label(
            self.info_frame,
            text="COLD-CHAIN RISK MONITOR",
            font=("Arial", 11, "bold"),
        ).pack(
            anchor="w",
            pady=(0, 3),
        )
        self.risk_text = tk.Text(
            self.info_frame,
            width=50,
            height=8,
            font=("Consolas", 8),
            state="disabled",
        )
        self.risk_text.pack(
            fill="x",
        )
        ttk.Separator(
            self.info_frame,
            orient="horizontal",
        ).pack(
            fill="x",
            pady=7,
        )
        # --------------------------------------------------
        # TASK STATUS
        # --------------------------------------------------
        ttk.Label(
            self.info_frame,
            text="TASK STATUS",
            font=("Arial", 11, "bold"),
        ).pack(
            anchor="w",
            pady=(0, 3),
        )
        self.task_text = tk.Text(
            self.info_frame,
            width=50,
            height=8,
            font=("Consolas", 8),
            state="disabled",
        )
        self.task_text.pack(
            fill="x",
        )
        # --------------------------------------------------
        # BUTTONS
        # --------------------------------------------------
        ttk.Separator(
            self.info_frame,
            orient="horizontal",
        ).pack(
            fill="x",
            pady=4,
        )
        button_frame = ttk.Frame(
            self.info_frame
        )
        button_frame.pack(
            fill="x",
            pady=(0, 6),
        )
        self.start_button = ttk.Button(
            button_frame,
            text="START DEMO",
            command=self.start,
        )
        self.start_button.pack(
            side="left",
            expand=True,
            fill="x",
            padx=(0, 3),
        )
        self.reset_button = ttk.Button(
            button_frame,
            text="RESET",
            command=self.reset,
        )
        self.reset_button.pack(
            side="right",
            expand=True,
            fill="x",
            padx=(3, 0),
        )
        ttk.Label(
            self.info_frame,
            text=(
                "Legend: Green=low risk | Yellow=moderate | "
                "Orange=high | Red=critical"
            ),
            font=("Arial", 8),
            wraplength=400,
        ).pack(
            anchor="w",
            pady=(3, 0),
        )
        self.draw_warehouse()
        self.update_information()
    # ======================================================
    # RESET
    # ======================================================
    def reset_environment(self):
        result = env.reset(seed=42)
        if isinstance(result, tuple):
            self.state = result[0]
        else:
            self.state = result
        self.state = np.asarray(
            self.state,
            dtype=np.float32,
        )
    def reset(self):
        self.running = False
        self.start_button.config(
            state="normal"
        )
        self.step_number = 0
        self.total_reward = 0.0
        self.collisions = 0
        self.completed = 0
        self.pickups = 0
        self.emergency_triggered = False
        self.reset_environment()
        self.mode_label.config(
            text="MODE: NORMAL"
        )
        self.emergency_label.config(
            text="Thermal emergency: NOT TRIGGERED"
        )
        self.draw_warehouse()
        self.update_information()
    # ======================================================
    # ZONE HELPERS
    # ======================================================
    def zone_id_from_position(self, position):
        x, y = map(int, position)
        col = min(max(x // 5, 0), 3)
        row = min(max(y // 5, 0), 3)
        return row * 4 + col + 1
    def risk_fill(self, risk):
        risk = float(risk)
        if risk < 0.30:
            return RISK_LOW
        if risk < 0.60:
            return RISK_MODERATE
        if risk < 0.80:
            return RISK_HIGH
        return RISK_CRITICAL
    # ======================================================
    # WAREHOUSE DRAWING
    # ======================================================
    def draw_warehouse(self):
        self.canvas.delete("all")
        risks = env.warehouse.get_all_zone_risks()
        thermal = getattr(
            env.warehouse,
            "_thermal_state",
            {},
        )
        cooling = env.warehouse.get_all_cooling_status()
        # --------------------------------------------------
        # GRID + THERMAL ZONES
        # --------------------------------------------------
        for zone_id in range(1, 17):
            row = (zone_id - 1) // 4
            col = (zone_id - 1) % 4
            x1 = col * 5 * CELL_SIZE
            y1 = row * 5 * CELL_SIZE
            x2 = x1 + 5 * CELL_SIZE
            y2 = y1 + 5 * CELL_SIZE
            zone_data = risks.get(zone_id, {})
            risk = float(
                zone_data.get(
                    "dynamic_risk",
                    0.0,
                )
            )
            fill = self.risk_fill(risk)
            self.canvas.create_rectangle(
                x1,
                y1,
                x2,
                y2,
                fill=fill,
                outline="#777777",
                width=2,
            )
            temp = float(
                thermal.get(
                    zone_id,
                    {},
                ).get(
                    "temperature",
                    0.0,
                )
            )
            cooling_status = str(
                cooling.get(
                    zone_id,
                    {},
                ).get(
                    "status",
                    "UNKNOWN",
                )
            )
            self.canvas.create_text(
                x1 + 5,
                y1 + 5,
                text=f"Z{zone_id}",
                anchor="nw",
                font=("Arial", 8, "bold"),
            )
            self.canvas.create_text(
                x1 + 5,
                y1 + 20,
                text=f"{temp:.1f} C",
                anchor="nw",
                font=("Arial", 8),
            )
            self.canvas.create_text(
                x1 + 5,
                y1 + 35,
                text=f"R={risk:.2f}",
                anchor="nw",
                font=("Arial", 8),
            )
            self.canvas.create_text(
                x1 + 5,
                y2 - 5,
                text=cooling_status,
                anchor="sw",
                font=("Arial", 7),
            )
        # --------------------------------------------------
        # CELL GRID
        # --------------------------------------------------
        for x in range(GRID_SIZE + 1):
            px = x * CELL_SIZE
            self.canvas.create_line(
                px,
                0,
                px,
                WAREHOUSE_PIXELS,
                fill=GRID_COLOR,
            )
        for y in range(GRID_SIZE + 1):
            py = y * CELL_SIZE
            self.canvas.create_line(
                0,
                py,
                WAREHOUSE_PIXELS,
                py,
                fill=GRID_COLOR,
            )
        # --------------------------------------------------
        # TASK LOCATIONS
        # --------------------------------------------------
        for task in env.tasks:
            pickup_x, pickup_y = task.pickup
            delivery_x, delivery_y = task.delivery
            px1 = pickup_x * CELL_SIZE + 5
            py1 = pickup_y * CELL_SIZE + 5
            px2 = px1 + CELL_SIZE - 10
            py2 = py1 + CELL_SIZE - 10
            self.canvas.create_rectangle(
                px1,
                py1,
                px2,
                py2,
                fill=PICKUP_COLOR,
                outline="black",
                width=2,
            )
            self.canvas.create_text(
                pickup_x * CELL_SIZE + CELL_SIZE / 2,
                pickup_y * CELL_SIZE + CELL_SIZE / 2,
                text=f"P{task.task_id}",
                font=("Arial", 7, "bold"),
            )
            dx1 = delivery_x * CELL_SIZE + 5
            dy1 = delivery_y * CELL_SIZE + 5
            dx2 = dx1 + CELL_SIZE - 10
            dy2 = dy1 + CELL_SIZE - 10
            delivery_fill = (
                COMPLETED_COLOR
                if str(task.status).upper()
                in ("COMPLETED", "DELIVERED")
                else DELIVERY_COLOR
            )
            self.canvas.create_rectangle(
                dx1,
                dy1,
                dx2,
                dy2,
                fill=delivery_fill,
                outline="black",
                width=2,
            )
            self.canvas.create_text(
                delivery_x * CELL_SIZE + CELL_SIZE / 2,
                delivery_y * CELL_SIZE + CELL_SIZE / 2,
                text=f"D{task.task_id}",
                font=("Arial", 7, "bold"),
            )
            # Rescue destination
            rescue = getattr(
                task,
                "rescue_destination",
                None,
            )
            if rescue is not None:
                rx, ry = rescue
                self.canvas.create_oval(
                    rx * CELL_SIZE + 5,
                    ry * CELL_SIZE + 5,
                    rx * CELL_SIZE + CELL_SIZE - 5,
                    ry * CELL_SIZE + CELL_SIZE - 5,
                    outline="#D32F2F",
                    width=3,
                )
                self.canvas.create_text(
                    rx * CELL_SIZE + CELL_SIZE / 2,
                    ry * CELL_SIZE + CELL_SIZE / 2,
                    text="SAFE",
                    font=("Arial", 6, "bold"),
                )
        # --------------------------------------------------
        # AGVS
        # --------------------------------------------------
        for i in range(1, NUM_AGVS + 1):
            agv = env.system.agvs[i]
            x, y = map(int, agv.position)
            cx = x * CELL_SIZE + CELL_SIZE / 2
            cy = y * CELL_SIZE + CELL_SIZE / 2
            radius = 10
            self.canvas.create_oval(
                cx - radius,
                cy - radius,
                cx + radius,
                cy + radius,
                fill=AGV_COLORS[i - 1],
                outline="black",
                width=2,
            )
            self.canvas.create_text(
                cx,
                cy,
                text=str(i),
                fill="white",
                font=("Arial", 8, "bold"),
            )
    # ======================================================
    # INFORMATION UPDATE
    # ======================================================
    def update_information(self):
        self.step_label.config(
            text=f"Step: {self.step_number}"
        )
        self.reward_label.config(
            text=f"Reward: {self.total_reward:.2f}"
        )
        self.pickup_label.config(
            text=f"Pickups: {self.pickups}"
        )
        self.completed_label.config(
            text=f"Completed: {self.completed}"
        )
        self.collision_label.config(
            text=f"Collisions: {self.collisions}"
        )
        weather = env.warehouse.get_external_weather()
        self.weather_label.config(
            text=(
                f"Outdoor temperature: "
                f"{float(weather.get('temperature', 0)):.1f} C    "
                f"Humidity: "
                f"{float(weather.get('humidity', 0)):.1f}%"
            )
        )
        # --------------------------------------------------
        # AGVS
        # --------------------------------------------------
        for i in range(NUM_AGVS):
            agv = env.system.agvs[i + 1]
            task_id = "--"
            for task in env.tasks:
                if task.assigned_agv == i + 1:
                    if str(task.status).upper() not in (
                        "COMPLETED",
                        "DELIVERED",
                    ):
                        task_id = str(task.task_id)
                        break
            zone_id = self.zone_id_from_position(
                agv.position
            )
            self.agv_labels[i].config(
                text=(
                    f"AGV {i + 1}: "
                    f"pos={agv.position}  "
                    f"zone={zone_id}  "
                    f"status={agv.status}  "
                    f"task={task_id}"
                )
            )
        # --------------------------------------------------
        # RISK MONITOR
        # --------------------------------------------------
        risks = env.warehouse.get_all_zone_risks()
        thermal = getattr(
            env.warehouse,
            "_thermal_state",
            {},
        )
        cooling = env.warehouse.get_all_cooling_status()
        self.risk_text.config(state="normal")
        self.risk_text.delete("1.0", "end")
        for zone_id in range(1, 17):
            r = float(
                risks.get(
                    zone_id,
                    {},
                ).get(
                    "dynamic_risk",
                    0.0,
                )
            )
            t = float(
                thermal.get(
                    zone_id,
                    {},
                ).get(
                    "temperature",
                    0.0,
                )
            )
            status = str(
                cooling.get(
                    zone_id,
                    {},
                ).get(
                    "status",
                    "UNKNOWN",
                )
            )
            if r >= 0.80:
                level = "CRITICAL"
            elif r >= 0.60:
                level = "HIGH"
            elif r >= 0.30:
                level = "MODERATE"
            else:
                level = "LOW"
            self.risk_text.insert(
                "end",
                (
                    f"Z{zone_id:02d}  "
                    f"T={t:5.1f}C  "
                    f"Risk={r:.2f}  "
                    f"{level:<8} "
                    f"{status}\n"
                ),
            )
        self.risk_text.config(state="disabled")
        # --------------------------------------------------
        # TASK MONITOR
        # --------------------------------------------------
        self.task_text.config(state="normal")
        self.task_text.delete("1.0", "end")
        for task in env.tasks:
            risk = float(
                getattr(task, "risk_score", 0.0)
            )
            urgency = float(
                getattr(task, "urgency_score", 0.0)
            )
            rset = float(
                getattr(task, "rset_minutes", 0.0)
            )
            rescue = bool(
                getattr(task, "in_rescue", False)
            )
            rescue_destination = getattr(
                task,
                "rescue_destination",
                None,
            )
            self.task_text.insert(
                "end",
                (
                    f"Task {task.task_id} | "
                    f"{task.status} | "
                    f"AGV={task.assigned_agv}\n"
                    f"  Risk={risk:.2f} "
                    f"Urgency={urgency:.2f} "
                    f"RSET={rset:.1f} min\n"
                    f"  Rescue={rescue} "
                    f"Destination={rescue_destination}\n"
                ),
            )
        self.task_text.config(state="disabled")
    # ======================================================
    # START
    # ======================================================
    def start(self):
        if self.running:
            return
        self.running = True
        self.start_button.config(
            state="disabled"
        )
        self.run_step()
    # ======================================================
    # STEP
    # ======================================================
    def run_step(self):
        if not self.running:
            return
        # --------------------------------------------------
        # Trigger controlled thermal emergency.
        # Zone 1 contains Pickup 1 at (3,3).
        # --------------------------------------------------
        if (
            not self.emergency_triggered
            and self.step_number >= THERMAL_EMERGENCY_STEP
        ):
            try:
                env.warehouse.trigger_cooling_failure(
                    EMERGENCY_ZONE
                )
                self.emergency_triggered = True
                self.mode_label.config(
                    text=(
                        "MODE: THERMAL EMERGENCY"
                    )
                )
                self.emergency_label.config(
                    text=(
                        f"Cooling failure triggered "
                        f"in Zone {EMERGENCY_ZONE}"
                    )
                )
            except Exception as exc:
                self.emergency_label.config(
                    text=f"Emergency trigger error: {exc}"
                )
        # --------------------------------------------------
        # DQN proposal
        # --------------------------------------------------
        dqn_actions = agent.select_action(
            self.state,
            training=False,
        )
        dqn_actions = np.asarray(
            dqn_actions,
            dtype=np.int64,
        ).reshape(NUM_AGVS)
        # --------------------------------------------------
        # Validated operational controller
        # --------------------------------------------------
        actions = select_hybrid_actions(
            env,
            dqn_actions,
        )
        result = env.step(actions)
        (
            next_state,
            reward,
            terminated,
            truncated,
            info,
        ) = result
        self.state = np.asarray(
            next_state,
            dtype=np.float32,
        )
        self.step_number += 1
        self.total_reward += float(reward)
        if isinstance(info, dict):
            self.pickups += int(
                info.get(
                    "pickup_tasks_this_step",
                    0,
                )
            )
            self.completed += int(
                info.get(
                    "completed_tasks_this_step",
                    0,
                )
            )
            self.collisions += int(
                info.get(
                    "collision_count",
                    0,
                )
            )
        self.draw_warehouse()
        self.update_information()
        # Highlight rescue status
        rescue_active = any(
            bool(getattr(task, "in_rescue", False))
            for task in env.tasks
        )
        if rescue_active:
            self.mode_label.config(
                text="MODE: ACTIVE PRODUCT RESCUE"
            )
        elif self.emergency_triggered:
            self.mode_label.config(
                text="MODE: THERMAL EMERGENCY"
            )
        if bool(terminated) or bool(truncated):
            self.running = False
            self.start_button.config(
                state="normal"
            )
            self.mode_label.config(
                text="MODE: EPISODE COMPLETE"
            )
            return
        self.root.after(
            UPDATE_DELAY,
            self.run_step,
        )
# ==========================================================
# MAIN
# ==========================================================
if __name__ == "__main__":
    root = tk.Tk()
    app = WarehouseWindow(root)
    root.mainloop()




