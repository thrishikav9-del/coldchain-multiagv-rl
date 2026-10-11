"""
FINAL PRESENTATION VISUAL DEMO
Risk-Aware Multi-AGV Cold-Chain Warehouse
Presentation layer over the existing validated MultiAGVEnv.
Validated operational execution:
    Multi-AGV task-directed controller
    -> 3 pickups
    -> 3 completions
    -> 0 collisions
The trained DQN model is loaded and queried for the RL component,
while the stable task-directed operational controller is used for
the visual demonstration.
This file does not modify the environment, risk engine, DQN model,
Q-learning implementation, or experiment results.
"""
import tkinter as tk
from tkinter import ttk
import itertools
from collections import deque
import numpy as np
from environment.multi_agv_env import MultiAGVEnv
from algorithms.dqn_agent import DQNAgent
# ==========================================================
# CONFIGURATION
# ==========================================================
NUM_AGVS = 3
MAX_STEPS = 300
MODEL_PATH = r"models\dqn_ep2000_test.pt"
GRID_SIZE = 20
CELL_SIZE = 27
WAREHOUSE_PIXELS = GRID_SIZE * CELL_SIZE
WINDOW_WIDTH = 1160
WINDOW_HEIGHT = 720
UPDATE_DELAY = 75
EMERGENCY_ZONE = 1
EMERGENCY_TRIGGER_STEP = 4
TRAIL_LENGTH = 100
# ==========================================================
# COLORS
# ==========================================================
AGV_COLORS = [
    "#1565C0",
    "#2E7D32",
    "#EF6C00",
]
RISK_LOW = "#C8E6C9"
RISK_MODERATE = "#FFF59D"
RISK_HIGH = "#FFCC80"
RISK_CRITICAL = "#EF9A9A"
ZONE_BORDER = "#555555"
PICKUP_COLOR = "#FFD54F"
DELIVERY_COLOR = "#AB47BC"
COMPLETED_COLOR = "#66BB6A"
RESCUE_COLOR = "#D32F2F"
WAREHOUSE_COLOR = "#FAFAFA"
GRID_COLOR = "#C8C8C8"
# ==========================================================
# ENVIRONMENT
# ==========================================================
env = MultiAGVEnv(
    num_agvs=NUM_AGVS,
    max_steps=MAX_STEPS,
    risk_aware_assignment=True,
)
STATE_SIZE = int(
    env.observation_space.shape[0]
)
# ==========================================================
# DQN
# ==========================================================
agent = DQNAgent(
    state_size=STATE_SIZE,
    num_agvs=NUM_AGVS,
    action_size=5,
)
agent._evaluation_env = env
agent.load(MODEL_PATH)
agent.evaluation_mode()
# ==========================================================
# HELPERS
# ==========================================================
def zone_from_position(position):
    x, y = map(int, position)
    col = min(
        max(x // 5, 0),
        3,
    )
    row = min(
        max(y // 5, 0),
        3,
    )
    return row * 4 + col + 1
def risk_level(risk):
    risk = float(risk)
    if risk >= 0.80:
        return "CRITICAL"
    if risk >= 0.60:
        return "HIGH"
    if risk >= 0.30:
        return "MODERATE"
    return "LOW"
def risk_fill(risk):
    risk = float(risk)
    if risk >= 0.80:
        return RISK_CRITICAL
    if risk >= 0.60:
        return RISK_HIGH
    if risk >= 0.30:
        return RISK_MODERATE
    return RISK_LOW
def action_name(action):
    names = {
        0: "STAY",
        1: "RIGHT",
        2: "LEFT",
        3: "UP",
        4: "DOWN",
    }
    return names.get(
        int(action),
        "?",
    )
# ==========================================================
# GUI
# ==========================================================
class WarehouseWindow:
    def __init__(self, root):
        self.root = root
        self.root.title(
            "Risk-Aware Cold-Chain Multi-AGV Warehouse | Final Demonstration"
        )
        self.root.geometry(
            f"{WINDOW_WIDTH}x{WINDOW_HEIGHT}"
        )
        self.root.resizable(
            False,
            False,
        )
        # --------------------------------------------------
        # Runtime state
        # --------------------------------------------------
        self.step_number = 0
        self.total_reward = 0.0
        self.pickups = 0
        self.completed = 0
        self.collisions = 0
        self.running = False
        self.finished = False
        self.emergency_triggered = False
        self.rescue_seen = False
        self.dqn_actions = np.zeros(
            NUM_AGVS,
            dtype=np.int64,
        )
        self.last_actions = np.zeros(
            NUM_AGVS,
            dtype=np.int64,
        )
        self.trails = {
            1: deque(maxlen=TRAIL_LENGTH),
            2: deque(maxlen=TRAIL_LENGTH),
            3: deque(maxlen=TRAIL_LENGTH),
        }
        self.event_history = []
        self.reset_environment()
        self.build_interface()
        self.draw_warehouse()
        self.update_information()
    # ======================================================
    # INTERFACE
    # ======================================================
    def build_interface(self):
        # ==================================================
        # FIXED TOP CONTROL BAR
        # ==================================================
        self.control_bar = ttk.Frame(
            self.root
        )
        self.control_bar.pack(
            side="top",
            fill="x",
            padx=10,
            pady=(8, 4),
        )
        self.start_button = ttk.Button(
            self.control_bar,
            text="▶ START DEMO",
            command=self.start,
        )
        self.start_button.pack(
            side="left",
            padx=(0, 6),
        )
        self.reset_button = ttk.Button(
            self.control_bar,
            text="↻ RESET",
            command=self.reset,
        )
        self.reset_button.pack(
            side="left",
        )
        self.control_status = ttk.Label(
            self.control_bar,
            text="READY",
            font=("Arial", 10, "bold"),
        )
        self.control_status.pack(
            side="right",
            padx=8,
        )
        # ==================================================
        # MAIN INTERFACE
        # ==================================================
        self.main_frame = ttk.Frame(
            self.root
        )
        self.main_frame.pack(
            fill="both",
            expand=True,
        )
        # --------------------------------------------------
        # LEFT WAREHOUSE
        # --------------------------------------------------
        self.canvas = tk.Canvas(
            self.main_frame,
            width=WAREHOUSE_PIXELS,
            height=WAREHOUSE_PIXELS,
            bg=WAREHOUSE_COLOR,
            highlightthickness=1,
            highlightbackground="#333333",
        )
        self.canvas.pack(
            side="left",
            padx=10,
            pady=10,
            anchor="n",
        )
        # --------------------------------------------------
        # RIGHT PANEL
        # --------------------------------------------------
        self.right_frame = ttk.Frame(
            self.main_frame,
            width=510,
        )
        self.right_frame.pack(
            side="right",
            fill="both",
            expand=True,
            padx=(0, 10),
            pady=10,
        )
        # --------------------------------------------------
        # TITLE
        # --------------------------------------------------
        ttk.Label(
            self.right_frame,
            text="RISK-AWARE COLD-CHAIN",
            font=("Arial", 17, "bold"),
        ).pack(
            anchor="w",
        )
        ttk.Label(
            self.right_frame,
            text="Multi-AGV Emergency Demonstration",
            font=("Arial", 10),
        ).pack(
            anchor="w",
            pady=(0, 6),
        )
        # --------------------------------------------------
        # MODE
        # --------------------------------------------------
        self.mode_label = ttk.Label(
            self.right_frame,
            text="MODE: NORMAL OPERATION",
            font=("Arial", 12, "bold"),
        )
        self.mode_label.pack(
            anchor="w",
            pady=(0, 5),
        )
        # --------------------------------------------------
        # BUTTONS ALWAYS VISIBLE
        # --------------------------------------------------
        controls = ttk.Frame(
            self.right_frame
        )
        controls.pack(
            fill="x",
            pady=(0, 8),
        )
        self.start_button = ttk.Button(
            controls,
            text="START DEMO",
            command=self.start,
        )
        self.start_button.pack(
            side="left",
            fill="x",
            expand=True,
            padx=(0, 4),
        )
        self.reset_button = ttk.Button(
            controls,
            text="RESET",
            command=self.reset,
        )
        self.reset_button.pack(
            side="left",
            fill="x",
            expand=True,
            padx=(4, 0),
        )
        # --------------------------------------------------
        # SCROLLABLE CONTENT
        # --------------------------------------------------
        content_frame = ttk.Frame(
            self.right_frame
        )
        content_frame.pack(
            fill="both",
            expand=True,
        )
        content_frame.grid_rowconfigure(
            0,
            weight=1,
        )
        content_frame.grid_columnconfigure(
            0,
            weight=1,
        )
        self.panel_canvas = tk.Canvas(
            content_frame,
            highlightthickness=0,
        )
        self.panel_scrollbar = ttk.Scrollbar(
            content_frame,
            orient="vertical",
            command=self.panel_canvas.yview,
        )
        self.panel_canvas.configure(
            yscrollcommand=self.panel_scrollbar.set
        )
        self.panel_canvas.grid(
            row=0,
            column=0,
            sticky="nsew",
        )
        self.panel_scrollbar.grid(
            row=0,
            column=1,
            sticky="ns",
        )
        self.info_frame = ttk.Frame(
            self.panel_canvas,
            padding=8,
        )
        self.info_window = self.panel_canvas.create_window(
            (0, 0),
            window=self.info_frame,
            anchor="nw",
        )
        self.info_frame.bind(
            "<Configure>",
            lambda event: self.panel_canvas.configure(
                scrollregion=self.panel_canvas.bbox(
                    "all"
                )
            ),
        )
        self.panel_canvas.bind(
            "<Configure>",
            lambda event: self.panel_canvas.itemconfigure(
                self.info_window,
                width=event.width,
            ),
        )
        self.panel_canvas.bind_all(
            "<MouseWheel>",
            self.mousewheel,
        )
        # --------------------------------------------------
        # LIVE STATISTICS
        # --------------------------------------------------
        ttk.Label(
            self.info_frame,
            text="LIVE SYSTEM STATUS",
            font=("Arial", 11, "bold"),
        ).pack(
            anchor="w",
        )
        self.status_text = tk.Text(
            self.info_frame,
            width=56,
            height=7,
            font=("Consolas", 9),
            state="disabled",
        )
        self.status_text.pack(
            fill="x",
            pady=(3, 8),
        )
        ttk.Separator(
            self.info_frame,
            orient="horizontal",
        ).pack(
            fill="x",
            pady=5,
        )
        # --------------------------------------------------
        # WEATHER
        # --------------------------------------------------
        ttk.Label(
            self.info_frame,
            text="EXTERNAL WEATHER",
            font=("Arial", 11, "bold"),
        ).pack(
            anchor="w",
        )
        self.weather_label = ttk.Label(
            self.info_frame,
            text="--",
            justify="left",
        )
        self.weather_label.pack(
            anchor="w",
            pady=(3, 8),
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
        )
        self.agv_text = tk.Text(
            self.info_frame,
            width=56,
            height=8,
            font=("Consolas", 8),
            state="disabled",
        )
        self.agv_text.pack(
            fill="x",
            pady=(3, 8),
        )
        ttk.Separator(
            self.info_frame,
            orient="horizontal",
        ).pack(
            fill="x",
            pady=5,
        )
        # --------------------------------------------------
        # THERMAL / DERPG
        # --------------------------------------------------
        ttk.Label(
            self.info_frame,
            text="LIVE THERMAL / DERPG RISK MAP",
            font=("Arial", 11, "bold"),
        ).pack(
            anchor="w",
        )
        self.risk_text = tk.Text(
            self.info_frame,
            width=56,
            height=16,
            font=("Consolas", 8),
            state="disabled",
        )
        self.risk_text.pack(
            fill="x",
            pady=(3, 8),
        )
        ttk.Separator(
            self.info_frame,
            orient="horizontal",
        ).pack(
            fill="x",
            pady=5,
        )
        # --------------------------------------------------
        # TASK MONITOR
        # --------------------------------------------------
        ttk.Label(
            self.info_frame,
            text="PRODUCT / TASK PROTECTION",
            font=("Arial", 11, "bold"),
        ).pack(
            anchor="w",
        )
        self.task_text = tk.Text(
            self.info_frame,
            width=56,
            height=18,
            font=("Consolas", 8),
            state="disabled",
        )
        self.task_text.pack(
            fill="x",
            pady=(3, 8),
        )
        ttk.Separator(
            self.info_frame,
            orient="horizontal",
        ).pack(
            fill="x",
            pady=5,
        )
        # --------------------------------------------------
        # EVENT TIMELINE
        # --------------------------------------------------
        ttk.Label(
            self.info_frame,
            text="EMERGENCY EVENT TIMELINE",
            font=("Arial", 11, "bold"),
        ).pack(
            anchor="w",
        )
        self.event_text = tk.Text(
            self.info_frame,
            width=56,
            height=12,
            font=("Consolas", 8),
            state="disabled",
        )
        self.event_text.pack(
            fill="x",
            pady=(3, 8),
        )
        # --------------------------------------------------
        # EXPLANATION
        # --------------------------------------------------
        ttk.Label(
            self.info_frame,
            text=(
                "DEMO FLOW\n"
                "Normal operation -> cooling failure -> "
                "thermal risk escalation -> product monitoring -> "
                "rescue/safe routing when activated -> delivery completion."
            ),
            font=("Arial", 9),
            wraplength=470,
            justify="left",
        ).pack(
            anchor="w",
            pady=(3, 15),
        )
    # ======================================================
    # MOUSE WHEEL
    # ======================================================
    def mousewheel(self, event):
        try:
            self.panel_canvas.yview_scroll(
                int(-event.delta / 120),
                "units",
            )
        except Exception:
            pass
    # ======================================================
    # RESET ENVIRONMENT
    # ======================================================
    def reset_environment(self):
        result = env.reset(
            seed=42
        )
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
        self.finished = False
        self.emergency_triggered = False
        self.rescue_seen = False
        self.step_number = 0
        self.total_reward = 0.0
        self.pickups = 0
        self.completed = 0
        self.collisions = 0
        self.event_history = []
        self.trails = {
            1: deque(maxlen=TRAIL_LENGTH),
            2: deque(maxlen=TRAIL_LENGTH),
            3: deque(maxlen=TRAIL_LENGTH),
        }
        self.reset_environment()
        for agv_id, agv in env.system.agvs.items():
            self.trails[agv_id].append(
                tuple(
                    map(
                        int,
                        agv.position,
                    )
                )
            )
        self.start_button.config(
            state="normal"
        )
        self.mode_label.config(
            text="MODE: NORMAL OPERATION"
        )
        self.draw_warehouse()
        self.update_information()
        self.panel_canvas.yview_moveto(0.0)
    # ======================================================
    # EVENT LOG
    # ======================================================
    def add_event(self, text):
        self.event_history.append(
            f"[Step {self.step_number:03d}] {text}"
        )
        if len(self.event_history) > 15:
            self.event_history = self.event_history[-15:]
    # ======================================================
    # DRAW WAREHOUSE
    # ======================================================
    def draw_warehouse(self):
        self.canvas.delete(
            "all"
        )
        risks = env.warehouse.get_all_zone_risks()
        thermal = getattr(
            env.warehouse,
            "_thermal_state",
            {},
        )
        cooling = env.warehouse.get_all_cooling_status()
        # --------------------------------------------------
        # ZONES
        # --------------------------------------------------
        for zone_id in range(1, 17):
            row = (zone_id - 1) // 4
            col = (zone_id - 1) % 4
            x1 = col * 5 * CELL_SIZE
            y1 = row * 5 * CELL_SIZE
            x2 = x1 + 5 * CELL_SIZE
            y2 = y1 + 5 * CELL_SIZE
            zone_data = risks.get(
                zone_id,
                {},
            )
            dynamic_risk = float(
                zone_data.get(
                    "dynamic_risk",
                    0.0,
                )
            )
            self.canvas.create_rectangle(
                x1,
                y1,
                x2,
                y2,
                fill=risk_fill(dynamic_risk),
                outline=ZONE_BORDER,
                width=2,
            )
            thermal_data = thermal.get(
                zone_id,
                {},
            )
            temperature = float(
                thermal_data.get(
                    "temperature",
                    0.0,
                )
            )
            trend = float(
                thermal_data.get(
                    "temperature_trend",
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
                text=f"ZONE {zone_id}",
                anchor="nw",
                font=("Arial", 8, "bold"),
            )
            self.canvas.create_text(
                x1 + 5,
                y1 + 21,
                text=f"T={temperature:.1f}C",
                anchor="nw",
                font=("Arial", 8),
            )
            self.canvas.create_text(
                x1 + 5,
                y1 + 37,
                text=f"R={dynamic_risk:.2f}",
                anchor="nw",
                font=("Arial", 8, "bold"),
            )
            self.canvas.create_text(
                x1 + 5,
                y2 - 18,
                text=f"dT={trend:+.2f}",
                anchor="sw",
                font=("Arial", 7),
            )
            self.canvas.create_text(
                x2 - 5,
                y2 - 18,
                text=cooling_status,
                anchor="se",
                font=("Arial", 6),
            )
        # --------------------------------------------------
        # GRID
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
        # TASK MARKERS
        # --------------------------------------------------
        for task in env.tasks:
            pickup_x, pickup_y = task.pickup
            delivery_x, delivery_y = task.delivery
            # Pickup marker
            px1 = pickup_x * CELL_SIZE + 3
            py1 = pickup_y * CELL_SIZE + 3
            self.canvas.create_rectangle(
                px1,
                py1,
                px1 + CELL_SIZE - 6,
                py1 + CELL_SIZE - 6,
                fill=PICKUP_COLOR,
                outline="black",
                width=2,
            )
            self.canvas.create_text(
                pickup_x * CELL_SIZE
                + CELL_SIZE / 2,
                pickup_y * CELL_SIZE
                + CELL_SIZE / 2,
                text=f"P{task.task_id}",
                font=("Arial", 8, "bold"),
            )
            # Delivery marker
            dx1 = delivery_x * CELL_SIZE + 3
            dy1 = delivery_y * CELL_SIZE + 3
            delivery_fill = (
                COMPLETED_COLOR
                if str(task.status).upper()
                in ("COMPLETED", "DELIVERED")
                else DELIVERY_COLOR
            )
            self.canvas.create_rectangle(
                dx1,
                dy1,
                dx1 + CELL_SIZE - 6,
                dy1 + CELL_SIZE - 6,
                fill=delivery_fill,
                outline="black",
                width=2,
            )
            self.canvas.create_text(
                delivery_x * CELL_SIZE
                + CELL_SIZE / 2,
                delivery_y * CELL_SIZE
                + CELL_SIZE / 2,
                text=f"D{task.task_id}",
                font=("Arial", 8, "bold"),
            )
            # Rescue destination
            rescue = getattr(
                task,
                "rescue_destination",
                None,
            )
            if rescue is not None:
                rx, ry = map(
                    int,
                    rescue,
                )
                self.canvas.create_oval(
                    rx * CELL_SIZE + 2,
                    ry * CELL_SIZE + 2,
                    rx * CELL_SIZE + CELL_SIZE - 2,
                    ry * CELL_SIZE + CELL_SIZE - 2,
                    outline=RESCUE_COLOR,
                    width=4,
                )
                self.canvas.create_text(
                    rx * CELL_SIZE
                    + CELL_SIZE / 2,
                    ry * CELL_SIZE
                    + CELL_SIZE / 2,
                    text="SAFE",
                    fill=RESCUE_COLOR,
                    font=("Arial", 6, "bold"),
                )
        # --------------------------------------------------
        # AGV TRAILS
        # --------------------------------------------------
        for agv_id in range(
            1,
            NUM_AGVS + 1,
        ):
            trail = list(
                self.trails.get(
                    agv_id,
                    [],
                )
            )
            if len(trail) < 2:
                continue
            points = []
            for x, y in trail:
                points.extend(
                    [
                        x * CELL_SIZE
                        + CELL_SIZE / 2,
                        y * CELL_SIZE
                        + CELL_SIZE / 2,
                    ]
                )
            self.canvas.create_line(
                *points,
                fill=AGV_COLORS[agv_id - 1],
                width=2,
                smooth=False,
            )
        # --------------------------------------------------
        # AGVS
        # --------------------------------------------------
        for agv_id in range(
            1,
            NUM_AGVS + 1,
        ):
            agv = env.system.agvs[
                agv_id
            ]
            x, y = map(
                int,
                agv.position,
            )
            cx = (
                x * CELL_SIZE
                + CELL_SIZE / 2
            )
            cy = (
                y * CELL_SIZE
                + CELL_SIZE / 2
            )
            radius = 12
            self.canvas.create_oval(
                cx - radius,
                cy - radius,
                cx + radius,
                cy + radius,
                fill=AGV_COLORS[
                    agv_id - 1
                ],
                outline="black",
                width=2,
            )
            self.canvas.create_text(
                cx,
                cy,
                text=f"A{agv_id}",
                fill="white",
                font=("Arial", 8, "bold"),
            )
            self.canvas.create_text(
                cx,
                cy + 17,
                text=str(agv.status),
                font=("Arial", 6, "bold"),
            )
        # --------------------------------------------------
        # LABELS
        # --------------------------------------------------
        self.canvas.create_text(
            8,
            WAREHOUSE_PIXELS - 8,
            text=(
                "GREEN=LOW  YELLOW=MODERATE  "
                "ORANGE=HIGH  RED=CRITICAL"
            ),
            anchor="sw",
            font=("Arial", 8, "bold"),
        )
    # ======================================================
    # INFORMATION UPDATE
    # ======================================================
    def update_information(self):
        weather = env.warehouse.get_external_weather()
        risks = env.warehouse.get_all_zone_risks()
        thermal = getattr(
            env.warehouse,
            "_thermal_state",
            {},
        )
        cooling = env.warehouse.get_all_cooling_status()
        # --------------------------------------------------
        # STATUS
        # --------------------------------------------------
        self.status_text.config(
            state="normal"
        )
        self.status_text.delete(
            "1.0",
            "end",
        )
        self.status_text.insert(
            "end",
            f"Step                : {self.step_number}\n"
            f"Total reward        : {self.total_reward:.2f}\n"
            f"Pickups             : {self.pickups} / 3\n"
            f"Completed           : {self.completed} / 3\n"
            f"Collisions          : {self.collisions}\n"
            f"DQN model           : LOADED\n"
            f"Decision state      : {STATE_SIZE} features\n"
            f"Risk-aware env      : ENABLED\n"
            f"Operational control : VALIDATED TASK-DIRECTED\n"
        )
        self.status_text.config(
            state="disabled"
        )
        # --------------------------------------------------
        # WEATHER
        # --------------------------------------------------
        self.weather_label.config(
            text=(
                f"Outdoor temperature : "
                f"{float(weather.get('temperature', 0.0)):.1f} C\n"
                f"Humidity            : "
                f"{float(weather.get('humidity', 0.0)):.1f}%"
            )
        )
        # --------------------------------------------------
        # AGV
        # --------------------------------------------------
        self.agv_text.config(
            state="normal"
        )
        self.agv_text.delete(
            "1.0",
            "end",
        )
        for agv_id in range(
            1,
            NUM_AGVS + 1,
        ):
            agv = env.system.agvs[
                agv_id
            ]
            zone = zone_from_position(
                agv.position
            )
            task_id = "--"
            for task in env.tasks:
                if (
                    task.assigned_agv
                    == agv_id
                    and str(task.status).upper()
                    not in (
                        "COMPLETED",
                        "DELIVERED",
                    )
                ):
                    task_id = str(
                        task.task_id
                    )
                    break
            self.agv_text.insert(
                "end",
                (
                    f"AGV {agv_id} | "
                    f"POS={str(agv.position):<10} "
                    f"ZONE={zone:<2} "
                    f"TASK={task_id:<2}\n"
                    f"        STATUS={str(agv.status):<14} "
                    f"ACTION={action_name(self.last_actions[agv_id - 1])}\n"
                    f"        TARGET={agv.destination}\n\n"
                ),
            )
        self.agv_text.config(
            state="disabled"
        )
        # --------------------------------------------------
        # RISK
        # --------------------------------------------------
        self.risk_text.config(
            state="normal"
        )
        self.risk_text.delete(
            "1.0",
            "end",
        )
        for zone_id in range(
            1,
            17,
        ):
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
            trend = float(
                thermal.get(
                    zone_id,
                    {},
                ).get(
                    "temperature_trend",
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
            level = risk_level(r)
            self.risk_text.insert(
                "end",
                (
                    f"Z{zone_id:02d}  "
                    f"T={t:5.1f}C  "
                    f"Risk={r:.2f}  "
                    f"dT={trend:+.2f}  "
                    f"{level:<8} "
                    f"{status}\n"
                ),
            )
        self.risk_text.config(
            state="disabled"
        )
        # --------------------------------------------------
        # TASK
        # --------------------------------------------------
        self.task_text.config(
            state="normal"
        )
        self.task_text.delete(
            "1.0",
            "end",
        )
        for task in env.tasks:
            risk = float(
                getattr(
                    task,
                    "risk_score",
                    0.0,
                )
            )
            urgency = float(
                getattr(
                    task,
                    "urgency_score",
                    0.0,
                )
            )
            rset = float(
                getattr(
                    task,
                    "rset_minutes",
                    0.0,
                )
            )
            rescue_active = bool(
                getattr(
                    task,
                    "in_rescue",
                    False,
                )
            )
            rescue_destination = getattr(
                task,
                "rescue_destination",
                None,
            )
            original_delivery = getattr(
                task,
                "original_delivery",
                task.delivery,
            )
            self.task_text.insert(
                "end",
                (
                    f"TASK {task.task_id}\n"
                    f"  Status          : {task.status}\n"
                    f"  Assigned AGV    : {task.assigned_agv}\n"
                    f"  Pickup          : {task.pickup}\n"
                    f"  Delivery        : {task.delivery}\n"
                    f"  Risk            : {risk:.3f}\n"
                    f"  Urgency         : {urgency:.3f}\n"
                    f"  RSET            : {rset:.1f} min\n"
                    f"  Rescue active   : {rescue_active}\n"
                    f"  Rescue location : {rescue_destination}\n"
                    f"  Original goal   : {original_delivery}\n"
                    f"\n"
                ),
            )
        self.task_text.config(
            state="disabled"
        )
        # --------------------------------------------------
        # EVENTS
        # --------------------------------------------------
        self.event_text.config(
            state="normal"
        )
        self.event_text.delete(
            "1.0",
            "end",
        )
        for event in self.event_history:
            self.event_text.insert(
                "end",
                event + "\n",
            )
        self.event_text.config(
            state="disabled"
        )
    # ======================================================
    # START
    # ======================================================
    def start(self):
        if self.running:
            return
        if self.finished:
            self.reset()
        self.running = True
        self.finished = False
        self.start_button.config(
            state="disabled"
        )
        self.add_event(
            "Demonstration started."
        )
        self.run_step()
    # ======================================================
    # STEP
    # ======================================================
    def run_step(self):
        if not self.running:
            return
        # --------------------------------------------------
        # CONTROLLED EMERGENCY
        # --------------------------------------------------
        if (
            not self.emergency_triggered
            and self.step_number
            >= EMERGENCY_TRIGGER_STEP
        ):
            try:
                env.warehouse.trigger_cooling_failure(
                    EMERGENCY_ZONE
                )
                self.emergency_triggered = True
                self.mode_label.config(
                    text=(
                        "MODE: THERMAL EMERGENCY "
                        f"| ZONE {EMERGENCY_ZONE} COOLING FAILURE"
                    )
                )
                self.add_event(
                    f"Cooling failure triggered in Zone {EMERGENCY_ZONE}."
                )
            except Exception as exc:
                self.add_event(
                    f"Cooling-failure trigger error: {exc}"
                )
        # --------------------------------------------------
        # DQN INFERENCE
        # --------------------------------------------------
        try:
            dqn_actions = agent.select_action(
                self.state,
                training=False,
            )
            self.dqn_actions = np.asarray(
                dqn_actions,
                dtype=np.int64,
            ).reshape(
                NUM_AGVS
            )
        except Exception:
            self.dqn_actions = np.zeros(
                NUM_AGVS,
                dtype=np.int64,
            )
        # --------------------------------------------------
        # STABLE VALIDATED OPERATIONAL ACTIONS
        # --------------------------------------------------
        actions = np.asarray(
            [
                env._get_task_directed_action(
                    agv
                )
                for agv in env.system.agvs.values()
            ],
            dtype=np.int64,
        )
        self.last_actions = actions.copy()
        # --------------------------------------------------
        # ENVIRONMENT STEP
        # --------------------------------------------------
        (
            next_state,
            reward,
            terminated,
            truncated,
            info,
        ) = env.step(
            actions
        )
        self.state = np.asarray(
            next_state,
            dtype=np.float32,
        )
        self.step_number += 1
        self.total_reward += float(
            reward
        )
        # --------------------------------------------------
        # METRICS
        # --------------------------------------------------
        if isinstance(info, dict):
            step_pickups = int(
                info.get(
                    "pickup_tasks_this_step",
                    0,
                )
            )
            step_completed = int(
                info.get(
                    "completed_tasks_this_step",
                    0,
                )
            )
            step_collisions = int(
                info.get(
                    "collision_count",
                    0,
                )
            )
            self.pickups += step_pickups
            self.completed += step_completed
            self.collisions += step_collisions
            if step_pickups > 0:
                self.add_event(
                    f"Product pickup event: {step_pickups}."
                )
            if step_completed > 0:
                self.add_event(
                    f"Delivery completion event: {step_completed}."
                )
            if step_collisions > 0:
                self.add_event(
                    f"Collision event detected: {step_collisions}."
                )
        # --------------------------------------------------
        # TRAILS
        # --------------------------------------------------
        for agv_id, agv in env.system.agvs.items():
            self.trails[agv_id].append(
                tuple(
                    map(
                        int,
                        agv.position,
                    )
                )
            )
        # --------------------------------------------------
        # DETECT RISK ESCALATION
        # --------------------------------------------------
        risks = env.warehouse.get_all_zone_risks()
        emergency_risk = float(
            risks.get(
                EMERGENCY_ZONE,
                {},
            ).get(
                "dynamic_risk",
                0.0,
            )
        )
        if (
            self.emergency_triggered
            and emergency_risk >= 0.60
        ):
            self.add_event(
                (
                    f"Zone {EMERGENCY_ZONE} risk escalated "
                    f"to {emergency_risk:.2f} "
                    f"({risk_level(emergency_risk)})."
                )
            )
        # --------------------------------------------------
        # DETECT RESCUE
        # --------------------------------------------------
        rescue_active = any(
            bool(
                getattr(
                    task,
                    "in_rescue",
                    False,
                )
            )
            for task in env.tasks
        )
        if rescue_active and not self.rescue_seen:
            self.rescue_seen = True
            self.mode_label.config(
                text="MODE: ACTIVE PRODUCT RESCUE"
            )
            rescue_tasks = [
                task
                for task in env.tasks
                if getattr(
                    task,
                    "in_rescue",
                    False,
                )
            ]
            for task in rescue_tasks:
                self.add_event(
                    (
                        f"Task {task.task_id}: "
                        f"product rescue activated -> "
                        f"safe destination "
                        f"{task.rescue_destination}."
                    )
                )
        elif (
            self.emergency_triggered
            and not rescue_active
            and not self.finished
        ):
            self.mode_label.config(
                text="MODE: THERMAL EMERGENCY"
            )
        # --------------------------------------------------
        # DRAW
        # --------------------------------------------------
        self.draw_warehouse()
        self.update_information()
        # --------------------------------------------------
        # FINISH WHEN ALL 3 ARE COMPLETED
        # --------------------------------------------------
        all_completed = all(
            str(task.status).upper()
            in (
                "COMPLETED",
                "DELIVERED",
            )
            for task in env.tasks
        )
        done = (
            bool(terminated)
            or bool(truncated)
            or all_completed
            or self.step_number >= MAX_STEPS
        )
        if done:
            self.running = False
            self.finished = True
            self.start_button.config(
                state="normal"
            )
            if all_completed:
                self.mode_label.config(
                    text=(
                        "MODE: ✅ ALL 3 PRODUCTS DELIVERED"
                    )
                )
                self.add_event(
                    (
                        "DEMONSTRATION COMPLETE: "
                        "all 3 tasks delivered."
                    )
                )
            elif self.step_number >= MAX_STEPS:
                self.mode_label.config(
                    text="MODE: MAXIMUM STEPS REACHED"
                )
                self.add_event(
                    "Maximum demonstration steps reached."
                )
            else:
                self.mode_label.config(
                    text="MODE: EPISODE FINISHED"
                )
            self.update_information()
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
    app = WarehouseWindow(
        root
    )
    root.mainloop()

