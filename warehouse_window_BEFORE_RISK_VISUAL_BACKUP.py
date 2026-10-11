"""
Warehouse Visualization for ColdChain Multi-AGV RL.

Features:
- 20x20 warehouse
- 3 AGVs
- Pickup locations
- Delivery locations
- Live AGV movement
- Task status
- Reward
- Collision count
- Completed tasks
- DQN model integration
"""

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
MAX_STEPS = 200

MODEL_PATH = "models/dqn_model.pt"

GRID_SIZE = 20

CELL_SIZE = 30

WINDOW_WIDTH = GRID_SIZE * CELL_SIZE + 360
WINDOW_HEIGHT = GRID_SIZE * CELL_SIZE + 80

UPDATE_DELAY = 150


# ==========================================================
# COLORS / DISPLAY
# ==========================================================

AGV_COLORS = [
    "#1565C0",
    "#2E7D32",
    "#EF6C00",
]

PICKUP_COLOR = "#FFD54F"
DELIVERY_COLOR = "#AB47BC"
COMPLETED_COLOR = "#66BB6A"
WAREHOUSE_COLOR = "#F5F5F5"
GRID_COLOR = "#BDBDBD"


# ==========================================================
# ENVIRONMENT
# ==========================================================

env = MultiAGVEnv(
    num_agvs=NUM_AGVS,
    max_steps=MAX_STEPS,
)


# ==========================================================
# STATE SIZE
# ==========================================================

STATE_SIZE = int(
    env.observation_space.shape[0]
)


# ==========================================================
# DQN AGENT
# ==========================================================

agent = DQNAgent(
    state_size=STATE_SIZE,
    num_agvs=NUM_AGVS,
    action_size=ACTION_SIZE,
)

agent.load(
    MODEL_PATH
)

agent.evaluation_mode()


# ==========================================================
# APPLICATION
# ==========================================================

class WarehouseWindow:

    def __init__(self, root):

        self.root = root

        self.root.title(
            "ColdChain Multi-AGV DQN Warehouse"
        )

        self.root.geometry(
            f"{WINDOW_WIDTH}x{WINDOW_HEIGHT}"
        )

        self.root.resizable(
            False,
            False,
        )

        # --------------------------------------------------
        # Environment state
        # --------------------------------------------------

        reset_result = env.reset(
            seed=42
        )

        if isinstance(
            reset_result,
            tuple,
        ):

            self.state = reset_result[0]

        else:

            self.state = reset_result

        self.state = np.asarray(
            self.state,
            dtype=np.float32,
        )

        self.step_number = 0
        self.total_reward = 0.0
        self.collisions = 0
        self.completed = 0
        self.pickups = 0

        self.running = False

        # --------------------------------------------------
        # Main layout
        # --------------------------------------------------

        self.main_frame = ttk.Frame(
            self.root
        )

        self.main_frame.pack(
            fill="both",
            expand=True,
        )

        # --------------------------------------------------
        # Warehouse canvas
        # --------------------------------------------------

        self.canvas = tk.Canvas(
            self.main_frame,
            width=GRID_SIZE * CELL_SIZE,
            height=GRID_SIZE * CELL_SIZE,
            bg=WAREHOUSE_COLOR,
            highlightthickness=1,
        )

        self.canvas.pack(
            side="left",
            padx=15,
            pady=15,
        )

        # --------------------------------------------------
        # Information panel
        # --------------------------------------------------

        self.info_frame = ttk.Frame(
            self.main_frame,
            width=320,
        )

        self.info_frame.pack(
            side="right",
            fill="y",
            padx=10,
            pady=15,
        )

        # --------------------------------------------------
        # Title
        # --------------------------------------------------

        ttk.Label(
            self.info_frame,
            text="COLDCHAIN MULTI-AGV",
            font=("Arial", 16, "bold"),
        ).pack(
            pady=(5, 2)
        )

        ttk.Label(
            self.info_frame,
            text="DQN Warehouse Simulation",
            font=("Arial", 10),
        ).pack(
            pady=(0, 15)
        )

        # --------------------------------------------------
        # Statistics
        # --------------------------------------------------

        self.step_label = ttk.Label(
            self.info_frame,
            text="Step: 0",
            font=("Arial", 11),
        )

        self.step_label.pack(
            anchor="w",
            pady=3,
        )

        self.reward_label = ttk.Label(
            self.info_frame,
            text="Reward: 0.00",
            font=("Arial", 11),
        )

        self.reward_label.pack(
            anchor="w",
            pady=3,
        )

        self.pickup_label = ttk.Label(
            self.info_frame,
            text="Pickups: 0",
            font=("Arial", 11),
        )

        self.pickup_label.pack(
            anchor="w",
            pady=3,
        )

        self.completed_label = ttk.Label(
            self.info_frame,
            text="Completed: 0",
            font=("Arial", 11),
        )

        self.completed_label.pack(
            anchor="w",
            pady=3,
        )

        self.collision_label = ttk.Label(
            self.info_frame,
            text="Collisions: 0",
            font=("Arial", 11),
        )

        self.collision_label.pack(
            anchor="w",
            pady=3,
        )

        ttk.Separator(
            self.info_frame,
            orient="horizontal",
        ).pack(
            fill="x",
            pady=15,
        )

        # --------------------------------------------------
        # AGV information
        # --------------------------------------------------

        ttk.Label(
            self.info_frame,
            text="AGV STATUS",
            font=("Arial", 12, "bold"),
        ).pack(
            anchor="w",
            pady=(0, 8),
        )

        self.agv_labels = []

        for i in range(NUM_AGVS):

            label = ttk.Label(
                self.info_frame,
                text=f"AGV {i + 1}: --",
                font=("Arial", 10),
            )

            label.pack(
                anchor="w",
                pady=3,
            )

            self.agv_labels.append(
                label
            )

        ttk.Separator(
            self.info_frame,
            orient="horizontal",
        ).pack(
            fill="x",
            pady=15,
        )

        # --------------------------------------------------
        # Task information
        # --------------------------------------------------

        ttk.Label(
            self.info_frame,
            text="TASK STATUS",
            font=("Arial", 12, "bold"),
        ).pack(
            anchor="w",
            pady=(0, 8),
        )

        self.task_text = tk.Text(
            self.info_frame,
            width=38,
            height=10,
            font=("Consolas", 9),
            state="disabled",
        )

        self.task_text.pack(
            fill="x",
        )

        # --------------------------------------------------
        # Buttons
        # --------------------------------------------------

        ttk.Separator(
            self.info_frame,
            orient="horizontal",
        ).pack(
            fill="x",
            pady=15,
        )

        self.start_button = ttk.Button(
            self.info_frame,
            text="START DQN",
            command=self.start,
        )

        self.start_button.pack(
            fill="x",
            pady=4,
        )

        self.reset_button = ttk.Button(
            self.info_frame,
            text="RESET",
            command=self.reset,
        )

        self.reset_button.pack(
            fill="x",
            pady=4,
        )

        # --------------------------------------------------
        # Initial drawing
        # --------------------------------------------------

        self.draw_warehouse()

        self.update_information()

    # ======================================================
    # DRAW WAREHOUSE
    # ======================================================

    def draw_warehouse(self):

        self.canvas.delete(
            "all"
        )

        # --------------------------------------------------
        # Grid
        # --------------------------------------------------

        for x in range(
            GRID_SIZE + 1
        ):

            px = x * CELL_SIZE

            self.canvas.create_line(
                px,
                0,
                px,
                GRID_SIZE * CELL_SIZE,
                fill=GRID_COLOR,
            )

        for y in range(
            GRID_SIZE + 1
        ):

            py = y * CELL_SIZE

            self.canvas.create_line(
                0,
                py,
                GRID_SIZE * CELL_SIZE,
                py,
                fill=GRID_COLOR,
            )

        # --------------------------------------------------
        # Coordinates
        # --------------------------------------------------

        for x in range(
            GRID_SIZE
        ):

            for y in range(
                GRID_SIZE
            ):

                if (
                    x % 5 == 0
                    and y % 5 == 0
                ):

                    self.canvas.create_text(
                        x * CELL_SIZE + 3,
                        y * CELL_SIZE + 3,
                        text=f"{x},{y}",
                        anchor="nw",
                        font=("Arial", 6),
                        fill="#757575",
                    )

        # --------------------------------------------------
        # Tasks
        # --------------------------------------------------

        for task in env.tasks:

            pickup_x, pickup_y = (
                task.pickup
            )

            delivery_x, delivery_y = (
                task.delivery
            )

            # Pickup

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
                pickup_x * CELL_SIZE
                + CELL_SIZE / 2,
                pickup_y * CELL_SIZE
                + CELL_SIZE / 2,
                text=f"P{task.task_id}",
                font=("Arial", 8, "bold"),
            )

            # Delivery

            dx1 = delivery_x * CELL_SIZE + 5
            dy1 = delivery_y * CELL_SIZE + 5
            dx2 = dx1 + CELL_SIZE - 10
            dy2 = dy1 + CELL_SIZE - 10

            delivery_fill = (
                COMPLETED_COLOR
                if str(task.status).upper()
                in (
                    "COMPLETED",
                    "DELIVERED",
                )
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
                delivery_x * CELL_SIZE
                + CELL_SIZE / 2,
                delivery_y * CELL_SIZE
                + CELL_SIZE / 2,
                text=f"D{task.task_id}",
                font=("Arial", 8, "bold"),
            )

        # --------------------------------------------------
        # AGVs
        # --------------------------------------------------

        for i in range(
            1,
            NUM_AGVS + 1,
        ):

            position = (
                env.system.agvs[i].position
            )

            x, y = position

            cx = (
                x * CELL_SIZE
                + CELL_SIZE / 2
            )

            cy = (
                y * CELL_SIZE
                + CELL_SIZE / 2
            )

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
                font=("Arial", 9, "bold"),
            )

    # ======================================================
    # INFORMATION
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

        # --------------------------------------------------
        # AGVs
        # --------------------------------------------------

        for i in range(
            NUM_AGVS
        ):

            position = (
                env.system.agvs[i + 1].position
            )

            self.agv_labels[i].config(
                text=(
                    f"AGV {i + 1}: "
                    f"Position = {position}"
                )
            )

        # --------------------------------------------------
        # Tasks
        # --------------------------------------------------

        self.task_text.config(
            state="normal"
        )

        self.task_text.delete(
            "1.0",
            "end",
        )

        for task in env.tasks:

            self.task_text.insert(
                "end",
                (
                    f"Task {task.task_id}\n"
                    f"  Pickup   : {task.pickup}\n"
                    f"  Delivery : {task.delivery}\n"
                    f"  Status   : {task.status}\n\n"
                ),
            )

        self.task_text.config(
            state="disabled"
        )

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
    # RUN ONE STEP
    # ======================================================

    def run_step(self):

        if not self.running:
            return

        # --------------------------------------------------
        # Select DQN action
        # --------------------------------------------------

        actions = agent.select_action(
            self.state,
            training=False,
        )

        actions = np.asarray(
            actions,
            dtype=np.int64,
        ).reshape(
            NUM_AGVS
        )

        # --------------------------------------------------
        # Environment step
        # --------------------------------------------------

        result = env.step(
            actions
        )

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

        self.total_reward += float(
            reward
        )

        # --------------------------------------------------
        # Info
        # --------------------------------------------------

        if isinstance(
            info,
            dict,
        ):

            self.pickups += int(
                info.get(
                    "pickup_tasks",
                    info.get(
                        "pickups",
                        0,
                    ),
                )
            )

            self.completed += int(
                info.get(
                    "completed_tasks",
                    info.get(
                        "completed",
                        0,
                    ),
                )
            )

            self.collisions += int(
                info.get(
                    "collision_count",
                    info.get(
                        "collisions",
                        0,
                    ),
                )
            )

        # --------------------------------------------------
        # Draw
        # --------------------------------------------------

        self.draw_warehouse()

        self.update_information()

        # --------------------------------------------------
        # Episode end
        # --------------------------------------------------

        done = (
            bool(terminated)
            or bool(truncated)
        )

        if done:

            self.running = False

            self.start_button.config(
                state="normal"
            )

            return

        self.root.after(
            UPDATE_DELAY,
            self.run_step,
        )

    # ======================================================
    # RESET
    # ======================================================

    def reset(self):

        self.running = False

        self.start_button.config(
            state="normal"
        )

        reset_result = env.reset(
            seed=42
        )

        if isinstance(
            reset_result,
            tuple,
        ):

            self.state = reset_result[0]

        else:

            self.state = reset_result

        self.state = np.asarray(
            self.state,
            dtype=np.float32,
        )

        self.step_number = 0
        self.total_reward = 0.0
        self.collisions = 0
        self.completed = 0
        self.pickups = 0

        self.draw_warehouse()

        self.update_information()


# ==========================================================
# MAIN
# ==========================================================

if __name__ == "__main__":

    root = tk.Tk()

    app = WarehouseWindow(
        root
    )

    root.mainloop()
    