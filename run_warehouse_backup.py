import tkinter as tk
from tkinter import ttk

from environment.multi_agv_env import MultiAGVEnv


# ==========================================================
# CONFIGURATION
# ==========================================================

WIDTH = 20
HEIGHT = 20
NUM_AGVS = 3
MAX_STEPS = 200

CELL_SIZE = 32


# ==========================================================
# COLORS
# ==========================================================

BACKGROUND = "#202124"
GRID_COLOR = "#555555"
EMPTY_COLOR = "#F5F5F5"

AGV_COLORS = [
    "#2196F3",
    "#4CAF50",
    "#FF9800",
]

PICKUP_COLOR = "#FFD54F"
DELIVERY_COLOR = "#EF5350"


# ==========================================================
# WAREHOUSE WINDOW
# ==========================================================

class WarehouseWindow:

    def __init__(self, root):

        self.root = root

        self.root.title(
            "Cold Chain Multi-AGV Warehouse"
        )

        self.root.resizable(
            False,
            False,
        )

        # --------------------------------------------------
        # ENVIRONMENT
        # --------------------------------------------------

        self.env = MultiAGVEnv(
            width=WIDTH,
            height=HEIGHT,
            num_agvs=NUM_AGVS,
            max_steps=MAX_STEPS,
        )

        self.state = None
        self.running = False

        self.step_count = 0
        self.total_reward = 0.0
        self.completed = 0
        self.pickups = 0
        self.collisions = 0

        # --------------------------------------------------
        # MAIN FRAME
        # --------------------------------------------------

        main_frame = ttk.Frame(
            self.root,
            padding=10,
        )

        main_frame.pack()

        # --------------------------------------------------
        # TITLE
        # --------------------------------------------------

        title = ttk.Label(
            main_frame,
            text="COLD CHAIN MULTI-AGV WAREHOUSE",
            font=("Arial", 18, "bold"),
        )

        title.pack(
            pady=(0, 10)
        )

        # --------------------------------------------------
        # CANVAS
        # --------------------------------------------------

        canvas_width = (
            WIDTH * CELL_SIZE
        )

        canvas_height = (
            HEIGHT * CELL_SIZE
        )

        self.canvas = tk.Canvas(
            main_frame,
            width=canvas_width,
            height=canvas_height,
            bg=BACKGROUND,
            highlightthickness=1,
            highlightbackground="black",
        )

        self.canvas.pack()

        # --------------------------------------------------
        # STATUS
        # --------------------------------------------------

        self.status_frame = ttk.Frame(
            main_frame
        )

        self.status_frame.pack(
            pady=10
        )

        self.step_label = ttk.Label(
            self.status_frame,
            text="Step: 0",
            font=("Arial", 11),
        )

        self.step_label.grid(
            row=0,
            column=0,
            padx=10,
        )

        self.reward_label = ttk.Label(
            self.status_frame,
            text="Reward: 0.00",
            font=("Arial", 11),
        )

        self.reward_label.grid(
            row=0,
            column=1,
            padx=10,
        )

        self.pickup_label = ttk.Label(
            self.status_frame,
            text="Pickups: 0",
            font=("Arial", 11),
        )

        self.pickup_label.grid(
            row=0,
            column=2,
            padx=10,
        )

        self.completed_label = ttk.Label(
            self.status_frame,
            text="Completed: 0",
            font=("Arial", 11),
        )

        self.completed_label.grid(
            row=0,
            column=3,
            padx=10,
        )

        self.collision_label = ttk.Label(
            self.status_frame,
            text="Collisions: 0",
            font=("Arial", 11),
        )

        self.collision_label.grid(
            row=0,
            column=4,
            padx=10,
        )

        # --------------------------------------------------
        # BUTTONS
        # --------------------------------------------------

        button_frame = ttk.Frame(
            main_frame
        )

        button_frame.pack(
            pady=5
        )

        self.start_button = ttk.Button(
            button_frame,
            text="Start",
            command=self.start,
        )

        self.start_button.grid(
            row=0,
            column=0,
            padx=5,
        )

        self.stop_button = ttk.Button(
            button_frame,
            text="Stop",
            command=self.stop,
        )

        self.stop_button.grid(
            row=0,
            column=1,
            padx=5,
        )

        self.reset_button = ttk.Button(
            button_frame,
            text="Reset",
            command=self.reset,
        )

        self.reset_button.grid(
            row=0,
            column=2,
            padx=5,
        )

        # --------------------------------------------------
        # LEGEND
        # --------------------------------------------------

        legend = ttk.Label(
            main_frame,
            text=(
                "AGV1 = Blue    "
                "AGV2 = Green    "
                "AGV3 = Orange    "
                "Pickup = Yellow    "
                "Delivery = Red"
            ),
            font=("Arial", 10),
        )

        legend.pack(
            pady=(5, 0)
        )

        # --------------------------------------------------
        # INITIAL RESET
        # --------------------------------------------------

        self.reset()

    # ======================================================
    # RESET
    # ======================================================

    def reset(self):

        self.running = False

        result = self.env.reset(
            seed=42
        )

        self.state = result[0]

        self.step_count = 0
        self.total_reward = 0.0
        self.completed = 0
        self.pickups = 0
        self.collisions = 0

        self.draw()

        self.update_labels()

    # ======================================================
    # START
    # ======================================================

    def start(self):

        if self.running:
            return

        self.running = True

        self.run_step()

    # ======================================================
    # STOP
    # ======================================================

    def stop(self):

        self.running = False

    # ======================================================
    # STEP
    # ======================================================

    def run_step(self):

        if not self.running:
            return

        if self.step_count >= MAX_STEPS:
            self.running = False
            return

        # --------------------------------------------------
        # RANDOM ACTIONS
        #
        # This first version tests the warehouse GUI.
        # DQN will be connected after the GUI is verified.
        # --------------------------------------------------

        actions = self.env.action_space.sample()

        (
            next_state,
            reward,
            terminated,
            truncated,
            info,
        ) = self.env.step(
            actions
        )

        self.state = next_state

        self.step_count += 1

        self.total_reward += float(
            reward
        )

        self.completed = max(
            self.completed,
            int(
                info.get(
                    "completed_tasks",
                    0,
                )
            ),
        )

        self.pickups += int(
            info.get(
                "pickup_tasks",
                0,
            )
        )

        self.collisions += int(
            info.get(
                "collision_count",
                0,
            )
        )

        self.draw()

        self.update_labels()

        if terminated or truncated:

            self.running = False

            return

        self.root.after(
            150,
            self.run_step,
        )

    # ======================================================
    # DRAW WAREHOUSE
    # ======================================================

    def draw(self):

        self.canvas.delete(
            "all"
        )

        # --------------------------------------------------
        # GRID
        # --------------------------------------------------

        for x in range(
            WIDTH + 1
        ):

            px = (
                x * CELL_SIZE
            )

            self.canvas.create_line(
                px,
                0,
                px,
                HEIGHT * CELL_SIZE,
                fill=GRID_COLOR,
            )

        for y in range(
            HEIGHT + 1
        ):

            py = (
                y * CELL_SIZE
            )

            self.canvas.create_line(
                0,
                py,
                WIDTH * CELL_SIZE,
                py,
                fill=GRID_COLOR,
            )

        # --------------------------------------------------
        # TASKS
        # --------------------------------------------------

        for task in self.env.tasks:

            if task.is_completed:
                continue

            # --------------------------------------------------
            # PICKUP
            # --------------------------------------------------

            if not task.is_picked_up:

                self.draw_cell(
                    task.pickup,
                    PICKUP_COLOR,
                    f"P{task.task_id}",
                )

            # --------------------------------------------------
            # DELIVERY
            # --------------------------------------------------

            self.draw_cell(
                task.delivery,
                DELIVERY_COLOR,
                f"D{task.task_id}",
            )

        # --------------------------------------------------
        # AGVS
        # --------------------------------------------------

        for index in range(
            NUM_AGVS
        ):

            agv_id = index + 1

            if agv_id not in self.env.system.agvs:
                continue

            agv = self.env.system.agvs[
                agv_id
            ]

            self.draw_agv(
                agv.position,
                AGV_COLORS[index],
                f"AGV {agv_id}",
            )

    # ======================================================
    # DRAW CELL
    # ======================================================

    def draw_cell(
        self,
        position,
        color,
        text,
    ):

        x, y = position

        left = (
            x * CELL_SIZE
        )

        top = (
            (HEIGHT - 1 - y)
            * CELL_SIZE
        )

        right = (
            left + CELL_SIZE
        )

        bottom = (
            top + CELL_SIZE
        )

        self.canvas.create_rectangle(
            left + 3,
            top + 3,
            right - 3,
            bottom - 3,
            fill=color,
            outline="black",
        )

        self.canvas.create_text(
            (left + right) / 2,
            (top + bottom) / 2,
            text=text,
            font=("Arial", 8, "bold"),
        )

    # ======================================================
    # DRAW AGV
    # ======================================================

    def draw_agv(
        self,
        position,
        color,
        text,
    ):

        x, y = position

        center_x = (
            x * CELL_SIZE
            + CELL_SIZE / 2
        )

        center_y = (
            (HEIGHT - 1 - y)
            * CELL_SIZE
            + CELL_SIZE / 2
        )

        radius = (
            CELL_SIZE * 0.32
        )

        self.canvas.create_oval(
            center_x - radius,
            center_y - radius,
            center_x + radius,
            center_y + radius,
            fill=color,
            outline="black",
            width=2,
        )

        self.canvas.create_text(
            center_x,
            center_y,
            text=text,
            fill="white",
            font=("Arial", 7, "bold"),
        )

    # ======================================================
    # UPDATE LABELS
    # ======================================================

    def update_labels(self):

        self.step_label.config(
            text=f"Step: {self.step_count}"
        )

        self.reward_label.config(
            text=f"Reward: {self.total_reward:.2f}"
        )

        self.pickup_label.config(
            text=f"Pickups: {self.pickups}"
        )

        self.completed_label.config(
            text=f"Completed: {self.completed}/{NUM_AGVS}"
        )

        self.collision_label.config(
            text=f"Collisions: {self.collisions}"
        )


# ==========================================================
# MAIN
# ==========================================================

def main():

    root = tk.Tk()

    WarehouseWindow(
        root
    )

    root.mainloop()


if __name__ == "__main__":
    main()
    