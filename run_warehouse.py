import tkinter as tk

from warehouse_window import WarehouseWindow


def main():
    root = tk.Tk()

    app = WarehouseWindow(root)

    root.mainloop()


if __name__ == "__main__":
    main()