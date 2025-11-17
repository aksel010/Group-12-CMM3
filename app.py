import tkinter as tk
from tkinter import ttk

# Example main window
root = tk.Tk()
root.title("EV Battery Fast-Charge Optimizer")
root.geometry("600x400")

lbl = ttk.Label(root, text="Group-12-CMM3 Control Panel", font=("Arial", 18))
lbl.pack(pady=20)

root.mainloop()
