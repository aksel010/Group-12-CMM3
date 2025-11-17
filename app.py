import tkinter as tk
from tkinter import ttk

class CMM3App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Group-12-CMM3 Fast-Charge Academic Analysis")
        self.geometry("1080x720")
        self.configure(bg="white")

        self.create_title()
        self.create_nav()
        self.create_tabs()

    def create_title(self):
        title = ttk.Label(self, text="Group 12 CMM3 — EV Battery Fast-Charge Optimization", font=("Arial", 22, "bold"))
        title.pack(pady=15)

    def create_nav(self):
        nav = ttk.Frame(self)
        nav.pack(fill="x", padx=10, pady=5)

        for tab_name in ["ODE", "Mass Flow", "Optimum Current", "Charging Time", "Heptane", "Cooling"]:
            btn = ttk.Button(nav, text=tab_name)
            btn.pack(side="left", padx=6)

    def create_tabs(self):
        notebook = ttk.Notebook(self)
        notebook.pack(fill="both", expand=True, padx=20, pady=15)

        # ODE Analysis Tab
        ode_tab = ttk.Frame(notebook)
        notebook.add(ode_tab, text="ODE Validation")
        ttk.Label(ode_tab, text="ODE (RK4 vs SciPy)", font=("Arial", 16)).pack(pady=10)
        ttk.Label(ode_tab, text="Plot Battery Temperature over Time", font=("Arial", 12)).pack()
        ttk.Label(ode_tab, text="Details, equations, and validation results go here...", font=("Arial", 10)).pack(pady=10)

        # Mass Flowrate Tab
        mf_tab = ttk.Frame(notebook)
        notebook.add(mf_tab, text="Mass Flowrate")
        ttk.Label(mf_tab, text="Pressure Balance Residuals", font=("Arial", 16)).pack(pady=10)
        ttk.Label(mf_tab, text="Display Mass Flow vs Pressure Residuals graphs", font=("Arial", 12)).pack()
        ttk.Label(mf_tab, text="Solver output, data, interpretation...", font=("Arial", 10)).pack(pady=10)

        # Optimum Current Tab
        oc_tab = ttk.Frame(notebook)
        notebook.add(oc_tab, text="Optimum Current")
        ttk.Label(oc_tab, text="Cubic Spline Analysis / Critical Current", font=("Arial", 16)).pack(pady=10)
        ttk.Label(oc_tab, text="Visualize Current vs ∆T", font=("Arial", 12)).pack()
        ttk.Label(oc_tab, text="Critical point and method explanation...", font=("Arial", 10)).pack(pady=10)

        # Charging Time Tab
        rct_tab = ttk.Frame(notebook)
        notebook.add(rct_tab, text="Charging Time")
        ttk.Label(rct_tab, text="Real Charging Time Study", font=("Arial", 16)).pack(pady=10)
        ttk.Label(rct_tab, text="Show charging simulation and relevant data", font=("Arial", 12)).pack()

        # Heptane Tab
        hi_tab = ttk.Frame(notebook)
        notebook.add(hi_tab, text="Heptane Fluid Properties")
        ttk.Label(hi_tab, text="Properties & Analysis", font=("Arial", 16)).pack(pady=10)
        ttk.Label(hi_tab, text="Display calculation and analysis results", font=("Arial", 12)).pack()

        # Cooling System Tab
        ca_tab = ttk.Frame(notebook)
        notebook.add(ca_tab, text="Cooling System")
        ttk.Label(ca_tab, text="Cooling Analysis Results", font=("Arial", 16)).pack(pady=10)
        ttk.Label(ca_tab, text="Detailed study, data, and visualizations", font=("Arial", 12)).pack()

        # You can add placeholders for plot canvases, input widgets, or data tables as your project evolves.

if __name__ == "__main__":
    app = CMM3App()
    app.mainloop()
