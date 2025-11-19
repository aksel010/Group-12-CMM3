import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import tkinter as tk
from tkinter import ttk
from tkinter import messagebox
import threading
import queue
import importlib
import matplotlib
matplotlib.use("TkAgg")
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
import numpy as np
try:
    import src.models.battery_temperature_ode as ODE
    from src.models.battery_temperature_ode import get_tb, d_tb_dt, get_tb_scipy
    import src.models.rk4_error as rk4e
    import src.models.mass_flowrate as mf
    import src.models.optimum_current as oc
    from src.models.optimum_current import current_params
    import scripts.charging_time_analysis as rct
    import src.models.cooling_analysis as ca
    import src.utils.heptane_interpolater as hi
    import src.config as config
    from src.config import current_store, current_error
except ImportError as e:
    print(f"Warning: Could not import module: {e}")

class CMM3App(tk.Tk):
    """
    Main application class for Group-12-CMM3 Fast-Charge Academic Analysis GUI.
    Inherits from tkinter.Tk and sets up all frames, widgets, and event handlers.
    """
    def __init__(self):
        """
        Initialize the main Tkinter window, widgets, thread communication.
        """
        super().__init__()
        self.title("Group-12-CMM3 Fast-Charge Academic Analysis")
        self.state('zoomed')  # Fullscreen on launch
        self.configure(bg="white")
        self.create_widgets()
        self.grid_rowconfigure(1, weight=3)
        self.grid_rowconfigure(2, weight=1)
        self.grid_columnconfigure(0, weight=1)
        self.grid_columnconfigure(1, weight=3)
        self.result_queue = queue.Queue()
        self.running = False
        self.cancel_event = threading.Event()

    def create_widgets(self):
        """
        Creates all frames, input widgets, buttons, output display, and diagnostics areas.
        """
        # Title
        title = ttk.Label(self, text="Group 12 CMM3 — PHEV Battery Fast-Charge Optimization", font=("Arial", 22, "bold"))
        title.grid(row=0, column=0, columnspan=2, pady=15, sticky="ew")
        # Top left: Customizable input section
        self.input_frame = ttk.LabelFrame(self, text="Input Parameters", padding=(15,15))
        self.input_frame.grid(row=1, column=0, sticky="nsew", padx=10, pady=5)
        
        # Row 0: Convergence Threshold
        ttk.Label(self.input_frame, text="Convergence Threshold:", font=("Arial", 10, "bold")).grid(row=0, column=0, sticky="w", pady=5)
        self.threshold_entry = ttk.Entry(self.input_frame, width=15)
        self.threshold_entry.insert(0, "1e-6")
        self.threshold_entry.grid(row=0, column=1, sticky="ew", pady=5, padx=5)
        ttk.Label(self.input_frame, text="[1e-8 to 1e-4]", font=("Arial", 8, "italic"), foreground="gray").grid(row=0, column=2, sticky="w", padx=5)
        
        # Row 1: Initial Current
        ttk.Label(self.input_frame, text="Initial Current (A):", font=("Arial", 10, "bold")).grid(row=1, column=0, sticky="w", pady=5)
        self.current_entry = ttk.Entry(self.input_frame, width=15)
        self.current_entry.insert(0, "17")
        self.current_entry.grid(row=1, column=1, sticky="ew", pady=5, padx=5)
        ttk.Label(self.input_frame, text="[1 to 17.5 A]", font=("Arial", 8, "italic"), foreground="gray").grid(row=1, column=2, sticky="w", padx=5)
        
        # Row 2: RK4 Step Size
        ttk.Label(self.input_frame, text="RK4 Step Size (s):", font=("Arial", 10, "bold")).grid(row=2, column=0, sticky="w", pady=5)
        self.step_entry = ttk.Entry(self.input_frame, width=15)
        self.step_entry.insert(0, "30")
        self.step_entry.grid(row=2, column=1, sticky="ew", pady=5, padx=5)
        ttk.Label(self.input_frame, text="[10 to 60 s]", font=("Arial", 8, "italic"), foreground="gray").grid(row=2, column=2, sticky="w", padx=5)
        
        # Row 3: Max Battery Temperature
        ttk.Label(self.input_frame, text="Max Battery Temp (°C):", font=("Arial", 10, "bold")).grid(row=3, column=0, sticky="w", pady=5)
        self.t_b_max_entry = ttk.Entry(self.input_frame, width=15)
        self.t_b_max_entry.insert(0, "40")
        self.t_b_max_entry.grid(row=3, column=1, sticky="ew", pady=5, padx=5)
        ttk.Label(self.input_frame, text="[35 to 50 °C]", font=("Arial", 8, "italic"), foreground="gray").grid(row=3, column=2, sticky="w", padx=5)
        
        # Row 4: Coolant Inlet Temperature
        ttk.Label(self.input_frame, text="Coolant Inlet Temp (°C):", font=("Arial", 10, "bold")).grid(row=4, column=0, sticky="w", pady=5)
        self.t_in_entry = ttk.Entry(self.input_frame, width=15)
        self.t_in_entry.insert(0, "15")
        self.t_in_entry.grid(row=4, column=1, sticky="ew", pady=5, padx=5)
        ttk.Label(self.input_frame, text="[10 to 25 °C]", font=("Arial", 8, "italic"), foreground="gray").grid(row=4, column=2, sticky="w", padx=5)
        
        ttk.Separator(self.input_frame, orient="horizontal").grid(row=5, column=0, columnspan=3, sticky="ew", pady=10)
        buttons_frame = ttk.Frame(self.input_frame)
        buttons_frame.grid(row=6, column=0, columnspan=3, sticky="ew", pady=(10, 0))
        buttons_frame.grid_columnconfigure(0, weight=1)
        buttons_frame.grid_columnconfigure(1, weight=1)
        buttons_frame.grid_columnconfigure(2, weight=1)
        self.run_btn = ttk.Button(buttons_frame, text="Run Complete Analysis", command=self.run_analysis)
        self.run_btn.grid(row=0, column=0, sticky="ew", padx=(0, 5))
        self.cancel_btn = ttk.Button(buttons_frame, text="Cancel", command=self.cancel_analysis, state="disabled")
        self.cancel_btn.grid(row=0, column=1, sticky="ew", padx=(5, 0))
        self.reset_btn = ttk.Button(buttons_frame, text="Reset", command=self.reset_app)
        self.reset_btn.grid(row=0, column=2, sticky="ew", padx=(5, 0))

        self.progress_frame = ttk.Frame(self.input_frame)
        self.progress_frame.grid(row=7, column=0, columnspan=3, sticky="ew", pady=5)
        self.progress_bar = ttk.Progressbar(self.progress_frame, mode='indeterminate', length=250)
        self.progress_label = ttk.Label(self.progress_frame, text="", font=("Arial", 9))
        ttk.Label(self.input_frame, text="Key Results:", font=("Arial", 11, "bold")).grid(row=8, column=0, columnspan=3, sticky="w", pady=(15,5))
        self.results_text = tk.Text(self.input_frame, height=12, width=30, wrap="word", font=("Courier", 9))
        self.results_text.grid(row=9, column=0, columnspan=3, sticky="nsew", pady=5)
        self.input_frame.grid_rowconfigure(9, weight=1)
        self.input_frame.grid_columnconfigure(1, weight=1)
        self.output_frame = ttk.LabelFrame(self, text="Analysis Outputs & Graphs", padding=(15,15))
        self.output_frame.grid(row=1, column=1, sticky="nsew", padx=10, pady=5)
        self.fig = Figure(figsize=(12,6), dpi=90)
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.output_frame)
        self.canvas.get_tk_widget().pack(fill="both", expand=True)
        self.fig.text(0.5, 0.5, 'Click "Run Complete Analysis" to begin', ha='center', va='center', fontsize=16, color='gray')
        self.canvas.draw()
        self.diagnostics_frame = ttk.LabelFrame(self, text="Diagnostics & Execution Log", padding=(15,10))
        self.diagnostics_frame.grid(row=2, column=0, columnspan=2, sticky="nsew", padx=10, pady=10)
        diag_scroll = ttk.Scrollbar(self.diagnostics_frame)
        diag_scroll.pack(side="right", fill="y")
        self.diagnostics_text = tk.Text(self.diagnostics_frame, height=10, wrap="word", font=("Consolas", 9), yscrollcommand=diag_scroll.set)
        self.diagnostics_text.pack(fill="both", expand=True)
        diag_scroll.config(command=self.diagnostics_text.yview)
        self.diagnostics_text.insert("end", "="*100 + "\n")
        self.diagnostics_text.insert("end", "Group 12 - CMM3 EV Battery Fast-Charge Optimization System\n")
        self.diagnostics_text.insert("end", "="*100 + "\n")
        self.diagnostics_text.insert("end", "Status: Ready. Awaiting analysis run...\n\n")

    def log(self, message):
        """
        Log messages to the diagnostics frame at the bottom of the application.
        Args:
            message (str): Message text to log
        """
        self.diagnostics_text.insert("end", message + "\n")
        self.diagnostics_text.see("end")

    def update_progress(self, message):
        """
        Update the progress label text.
        Args:
            message (str): Progress message
        """
        self.progress_label.config(text=message)

    def cancel_analysis(self):
        """
        Signal the analysis thread to cancel processing.
        """
        if self.running:
            self.cancel_event.set()
            self.cancel_btn.config(state="disabled")
            self.log("\n[INFO] Cancellation signal sent. Waiting for thread to terminate...")

    def run_analysis(self):
        """
        Start the analysis run in a separate thread and manage UI state changes.
        """
        if self.running:
            messagebox.showwarning("Analysis Running", "An analysis is already in progress.")
            return
        try:
            threshold = float(self.threshold_entry.get())
            initial_current = float(self.current_entry.get())
            step_size = float(self.step_entry.get())
            t_b_max = float(self.t_b_max_entry.get()) + 273.13  # Convert to Kelvin
            t_in = float(self.t_in_entry.get()) + 273.13  # Convert to Kelvin
        except ValueError as e:
            messagebox.showerror("Input Error", f"Invalid input: {str(e)}. Please check all fields are numbers.")
            return
        
        # Validate ranges
        if not (1e-8 <= threshold <= 1e-4):
            messagebox.showwarning("Range Warning", "Convergence threshold recommended: [1e-8 to 1e-4]")
        if not (1 <= initial_current <= 17.5):
            messagebox.showwarning("Range Warning", "Initial current recommended: [1 to 17.5 A]")
        if not (10 <= step_size <= 60):
            messagebox.showwarning("Range Warning", "RK4 step size recommended: [10 to 60 s]")
        if not (35 <= (t_b_max - 273.13) <= 50):
            messagebox.showwarning("Range Warning", "Max battery temp recommended: [35 to 50 °C]")
        if not (10 <= (t_in - 273.13) <= 25):
            messagebox.showwarning("Range Warning", "Coolant inlet temp recommended: [10 to 25 °C]")
        
        self.running = True
        self.cancel_event.clear()
        self.run_btn.config(state="disabled", text="Running...")
        self.cancel_btn.config(state="normal")
        self.results_text.delete("1.0", "end")
        self.progress_bar.pack(side="left", padx=5)
        self.progress_label.pack(side="left", padx=5)
        self.progress_bar.start(10)
        self.update_progress("Initializing...")
        analysis_thread = threading.Thread(target=self.perform_analysis, args=(threshold, initial_current, step_size, t_b_max, t_in, self.cancel_event), daemon=True)
        analysis_thread.start()
        self.check_results()

    def perform_analysis(self, threshold, initial_current, step_size, t_b_max, t_in, cancel_event):
        """
        Main analysis routine following exact workflow from main.py.
        Iteratively solves for optimum current, then runs all module analyses.
        Uses GUI inputs to override config values dynamically.
        """
        global config, ODE, rk4e, mf, oc, rct, ca, hi
        
        result = {"success": False, "data": {}, "error": None, "status": "running"}
        try:
            # Force reimport of config in all dependent modules by deleting cached modules FIRST
            # This ensures all modules read the fresh config values
            modules_to_reload = [
                'src.config',
                'src.models.battery_temperature_ode',
                'src.models.rk4_error', 
                'src.models.mass_flowrate',
                'src.models.optimum_current',
                'scripts.charging_time_analysis',
                'src.models.cooling_analysis',
                'src.utils.heptane_interpolater'
            ]
            for mod_name in modules_to_reload:
                if mod_name in sys.modules:
                    del sys.modules[mod_name]
            
            # Re-import config first
            import src.config as config
            
            # Now update config with GUI inputs AFTER reimporting
            config.current_0 = initial_current
            config.H = step_size
            config.t_b_max = t_b_max
            config.t_in = t_in
            
            # Now re-import all other modules with updated config
            import src.models.battery_temperature_ode as ODE
            from src.models.battery_temperature_ode import get_tb, d_tb_dt, get_tb_scipy
            import src.models.rk4_error as rk4e
            import src.models.mass_flowrate as mf
            import src.models.optimum_current as oc
            from src.models.optimum_current import current_params
            import scripts.charging_time_analysis as rct
            import src.models.cooling_analysis as ca
            import src.utils.heptane_interpolater as hi
            
            if cancel_event.is_set(): return
            required_modules = {'oc', 'ODE', 'mf', 'rk4e', 'rct', 'hi'}
            for mod_name in required_modules:
                if mod_name not in globals():
                    raise ImportError(f"Required analysis module '{mod_name}' failed to import.")
            
            self.result_queue.put({"type": "log", "message": "\n" + "="*100})
            self.result_queue.put({"type": "log", "message": "[STARTING] Group 12 - CMM3 Consolidated Analysis"})
            self.result_queue.put({"type": "log", "message": f"Parameters: T_b_max={t_b_max-273.13:.1f}°C, T_in={t_in-273.13:.1f}°C, I_0={initial_current:.1f}A, H={step_size:.0f}s"})
            self.result_queue.put({"type": "log", "message": f"Config values after update: current_0={config.current_0}, H={config.H}, t_b_max={config.t_b_max}, t_in={config.t_in}"})
            self.result_queue.put({"type": "log", "message": "="*100})
            
            # Step 1: Compute optimum current
            if cancel_event.is_set(): return
            self.result_queue.put({"type": "progress", "message": "Step 1/7: Optimum Current Convergence..."})
            self.result_queue.put({"type": "log", "message": "\n[1/7] Running Optimum Current convergence loop... (~60-180s)"})
            
            # Clear and initialize current_store from the newly reloaded config
            config.current_store.clear()
            config.current_error.clear()
            result_oc = oc.run()
            current = result_oc['critical'][0]
            config.current_store.append(current)
            
            iteration = 1
            converged = False
            while not converged:
                if cancel_event.is_set(): return
                result_oc = oc.run()
                new_current = result_oc['critical'][0]
                config.current_store.append(new_current)
                iteration += 1
                # Convergence check
                if len(config.current_store) > 1 and abs(config.current_store[-1] - config.current_store[-2]) / config.current_store[-1] < threshold:
                    converged = True
                if iteration > 17.5:
                    self.result_queue.put({"type": "log", "message": "      Warning: Max iterations reached"})
                    break
            
            if cancel_event.is_set(): return
            optimum_current = config.current_store[-1]
            current_error_avg = np.mean(config.current_error) if config.current_error else 0.0
            self.result_queue.put({"type": "log", "message": f"      ✓ Converged after {len(config.current_store)} iterations"})
            self.result_queue.put({"type": "log", "message": f"      Critical Current: {optimum_current:.4f} ± {current_error_avg:.4f}A"})
            self.result_queue.put({"type": "result", "message": f"Critical Current:\n  {optimum_current:.4f} ±{current_error_avg:.4f} A\n\n"})
            
            # Step 2: Run all analyses
            if cancel_event.is_set(): return
            self.result_queue.put({"type": "progress", "message": "Step 2/7: RK4 Error Analysis..."})
            self.result_queue.put({"type": "log", "message": "\n[2/7] RK4 Error Analysis"})
            rk4e.run()
            self.result_queue.put({"type": "log", "message": "      ✓ RK4 error analysis complete"})
            
            if cancel_event.is_set(): return
            self.result_queue.put({"type": "progress", "message": "Step 3/7: Mass Flowrate Solver..."})
            self.result_queue.put({"type": "log", "message": "\n[3/7] Mass Flowrate Solver"})
            mf_data = mf.run()
            self.result_queue.put({"type": "log", "message": "      ✓ Mass flowrate computed"})
            self.result_queue.put({"type": "result", "message": f"Mass Flowrate: \n\n"})
            
            if cancel_event.is_set(): return
            self.result_queue.put({"type": "progress", "message": "Step 4/7: Optimum Current Analysis..."})
            self.result_queue.put({"type": "log", "message": "\n[4/7] Optimum Current Analysis"})
            oc_data = oc.run()
            self.result_queue.put({"type": "log", "message": "      ✓ Optimum current interpolation complete"})
            
            if cancel_event.is_set(): return
            self.result_queue.put({"type": "progress", "message": "Step 5/7: Real Charging Time..."})
            self.result_queue.put({"type": "log", "message": "\n[5/7] Real Charging Time"})
            rct_results = rct.run()
            self.result_queue.put({"type": "log", "message": "      ✓ Real charging time computed"})
            if rct_results:
                self.result_queue.put({"type": "result", "message": f"Charging Performance Analysis:\n\n"})
                cr_val = rct_results.get('critical_C_rate', 'N/A')
                cr_err = rct_results.get('critical_C_rate_err', 'N/A')
                self.result_queue.put({"type": "result", "message": f"  Critical C-Rate:\n    {cr_val:.2f} ± {cr_err:.2f} C\n\n"})
                ft_val = rct_results.get('fastest_charge_min', 'N/A')
                ft_err = rct_results.get('fastest_charge_min_err', 'N/A')
                self.result_queue.put({"type": "result", "message": f"  Fastest Charge Time (Theoretical):\n    {ft_val:.1f} ± {ft_err:.1f} min\n\n"})
                rc_val = rct_results.get('recommended_C_rate', 'N/A')
                rc_err = rct_results.get('recommended_C_rate_err', 'N/A')
                self.result_queue.put({"type": "result", "message": f"  Recommended C-Rate (Practical):\n    {rc_val:.2f} ± {rc_err:.2f} C\n\n"})
                rct_val = rct_results.get('recommended_charge_min', 'N/A')
                rct_err = rct_results.get('recommended_charge_min_err', 'N/A')
                self.result_queue.put({"type": "result", "message": f"  Recommended Charge Time:\n    {rct_val:.1f} ± {rct_err:.1f} min\n\n"})
            else:
                self.result_queue.put({"type": "log", "message": "      Warning: Could not retrieve charging time results."})
            
            if cancel_event.is_set(): return
            self.result_queue.put({"type": "progress", "message": "Step 6/7: Heptane Fluid Properties..."})
            self.result_queue.put({"type": "log", "message": "\n[6/7] Heptane Fluid Properties"})
            hi_data = hi.run()
            self.result_queue.put({"type": "log", "message": "      ✓ Heptane properties analyzed"})
            
            # Step 3: Generate ODE solutions
            if cancel_event.is_set(): return
            self.result_queue.put({"type": "progress", "message": "Step 7/7: ODE Solution (RK4 vs SciPy)..."})
            self.result_queue.put({"type": "log", "message": "\n[7/7] ODE Solution Comparison"})
            time_rk4, temp_rk4 = get_tb(d_tb_dt, current_params(config.current_store[-1]), stepsize=step_size)
            time_scipy, temp_scipy = get_tb_scipy(d_tb_dt, current_params(config.current_store[-1]))
            self.result_queue.put({"type": "log", "message": f"      ✓ ODE solved: {len(time_rk4)} RK4 points, {len(time_scipy)} SciPy points"})
            
            result["success"] = True
            result["status"] = "success"
            result["data"] = {
                "time_rk4": time_rk4,
                "temp_rk4": temp_rk4,
                "time_scipy": time_scipy,
                "temp_scipy": temp_scipy,
                "mf_data": mf_data,
                "oc_data": oc_data,
                "optimum_current": optimum_current,
                "step_size": step_size,
                "t_b_max": t_b_max,
                "t_in": t_in
            }
            self.result_queue.put({"type": "progress", "message": "Generating plots..."})
            self.result_queue.put({"type": "log", "message": "\n[PLOTTING] Generating visualization..."})
            
        except Exception as e:
            result["status"] = "error"
            result["error"] = str(e)
            self.result_queue.put({"type": "log", "message": f"\n[ERROR] Analysis failed: {str(e)}"})
            import traceback
            self.result_queue.put({"type": "log", "message": traceback.format_exc()})
        finally:
            if cancel_event.is_set():
                result["status"] = "cancelled"
                self.result_queue.put({"type": "log", "message": "\n[CANCELLED] Analysis was cancelled by the user."})
            self.result_queue.put({"type": "complete", "result": result})

    def check_results(self):
        """
        Monitor result queue for logs, progress, and final output from the analysis thread.
        """
        try:
            while True:
                message = self.result_queue.get_nowait()
                if message["type"] == "log":
                    self.log(message["message"])
                elif message["type"] == "progress":
                    self.update_progress(message["message"])
                elif message["type"] == "result":
                    self.results_text.insert("end", message["message"])
                elif message["type"] == "complete":
                    self.handle_completion(message["result"])
                    return
        except queue.Empty:
            pass
        if self.running:
            self.after(100, self.check_results)

    def reset_app(self):
        """
        Resets the GUI to its initial state, clearing results, logs, and plots.
        """
        if self.running:
            messagebox.showwarning("Analysis Running", "Please cancel the current analysis before resetting.")
            return
        self.results_text.delete("1.0", "end")
        self.diagnostics_text.delete("1.0", "end")
        self.diagnostics_text.insert("end", "="*100 + "\n")
        self.diagnostics_text.insert("end", "Group 12 - CMM3 EV Battery Fast-Charge Optimization System\n")
        self.diagnostics_text.insert("end", "="*100 + "\n")
        self.diagnostics_text.insert("end", "Status: Ready. Awaiting analysis run...\n\n")
        self.run_btn.config(state="normal", text="Run Complete Analysis")
        self.cancel_btn.config(state="disabled")
        self.fig.clf()
        self.fig.text(0.5, 0.5, 'Click "Run Complete Analysis" to begin', ha='center', va='center', fontsize=16, color='gray')
        self.canvas.draw()
        # Clear global state variables for the next analysis
        config.current_store.clear()
        config.current_error.clear()
        self.log("[INFO] Application reset to initial state.")

    def handle_completion(self, result):
        """
        Handle completion logic, plotting, and final status after analysis thread ends.
        Uses same plot layout as main.py.
        """
        self.progress_bar.stop()
        self.progress_bar.pack_forget()
        self.progress_label.pack_forget()
        self.running = False
        self.cancel_btn.config(state="disabled")
        self.run_btn.config(state="normal", text="Run Complete Analysis")
        status = result.get("status", "error")
        if status == "error":
            messagebox.showerror("Analysis Error", f"An error occurred:\n{result['error']}")
            self.log("\n" + "="*100)
            self.log("[FAILED] Analysis completed with errors.")
            self.log("="*100 + "\n")
            return
        elif status == "cancelled":
            self.log("\n" + "="*100)
            self.log("[CANCELLED] Analysis was successfully stopped.")
            self.log("="*100 + "\n")
        elif status == "success":
            try:
                # Extract data from result 
                time_rk4 = result["data"]["time_rk4"]
                temp_rk4 = result["data"]["temp_rk4"]
                time_scipy = result["data"]["time_scipy"]
                temp_scipy = result["data"]["temp_scipy"]
                mf_data = result["data"]["mf_data"]
                oc_data = result["data"]["oc_data"]
                optimum_current = result["data"]["optimum_current"]
                
                # Create 1x3 subplot layout matching main.py
                self.fig.clf()
                self.fig.suptitle('Group 12 - CMM3 Complete Analysis', fontsize=18, fontweight='bold')
                
                # Plot 1: ODE Solution Validation (RK4 vs SciPy)
                ax1 = self.fig.add_subplot(131)
                ax1.plot(time_rk4, temp_rk4, 'b--', label=f'RK4 (h={result["data"]["step_size"]:.0f}s)', linewidth=1.5)
                ax1.plot(time_scipy, temp_scipy, 'r-', linewidth=3, alpha=0.6, label='SciPy LSODA')
                ax1.set_xlabel('Time (s)', fontsize=10)
                ax1.set_ylabel('Battery Temperature $T_b$ (K)', fontsize=10)
                ax1.set_title('ODE Solution Validation', fontsize=12, fontweight='bold')
                ax1.legend(fontsize=9)
                ax1.grid(True, alpha=0.3)
                
                # Plot 2: Mass Flowrate Pressure Balance
                ax2 = self.fig.add_subplot(132)
                ax2.plot(mf_data['mass_flow'], mf_data['residuals'], 'b-', linewidth=2)
                ax2.axhline(0, color='k', linestyle='--', linewidth=1)
                ax2.set_xlabel('Mass Flow Rate (kg/s)', fontsize=10)
                ax2.set_ylabel('Pressure Residual (Pa)', fontsize=10)
                ax2.set_title('Pressure Balance Residual', fontsize=12, fontweight='bold')
                ax2.grid(True, alpha=0.3)
                
                # Plot 3: Optimum Current Analysis
                ax3 = self.fig.add_subplot(133)
                current_smooth, delta_t_smooth = oc_data['smooth']
                critical_current, critical_y = oc_data['critical']
                ax3.plot(current_smooth, delta_t_smooth, 'r-', linewidth=2, label='Cubic Spline Interpolation')
                ax3.plot(critical_current, critical_y, 'ro', markersize=10, 
                         label=f'Critical Point: {critical_current:.1f} A', zorder=6)
                ax3.axhline(0, color='red', linestyle='--', linewidth=1)
                ax3.set_xlabel('Current (A)', fontsize=10)
                ax3.set_ylabel('$\\Delta T$ (K)', fontsize=10)
                ax3.set_title('Optimum Current Analysis', fontsize=12, fontweight='bold')
                ax3.legend(fontsize=9)
                ax3.grid(True, alpha=0.3)
                
                self.fig.tight_layout(rect=(0, 0, 1, 0.96))
                self.canvas.draw()
                self.log("\n[INFO] Plots generated successfully in GUI.")

                # --- Save plot and results to /results folder ---
                RESULTS_DIR = os.path.join(os.path.dirname(__file__), '..', 'results')
                FIGURES_DIR = os.path.join(RESULTS_DIR, 'figures')
                TABLES_DIR = os.path.join(RESULTS_DIR, 'tables')
                os.makedirs(FIGURES_DIR, exist_ok=True)
                os.makedirs(TABLES_DIR, exist_ok=True)

                plot_path = os.path.join(FIGURES_DIR, 'gui_analysis_plot.png')
                self.fig.savefig(plot_path, dpi=300)
                self.log(f"[INFO] Plot saved to {os.path.relpath(plot_path)}")

                summary_path = os.path.join(TABLES_DIR, 'gui_summary.txt')
                with open(summary_path, 'w') as f:
                    f.write("==== Group 12 - CMM3 GUI Analysis Summary ====\n\n")
                    f.write(self.results_text.get("1.0", "end"))
                self.log(f"[INFO] Results summary saved to {os.path.relpath(summary_path)}")
                self.log("\n" + "="*100)
                self.log("[COMPLETE] All computations and visualizations finished successfully!")
                self.log("="*100 + "\n")
                self.results_text.insert("end", "Status: COMPLETE\n\n")
            
            except Exception as e:
                self.log(f"\n[ERROR] Failed to generate plots: {str(e)}")
                import traceback
                self.log(traceback.format_exc())
                messagebox.showerror("Plotting Error", f"Failed to generate plots:\n{str(e)}")

if __name__ == "__main__":
    app = CMM3App()
    app.mainloop()
