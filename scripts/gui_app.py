import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import tkinter as tk
from tkinter import ttk
from tkinter import messagebox
import threading
import queue
import matplotlib
matplotlib.use("TkAgg")
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
import numpy as np
try:
    import src.models.ODE as ODE
    from src.models.ODE import get_tb, d_tb_dt, get_tb_scipy
    import src.models.RK4_Error as rk4e
    import src.models.Mass_flowrate as mf
    import src.models.Optimum_Current as oc
    from src.models.Optimum_Current import current_params
    import scripts.Real_Charging_Time as rct
    import src.models.cooling_analysis as ca
    import src.utils.heptane_itpl as hi
    from src.config import current_store, current_threshold, H
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
        self.state('zoomed')
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
        title = ttk.Label(self, text="Group 12 CMM3 — EV Battery Fast-Charge Optimization", font=("Arial", 22, "bold"))
        title.grid(row=0, column=0, columnspan=2, pady=15, sticky="ew")
        # Top left: Customizable input section
        self.input_frame = ttk.LabelFrame(self, text="Input Parameters", padding=(15,15))
        self.input_frame.grid(row=1, column=0, sticky="nsew", padx=10, pady=5)
        ttk.Label(self.input_frame, text="Convergence Threshold:", font=("Arial", 10, "bold")).grid(row=0, column=0, sticky="w", pady=5)
        self.threshold_entry = ttk.Entry(self.input_frame, width=20)
        self.threshold_entry.insert(0, "1e-6")
        self.threshold_entry.grid(row=0, column=1, sticky="ew", pady=5, padx=5)
        ttk.Label(self.input_frame, text="Initial Current (A):", font=("Arial", 10)).grid(row=1, column=0, sticky="w", pady=5)
        self.current_entry = ttk.Entry(self.input_frame, width=20)
        self.current_entry.insert(0, "15")
        self.current_entry.grid(row=1, column=1, sticky="ew", pady=5, padx=5)
        ttk.Label(self.input_frame, text="Step Size (s):", font=("Arial", 10)).grid(row=2, column=0, sticky="w", pady=5)
        self.step_entry = ttk.Entry(self.input_frame, width=20)
        self.step_entry.insert(0, "30")
        self.step_entry.grid(row=2, column=1, sticky="ew", pady=5, padx=5)
        ttk.Separator(self.input_frame, orient="horizontal").grid(row=3, column=0, columnspan=2, sticky="ew", pady=10)
        buttons_frame = ttk.Frame(self.input_frame)
        buttons_frame.grid(row=4, column=0, columnspan=2, sticky="ew", pady=10)
        buttons_frame.grid_columnconfigure(0, weight=1)
        buttons_frame.grid_columnconfigure(1, weight=1)
        self.run_btn = ttk.Button(buttons_frame, text="Run Complete Analysis", command=self.run_analysis)
        self.run_btn.grid(row=0, column=0, sticky="ew", padx=(0, 5))
        self.cancel_btn = ttk.Button(buttons_frame, text="Cancel", command=self.cancel_analysis, state="disabled")
        self.cancel_btn.grid(row=0, column=1, sticky="ew", padx=(5, 0))
        self.progress_frame = ttk.Frame(self.input_frame)
        self.progress_frame.grid(row=5, column=0, columnspan=2, sticky="ew", pady=5)
        self.progress_bar = ttk.Progressbar(self.progress_frame, mode='indeterminate', length=250)
        self.progress_label = ttk.Label(self.progress_frame, text="", font=("Arial", 9))
        ttk.Label(self.input_frame, text="Key Results:", font=("Arial", 11, "bold")).grid(row=6, column=0, columnspan=2, sticky="w", pady=(15,5))
        self.results_text = tk.Text(self.input_frame, height=12, width=30, wrap="word", font=("Courier", 9))
        self.results_text.grid(row=7, column=0, columnspan=2, sticky="nsew", pady=5)
        self.input_frame.grid_rowconfigure(7, weight=1)
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
        except ValueError:
            messagebox.showerror("Input Error", "Threshold must be a float (e.g., 1e-6)")
            return
        self.running = True
        self.cancel_event.clear()
        self.run_btn.config(state="disabled", text="Running...")
        self.cancel_btn.config(state="normal")
        self.results_text.delete("1.0", "end")
        self.progress_bar.pack(side="left", padx=5)
        self.progress_label.pack(side="left", padx=5)
        self.progress_bar.start(10)
        self.update_progress("Initializing...")
        analysis_thread = threading.Thread(target=self.perform_analysis, args=(threshold, self.cancel_event), daemon=True)
        analysis_thread.start()
        self.check_results()

    def perform_analysis(self, threshold, cancel_event):
        """
        Main analysis routine following exact workflow from main.py.
        Iteratively solves for optimum current, then runs all module analyses.
        """
        result = {"success": False, "data": {}, "error": None, "status": "running"}
        try:
            if cancel_event.is_set(): return
            required_modules = {'oc', 'ODE', 'mf', 'rk4e', 'rct', 'hi'}
            for mod_name in required_modules:
                if mod_name not in globals():
                    raise ImportError(f"Required analysis module '{mod_name}' failed to import.")
            
            self.result_queue.put({"type": "log", "message": "\n" + "="*100})
            self.result_queue.put({"type": "log", "message": "[STARTING] Group 12 - CMM3 Consolidated Analysis"})
            self.result_queue.put({"type": "log", "message": "="*100})
            
            # Step 1: Compute optimum current (convergence loop from main.py)
            if cancel_event.is_set(): return
            self.result_queue.put({"type": "progress", "message": "Step 1/7: Optimum Current Convergence..."})
            self.result_queue.put({"type": "log", "message": "\n[1/7] Running Optimum Current convergence loop..."})
            
            # Clear and initialize current_store (from main.py line 40)
            current_store.clear()
            result_oc = oc.run()
            current = result_oc['critical'][0]
            current_store.append(current)
            
            iteration = 1
            converged = False
            while not converged:
                if cancel_event.is_set(): return
                result_oc = oc.run()
                new_current = result_oc['critical'][0]
                current_store.append(new_current)
                iteration += 1
                # Convergence check from main.py line 44
                if len(current_store) > 1 and abs(current_store[-1] - current_store[-2]) < threshold:
                    converged = True
                if iteration > 100:
                    self.result_queue.put({"type": "log", "message": "      Warning: Max iterations reached"})
                    break
            
            if cancel_event.is_set(): return
            optimum_current = current_store[-1]
            self.result_queue.put({"type": "log", "message": f"      ✓ Converged after {len(current_store)} iterations"})
            self.result_queue.put({"type": "log", "message": f"      Optimum Current: {optimum_current:.4f} A"})
            self.result_queue.put({"type": "result", "message": f"Optimum Current:\n  {optimum_current:.4f} A\n\n"})
            
            # Step 2: Run all analyses (from main.py lines 47-58)
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
            self.result_queue.put({"type": "result", "message": f"Mass Flowrate:\n  Data collected\n\n"})
            
            if cancel_event.is_set(): return
            self.result_queue.put({"type": "progress", "message": "Step 4/7: Optimum Current Analysis..."})
            self.result_queue.put({"type": "log", "message": "\n[4/7] Optimum Current Analysis"})
            oc_data = oc.run()
            self.result_queue.put({"type": "log", "message": "      ✓ Optimum current interpolation complete"})
            
            if cancel_event.is_set(): return
            self.result_queue.put({"type": "progress", "message": "Step 5/7: Real Charging Time..."})
            self.result_queue.put({"type": "log", "message": "\n[5/7] Real Charging Time"})
            rct.run()
            self.result_queue.put({"type": "log", "message": "      ✓ Real charging time computed"})
            
            if cancel_event.is_set(): return
            self.result_queue.put({"type": "progress", "message": "Step 6/7: Heptane Fluid Properties..."})
            self.result_queue.put({"type": "log", "message": "\n[6/7] Heptane Fluid Properties"})
            hi_data = hi.run()
            self.result_queue.put({"type": "log", "message": "      ✓ Heptane properties analyzed"})
            
            # Step 3: Generate ODE solutions (from main.py lines 60-62)
            if cancel_event.is_set(): return
            self.result_queue.put({"type": "progress", "message": "Step 7/7: ODE Solution (RK4 vs SciPy)..."})
            self.result_queue.put({"type": "log", "message": "\n[7/7] ODE Solution Comparison"})
            time_rk4, temp_rk4 = get_tb(d_tb_dt, current_params(current_store[-1]), stepsize=H)
            time_scipy, temp_scipy = get_tb_scipy(d_tb_dt, current_params(current_store[-1]))
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
                "optimum_current": optimum_current
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
                # Extract data from result (from main.py structure)
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
                ax1.plot(time_rk4, temp_rk4, 'b--', label=f'RK4 (h={H}s)', linewidth=1.5)
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
                self.log("      ✓ Plots generated successfully")
                self.log("\n" + "="*100)
                self.log("[COMPLETE] All computations and visualizations finished successfully!")
                self.log("="*100 + "\n")
                self.results_text.insert("end", "Status: COMPLETE\n\n")
                self.results_text.insert("end", f"Final Optimum Current: {optimum_current:.4f} A")
            except Exception as e:
                self.log(f"\n[ERROR] Failed to generate plots: {str(e)}")
                import traceback
                self.log(traceback.format_exc())
                messagebox.showerror("Plotting Error", f"Failed to generate plots:\n{str(e)}")

if __name__ == "__main__":
    app = CMM3App()
    app.mainloop()
