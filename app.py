import tkinter as tk
from tkinter import ttk
from tkinter import messagebox

import matplotlib
matplotlib.use("TkAgg")
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.pyplot as plt

import numpy as np
try:
    import Optimum_Current as oc
    import Mass_flowrate as mf
    import ODE
    import RK4_Error as rk4e
    import Real_Charging_Time as rct
    import cooling_analysis as ca
    import heptane_itpl as hi
    from config import I_Threshold
except ImportError as e:
    print(f"Warning: Could not import module: {e}")

class CMM3App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Group-12-CMM3 Fast-Charge Academic Analysis")
        self.geometry("1400x900")
        self.configure(bg="white")
        self.create_widgets()
        self.grid_rowconfigure(1, weight=3)
        self.grid_rowconfigure(2, weight=1)
        self.grid_columnconfigure(0, weight=1)
        self.grid_columnconfigure(1, weight=3)

    def create_widgets(self):
        # Title
        title = ttk.Label(self, text="Group 12 CMM3 — EV Battery Fast-Charge Optimization", 
                         font=("Arial", 22, "bold"))
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
        
        # Separator
        ttk.Separator(self.input_frame, orient="horizontal").grid(row=3, column=0, columnspan=2, sticky="ew", pady=10)
        
        # Run button
        self.run_btn = ttk.Button(self.input_frame, text="Run Complete Analysis", 
                                  command=self.run_analysis)
        self.run_btn.grid(row=4, column=0, columnspan=2, pady=15, sticky="ew")
        
        # Results display
        ttk.Label(self.input_frame, text="Key Results:", font=("Arial", 11, "bold")).grid(row=5, column=0, columnspan=2, sticky="w", pady=(15,5))
        self.results_text = tk.Text(self.input_frame, height=12, width=30, wrap="word", font=("Courier", 9))
        self.results_text.grid(row=6, column=0, columnspan=2, sticky="nsew", pady=5)
        self.input_frame.grid_rowconfigure(6, weight=1)
        self.input_frame.grid_columnconfigure(1, weight=1)

        # Top right: Outputs and Graphs
        self.output_frame = ttk.LabelFrame(self, text="Analysis Outputs & Graphs", padding=(15,15))
        self.output_frame.grid(row=1, column=1, sticky="nsew", padx=10, pady=5)
        
        self.fig = plt.Figure(figsize=(12,6), dpi=90)
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.output_frame)
        self.canvas.get_tk_widget().pack(fill="both", expand=True)
        
        # Initialize with welcome message
        self.fig.text(0.5, 0.5, 'Click "Run Complete Analysis" to begin', 
                     ha='center', va='center', fontsize=16, color='gray')
        self.canvas.draw()

        # Bottom: Diagnostics/log section spans both columns
        self.diagnostics_frame = ttk.LabelFrame(self, text="Diagnostics & Execution Log", padding=(15,10))
        self.diagnostics_frame.grid(row=2, column=0, columnspan=2, sticky="nsew", padx=10, pady=10)
        
        # Add scrollbar
        diag_scroll = ttk.Scrollbar(self.diagnostics_frame)
        diag_scroll.pack(side="right", fill="y")
        
        self.diagnostics_text = tk.Text(self.diagnostics_frame, height=10, wrap="word", 
                                       font=("Consolas", 9), yscrollcommand=diag_scroll.set)
        self.diagnostics_text.pack(fill="both", expand=True)
        diag_scroll.config(command=self.diagnostics_text.yview)
        
        self.diagnostics_text.insert("end", "="*100 + "\n")
        self.diagnostics_text.insert("end", "Group 12 - CMM3 EV Battery Fast-Charge Optimization System\n")
        self.diagnostics_text.insert("end", "="*100 + "\n")
        self.diagnostics_text.insert("end", "Status: Ready. Awaiting analysis run...\n\n")

    def log(self, message):
        """Helper function to log messages to diagnostics"""
        self.diagnostics_text.insert("end", message + "\n")
        self.diagnostics_text.see("end")
        self.update_idletasks()

    def run_analysis(self):
        """Main analysis execution function"""
        try:
            threshold = float(self.threshold_entry.get())
        except ValueError:
            messagebox.showerror("Input Error", "Threshold must be a float (e.g., 1e-6)")
            return

        self.log("\n" + "="*100)
        self.log("[STARTING] Complete analysis run...")
        self.log("="*100)
        self.run_btn.config(state="disabled", text="Running...")
        self.results_text.delete("1.0", "end")

        try:
            # Optimum Current Convergence
            self.log("\n[1/6] Running Optimum Current convergence loop...")
            I_store = []
            oc_result = oc.run()
            current = oc_result['critical'][0]
            I_store.append(current)
            
            iteration = 1
            converged = False
            while not converged:
                new_result = oc.run()
                new_current = new_result['critical'][0]
                I_store.append(new_current)
                iteration += 1
                if len(I_store)>1 and abs(I_store[-1] - I_store[-2]) < threshold:
                    converged = True
                if iteration > 100:  # Safety break
                    self.log("      Warning: Max iterations reached")
                    break
            
            optimum_current = I_store[-1]
            self.log(f"      Converged after {len(I_store)} iterations")
            self.log(f"      Optimum Current: {optimum_current:.4f} A")
            self.results_text.insert("end", f"Optimum Current:\n  {optimum_current:.4f} A\n\n")

            # ODE Solution
            self.log("\n[2/6] Solving ODE (RK4 vs SciPy)...")
            ode_data = ODE.run()
            t_rk, T_rk = ode_data['rk4']
            t_scipy, T_scipy = ode_data['scipy']
            self.log(f"      ODE solved: {len(t_rk)} RK4 points, {len(t_scipy)} SciPy points")
            self.results_text.insert("end", f"ODE Solution:\n  RK4: {len(t_rk)} points\n  SciPy: {len(t_scipy)} points\n\n")

            # Mass Flowrate
            self.log("\n[3/6] Computing mass flowrate...")
            mf_data = mf.run()
            self.log(f"      Mass flowrate computed")
            self.results_text.insert("end", f"Mass Flowrate:\n  Computed successfully\n\n")

            # RK4 Error Analysis
            self.log("\n[4/6] Performing RK4 error analysis...")
            rk4e.run()
            self.log("      RK4 error analysis complete")

            # Real Charging Time
            self.log("\n[5/6] Computing real charging time...")
            rct.run()
            self.log("      Real charging time computed")

            # Heptane Properties
            self.log("\n[6/6] Analyzing heptane fluid properties...")
            hi_data = hi.run()
            self.log("      Heptane properties analyzed")

            # Optimum current data for plotting
            oc_data = oc.run()

            # Create plots
            self.log("\n[PLOTTING] Generating visualization...")
            self.fig.clf()
            self.fig.suptitle('Group 12 - CMM3 Complete Analysis', fontsize=18, fontweight='bold')
            
            # Plot 1: ODE Solution
            ax1 = self.fig.add_subplot(131)
            ax1.plot(t_rk, T_rk, 'b--', label="RK4", linewidth=1.5)
            ax1.plot(t_scipy, T_scipy, 'r-', label="SciPy LSODA", linewidth=2, alpha=0.7)
            ax1.set_xlabel('Time (s)', fontsize=10)
            ax1.set_ylabel('Battery Temperature Tb (K)', fontsize=10)
            ax1.set_title('ODE Solution Validation', fontsize=11, fontweight='bold')
            ax1.legend(fontsize=9)
            ax1.grid(True, alpha=0.3)

            # Plot 2: Mass Flowrate
            ax2 = self.fig.add_subplot(132)
            ax2.plot(mf_data['mass_flow'], mf_data['residuals'], 'b-', linewidth=2)
            ax2.axhline(0, color='k', linestyle='--', linewidth=1)
            ax2.set_xlabel('Mass Flow Rate (kg/s)', fontsize=10)
            ax2.set_ylabel('Pressure Residual (Pa)', fontsize=10)
            ax2.set_title('Pressure Balance Residual', fontsize=11, fontweight='bold')
            ax2.grid(True, alpha=0.3)

            # Plot 3: Optimum Current
            ax3 = self.fig.add_subplot(133)
            I_smooth, delta_T_smooth = oc_data['smooth']
            critical_current, critical_y = oc_data['critical']
            ax3.plot(I_smooth, delta_T_smooth, 'r-', linewidth=2, label='Cubic Spline')
            ax3.plot(critical_current, critical_y, 'ro', markersize=10, 
                    label=f'Critical: {critical_current:.1f} A', zorder=6)
            ax3.axhline(0, color='red', linestyle='--', linewidth=1)
            ax3.set_xlabel('Current (A)', fontsize=10)
            ax3.set_ylabel('Delta T (K)', fontsize=10)
            ax3.set_title('Optimum Current Analysis', fontsize=11, fontweight='bold')
            ax3.legend(fontsize=9)
            ax3.grid(True, alpha=0.3)

            self.fig.tight_layout(rect=[0, 0, 1, 0.96])
            self.canvas.draw()
            
            self.log("      Plots generated successfully")
            self.log("\n" + "="*100)
            self.log("[COMPLETE] All computations and visualizations finished successfully!")
            self.log("="*100 + "\n")
            
            self.results_text.insert("end", "Status: COMPLETE")

        except Exception as e:
            self.log(f"\n[ERROR] Analysis failed: {str(e)}")
            self.log(f"        {type(e).__name__}: {e}")
            messagebox.showerror("Analysis Error", f"An error occurred:\n{str(e)}")
        
        finally:
            self.run_btn.config(state="normal", text="Run Complete Analysis")

if __name__ == "__main__":
    app = CMM3App()
    app.mainloop()
