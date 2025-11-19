"""
Main execution script for Group-12-CMM3 consolidated analysis and visualization.

Runs all module analyses and presents a unified results dashboard.
"""
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import matplotlib.pyplot as plt
from src.config import *
import src.models.battery_temperature_ode as battery_temperature_ode
from src.models.battery_temperature_ode import get_tb, d_tb_dt, get_tb_scipy
import src.models.rk4_error as rk4e
import src.models.mass_flowrate as mf
import src.models.optimum_current as oc
from src.models.optimum_current import current_params
from src.config import current_store, current_threshold
import scripts.charging_time_analysis as rct
import src.models.cooling_analysis as ca
import src.utils.heptane_interpolater as hi

def compute_optimum_current(threshold=current_threshold):
    """
    Iteratively solves for critical optimum current until convergence below threshold.
    Args:
        threshold (float): Convergence tolerance.
    Returns:
        list: List of current guesses with final critical value last.
    """
    mf_data = mf.run()
    current_store.clear()
    result = oc.run()
    # Defensive checks: ensure 'critical' exists and has at least one value
    if 'critical' not in result or len(result['critical']) == 0:
        raise ValueError("oc.run() returned no 'critical' value to initialise current_store")
    current = result['critical'][0]
    current_store.append(current)
    # Ensure we check length first to avoid indexing when list has only one element
    while len(current_store) < 2 or abs(current_store[-1] - current_store[-2]) / current_store[-1] > threshold:
        result = oc.run()
        if 'critical' not in result or len(result['critical']) == 0:
            raise ValueError("oc.run() returned no 'critical' value during iteration")
        new_current = result['critical'][0]
        current_store.append(new_current)
    return current_store

if __name__ == "__main__":
    # --- 1. Define and create output directories ---
    RESULTS_DIR = os.path.join(os.path.dirname(__file__), '..', 'results')
    FIGURES_DIR = os.path.join(RESULTS_DIR, 'figures')
    TABLES_DIR = os.path.join(RESULTS_DIR, 'tables')
    os.makedirs(FIGURES_DIR, exist_ok=True)
    os.makedirs(TABLES_DIR, exist_ok=True)

    # Run convergence loop for optimum current
    current_store_result = compute_optimum_current(threshold=1e-6)
    optimum_current = current_store_result[-1]
    num_iterations = len(current_store_result)

    print(f'\n✓ Model converged after {num_iterations} iterations')
    print(f'Optimum Current: {optimum_current:.4f} A\n')
    print("==== Group 12 - CMM3 Consolidated Results ====\n")
    print("\n--- RK4 Error Analysis ---")
    rk4e.run()
    print("\n--- Mass Flowrate Solver ---")
    mf_data = mf.run()
    print("\n--- Optimum Current Analysis ---")
    oc_data = oc.run()
    print("\n--- Real Charging Time ---")
    rct_results = rct.run()
    print("\n--- Heptane Fluid Properties ---")
    hi_data = hi.run()

    # Create centralized subplots (1x3 grid = 3 plots)
    fig = plt.figure(figsize=(18, 6))
    fig.suptitle('Group 12 - CMM3 Complete Analysis', fontsize=20, fontweight='bold')
    # Plot 1: ODE (RK4 vs SciPy)
    ax1 = plt.subplot(1, 3, 1)
    time_rk4, temp_rk4 = get_tb(d_tb_dt, current_params(optimum_current), stepsize = H)
    time_scipy, temp_scipy = get_tb_scipy(d_tb_dt, current_params(optimum_current))
    ax1.plot(time_rk4, temp_rk4, 'b--', label=f'RK4 (h={H}s)', linewidth=1.5)
    ax1.plot(time_scipy, temp_scipy, 'r-', linewidth=3, alpha=0.6, label='SciPy LSODA')
    ax1.set_xlabel('Time (s)', fontsize=10)
    ax1.set_ylabel('Battery Temperature $T_b$ (K)', fontsize=10)
    ax1.set_title('ODE Solution Validation', fontsize=12, fontweight='bold')
    ax1.legend(fontsize=9)
    ax1.grid(True, alpha=0.3)
    # Plot 2: Mass Flowrate Pressure Balance
    ax2 = plt.subplot(1, 3, 2)
    ax2.plot(mf_data['mass_flow'], mf_data['residuals'], 'b-', linewidth=2)
    ax2.axhline(0, color='k', linestyle='--', linewidth=1)
    ax2.set_xlabel('Mass Flow Rate (kg/s)', fontsize=10)
    ax2.set_ylabel('Pressure Residual (Pa)', fontsize=10)
    ax2.set_title('Pressure Balance Residual', fontsize=12, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    # Plot 3: Optimum Current
    ax3 = plt.subplot(1, 3, 3)
    current_smooth, delta_t_smooth = oc_data['smooth']
    critical_current, critical_y = oc_data['critical']
    ax3.plot(current_smooth, delta_t_smooth, 'r-', linewidth=2, label='Cubic Spline Interpolation')
    ax3.plot(critical_current, critical_y, 'ro', markersize=10, label=f'Critical Point: {critical_current:.1f} A', zorder=6)
    ax3.axhline(0, color='red', linestyle='--', linewidth=1)
    ax3.set_xlabel('Current (A)', fontsize=10)
    ax3.set_ylabel('$\\Delta T$ (K)', fontsize=10)
    ax3.set_title('Optimum Current Analysis', fontsize=12, fontweight='bold')
    ax3.legend(fontsize=9)
    ax3.grid(True, alpha=0.3)
    plt.tight_layout(rect=(0, 0, 1, 0.96))

    # --- 3. Save plot and results summary ---
    plot_path = os.path.join(FIGURES_DIR, 'main_analysis_plot.png')
    fig.savefig(plot_path, dpi=300)
    print(f"\n✓ Plot saved to {os.path.relpath(plot_path)}")

    summary_path = os.path.join(TABLES_DIR, 'main_summary.txt')
    with open(summary_path, 'w') as f:
        f.write("==== Group 12 - CMM3 CLI Analysis Summary ====\n\n")
        f.write(f"Optimum Current: {optimum_current:.4f} A\n")
        f.write(f"Convergence Iterations: {num_iterations}\n\n")
        if rct_results:
            f.write("--- Charging Performance ---\n")
            f.write(f"  Recommended C-Rate: {rct_results.get('recommended_C_rate', 'N/A')}C\n")
            f.write(f"  Recommended Charge Time: {rct_results.get('recommended_charge_min', 'N/A')} min\n")
            f.write(f"  Critical C-Rate: {rct_results.get('critical_C_rate', 'N/A')}C\n")
            f.write(f"  Fastest Theoretical Charge Time: {rct_results.get('fastest_charge_min', 'N/A')} min\n")
    print(f"✓ Results summary saved to {os.path.relpath(summary_path)}")


    print("\n✓ All computations and plots complete!")
    try:
        manager = plt.get_current_fig_manager()
        if hasattr(manager.window, 'showMaximized'):
            # PyQt backend
            manager.window.showMaximized()
        elif hasattr(manager.window, 'state'):
            # Tkinter backend
            manager.window.state('zoomed')
    except Exception:
        # Fallback: if neither method works, just show the plot
        pass
    plt.show()
