"""
Main execution script for Group-12-CMM3 consolidated analysis and visualization.

Runs all module analyses and presents a unified results dashboard. PEP8-style docstrings and structured comments included.
"""
import numpy as np
import matplotlib.pyplot as plt
from config import *
import ODE
import RK4_Error as rk4e
import Mass_flowrate as mf
import Optimum_Current as oc
import Real_Charging_Time as rct
import cooling_analysis as ca
import heptane_itpl as hi

def compute_optimum_current(threshold=I_Threshold):
    """
    Iteratively solves for critical optimum current until convergence below threshold.
    Args:
        threshold (float): Convergence tolerance.
    Returns:
        list: List of current guesses with final critical value last.
    """
    I_store.clear()
    result = oc.run()
    current = result['critical'][0]
    I_store.append(current)
    while True:
        result = oc.run()
        new_current = result['critical'][0]
        I_store.append(new_current)
        if len(I_store) > 1 and abs(I_store[-1] - I_store[-2]) < threshold:
            break
    return I_store

if __name__ == "__main__":
    # Run convergence loop for optimum current
    I_store_result = compute_optimum_current(threshold=1e-6)
    print(f'\n✓ Model converged after {len(I_store_result)} iterations')
    print(f'Optimum Current: {I_store_result[-1]:.4f} A\n')
    print("==== Group 12 - CMM3 Consolidated Results ====\n")
    ode_data = ODE.run()
    print("\n--- RK4 Error Analysis ---")
    rk4e.run()
    print("\n--- Mass Flowrate Solver ---")
    mf_data = mf.run()
    print("\n--- Optimum Current Analysis ---")
    oc_data = oc.run()
    print("\n--- Real Charging Time ---")
    rct.run()
    print("\n--- Heptane Fluid Properties ---")
    hi_data = hi.run()

    # Create centralized subplots (1x3 grid = 3 plots)
    fig = plt.figure(figsize=(18, 6))
    fig.suptitle('Group 12 - CMM3 Complete Analysis', fontsize=20, fontweight='bold')
    # Plot 1: ODE (RK4 vs SciPy)
    ax1 = plt.subplot(1, 3, 1)
    t_rk, T_rk = ode_data['rk4']
    t_scipy, T_scipy = ode_data['scipy']
    ax1.plot(t_rk, T_rk, 'b--', label=f'RK4 (h={H}s)', linewidth=1.5)
    ax1.plot(t_scipy, T_scipy, 'r-', linewidth=3, alpha=0.6, label='SciPy LSODA')
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
    I_smooth, delta_T_smooth = oc_data['smooth']
    critical_current, critical_y = oc_data['critical']
    ax3.plot(I_smooth, delta_T_smooth, 'r-', linewidth=2, label='Cubic Spline Interpolation')
    ax3.plot(critical_current, critical_y, 'ro', markersize=10, label=f'Critical Point: {critical_current:.1f} A', zorder=6)
    ax3.axhline(0, color='red', linestyle='--', linewidth=1)
    ax3.set_xlabel('Current (A)', fontsize=10)
    ax3.set_ylabel('$\\Delta T$ (K)', fontsize=10)
    ax3.set_title('Optimum Current Analysis', fontsize=12, fontweight='bold')
    ax3.legend(fontsize=9)
    ax3.grid(True, alpha=0.3)
    plt.tight_layout(rect=(0, 0, 1, 0.96))
    print("\n✓ All computations and plots complete!")
    plt.show()
