"""
Group-12-CMM3 Consolidated Execution (main.py)

Main orchestration script for battery temperature, current optimization,
fluid analysis, and integrated plotting across all modules.
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

def compute_optimum_current(threshold: float = I_Threshold) -> list:
    """
    Iteratively compute and store optimum critical current values until convergence.

    Args:
        threshold (float): Convergence threshold for stopping condition (A).

    Returns:
        list: List of successive optimum current values (float, in A).
    """
    I_store.clear()  # Start fresh for each run
    result = oc.run()  # Returns a dict with critical current value list
    current = result['critical'][0]  # Extract the critical current value
    I_store.append(current)

    while True:
        result = oc.run()
        new_current = result['critical'][0]  # Extract the critical current value
        I_store.append(new_current)
        if len(I_store) > 1 and abs(I_store[-1] - I_store[-2]) < threshold:
            break
    return I_store

if __name__ == "__main__":
    # Run convergence loop for optimum current
    I_store_result = compute_optimum_current(threshold=1e-6)
    print(f"\n✓ Model converged after {len(I_store_result)} iterations")
    print(f"Optimum Current: {I_store_result[-1]:.4f} A\n")

    # Run all analyses and collect plot data
    print("==== Group 12 - CMM3 Consolidated Results ====\n")

    ode_data = ODE.run()  # Get ODE plot data

    print("\n--- RK4 Error Analysis ---")
    rk4e.run()

    print("\n--- Mass Flowrate Solver ---")
    mf_data = mf.run()  # Get Mass Flowrate plot data

    print("\n--- Optimum Current Analysis ---")
    oc_data = oc.run()  # Get Optimum Current plot data

    print("\n--- Real Charging Time ---")
    rct.run()

    print("\n--- Heptane Fluid Properties ---")
    hi_data = hi.run()  # Get Heptane properties plot data

# CREATE CENTRALIZED SUBPLOTS (1x3 grid = 3 plots)
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
ax3.plot(critical_current, critical_y, 'ro', markersize=10, 
         label=f'Critical Point: {critical_current:.1f} A', zorder=6)
ax3.axhline(0, color='red', linestyle='--', linewidth=1)
ax3.set_xlabel('Current (A)', fontsize=10)
ax3.set_ylabel('$\Delta T$ (K)', fontsize=10)
ax3.set_title('Optimum Current Analysis', fontsize=12, fontweight='bold')
ax3.legend(fontsize=9)
ax3.grid(True, alpha=0.3)

plt.tight_layout(rect=(0, 0, 1, 0.96))
print("\n✓ All computations and plots complete!")
plt.show()
