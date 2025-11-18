"""
Channel Width Sensitivity Analysis Script.

This script performs a sweep across a defined range of coolant channel widths
(w_branch) to determine the resulting optimum charging current (I_optimum)
and the maximum battery temperature achieved (T_b_max_achieved) for each width.

It relies on modules imported from the main project, dynamically updating
the geometry parameters in config.py for each iteration.
"""
import numpy as np
import matplotlib.pyplot as plt
from src.config import * # IMPORTANT: Import * to allow dynamic updating of globals
import ODE
import Mass_flowrate as mf
import Optimum_Current as oc

def compute_optimum_current_for_sweep(threshold=I_Threshold):
    """
    Iteratively solves for critical optimum current until convergence below threshold.
    Includes checks for breaching the maximum safe temperature (T_b_max).

    Returns:
        tuple: (List of current guesses with final critical value last, Max Temperature Achieved [K])
    """
    I_store.clear()
    
    # Initial run to get steady state and initial guess
    # Note: oc.run() calls ODE.Tb_scipy, which updates the global T_b_max_achieved
    result = oc.run() 
    current = result['critical'][0]
    I_store.append(current)
    
    # Check if the initial current already resulted in T_b_max_achieved >= T_b_max
    if T_b_max_achieved >= T_b_max:
         print(f"Initial flow calculation for {w_branch*1e3:.1f}mm width resulted in T_max={T_b_max_achieved:.2f}K > T_b_max. Stopping iteration.")
         return I_store, T_b_max_achieved

    # Convergence loop
    while True:
        # Rerunning oc.run() triggers Mass_flowrate and ODE to update based on the new I_store[-1]
        result = oc.run()
        new_current = result['critical'][0]
        I_store.append(new_current)
        
        # Check if the new current breached the safety limit
        if T_b_max_achieved >= T_b_max:
             print(f"Current {new_current:.2f} A resulted in T_max={T_b_max_achieved:.2f}K > T_b_max. Stopping convergence.")
             break

        # Check for convergence
        if len(I_store) > 1 and abs(I_store[-1] - I_store[-2]) < threshold:
            break
            
    return I_store, T_b_max_achieved

def run_optimization_with_geometry(w_branch_new, h_branch_val=h_branch, L_b_val=L_b, n_val=n):
    """
    Runs the full optimization routine for a specific channel width by dynamically
    updating global geometry variables (A_s and S_b).

    Args:
        w_branch_new (float): New branch width [m].
        h_branch_val (float): Branch height [m].
        L_b_val (float): Branch length [m].
        n_val (int): Number of branches.

    Returns:
        tuple: (Optimum Current [A], Maximum Battery Temperature Achieved [K])
    """
    global w_branch, A_s, S_b
    # 1. Update w_branch
    w_branch = w_branch_new
    
    # 2. Recalculate dependent geometric variables in config.py globals
    A_s = L_b_val * 2 * (w_branch + h_branch_val) * n_val  # Heat transfer area
    S_b = w_branch * h_branch_val * n_val  # Flow cross-section area
    
    print(f"\n--- Running Analysis for w_branch = {w_branch*1e3:.3f} mm ---")
    
    # 3. Run the mass flow rate calculation first (updates steady-state m_dot)
    mf.run() 
    
    # 4. Run convergence loop for optimum current
    I_store_result, T_max_achieved = compute_optimum_current_for_sweep(threshold=1e-6)
    I_optimum = I_store_result[-1]

    # Clean up I_store for the next run
    I_store.clear()
    
    return I_optimum, T_max_achieved

# --- SENSITIVITY ANALYSIS EXECUTION BLOCK ---
if __name__ == "__main__":
    # Define the range of channel widths to test (in meters)
    W_MIN = 10e-3  # 10 mm
    W_MAX = 50e-3  # 50 mm
    W_STEPS = 20   # Number of data points

    # Generate logarithmic spacing for channel width sweep
    width_range = np.logspace(np.log10(W_MIN), np.log10(W_MAX), W_STEPS)

    # --- Analysis Storage ---
    optimum_currents = []
    max_temperatures = []
    valid_widths = []

    print("--- Starting Channel Width Sensitivity Analysis ---")
    print(f"Sweeping w_branch from {W_MIN*1e3:.1f} mm to {W_MAX*1e3:.1f} mm ({W_STEPS} points).")

    # --- Sensitivity Loop ---
    for w in width_range:
        try:
            # Run the full optimization for the current channel width
            I_optimum, T_max = run_optimization_with_geometry(w)
            
            # Store results
            optimum_currents.append(I_optimum)
            max_temperatures.append(T_max)
            valid_widths.append(w * 1e3) # Store in mm for plotting
            
            # Display run status
            if T_max < T_b_max - 0.1: 
                 print(f"Width {w*1e3:.1f} mm: Overcooled (T_max: {T_max:.2f} K). I_optimum: {I_optimum:.2f} A.")
            elif T_max > T_b_max:
                 print(f"Width {w*1e3:.1f} mm: Exceeded limit (T_max: {T_max:.2f} K). I_optimum: {I_optimum:.2f} A (Limit Current).")
            else:
                 print(f"Width {w*1e3:.1f} mm: Near Optimum (T_max: {T_max:.2f} K). I_optimum: {I_optimum:.2f} A.")

        except Exception as e:
            print(f"Error encountered for width {w*1e3:.1f} mm: {e}")
            # Append NaN if a calculation fails
            optimum_currents.append(np.nan)
            max_temperatures.append(np.nan)
            valid_widths.append(w * 1e3)
            continue

    # --- Results Plotting ---
    currents_np = np.array(optimum_currents)
    temps_np = np.array(max_temperatures)
    widths_np = np.array(valid_widths)

    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax1 = plt.subplots(figsize=(10, 6))
    fig.suptitle('Sensitivity Analysis: Optimum Current and Max Temperature vs. Channel Width', 
                 fontsize=14, fontweight='bold')

    # Plot 1: Maximum Temperature (Left Y-axis)
    color_temp = 'tab:red'
    ax1.set_xlabel('Channel Branch Width $w_{branch}$ (mm)', fontsize=11)
    ax1.set_ylabel('Maximum Battery Temperature $T_{b,max}$ (K)', color=color_temp, fontsize=11)
    ax1.plot(widths_np, temps_np, color=color_temp, marker='o', linestyle='-', linewidth=2, label='$T_{b,max}$ Achieved')
    ax1.axhline(T_b_max, color='k', linestyle='--', linewidth=1.5, label=f'Safety Limit ({T_b_max:.1f} K)')
    ax1.tick_params(axis='y', labelcolor=color_temp)

    # Find the approximate transition point (the last safe width)
    exceeded_limit = temps_np > T_b_max
    if np.any(~exceeded_limit):
        # Find the index of the largest width that did NOT exceed the limit
        last_safe_index = np.where(~exceeded_limit)[0][-1]
        optimum_width_for_limit = widths_np[last_safe_index]
        optimum_I = currents_np[last_safe_index]

        ax1.axvline(optimum_width_for_limit, color='g', linestyle='-.', linewidth=1, label=f'Approx. Optimal Width: {optimum_width_for_limit:.2f} mm')
        print(f"\nApproximate optimum w_branch (last safe point): {optimum_width_for_limit:.2f} mm")
        print(f"Corresponding optimum current: {optimum_I:.2f} A")


    # Plot 2: Optimum Current (Right Y-axis)
    ax2 = ax1.twinx()  
    color_current = 'tab:blue'
    ax2.set_ylabel('Optimum Charging Current $I_{opt}$ (A)', color=color_current, fontsize=11)
    ax2.plot(widths_np, currents_np, color=color_current, marker='s', linestyle='--', linewidth=2, label='$I_{opt}$')
    ax2.tick_params(axis='y', labelcolor=color_current)

    # Add combined legend
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='center right', bbox_to_anchor=(0.95, 0.2))

    fig.tight_layout(rect=[0, 0.03, 1, 0.95]) # Adjust layout to make room for suptitle
    plt.show()

    print("\n--- Sensitivity Analysis Complete ---")