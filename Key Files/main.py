import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d

# --- Import from project files ---
from config import q_b, m_b, C_b, T_in, T_b_max, R_b
from ODE import Tb, dTb_dt
from Mass_flowrate import calculate_steady_state_mass_flow


def find_critical_current(current_range=range(5, 100, 5)):
    """
    Analyzes a range of currents to find the critical current where the final
    battery temperature equals T_b_max.

    This function encapsulates the logic from 'Current against delta T.py'.

    Returns:
        tuple: (critical_current, I_array, delta_T_array)
    """
    print("--- Running Critical Current Analysis ---")
    I_runs = []
    delta_T_list = []

    # Helper to package parameters for the ODE solver
    def get_ode_params(I, m_dot_c):
        return (m_b, C_b, I, R_b, 0.01, T_in, m_dot_c)

    for i in current_range:
        I_0 = float(i)
        # At each step, we need to find the corresponding steady-state mass flow
        Q_gen = I_0**2 * R_b
        m_dot_ss, _, _ = calculate_steady_state_mass_flow(Q_gen, T_in, guess_m_dot=0.01)

        if m_dot_ss <= 1e-6:
            print(f"Warning: Could not find a valid mass flow rate for I = {I_0}A. Skipping.")
            continue

        t_total = q_b / I_0  # Total time for charging
        params = get_ode_params(I_0, m_dot_ss)
        
        _, T_profile = Tb(dTb_dt, t_total, params)
        final_temp = T_profile[-1]

        I_runs.append(I_0)
        delta_T_list.append(final_temp - T_b_max)
        print(f"  - Current: {I_0: >3} A, Mass Flow: {m_dot_ss:.4f} kg/s, Final Temp: {final_temp:.2f} K, ΔT: {final_temp - T_b_max:+.2f} K")

    I_array = np.array(I_runs)
    delta_T_array = np.array(delta_T_list)

    if not np.any(delta_T_array <= 0) or not np.any(delta_T_array > 0):
        print("\nAnalysis complete. The system either always overheats or never reaches the maximum temperature.")
        return None, I_array, delta_T_array

    # Interpolate to find the current where delta_T is zero
    deltaT_vs_current = interp1d(I_array, delta_T_array, kind='cubic', bounds_error=False, fill_value="extrapolate")
    
    # Find the root using a simple search, as fsolve can be complex here
    from scipy.optimize import brentq
    critical_current = brentq(deltaT_vs_current, I_array.min(), I_array.max())
    
    print(f"\n--- Critical Current Found: {critical_current:.2f} A ---")
    return critical_current, I_array, delta_T_array


def run_final_simulation(critical_current):
    """
    Runs the final ODE simulation at the determined critical current and
    plots the results.
    """
    print("\n--- Running Final Simulation at Critical Current ---")
    Q_gen = critical_current**2 * R_b
    m_dot_final, T_c_avg, h_final = calculate_steady_state_mass_flow(Q_gen, T_in, guess_m_dot=0.01)
    
    t_total = q_b / critical_current
    params = (m_b, C_b, critical_current, R_b, 0.01, T_in, m_dot_final)
    
    t_sim, T_sim = Tb(dTb_dt, t_total, params)
    
    # --- Reporting ---
    print("\n" + "="*50)
    print("      FINAL OPTIMIZED SYSTEM RESULTS")
    print("="*50)
    print(f"Critical Charging Current: {critical_current:.2f} A")
    print(f"Steady-State Mass Flow Rate: {m_dot_final:.4f} kg/s")
    print(f"Average Coolant Temperature: {T_c_avg:.2f} K")
    print(f"Heat Transfer Coefficient (h): {h_final:.2f} W/(m²·K)")
    print(f"Final Battery Temperature: {T_sim[-1]:.2f} K (Target: {T_b_max:.2f} K)")
    print(f"Total Simulation Time: {t_total:.1f} s")
    print("="*50)

    # --- Plotting ---
    plt.figure(figsize=(10, 6))
    plt.plot(t_sim, T_sim, label='Battery Temperature (Tb)', color='navy')
    plt.axhline(T_b_max, color='red', linestyle='--', label=f'Max Safe Temp ({T_b_max} K)')
    plt.title(f'Battery Temperature During Charging @ {critical_current:.1f} A')
    plt.xlabel('Time (s)')
    plt.ylabel('Temperature (K)')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    # 1. Find the critical operating current
    crit_I, _, _ = find_critical_current()

    # 2. Run the final simulation and report results if a critical current was found
    if crit_I is not None:
        run_final_simulation(crit_I)
    else:
        print("\nCould not determine a critical current within the tested range.")