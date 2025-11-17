import numpy as np
import matplotlib.pyplot as plt
from config import *
from ODE import Tb
from heptane_itpl import Cp_func, mu_func, rho_func, lambda_func
from Mass_flowrate import get_steady_state_values

# -------------------------------------------------------------------
# Fetch steady-state parameters from solver
# -------------------------------------------------------------------
M_DOT_SS, T_C_AVG_K, H_SS = get_steady_state_values()
I_FIXED = I_0  # Fixed current from config.py
W_C_FIXED = w_branch  # Fixed channel width from config.py

# Define channel height sweep range (m)
H_BRANCH_MIN_M = 1e-3
H_BRANCH_MAX_M = 10e-3
N_POINTS = 20
h_branch_sweep_m = np.linspace(H_BRANCH_MIN_M, H_BRANCH_MAX_M, N_POINTS)


# -------------------------------------------------------------------
# Function: calculate_h_branch
# -------------------------------------------------------------------
def calculate_h_branch(h_branch_m, T_c_avg_K, m_dot_c):
    """
    Calculate heat transfer coefficient for a given channel height using
    laminar/turbulent flow correlations, incorporating changing geometry.

    Parameters
    ----------
    h_branch_m : float
        Channel height in meters.
    T_c_avg_K : float
        Average coolant temperature in Kelvin.
    m_dot_c : float
        Steady-state mass flow rate (kg/s).

    Returns
    -------
    h_ss : float
        Convective heat transfer coefficient (W/m²K).
    S_b : float
        Channel cross-sectional area (m²).
    d_H : float
        Hydraulic diameter (m).
    """
    # Geometry
    S_b = W_C_FIXED * h_branch_m
    P_b = 2 * (W_C_FIXED + h_branch_m)
    d_H = 4 * S_b / P_b

    # Fluid properties
    mu = mu_func(T_c_avg_K)
    rho = rho_func(T_c_avg_K)
    k_therm = lambda_func(T_c_avg_K)

    # Flow velocity
    m_dot_branch = m_dot_c / n  # Divide mass flow per branch
    V_branch = m_dot_branch / (rho * S_b)

    # Reynolds number
    Re = rho * V_branch * d_H / mu

    # Nusselt number correlation
    if Re < 2000:
        Nu = 4.86  # Laminar, rectangular duct, constant wall temp
    else:
        Pr = Cp_func(T_c_avg_K) * mu / k_therm
        Nu = 0.023 * Re**0.8 * Pr**0.4  # Turbulent Dittus-Boelter

    h_ss = Nu * k_therm / d_H
    return h_ss, S_b, d_H


# -------------------------------------------------------------------
# Function: run_sensitivity_analysis
# -------------------------------------------------------------------
def run_sensitivity_analysis():
    """
    Perform sensitivity analysis of maximum battery temperature with
    respect to channel height.

    Returns
    -------
    None
    """
    h_branch_final = []
    T_b_max_results = []

    print(f"--- Running Geometry Sensitivity Analysis ---")
    print(f"Fixed current: {I_FIXED:.2f} A, Fixed flow: {M_DOT_SS:.4f} kg/s\n")

    for idx, h_branch_m in enumerate(h_branch_sweep_m):
        # 1. Compute heat transfer coefficient
        h_ss, S_b_new, d_H_new = calculate_h_branch(h_branch_m, T_C_AVG_K, M_DOT_SS)
        P_b_new = 2 * (W_C_FIXED + h_branch_m)
        A_s_new = P_b_new * L_b * n  # Total wetted area for module

        # 2. Custom ODE parameters with fixed h
        params_mod_fixed_h = (
            m_cell * 24,  # Module mass (24 cells)
            C_b,          # Battery heat capacity
            I_FIXED,      # Fixed current
            DC_IR * 24,   # Module resistance
            A_s_new,      # Wetted area
            T_in,         # Inlet coolant temperature
            M_DOT_SS,     # Steady-state mass flow rate
            h_ss          # Fixed heat transfer coefficient
        )

        # 3. ODE with fixed h
        def dTb_dt_fixed_h(Tb, t, params):
            m, cp_b, I, R, A_s_geom, T_c_in, m_dot_c, h_fixed = params
            T_c_avg = (Tb + T_c_in) / 2
            cp_c = Cp_func(T_c_avg)
            heating = I**2 * R
            cooling_denom = 1 + (h_fixed * A_s_geom) / (2 * m_dot_c * cp_c)
            cooling_term = (h_fixed * A_s_geom * (Tb - T_c_in)) / cooling_denom
            return (heating - cooling_term) / (m * cp_b)

        t_i, T_i = Tb(dTb_dt_fixed_h, params_mod_fixed_h, stepsize=H)

        # 4. Store results
        T_b_max = T_i[-1]
        h_branch_final.append(h_branch_m * 1000)  # mm
        T_b_max_results.append(T_b_max)

        # Print progress every 20%
        if (idx + 1) % (N_POINTS // 5) == 0:
            print(f"Height: {h_branch_m*1000:.1f} mm -> "
                  f"h_ss: {h_ss:.0f} W/m²K -> T_b_max: {T_b_max-273.15:.2f} °C")

    # --- Plot results ---
    T_b_max_C = np.array(T_b_max_results) - 273.15

    plt.figure(figsize=(10, 6))
    plt.plot(h_branch_final, T_b_max_C, 'rD-', linewidth=2, markersize=6,
             label=f'Max $T_b$ at $I={I_FIXED}$ A')

    # Add constraint line
    T_b_max_C_constraint = T_b_max - 273.15
    plt.axhline(T_b_max_C_constraint, color='red', linestyle='--',
                label=f'Max Constraint ({T_b_max_C_constraint:.0f}°C)')

    plt.xlabel('Channel Height, $h_{\\mathrm{branch}}$ (mm)', fontsize=14)
    plt.ylabel('Maximum Battery Temperature, $T_{b,\\mathrm{max}}$ (°C)', fontsize=14)
    plt.title(f'Sensitivity of Max Battery Temperature to Channel Height\n'
              f'(Fixed Current $I={I_FIXED}$ A, Fixed Flow Rate)', fontsize=16)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend(fontsize=12)
    plt.tight_layout()
    plt.show()

    # Summary
    print("\n--- Sensitivity Results Summary ---")
    print(f"Lowest T_b_max ({T_b_max_C.min():.2f} °C) occurred at height: "
          f"{h_branch_final[np.argmin(T_b_max_C)]:.1f} mm")
    print(f"Highest T_b_max ({T_b_max_C.max():.2f} °C) occurred at height: "
          f"{h_branch_final[np.argmax(T_b_max_C)]:.1f} mm")


# -------------------------------------------------------------------
# Main execution
# -------------------------------------------------------------------
if __name__ == "__main__":
    run_sensitivity_analysis()
