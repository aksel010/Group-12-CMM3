"""
Sensitivity analysis: Impact of coolant channel height on maximum battery temperature in a cooling module.
PEP8-compliant docstrings, inline clarification for parameters, geometry, and summary reporting.
"""
import numpy as np
import matplotlib.pyplot as plt
from config import *
from ODE import Tb
from heptane_itpl import Cp_func, mu_func, rho_func, lambda_func
from Mass_flowrate import get_steady_state_values

# Steady-state and geometry bounds
M_DOT_SS, T_C_AVG_K, H_SS = get_steady_state_values()
I_FIXED = I_0
W_C_FIXED = w_branch
H_BRANCH_MIN_M = 1e-3
H_BRANCH_MAX_M = 10e-3
N_POINTS = 20
h_branch_sweep_m = np.linspace(H_BRANCH_MIN_M, H_BRANCH_MAX_M, N_POINTS)

def calculate_h_branch(h_branch_m, T_c_avg_K, m_dot_c):
    """
    Returns convective heat transfer coefficient for test channel height.
    Args:
        h_branch_m (float): Height [m].
        T_c_avg_K (float): Coolant temp [K].
        m_dot_c (float): Flow [kg/s].
    Returns:
        tuple: (h_ss [W/m²K], S_b [m²], d_H [m]).
    """
    S_b = W_C_FIXED * h_branch_m
    P_b = 2 * (W_C_FIXED + h_branch_m)
    d_H = 4 * S_b / P_b
    mu = mu_func(T_c_avg_K)
    rho = rho_func(T_c_avg_K)
    k_therm = lambda_func(T_c_avg_K)
    m_dot_branch = m_dot_c / n
    V_branch = m_dot_branch / (rho * S_b)
    Re = rho * V_branch * d_H / mu
    if Re < 2000:
        Nu = 4.86
    else:
        Pr = Cp_func(T_c_avg_K) * mu / k_therm
        Nu = 0.023 * Re**0.8 * Pr**0.4
    h_ss = Nu * k_therm / d_H
    return h_ss, S_b, d_H

def run_sensitivity_analysis():
    """
    Sweeps channel height, solves for limiting battery temperature, reports and plots.
    """
    h_branch_final = []
    T_b_max_results = []
    print(f"--- Running Geometry Sensitivity Analysis ---")
    print(f"Fixed current: {I_FIXED:.2f} A, Fixed flow: {M_DOT_SS:.4f} kg/s\n")
    for idx, h_branch_m in enumerate(h_branch_sweep_m):
        h_ss, S_b_new, d_H_new = calculate_h_branch(h_branch_m, T_C_AVG_K, M_DOT_SS)
        P_b_new = 2 * (W_C_FIXED + h_branch_m)
        A_s_new = P_b_new * L_b * n
        params_mod_fixed_h = (
            m_cell * 24,
            C_b,
            I_FIXED,
            DC_IR * 24,
            A_s_new,
            T_in,
            M_DOT_SS,
            h_ss
        )
        def dTb_dt_fixed_h(Tb, t, params):
            m, cp_b, I, R, A_s_geom, T_c_in, m_dot_c, h_fixed = params
            T_c_avg = (Tb + T_c_in) / 2
            cp_c = Cp_func(T_c_avg)
            heating = I**2 * R
            cooling_denom = 1 + (h_fixed * A_s_geom) / (2 * m_dot_c * cp_c)
            cooling_term = (h_fixed * A_s_geom * (Tb - T_c_in)) / cooling_denom
            return (heating - cooling_term) / (m * cp_b)
        t_i, T_i = Tb(dTb_dt_fixed_h, params_mod_fixed_h, stepsize=H)
        T_b_max = T_i[-1]
        h_branch_final.append(h_branch_m * 1000)
        T_b_max_results.append(T_b_max)
        if (idx + 1) % (N_POINTS // 5) == 0:
            print(f"Height: {h_branch_m*1000:.1f} mm -> "
                  f"h_ss: {h_ss:.0f} W/m²K -> T_b_max: {T_b_max-273.15:.2f} °C")
    T_b_max_C = np.array(T_b_max_results) - 273.15
    plt.figure(figsize=(10, 6))
    plt.plot(h_branch_final, T_b_max_C, 'rD-', linewidth=2, markersize=6,
             label=f'Max $T_b$ at $I={I_FIXED}$ A')
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
    print("\n--- Sensitivity Results Summary ---")
    print(f"Lowest T_b_max ({T_b_max_C.min():.2f} °C) occurred at height: "
          f"{h_branch_final[np.argmin(T_b_max_C)]:.1f} mm")
    print(f"Highest T_b_max ({T_b_max_C.max():.2f} °C) occurred at height: "
          f"{h_branch_final[np.argmax(T_b_max_C)]:.1f} mm")

if __name__ == "__main__":
    run_sensitivity_analysis()
