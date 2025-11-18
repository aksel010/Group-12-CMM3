"""
Thermohydraulic coolant loop and mass flow rate calculation utilities.

Key functionality:
- Compute mass flow rate, temperature, heat transfer.
- Solve coupled hydraulic/thermal balance equations for steady state.
- Provide system diagnostics (pressure/flow residuals).

Each function now includes a PEP8/PEP257-compliant docstring and clarifying block comments.
"""
import numpy as np
from scipy.optimize import fsolve
from matplotlib import pyplot as plt
from cooling_analysis import get_head_loss
from config import *
from heptane_itpl import calculate_h, Cp_func, rho_func, mu_func
from root_finders import newton

# Global heat load, tied to current calculation context
Q_heat = 0.0

# -------------------------------------------------------------------
# HYDRAULIC + THERMAL SAFETY LIMIT CALCULATIONS
# -------------------------------------------------------------------
# Maximum allowed mass flow (viscous/geometry limit)
m_dot_limit = 2000 * mu_func(T_b_max) * S_b / D
# Maximum allowable heat load (safety limit)
Q_limit = m_dot_limit * Cp_func(T_b_max) * (T_b_max - T_in)

# -------------------------------------------------------------------
# PUMP + SYSTEM HEAD FUNCTIONS
# -------------------------------------------------------------------
def pump_head_curve(m_dot, T_c_avg_K):
    """
    Models pump pressure rise as a function of flow.

    Args:
        m_dot (float): Mass flow rate [kg/s].
        T_c_avg_K (float): Average coolant temperature [K].

    Returns:
        float: Net pump pressure [Pa] at given flow.
    """
    H_max = 80.0
    P_max = H_max * rho_func(T_c_avg_K) * 9.81
    V_dot_max = 9.0 / 60000
    K = H_max / (V_dot_max ** 2)
    return P_max - K * m_dot ** 2

def system_head_loss_calc(m_dot, T_c_avg_K):
    """
    Hydraulic system head loss, considering fluid and channel properties.

    Args:
        m_dot (float): Mass flow rate [kg/s].
        T_c_avg_K (float): Average coolant temperature [K].

    Returns:
        float: System head loss [Pa].
    """
    V_dot = m_dot / rho_func(T_c_avg_K)
    return get_head_loss(V_dot) * rho_func(T_c_avg_K) * g

# -------------------------------------------------------------------
# COUPLED HYDRAULIC–THERMAL ROOT FUNCTION
# -------------------------------------------------------------------
def pressure_balance_couple(m_dot):
    """
    Combined pump/system pressure balance at a given mass flow rate.
    Returns positive when pump exceeds loss, negative if not.

    Args:
        m_dot (float): Flow assumption [kg/s].
    Returns:
        float: Pressure surplus [Pa] at this m_dot.
    """
    Cp_c_in = Cp_func(T_in)
    T_c_out_K = T_in + Q_heat / (max(m_dot, 1e-10) * Cp_c_in)
    T_c_avg_K = (T_in + T_c_out_K) / 2
    P_supplied = pump_head_curve(m_dot, T_c_avg_K)
    P_loss = system_head_loss_calc(m_dot, T_c_avg_K)
    return P_supplied - P_loss

def df(m_dot):
    """
    Finite-difference derivative of pressure_balance_couple().
    Args:
        m_dot (float): Current mass flow [kg/s].
    Returns:
        float: Approximated derivative.
    """
    H = 1e-8
    return (pressure_balance_couple(m_dot + H) - pressure_balance_couple(m_dot)) / H

# -------------------------------------------------------------------
# MAIN STEADY-STATE SOLVER
# -------------------------------------------------------------------
def calculate_steady_state_mass_flow(Q_gen, guess_m_dot):
    """
    Solve coupled thermohydraulic steady-state problem.
    Returns mass flow, mean temperature, and heat transfer coefficient.

    Args:
        Q_gen (float): Input heat load [W].
        guess_m_dot (float): Initial guess for mass flow [kg/s].
    Returns:
        tuple: (m_dot, T_c_avg_K, h_ss)
    """
    global Q_heat
    Q_heat = Q_gen
    if Q_gen >= Q_limit:
        print("Warning: Heat generation exceeds geometric cooling limits.")
    m_dot = newton(pressure_balance_couple, df, guess_m_dot, epsilon=1e-6, max_iter=100, args=()) / (2 * n)
    if m_dot is None:
        print("Warning: Newton-Raphson failed. Returning fallback value.")
        m_dot = 1e-6
    Cp_c_in = Cp_func(T_in)
    T_c_out_K = T_in + Q_heat / (max(m_dot, 1e-10) * Cp_c_in)
    T_c_avg_K = (T_in + T_c_out_K) / 2
    if m_dot == m_dot_limit:
        print("Mass flowrate limit reached.")
    h_ss = calculate_h(T_c_avg_K)
    return m_dot, T_c_avg_K, h_ss

# -------------------------------------------------------------------
# DRIVER FUNCTION (no plotting)
# -------------------------------------------------------------------
def run():
    """
    Run steady-state solution, return flow curve and residuals (diagnostic/sweep only).
    Returns:
        dict: {'mass_flow': m_range, 'residuals': residuals}
    """
    if len(I_store) == 0:
        I_store.append(I_0)
    Q_gen = I_store[-1] ** 2 * R_b
    m_dot_ss, T_c_avg_K, h_ss = calculate_steady_state_mass_flow(Q_gen, M_DOT)
    if m_dot_ss > 0:
        print(f"Mass Flow Rate: {m_dot_ss:.8f} kg/s")
        print(f"Average Coolant Temperature: {T_c_avg_K:.2f} K ({T_c_avg_K - 273.15:.2f} °C)")
    else:
        print("Solver failed: non-physical flow result.")
    m_range = np.linspace(M_DOT, 0.02, 1000)
    residuals = [pressure_balance_couple(m) for m in m_range]
    return {"mass_flow": m_range, "residuals": residuals}

def get_steady_state_values():
    """
    Convenience wrapper to fetch steady-state result, prevents circular import.
    Returns:
        tuple: (m_dot, T_c_avg_K, h_ss)
    """
    if len(I_store) == 0:
        I_store.append(I_0)
    Q_gen = I_store[-1] ** 2 * R_b
    return calculate_steady_state_mass_flow(Q_gen, M_DOT)

if __name__ == "__main__":
    run()
