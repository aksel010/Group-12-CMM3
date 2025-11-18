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
from heptane_itpl import calculate_h, cp_func, rho_func, mu_func
from root_finders import newton

# Global heat load, tied to current calculation context
heat_load = 0.0

# -------------------------------------------------------------------
# HYDRAULIC + THERMAL SAFETY LIMIT CALCULATIONS
# -------------------------------------------------------------------
# Maximum allowed mass flow (viscous/geometry limit)
mass_flow_limit = 2000 * mu_func(t_b_max) * s_b / d
# Maximum allowable heat load (safety limit)
heat_limit = mass_flow_limit * cp_func(t_b_max) * (t_b_max - t_in)

# -------------------------------------------------------------------
# PUMP + SYSTEM HEAD FUNCTIONS
# -------------------------------------------------------------------
def pump_head_curve(mass_flow, t_c_avg_k):
    """
    Models pump pressure rise as a function of flow.

    Args:
        mass_flow (float): Mass flow rate [kg/s].
        t_c_avg_k (float): Average coolant temperature [K].

    Returns:
        float: Net pump pressure [Pa] at given flow.
    """
    h_max = 80.0
    p_max = h_max * rho_func(t_c_avg_k) * 9.81
    v_dot_max = 9.0 / 60000
    k = h_max / (v_dot_max ** 2)
    return p_max - k * mass_flow ** 2

def system_head_loss_calc(mass_flow, t_c_avg_k):
    """
    Hydraulic system head loss, considering fluid and channel properties.

    Args:
        mass_flow (float): Mass flow rate [kg/s].
        T_c_avg_K (float): Average coolant temperature [K].

    Returns:
        float: System head loss [Pa].
    """
    v_dot = mass_flow / rho_func(t_c_avg_k)
    return get_head_loss(v_dot) * rho_func(t_c_avg_k) * g

# -------------------------------------------------------------------
# COUPLED HYDRAULIC–THERMAL ROOT FUNCTION
# -------------------------------------------------------------------
def pressure_balance_couple(mass_flow):
    """
    Combined pump/system pressure balance at a given mass flow rate.
    Returns positive when pump exceeds loss, negative if not.

    Args:
        mass_flow (float): Flow assumption [kg/s].
    Returns:
        float: Pressure surplus [Pa] at this mass_flow.
    """
    cp_c_in = cp_func(t_in)
    t_c_out_k = t_in + heat_load / (max(mass_flow, 1e-10) * cp_c_in)
    t_c_avg_k = (t_in + t_c_out_k) / 2
    p_supplied = pump_head_curve(mass_flow, t_c_avg_k)
    p_loss = system_head_loss_calc(mass_flow, t_c_avg_k)
    return p_supplied - p_loss

def pressure_root_deriv(mass_flow):
    """
    Finite-difference derivative of pressure_balance_couple().
    Args:
        mass_flow (float): Current mass flow [kg/s].
    Returns:
        float: Approximated derivative.
    """
    H = 1e-8
    return (pressure_balance_couple(mass_flow + H) - pressure_balance_couple(mass_flow)) / H

# -------------------------------------------------------------------
# MAIN STEADY-STATE SOLVER
# -------------------------------------------------------------------
def calculate_steady_state_mass_flow(generated_heat, mass_flow_initial):
    """
    Solve coupled thermohydraulic steady-state problem.
    Returns mass flow, mean temperature, and heat transfer coefficient.

    Args:
        generated_heat (float): Input heat load [W].
        mass_flow_initial (float): Initial guess for mass flow [kg/s].
    Returns:
        tuple: (mass_flow, T_c_avg_K, h_ss)
    """
    global heat_load
    heat_load = generated_heat
    if generated_heat >= heat_limit:
        print("Warning: Heat generation exceeds geometric cooling limits.")
    mass_flow = newton(pressure_balance_couple, pressure_root_deriv, mass_flow_initial, epsilon=1e-6, max_iter=100, args=()) / (2 * n)
    if mass_flow is None:
        print("Warning: Newton-Raphson failed. Returning fallback value.")
        mass_flow = 1e-6
    cp_c_in = cp_func(t_in)
    t_c_out_k = t_in + heat_load / (max(mass_flow, 1e-10) * cp_c_in)
    t_c_avg_k = (t_in + t_c_out_k) / 2
    if mass_flow == mass_flow_limit:
        print("Mass flowrate limit reached.")
    h_ss = calculate_h(t_c_avg_k)
    return mass_flow, t_c_avg_k, h_ss

# -------------------------------------------------------------------
# DRIVER FUNCTION (no plotting)
# -------------------------------------------------------------------
def run():
    """
    Run steady-state solution, return flow curve and residuals (diagnostic/sweep only).
    Returns:
        dict: {'mass_flow': m_range, 'residuals': residuals}
    """
    if len(current_store) == 0:
        current_store.append(current_0)
    generated_heat = current_store[-1] ** 2 * r_b
    mass_flow_ss, t_c_avg_k, h_ss = calculate_steady_state_mass_flow(generated_heat, mass_flow_initial)
    if mass_flow_ss > 0:
        print(f"Mass Flow Rate: {mass_flow_ss:.8f} kg/s")
        print(f"Average Coolant Temperature: {t_c_avg_k:.2f} K ({t_c_avg_k - 273.15:.2f} °C)")
    else:
        print("Solver failed: non-physical flow result.")
    m_range = np.linspace(mass_flow_initial, 0.02, 1000)
    residuals = [pressure_balance_couple(m) for m in m_range]
    return {"mass_flow": m_range, "residuals": residuals}

def get_steady_state_values():
    """
    Convenience wrapper to fetch steady-state result, prevents circular import.
    Returns:
        tuple: (mass_flow, t_c_avg_k, h_ss)
    """
    if len(current_store) == 0:
        current_store.append(current_0)
    generated_heat = current_store[-1] ** 2 * r_b
    return calculate_steady_state_mass_flow(generated_heat, mass_flow_initial)

if __name__ == "__main__":
    run()
