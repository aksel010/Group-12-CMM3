import numpy as np
from scipy.optimize import fsolve
from matplotlib import pyplot as plt
from cooling_analysis import get_head_loss
from config import *
from heptane_itpl import calculate_h, Cp_func, rho_func, mu_func
from root_finders import newton

# Global, updated inside calculate_steady_state_mass_flow()
Q_heat = 0.0  

# -------------------------------------------------------------------
# HYDRAULIC + THERMAL SAFETY LIMIT CALCULATIONS
# -------------------------------------------------------------------

# Max allowable mass flow based on viscosity and geometry
m_dot_limit = 2000 * mu_func(T_b_max) * S_b / D

# Max allowable heat load (safety threshold)
Q_limit = m_dot_limit * Cp_func(T_b_max) * (T_b_max - T_in)

# -------------------------------------------------------------------
# PUMP + SYSTEM HEAD FUNCTIONS
# -------------------------------------------------------------------

def pump_head_curve(m_dot, T_c_avg_K):
    """
    Pump head curve model (placeholder).
    Returns pump pressure rise (Pa) as a function of m_dot.
    """
    H_max = 80.0                           # maximum pump head (m)
    P_max = H_max * rho_func(T_c_avg_K) * 9.81  # convert head → pressure
    V_dot_max = 9.0 / 60000                # 9 L/min → m³/s
    K = H_max / (V_dot_max ** 2)           # proportionality constant
    return P_max - K * m_dot**2            # quadratic drop-off with flow rate


def system_head_loss_calc(m_dot, T_c_avg_K):
    """
    Computes system head loss (Pa), based on hydraulic model.
    """
    V_dot = m_dot / rho_func(T_c_avg_K)     # convert mass flow → volumetric flow
    return get_head_loss(V_dot) * rho_func(T_c_avg_K) * g


# -------------------------------------------------------------------
# COUPLED HYDRAULIC–THERMAL ROOT FUNCTION
# -------------------------------------------------------------------

def pressure_balance_couple(m_dot):
    """
    Coupled equation: pump pressure - system loss = 0.
    Root of this function => hydraulic equilibrium.
    """
    # --- Thermal balance (needed because pump performance depends on T_c_avg) ---
    Cp_c_in = Cp_func(T_in)

    # Outlet coolant temperature from energy balance
    T_c_out_K = T_in + Q_heat / (max(m_dot, 1e-10) * Cp_c_in)  # Protect against division by zero
    # Mean coolant temperature in loop
    T_c_avg_K = (T_in + T_c_out_K) / 2

    # --- Hydraulic balance ---
    P_supplied = pump_head_curve(m_dot, T_c_avg_K)
    P_loss = system_head_loss_calc(m_dot, T_c_avg_K)

    return P_supplied - P_loss


def df(m_dot):
    """
    Numerical derivative of pressure_balance_couple() using finite differences.
    """
    H = 1e-8
    return (pressure_balance_couple(m_dot + H) - pressure_balance_couple(m_dot)) / H


# -------------------------------------------------------------------
# MAIN STEADY-STATE SOLVER
# -------------------------------------------------------------------

def calculate_steady_state_mass_flow(Q_gen, guess_m_dot):
    """
    Computes steady-state:
        - mass flow rate m_dot_ss
        - average coolant temperature T_c_avg
        - heat transfer coefficient h_ss
    """
    global Q_heat
    Q_heat = Q_gen  # update global heat load

    # Safety check: too much heat for this channel geometry
    if Q_gen >= Q_limit:
        print("Warning: Heat generation exceeds geometric cooling limits.")

    # ---- Solve hydraulic balance for m_dot using Newton–Raphson ----
    m_dot = newton(pressure_balance_couple, df, guess_m_dot,
                   epsilon=1e-6, max_iter=100, args=()) / (2 * n)

    if m_dot is None:
        print("Warning: Newton-Raphson failed. Returning fallback value.")
        m_dot = 1e-6

    # ---- Compute resulting coolant temperature ----
    Cp_c_in = Cp_func(T_in)
    T_c_out_K = T_in + Q_heat / (m_dot * Cp_c_in)
    T_c_avg_K = (T_in + T_c_out_K) / 2

    # ---- Ensure physically meaningful m_dot (no runaway heating) ----
    T_c_out_K = T_in + Q_heat / (max(m_dot, 1e-10) * Cp_c_in)  # Protect against division by zero        m_dot += 1e-7  
        T_c_out_K = T_in + Q_heat / (max(m_dot, 1e-10) * Cp_c_in)  # Protect against division by zero        T_c_avg_K = (T_in + T_c_out_K) / 2

        if m_dot == m_dot_limit:
            print("Mass flowrate limit reached.")
            break

    # Heat transfer coefficient at steady state
    h_ss = calculate_h(T_c_avg_K)

    return m_dot, T_c_avg_K, h_ss


# -------------------------------------------------------------------
# DRIVER FUNCTION (no plotting)
# -------------------------------------------------------------------

def run():
    """
    Runs one steady-state calculation and returns flow–residual curves.
    """
    # Ensure current list is initialized
    if len(I_store) == 0:
        I_store.append(I_0)

    Q_gen = I_store[-1]**2 * R_b  # Electrical → heat

    # Solve steady state
    m_dot_ss, T_c_avg_K, h_ss = calculate_steady_state_mass_flow(Q_gen, M_DOT)

    # Output results
    if m_dot_ss > 0:
        print(f"Mass Flow Rate: {m_dot_ss:.8f} kg/s")
        print(f"Average Coolant Temperature: {T_c_avg_K:.2f} K ({T_c_avg_K - 273.15:.2f} °C)")
    else:
        print("Solver failed: non-physical flow result.")

    # Residual sweep for diagnostics (no plot)
    m_range = np.linspace(M_DOT, 0.02, 1000)
    residuals = [pressure_balance_couple(m) for m in m_range]

    return {"mass_flow": m_range, "residuals": residuals}


def get_steady_state_values():
    """
    External accessor for steady-state values.
    Prevents circular imports.
    """
    if len(I_store) == 0:
        I_store.append(I_0)

    Q_gen = I_store[-1]**2 * R_b
    return calculate_steady_state_mass_flow(Q_gen, M_DOT)


# Run directly if executed as a script
if __name__ == "__main__":
    run()
