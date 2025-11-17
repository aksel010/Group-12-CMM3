import numpy as np
import matplotlib.pyplot as plt
import time

from ODE import Tb, dTb_dt
from config import *
from Mass_flowrate import get_steady_state_values
from RK4_Error import get_rk4_error_val
from interpolater import *
from root_finders import bisection, newton


# -------------------------------------------------------------------------
# PARAMETER BUILDER FOR ODE SOLVER
# -------------------------------------------------------------------------
def I_params(I, m_dot_ss):
    """
    Build the parameter tuple needed by dTb_dt() for a given current I.

    I : discharge current [A]
    m_dot_ss : steady-state coolant mass flow rate [kg/s]
    """
    return (
        m_b,                          # bulk mass [kg]
        C_b,                          # bulk specific heat [J/kg·K]
        I,                            # electrical current [A]
        DC_IR * 24,                   # effective pack resistance [Ω]
        A_s,                          # cell surface area [m²]
        T_in,                         # inlet coolant temperature [K]
        get_steady_state_values()[0]  # coolant mass flow rate [kg/s]
    )


# -------------------------------------------------------------------------
# INTERPOLATION WRAPPERS
# -------------------------------------------------------------------------
def current_profile(I_array, delta_T_array, I_query):
    """
    Cubic spline interpolation of ΔT vs current (I).
    Returns interpolated ΔT at I_query.
    """
    return cubic_spline_interpolation(I_array, delta_T_array, I_query)


def current_profile_derivative(I_array, delta_T_array, I_query):
    """
    Returns d(ΔT)/dI using derivative of cubic spline.
    """
    return cubic_spline_derivative(I_array, delta_T_array, I_query)


# -------------------------------------------------------------------------
# MANUAL ROOT FINDING USING BISECTION ON THE SPLINE
# -------------------------------------------------------------------------
def find_root_cubic_spline(I_array, delta_T_array):
    """
    Simple bisection method to find zero of cubic spline.
    Searches across range of measured currents.
    """
    I_min, I_max = I_array.min(), I_array.max()
    tolerance = 0.01

    for _ in range(100):
        I_mid = (I_min + I_max) / 2
        delta_T_mid = current_profile(I_array, delta_T_array, I_mid)

        if abs(delta_T_mid) < tolerance:
            return I_mid

        delta_T_min = current_profile(I_array, delta_T_array, I_min)

        # Check sign change interval
        if delta_T_min * delta_T_mid < 0:
            I_max = I_mid
        else:
            I_min = I_mid

    return (I_min + I_max) / 2


# -------------------------------------------------------------------------
# MAIN DRIVER FUNCTION
# -------------------------------------------------------------------------
def run():
    """
    Computes ΔT vs current, interpolates results, finds critical current 
    using multiple root-finding methods, and returns smoothed curve data.

    Output structure:
        {
            'smooth': (I_smooth, delta_T_smooth),
            'critical': (I_critial, 0)
        }
    """

    # -------------------------------------------
    # 1. Retrieve steady-state flow conditions
    # -------------------------------------------
    steady_state_result = get_steady_state_values()

    # Extract scalar m_dot_ss cleanly
    if isinstance(steady_state_result, tuple):
        m_dot_ss = steady_state_result[0]
        if hasattr(m_dot_ss, '__len__'):
            m_dot_ss = m_dot_ss[-1]
    else:
        m_dot_ss = steady_state_result
        if hasattr(m_dot_ss, '__len__'):
            m_dot_ss = m_dot_ss[-1]

    # RK4 numerical error estimate for threshold correction
    rk4_error_val = get_rk4_error_val()

    # -------------------------------------------
    # 2. Generate ΔT data for different currents
    # -------------------------------------------
    I_runs = []          # currents tested
    delta_T = []         # ΔT relative to failure threshold
    final_temperatures = []

    print("Computing...")
    total_start_time = time.time()

    # Run from 6 A to 12 A (inclusive)
    for idx, I in enumerate(np.arange(6, 13, 1)):
        iter_start = time.time()

        # Total discharge time at current I
        t_total = q_b / I

        # Solve temperature ODE using the RK4 solver
        t_i, T_i = Tb(dTb_dt, I_params(I, m_dot_ss), stepsize=0.2)

        I_runs.append(I)
        final_temperatures.append(T_i[-1])

        # ΔT relative to the safe maximum
        delta_T.append(T_i[-1] - (T_b_max - rk4_error_val))

        iter_time = time.time() - iter_start

    total_time = time.time() - total_start_time
    print(f"\nTotal data generation time: {total_time:.2f} seconds")

    # Convert lists to arrays
    I_array = np.array(I_runs)
    delta_T_array = np.array(delta_T)

    # -------------------------------------------
    # 3. Interpolation accuracy comparison
    # -------------------------------------------
    comparison = compare_interpolation_accuracy(I_array, delta_T_array)
    print(f"Newton Divided Diff RMSE: {comparison['newton_rmse']:.6f}")

    # -------------------------------------------
    # 4. Find root (critical current)
    # -------------------------------------------
    I_min = I_array.min()
    I_max = I_array.max()
    tolerance = 0.01

    # --- Bisection root (with fallback) ---
    try:
        critical_current_bisection = bisection(
            f=lambda I: current_profile(I_array, delta_T_array, I),
            a=I_min,
            b=I_max,
            tolerance=tolerance
        )
        print(f"\nBisection result: {critical_current_bisection}")
    except NameError:
        print("Warning: bisection() not found in root_finders — using manual bisection.")
        critical_current_bisection = find_root_cubic_spline(I_array, delta_T_array)
        print(f"Manual bisection result: {critical_current_bisection}")

    # --- Newton-Raphson refinement (with fallback) ---
    try:
        critical_current_newton = newton(
            lambda I: current_profile(I_array, delta_T_array, I),
            lambda I: current_profile_derivative(I_array, delta_T_array, I),
            x0=critical_current_bisection,
            epsilon=0.01,
            max_iter=50
        )
        print(f"Newton result: {critical_current_newton}")
    except NameError:
        print("Warning: newton() not found — using bisection only.")
        critical_current_newton = critical_current_bisection

    # -------------------------------------------
    # 5. Final RMSE check for interpolation
    # -------------------------------------------
    delta_T_interpolated = np.array(
        [current_profile(I_array, delta_T_array, I) for I in I_array]
    )
    residuals = delta_T_interpolated - delta_T_array
    rmse = np.sqrt(np.mean(residuals**2))

    print(f"Critical Current: {critical_current_bisection:.2f} A")
    print(f"Newton result: {critical_current_newton:.2f} A")

    # -------------------------------------------
    # 6. Generate smooth interpolation curve
    # -------------------------------------------
    I_smooth = np.linspace(I_array.min(), I_array.max(), 100)
    delta_T_smooth = [current_profile(I_array, delta_T_array, I) for I in I_smooth]

    # -------------------------------------------
    # 7. Return analysis results (no plotting)
    # -------------------------------------------
    return {
        'smooth': (I_smooth, delta_T_smooth),
        'critical': (critical_current_bisection, 0)
    }
