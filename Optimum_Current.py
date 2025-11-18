"""
Determine optimum battery charging current by interpolating and root-finding the deltaT-vs-current curve.

Includes cubic spline interpolation, manual and root_finders-based root-finding, and result packaging for downstream plotters or analytics.
All functions and blocks include PEP8 docstrings and clarifying comments.
"""
import numpy as np
import matplotlib.pyplot as plt
import time
from ODE import Tb, dTb_dt
from config import *
from Mass_flowrate import get_steady_state_values
from RK4_Error import get_rk4_error_val
from interpolater import *
from root_finders import bisection, newton

def I_params(I, m_dot_ss):
    """
    Builds ODE parameter tuple for a specific current value and coolant flow.
    Args:
        I (float): Discharge current [A].
        m_dot_ss (float): Steady-state mass flow rate [kg/s].
    Returns:
        tuple: Parameters for dTb_dt.
    """
    return (
        m_b,
        C_b,
        I,
        DC_IR * 24,
        A_s,
        T_in,
        get_steady_state_values()[0]
    )

def current_profile(I_array, delta_T_array, I_query):
    """
    Interpolates ΔT vs. I (cubic spline) and returns ΔT for queried current.

    Args:
        I_array (array): Current values (A).
        delta_T_array (array): ΔT values for those currents.
        I_query (float or array): Query current(s).
    Returns:
        float or array: Interpolated ΔT at each query point.
    """
    return cubic_spline_interpolation(I_array, delta_T_array, I_query)

def current_profile_derivative(I_array, delta_T_array, I_query):
    """
    Derivative of interpolated ΔT-vs-I curve at given current(s).
    Args/Returns as for current_profile.
    """
    return cubic_spline_derivative(I_array, delta_T_array, I_query)

def find_root_cubic_spline(I_array, delta_T_array):
    """
    Root finder for cubic spline ΔT profile using bisection across I range.
    Args:
        I_array (array): Current values (A).
        delta_T_array (array): ΔT values for those currents.
    Returns:
        float: Approximate crossing current for ΔT=0.
    """
    I_min, I_max = I_array.min(), I_array.max()
    tolerance = 0.01
    for _ in range(100):
        I_mid = (I_min + I_max) / 2
        delta_T_mid = current_profile(I_array, delta_T_array, I_mid)
        if abs(delta_T_mid) < tolerance:
            return I_mid
        delta_T_min = current_profile(I_array, delta_T_array, I_min)
        if delta_T_min * delta_T_mid < 0:
            I_max = I_mid
        else:
            I_min = I_mid
    return (I_min + I_max) / 2

def run():
    """
    Main driver: computes ΔT vs I and finds critical (optimum) current by interpolation/root-finding.

    Returns:
        dict: {'smooth': (I_smooth, delta_T_smooth), 'critical': (critical_current, 0)}
    """
    # 1. Get steady-state flow (safe cooling for ODE)
    steady_state_result = get_steady_state_values()
    if isinstance(steady_state_result, tuple):
        m_dot_ss = steady_state_result[0]
        if hasattr(m_dot_ss, '__len__'):
            m_dot_ss = m_dot_ss[-1]
    else:
        m_dot_ss = steady_state_result
        if hasattr(m_dot_ss, '__len__'):
            m_dot_ss = m_dot_ss[-1]
    rk4_error_val = get_rk4_error_val()
    I_runs = []
    delta_T = []
    final_temperatures = []
    print("Computing...")
    total_start_time = time.time()
    for I in np.arange(6, 13, 1):
        # Discharge simulation for each current
        t_total = q_b / I
        t_i, T_i = Tb(dTb_dt, I_params(I, m_dot_ss), stepsize=0.2)
        I_runs.append(I)
        final_temperatures.append(T_i[-1])
        delta_T.append(T_i[-1] - (T_b_max - rk4_error_val))
    total_time = time.time() - total_start_time
    print(f"\nTotal data generation time: {total_time:.2f} seconds")
    I_array = np.array(I_runs)
    delta_T_array = np.array(delta_T)
    comparison = compare_interpolation_accuracy(I_array, delta_T_array)
    print(f"Newton Divided Diff RMSE: {comparison['newton_rmse']:.6f}")
    I_min = I_array.min()
    I_max = I_array.max()
    tolerance = 0.01
    # Root (critical current) via bisection (or fallback manual method)
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
    delta_T_interpolated = np.array([
        current_profile(I_array, delta_T_array, I) for I in I_array
    ])
    residuals = delta_T_interpolated - delta_T_array
    rmse = np.sqrt(np.mean(residuals**2))
    print(f"Critical Current: {critical_current_bisection:.2f} A")
    print(f"Newton result: {critical_current_newton:.2f} A")
    I_smooth = np.linspace(I_array.min(), I_array.max(), 100)
    delta_T_smooth = [current_profile(I_array, delta_T_array, I) for I in I_smooth]
    return {
        'smooth': (I_smooth, delta_T_smooth),
        'critical': (critical_current_bisection, 0)
    }
