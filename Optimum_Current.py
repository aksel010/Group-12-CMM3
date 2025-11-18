import numpy as np
import matplotlib.pyplot as plt
import time

from ODE import get_tb, d_tb_dt
from config import *
from Mass_flowrate import get_steady_state_values
from RK4_Error import get_rk4_error_val
from interpolater import *
from root_finders import bisection, newton_method

# -------------------------------------------------------------------------
# PARAMETER BUILDER FOR ODE SOLVER
# -------------------------------------------------------------------------
def build_i_params(I, m_dot_ss):
    """
    Build the parameter tuple needed by d_tb_dt() for a given current I.
    """
    return (
        m_b,
        C_b,
        I,
        dc_ir * 24,
        A_s,
        T_in,
        get_steady_state_values()[0]
    )

# -------------------------------------------------------------------------
# INTERPOLATION WRAPPERS
# -------------------------------------------------------------------------
def interpolate_current_delta_T(I_array, delta_T_array, I_query):
    """
    Cubic spline interpolation of ΔT vs current (I).
    Returns interpolated ΔT at I_query.
    """
    return cubic_spline_interpolation(I_array, delta_T_array, I_query)

def interpolate_current_delta_T_deriv(I_array, delta_T_array, I_query):
    """
    Returns d(ΔT)/dI using derivative of cubic spline.
    """
    return cubic_spline_derivative(I_array, delta_T_array, I_query)

# -------------------------------------------------------------------------
# MANUAL ROOT FINDING USING BISECTION ON THE SPLINE
# -------------------------------------------------------------------------
def find_cubic_spline_root(I_array, delta_T_array):
    """
    Simple bisection method to find zero of cubic spline.
    """
    I_min, I_max = I_array.min(), I_array.max()
    tolerance = 0.01
    for _ in range(100):
        I_mid = (I_min + I_max) / 2
        delta_T_mid = interpolate_current_delta_T(I_array, delta_T_array, I_mid)
        if abs(delta_T_mid) < tolerance:
            return I_mid
        delta_T_min = interpolate_current_delta_T(I_array, delta_T_array, I_min)
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
    Computes ΔT vs current, interpolates results, finds critical current,
    using multiple root-finding methods, and returns smoothed curve data.
    """
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
    current_tests = []
    delta_t = []
    final_temps = []
    print("Computing...")
    total_start_time = time.time()
    for idx, I in enumerate(np.arange(6, 13, 1)):
        iter_start = time.time()
        t_total = q_b / I
        t_i, T_i = get_tb(d_tb_dt, build_i_params(I, m_dot_ss), stepsize=0.2)
        current_tests.append(I)
        final_temps.append(T_i[-1])
        delta_t.append(T_i[-1] - (T_b_max - rk4_error_val))
        iter_time = time.time() - iter_start
    total_time = time.time() - total_start_time
    print(f"\nTotal data generation time: {total_time:.2f} seconds")
    I_array = np.array(current_tests)
    delta_T_array = np.array(delta_t)
    comparison = compare_interpolation_accuracy(I_array, delta_T_array)
    print(f"Newton Divided Diff RMSE: {comparison['newton_rmse']:.6f}")
    I_min = I_array.min()
    I_max = I_array.max()
    tolerance = 0.01
    try:
        critical_current_bisection = bisection(
            f=lambda I: interpolate_current_delta_T(I_array, delta_T_array, I),
            a=I_min,
            b=I_max,
            tolerance=tolerance
        )
        print(f"\nBisection result: {critical_current_bisection}")
    except NameError:
        print("Warning: bisection() not found in root_finders — using manual bisection.")
        critical_current_bisection = find_cubic_spline_root(I_array, delta_T_array)
        print(f"Manual bisection result: {critical_current_bisection}")
    try:
        critical_current_newton = newton_method(
            lambda I: interpolate_current_delta_T(I_array, delta_T_array, I),
            lambda I: interpolate_current_delta_T_deriv(I_array, delta_T_array, I),
            x0=critical_current_bisection,
            epsilon=0.01,
            max_iter=50
        )
        print(f"Newton result: {critical_current_newton}")
    except NameError:
        print("Warning: newton_method() not found — using bisection only.")
        critical_current_newton = critical_current_bisection
    delta_T_interpolated = np.array(
        [interpolate_current_delta_T(I_array, delta_T_array, I) for I in I_array]
    )
    residuals = delta_T_interpolated - delta_T_array
    rmse = np.sqrt(np.mean(residuals**2))
    print(f"Critical Current: {critical_current_bisection:.2f} A")
    print(f"Newton result: {critical_current_newton:.2f} A")
    I_smooth = np.linspace(I_array.min(), I_array.max(), 100)
    delta_T_smooth = [interpolate_current_delta_T(I_array, delta_T_array, I) for I in I_smooth]
    return {
        'smooth': (I_smooth, delta_T_smooth),
        'critical': (critical_current_bisection, 0)
    }
