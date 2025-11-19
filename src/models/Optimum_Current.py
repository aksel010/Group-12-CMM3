"""
Battery Charging Current Optimization Module.

Determines optimum battery charging current via interpolation and root-finding
of the temperature deviation (deltaT) vs. current curve using cubic splines
and bisection/Newton methods.
"""

import numpy as np
import matplotlib.pyplot as plt
import time

from src.models.ODE import get_tb, d_tb_dt
from src.config import *
from src.models.Mass_flowrate import get_steady_state_values, calculate_steady_state_mass_flow
from src.models.RK4_Error import get_rk4_error_val
from src.utils.interpolater import *
from src.utils.root_finders import bisection, newton


def current_params(current):
    """
    Build ODE parameter tuple for given current and coolant flow.

    Args:
        current (float): Discharge current (A).
        mass_flow_ss (float): Steady-state mass flow rate (kg/s).

    Returns:
        tuple: Parameters (m_b, c_b, current, losses, a_s, t_in, ss_value)
            for dTb_dt ODE function.
    """
    return (
        m_cell,
        c_b,
        current,
        dc_ir*24,
        a_s,
        t_in,
        calculate_steady_state_mass_flow(current**2*r_b, mass_flow_initial)[0]
    )


def current_profile(current_array, delta_temp_array, current_query):
    """
    Interpolate deltaT vs. current using cubic spline.

    Args:
        current_array (np.ndarray): Current values (A).
        delta_temp_array (np.ndarray): Temperature deviations (K).
        current_query (float or np.ndarray): Query current(s) (A).

    Returns:
        float or np.ndarray: Interpolated deltaT at query point(s).
    """
    return cubic_spline_interpolation(current_array, delta_temp_array, current_query)


def current_profile_derivative(current_array, delta_temp_array, current_query):
    """
    Compute derivative of interpolated deltaT-vs-current curve.

    Args:
        current_array (np.ndarray): Current values (A).
        delta_temp_array (np.ndarray): Temperature deviations (K).
        current_query (float or np.ndarray): Query current(s) (A).

    Returns:
        float or np.ndarray: Derivative dΔT/dI (K/A) at query point(s).
    """
    return cubic_spline_derivative(current_array, delta_temp_array, current_query)


def find_root_cubic_spline(current_array, delta_temp_array):
    """
    Find root of cubic spline deltaT profile using manual bisection.

    Args:
        current_array (np.ndarray): Current values (A).
        delta_temp_array (np.ndarray): Temperature deviations (K).

    Returns:
        float: Critical current where deltaT = 0 (A). Tolerance: 0.01 K,
            max 100 iterations.
    """
    current_min, current_max = current_array.min(), current_array.max()
    tolerance = 0.01
    
    for _ in range(100):
        current_mid = (current_min + current_max) / 2
        delta_temp_mid = current_profile(current_array, delta_temp_array, current_mid)
        
        if abs(delta_temp_mid) < tolerance:
            return current_mid
            
        delta_temp_min = current_profile(current_array, delta_temp_array, current_min)
        if delta_temp_min * delta_temp_mid < 0:
            current_max = current_mid
        else:
            current_min = current_mid
            
    return (current_min + current_max) / 2


def run():
    """
    Find critical battery charging current via interpolation and root-finding.

    Workflow:
        1. Compute battery temperature for currents 6-13 A (1 A steps)
        2. Calculate temperature deviations from target
        3. Perform cubic spline interpolation
        4. Find critical current using bisection and Newton methods

    Returns:
        dict: Results with keys:
            - 'smooth': (current_array, delta_temp_array) - 100 interpolated points
            - 'critical': (critical_current, 0) - optimum current where deltaT = 0

    Side Effects:
        Prints progress, timing, RMSE, and root-finding results. Displays plot.
    """

    rk4_error_val = get_rk4_error_val()

    print("Computing...")
    total_start_time = time.time()

    # Storage for results
    current_runs = []
    final_temperatures = []
    delta_temp = []

    # Run from 6 A to 13 A (adjust range as needed for zero crossing)
    for current in np.arange(6, 13, 1):
        iter_start = time.time()
        
        # Total discharge time at this current
        t_total = q_b / current
        
        # Solve temperature ODE using RK4 solver
        time_points, temp_battery = get_tb(d_tb_dt, current_params(current), stepsize=0.2)

        current_runs.append(current)
        final_temperatures.append(temp_battery[-1])
        
        # ΔT relative to the safe maximum
        target_temp = t_b_max - rk4_error_val
        temp_deviation = temp_battery[-1] - target_temp
        delta_temp.append(temp_deviation)
        
        iter_time = time.time() - iter_start
        # Optional: print iteration time if needed
        # print(f"  Iteration {current}A completed in {iter_time:.3f}s")

    total_time = time.time() - total_start_time
    print(f"\nTotal data generation time: {total_time:.2f} seconds")

    # Convert lists to arrays
    current_array = np.array(current_runs)
    delta_temp_array = np.array(delta_temp)
    
    # Validate interpolation
    comparison = compare_interpolation_accuracy(current_array, delta_temp_array)
    print(f"Newton Divided Diff RMSE: {comparison['newton_rmse']:.6f}")
    
    current_min = current_array.min()
    current_max = current_array.max()
    tolerance = 0.01
    
    # Find root via bisection
    try:
        print("f(a) =", current_profile(current_array, delta_temp_array, current_min))
        print("f(b) =", current_profile(current_array, delta_temp_array, current_max))
        
        current_values = np.linspace(current_min, current_max, 100)
        profile_vals = [current_profile(current_array, delta_temp_array, current) for current in current_values]
        
        plt.plot(current_values, profile_vals)
        plt.axhline(0, color='gray', linestyle='--')
        plt.xlabel('Current (I)')
        plt.ylabel('current_profile')
        plt.show()
        
        critical_current_bisection = bisection(
            f=lambda current: current_profile(current_array, delta_temp_array, current),
            a=current_min,
            b=current_max,
            tolerance=tolerance
        )
        print(f"\nBisection result: {critical_current_bisection}")
    except NameError:
        print("Warning: bisection() not found in root_finders — using manual bisection.")
        critical_current_bisection = find_root_cubic_spline(current_array, delta_temp_array)
        print(f"Manual bisection result: {critical_current_bisection}")
    
    # Refine with Newton's method
    try:
        critical_current_newton = newton(
            lambda current: current_profile(current_array, delta_temp_array, current),
            lambda current: current_profile_derivative(current_array, delta_temp_array, current),
            x0=critical_current_bisection,
            epsilon=0.01,
            max_iter=50
        )
        print(f"Newton result: {critical_current_newton}")
    except NameError:
        print("Warning: newton() not found — using bisection only.")
        critical_current_newton = critical_current_bisection
    
    # Calculate RMSE
    delta_temp_interpolated = np.array([
        current_profile(current_array, delta_temp_array, current) for current in current_array
    ])
    residuals = delta_temp_interpolated - delta_temp_array
    rmse = np.sqrt(np.mean(residuals**2))
    
    print(f"Critical Current: {critical_current_bisection:.2f} A")
    print(f"Newton result: {critical_current_newton:.2f} A")
    
    # Generate smooth interpolated curve
    current_smooth = np.linspace(current_array.min(), current_array.max(), 100)
    delta_temp_smooth = [current_profile(current_array, delta_temp_array, current) for current in current_smooth]
    
    return {
        'smooth': (current_smooth, delta_temp_smooth),
        'critical': (critical_current_bisection, 0)
    }


run()