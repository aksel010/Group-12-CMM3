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
    return (
        m_b,      # m [kg]
        C_b,      # cp_b [J/(kg·K)]
        I,        # I [A]
        DC_IR *24,      # R_cell * n _cell it passes through[Ω]
        A_s,     # A_s [m²]
        T_in,     # T_c_in [K]
        get_steady_state_values()[0]     # m_dot_c [kg/s]
    )


def current_profile(I_array, delta_T_array, I_query):
    """Returns the interpolated Delta T using cubic spline."""
    return cubic_spline_interpolation(I_array, delta_T_array, I_query)


def current_profile_derivative(I_array, delta_T_array, I_query):
    """Returns the derivative of the interpolated Delta T using cubic spline."""
    return cubic_spline_derivative(I_array, delta_T_array, I_query)


def find_root_cubic_spline(I_array, delta_T_array):
    """Find where cubic spline crosses zero using bisection"""
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
    print("Starting simulation...")
    
    # Get steady state values (only computed once, when run() is called)
    steady_state_result = get_steady_state_values()
    
    # Handle different return types and extract scalar value
    if isinstance(steady_state_result, tuple):
        m_dot_ss = steady_state_result[0]
        if hasattr(m_dot_ss, '__len__'):  # If it's an array
            m_dot_ss = m_dot_ss[-1]  # Get the last (steady-state) value
    else:
        m_dot_ss = steady_state_result
        if hasattr(m_dot_ss, '__len__'):  # If it's an array
            m_dot_ss = m_dot_ss[-1]  # Get the last (steady-state) value
    
    rk4_error_val = get_rk4_error_val()
    
    print(f"m_dot_ss: {m_dot_ss}, rk4_error_val: {rk4_error_val}")
    
    # Analysis - generate data points
    I_runs = []
    delta_T = []
    final_temperatures = []
    
    print("Generating data points...")
    total_start_time = time.time()
    
    for idx, i in enumerate(np.arange(2, 16, 1)):
        I_0 = i
        iter_start = time.time()
        print(f"  Computing for I = {I_0} A (point {idx+1}/20)...", end=" ")
        
        t_total = q_b / I_0  # total time [s]
        t_i, T_i = Tb(dTb_dt, I_params(I_0, m_dot_ss), stepsize=0.2)
        I_runs.append(I_0)
        delta_T.append(T_i[-1] - (T_b_max - rk4_error_val))
        final_temperatures.append(T_i[-1])
        
        iter_time = time.time() - iter_start
        print(f"done in {iter_time:.2f}s")
    
    total_time = time.time() - total_start_time
    print(f"\nTotal data generation time: {total_time:.2f} seconds")

    I_array = np.array(I_runs)
    delta_T_array = np.array(delta_T)

    print("Comparing interpolation methods...")
    # Compare both methods
    comparison = compare_interpolation_accuracy(I_array, delta_T_array)

    print(f"Newton Divided Diff RMSE: {comparison['newton_rmse']:.6f}")

    # Find critical current using bisection
    I_min = I_array.min()
    I_max = I_array.max()
    tolerance = 0.01

    print("Finding critical current using bisection...")
    try:
        critical_current_bisection = bisection(
            f=lambda I: current_profile(I_array, delta_T_array, I), 
            a=I_min, 
            b=I_max, 
            tolerance=tolerance
        )
        print(f"Bisection result: {critical_current_bisection}")
    except NameError:
        print("Warning: bisection function not found, using manual implementation")
        critical_current_bisection = find_root_cubic_spline(I_array, delta_T_array)
        print(f"Manual bisection result: {critical_current_bisection}")

    # Use the bisection result as a good initial guess for the Newton-Raphson method
    print("Finding critical current using Newton-Raphson...")
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
        print("Warning: newton function not found, using bisection result only")
        critical_current_newton = critical_current_bisection

    # Error calculation
    critical_current = find_root_cubic_spline(I_array, delta_T_array)
    delta_T_interpolated = np.array([current_profile(I_array, delta_T_array, I) for I in I_array])
    residuals = delta_T_interpolated - delta_T_array
    rmse = np.sqrt(np.mean(residuals**2))

    # Plot
    print("Generating plot...")
    plt.figure(figsize=(10, 6))
    plt.scatter(I_runs, delta_T, color='blue', s=50, label='Simulation Data', zorder=5)
    I_smooth = np.linspace(I_array.min(), I_array.max(), 100)
    delta_T_smooth = [cubic_spline_interpolation(I_array, delta_T_array, I) for I in I_smooth] 

    plt.plot(I_smooth, delta_T_smooth, 'r-', linewidth=2, label='Cubic Spline Interpolation')
    plt.plot(critical_current, 0, 'ro', markersize=10, 
            label=f'Critical Point: {critical_current:.1f} A', zorder=6)
    plt.axhline(0, color='red', linestyle='--', linewidth=1)

    plt.xlabel('Current (A)')
    plt.ylabel('Delta T (K)')
    plt.title('Critical Current Analysis - Cubic Spline Interpolation')
    plt.legend()

    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

    print(f"Critical Current: {critical_current:.2f} A")
    print(f"Newton result: {critical_current_newton}")

    # Return the critical current so other modules can use it
    return critical_current

if __name__ == "__main__":
    run()
