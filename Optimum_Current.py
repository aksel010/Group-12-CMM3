import numpy as np
import matplotlib.pyplot as plt
from ODE import Tb, dTb_dt
from config import *
from Mass_flowrate import get_steady_state_values
from RK4_Error import get_rk4_error_val
from interpolater import *

# Analysis
I_runs = []
delta_T = []
final_temperatures = []

def I_params(I):
    return (
        m_b,      # m [kg]
        C_b,      # cp_b [J/(kg·K)]
        I,        # I [A]
        DC_IR *24,      # R_cell * n _cell it passes through[Ω]
        A_s,     # A_s [m²]
        T_in,     # T_c_in [K]
        m_dot_ss     # m_dot_c [kg/s]
    )

# Generate data points
for idx, i in enumerate(range(1, 30, 3)):
    I_0 = i
    t_total = q_b / I_0  # total time [s]
    t_i, T_i = Tb(dTb_dt, I_params(I_0), stepsize=0.2)
    I_runs.append(I_0)
    delta_T.append(T_i[-1] - (T_b_max-rk4_error_val))
    final_temperatures.append(T_i[-1])

I_array = np.array(I_runs)
delta_T_array = np.array(delta_T)

def current_profile(I_query):
    """Returns the interpolated Delta T using cubic spline."""
    return cubic_spline_interpolation(I_array, delta_T_array, I_query)

def current_profile_derivative(I_query):
    """Returns the derivative of the interpolated Delta T using cubic spline."""
    return cubic_spline_derivative(I_array, delta_T_array, I_query)

# Compare both methods
comparison = compare_interpolation_accuracy(I_array, delta_T_array)

print(f"Spline RMSE: {comparison['spline_rmse']:.6f}")
print(f"Newton Divided Diff RMSE: {comparison['newton_rmse']:.6f}")

# Find critical current
def find_root_cubic_spline():
    """Find where cubic spline crosses zero using bisection"""
    I_min, I_max = I_array.min(), I_array.max()
    tolerance = 0.01
    
    for _ in range(100):
        I_mid = (I_min + I_max) / 2
        
        delta_T_mid = current_profile(I_mid) 
        
        if abs(delta_T_mid) < tolerance:
            return I_mid
        
        delta_T_min = current_profile(I_min) 
        
        if delta_T_min * delta_T_mid < 0:
            I_max = I_mid
        else:
            I_min = I_mid
    
    return (I_min + I_max) / 2

I_min = I_array.min()
I_max = I_array.max()
tolerance = 0.01

#root finding validation with NR

critical_current_bisection = bisection(
    f=current_profile, 
    a= I_min, 
    b= I_max, 
    tolerance= tolerance
)

print(f"Bisection result: {critical_current_bisection}") # Added print for bisection result

def f_to_test(I):
    return current_profile(I)
    
def Df_to_test(I): 
    return current_profile_derivative(I)

# Use the bisection result as a good initial guess for the Newton-Raphson method
critical_current_newton = newton(f_to_test, Df_to_test, x0=critical_current_bisection, epsilon=0.01, max_iter=50) 


#Error calculation
critical_current = find_root_cubic_spline()
delta_T_interpolated = np.array([current_profile(I) for I in I_array])
residuals = delta_T_interpolated - delta_T_array
rmse = np.sqrt(np.mean(residuals**2))

def run():
    # Plot
    plt.figure(figsize=(10, 6))
    plt.scatter(I_runs, delta_T, color='blue', s=50, label='Simulation Data', zorder=5)
    I_smooth = np.linspace(I_array.min(), I_array.max(), 100)
    # This part of the plot uses the old list comprehension style, which is fine
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

if __name__ == "__main__":
    run()
