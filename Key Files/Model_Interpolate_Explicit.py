import numpy as np
import matplotlib.pyplot as plt
from ODE import Tb, dTb_dt
from config import q_b, m_b, C_b, T_in, M_DOT, T_b_max, R_b, S_b

# Cubic Spline Interpolation
def cubic_spline_coefficients(x_data, y_data):
    n = len(x_data) - 1
    
    # Step 1: Calculate h (interval widths)
    h = [x_data[i+1] - x_data[i] for i in range(n)]
    
    # Step 2: Set up tridiagonal system for second derivatives
    A = np.zeros((n+1, n+1))
    b = np.zeros(n+1)
    
    # Main diagonal
    for i in range(1, n):
        A[i, i] = 2 * (h[i-1] + h[i])
    
    # Upper diagonal
    for i in range(1, n):
        A[i, i+1] = h[i]
    
    # Lower diagonal  
    for i in range(1, n):
        A[i, i-1] = h[i-1]
    
    # Boundary conditions (natural spline)
    A[0, 0] = 1
    A[n, n] = 1
    
    # Right-hand side
    for i in range(1, n):
        b[i] = 6 * ((y_data[i+1] - y_data[i]) / h[i] - (y_data[i] - y_data[i-1]) / h[i-1])
    
    # Step 3: Solve for second derivatives M
    M = np.linalg.solve(A, b)
    
    # Step 4: Calculate coefficients for each segment
    coefficients = []
    for i in range(n):
        a = y_data[i]
        b = (y_data[i+1] - y_data[i]) / h[i] - h[i] * (2 * M[i] + M[i+1]) / 6
        c = M[i] / 2
        d = (M[i+1] - M[i]) / (6 * h[i])
        coefficients.append((a, b, c, d, x_data[i], x_data[i+1]))
    
    return coefficients

def cubic_spline_interpolation(x_data, y_data, x_query):
    coefficients = cubic_spline_coefficients(x_data, y_data)
    
    # Find the correct segment
    for coeff in coefficients:
        a, b, c, d, x_start, x_end = coeff
        if x_start <= x_query <= x_end:
            dx = x_query - x_start
            return a + b * dx + c * dx**2 + d * dx**3
    
    # Extrapolation handling
    if x_query < x_data[0]:
        coeff = coefficients[0]
        a, b, c, d, x_start, x_end = coeff
        dx = x_query - x_start
        return a + b * dx + c * dx**2 + d * dx**3
    else:
        coeff = coefficients[-1]
        a, b, c, d, x_start, x_end = coeff
        dx = x_query - x_start
        return a + b * dx + c * dx**2 + d * dx**3

# Analysis
I_runs = []
delta_T = []
final_temperatures = []

def I_params(I):
    return (
        m_b,      # m [kg]
        C_b,      # cp_b [J/(kg·K)]
        I,        # I [A]
        R_b,      # R [Ω]
        S_b*10,     # A_s [m²]
        T_in,     # T_c_in [K]
        M_DOT     # m_dot_c [kg/s]
    )

# Generate data points
for idx, i in enumerate(range(5, 130, 5)):
    I_0 = i
    t_total = q_b / I_0  # total time [s]
    t_i, T_i = Tb(dTb_dt, I_params(I_0))
    I_runs.append(I_0)
    delta_T.append(T_i[-1] - T_b_max)
    final_temperatures.append(T_i[-1])

I_array = np.array(I_runs)
delta_T_array = np.array(delta_T)

# Find critical current
def find_root_cubic_spline():
    """Find where cubic spline crosses zero using bisection"""
    I_min, I_max = I_array.min(), I_array.max()
    tolerance = 0.01
    
    for _ in range(100):
        I_mid = (I_min + I_max) / 2
        delta_T_mid = cubic_spline_interpolation(I_array, delta_T_array, I_mid)
        
        if abs(delta_T_mid) < tolerance:
            return I_mid
        
        delta_T_min = cubic_spline_interpolation(I_array, delta_T_array, I_min)
        
        if delta_T_min * delta_T_mid < 0:
            I_max = I_mid
        else:
            I_min = I_mid
    
    return (I_min + I_max) / 2

critical_current = find_root_cubic_spline()

# Plot
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
