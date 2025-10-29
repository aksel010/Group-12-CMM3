import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from ODE import Tb, dTb_dt
from config import q_b, m_b, C_b, T_in, M_DOT, T_b_max
 
I_runs = []
delta_T = []
final_temperatures = []
 
def I_params(I):
    return (
        m_b,      # m [kg]
        C_b,      # cp_b [J/(kg·K)]
        I,        # I [A]
        0.1,      # R [Ω]
        0.01,     # A_s [m²]
        T_in,     # T_c_in [K]
        M_DOT     # m_dot_c [kg/s]
    )
 
for idx, i in enumerate(range(5, 100, 5)):
    I_0 = i
    t_total = q_b / I_0  # total time [s]
    t_i, T_i = Tb(dTb_dt, t_total, I_params(I_0))
    
    final_temp = T_i[-1]
    print(final_temp)
    I_runs.append(I_0)
    delta_T.append(final_temp - T_b_max)
    final_temperatures.append(final_temp)
 
I_array = np.array(I_runs)
delta_T_array = np.array(delta_T)
 
print("\n ---INTERPOLATION RESULTS--- ")
print(f"Current range: {I_array.min()}A to {I_array.max()}A")
print(f"Delta T range: {delta_T_array.min():.2f}K to {delta_T_array.max():.2f}K")
 
 
if np.any(delta_T_array <= 0) and np.any(delta_T_array >= 0):
    
    current_vs_deltaT = interp1d(delta_T_array, I_array, kind='linear', 
                                bounds_error=False, fill_value="extrapolate")
 
    critical_current = float(current_vs_deltaT(0))
 
    print(f"CRITICAL CURRENT: {critical_current:.2f} A")
    
 
    '''temp_vs_current = interp1d(I_array, final_temperatures, kind='linear')
    predicted_final_temp = float(temp_vs_current(critical_current))
    print(f"Predicted final temp at critical current: {predicted_final_temp:.2f} K")'''
    
else:
    print("Never reaches T_b_max")
    if delta_T_array.min() > 0:
        print("All currents cause overheating (ΔT > 0)")
    else:
        print("System never reaches T_b_max (ΔT < 0 for all currents)")
 
 
plt.figure(figsize=(10, 6))
plt.scatter(I_runs, delta_T, color='blue', s=50, label='Simulation Data', zorder=5)
 
# Plot interpolation line 
if 'critical_current' in locals() and np.any(delta_T_array <= 0) and np.any(delta_T_array >= 0):
    
    I_smooth = np.linspace(I_array.min(), I_array.max(), 100)
    delta_T_smooth = interp1d(I_array, delta_T_array, kind='cubic')(I_smooth)
    plt.plot(I_smooth, delta_T_smooth, 'r-', alpha=0.7, label='Interpolation', linewidth=2)
    
    # Mark critical point
    plt.plot(critical_current, 0, 'ro', markersize=10, 
             label=f'Critical Point: {critical_current:.1f} A', zorder=6)
 
plt.xlabel('Current (A)')
plt.ylabel('Delta T (K)')       
plt.axhline(0, color='red', linestyle='--', linewidth=2, label='Safe Limit (ΔT=0)')
plt.axvline(critical_current if 'critical_current' in locals() else 0, 
            color='green', linestyle=':', linewidth=2, 
            label=f'Critical Current: {critical_current:.2f} A' if 'critical_current' in locals() else 'No critical current found')
 
plt.title('Delta T vs Current - Critical Current Analysis')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()
 