import math
from heptane_itpl import M_DOT, D, Cp_func,calculate_h
from config import C_b, T_in, m_b
import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

# Parameters
I_0 = 50
R_0 = 240/I_0

params_initial = (
    1.0,      # m [kg]
    1306,     # cp_b [J/(kg·K)]
    I_0,      # I [A]
    R_0,      # R [Ω]
    1.9634954084936207e-05,  # A_s [m²]
    298.13,   # T_c_in [K]
    0.001     # m_dot_c [kg/s] - reduced from 1 to more reasonable value
)

def dTb_dt_simple(Tb, t, params):
    """
    Simplified differential equation for bulk temperature Tb
    Using Newton's law of cooling
    """
    # Unpack parameters
    m, cp_b, I, R, A_s, T_c_in, m_dot_c = params

    # Calculate h based on current bulk temperature
    h = calculate_h(Tb)
    
    # Electrical heating term
    electrical_heating = I**2 * R
    
    # Simple cooling term (Newton's law of cooling)
    cooling = h * A_s * (Tb - T_c_in)
    
    # Rate of change of temperature
    dTb_dt_val = (electrical_heating - cooling) / (m * cp_b)
    
    # Debug printing
    if t % 100 == 0:  # Print every 100 seconds
        print(f"t={t:6.1f}s, Tb={Tb:7.2f}K, heating={electrical_heating:7.2f}W, "
              f"cooling={cooling:7.2f}W, h={h:5.2f}W/m²K, dTb_dt={dTb_dt_val:8.4f}K/s")
    
    return dTb_dt_val

def dTb_dt_effectiveness(Tb, t, params):
    """
    Differential equation using heat exchanger effectiveness method
    """
    # Unpack parameters
    m, cp_b, I, R, A_s, T_c_in, m_dot_c = params

    # Calculate properties - using average temperature for coolant properties
    T_avg = (Tb + T_c_in) / 2
    h = calculate_h(T_avg)
    cp_c = Cp_func(T_avg)
    
    # Electrical heating term
    electrical_heating = I**2 * R
    
    # NTU method for counter-flow heat exchanger
    NTU = (h * A_s) / (m_dot_c * cp_c) if m_dot_c * cp_c > 0 else 0
    effectiveness = (1 - math.exp(-NTU)) if NTU > 0 else 0
    
    # Maximum possible heat transfer
    Q_max = m_dot_c * cp_c * (Tb - T_c_in) if Tb > T_c_in else 0
    
    # Actual cooling
    cooling = effectiveness * Q_max if Q_max > 0 else 0
    
    # Rate of change of temperature
    dTb_dt_val = (electrical_heating - cooling) / (m * cp_b)
    
    return dTb_dt_val

def run_cooling_model(method='simple'):
    """
    Run the cooling model with specified method
    """
    # Initial conditions
    t0 = 0
    T0 = 298.13  # Initial bulk temperature [K]
    
    # Total solution interval
    t_final = 1000
    
    # Step size
    H = 0.2
    
    # Number of steps
    n_step = math.ceil(t_final / H)
    
    # Initialize arrays
    T_rk = np.zeros(n_step + 1)
    t_rk = np.zeros(n_step + 1)
    
    # Set initial conditions
    T_rk[0] = T0
    t_rk[0] = t0
    
    # Choose the method
    if method == 'simple':
        dTb_dt_func = dTb_dt_simple
    else:
        dTb_dt_func = dTb_dt_effectiveness
    
    print(f"Running cooling model with {method} method")
    print(f"Initial conditions: T0 = {T0}K, t_final = {t_final}s")
    print(f"Parameters: m = {params_initial[0]}kg, cp_b = {params_initial[1]}J/kgK")
    print(f"            I = {params_initial[2]}A, R = {params_initial[3]:.4f}Ω")
    print(f"            A_s = {params_initial[4]:.2e}m², T_c_in = {params_initial[5]}K")
    print(f"            m_dot_c = {params_initial[6]}kg/s")
    print("-" * 80)
    
    # Runge-Kutta integration
    for i in range(n_step):
        t_rk[i + 1] = t_rk[i] + H
        
        # Compute the four slopes
        k1 = dTb_dt_func(T_rk[i], t_rk[i], params_initial)
        k2 = dTb_dt_func(T_rk[i] + k1 * H / 2, t_rk[i] + H / 2, params_initial)
        k3 = dTb_dt_func(T_rk[i] + k2 * H / 2, t_rk[i] + H / 2, params_initial)
        k4 = dTb_dt_func(T_rk[i] + k3 * H, t_rk[i] + H, params_initial)
        
        # Weighted average slope
        slope = (k1 + 2 * k2 + 2 * k3 + k4) / 6
        
        # Update temperature
        T_rk[i + 1] = T_rk[i] + H * slope
        
        # Check for unrealistic values
        if T_rk[i + 1] < 0 or T_rk[i + 1] > 1000:
            print(f"Warning: Unrealistic temperature {T_rk[i + 1]}K at t = {t_rk[i + 1]}s")
            break
    
    return t_rk, T_rk

def analyze_results(t, T):
    """
    Analyze and plot the results
    """
    print("\n" + "="*80)
    print("RESULTS ANALYSIS")
    print("="*80)
    print(f"Initial temperature: {T[0]:.2f} K")
    print(f"Final temperature: {T[-1]:.2f} K")
    print(f"Total temperature change: {T[-1] - T[0]:.2f} K")
    print(f"Maximum temperature: {np.max(T):.2f} K")
    print(f"Minimum temperature: {np.min(T):.2f} K")
    
    # Calculate heating and cooling at final state
    m, cp_b, I, R, A_s, T_c_in, m_dot_c = params_initial
    heating_final = I**2 * R
    h_final = calculate_h(T[-1])
    cooling_final = h_final * A_s * (T[-1] - T_c_in)
    
    print(f"\nFinal state balance:")
    print(f"Heating power: {heating_final:.2f} W")
    print(f"Cooling power: {cooling_final:.2f} W")
    print(f"Net power: {heating_final - cooling_final:.2f} W")
    print(f"Heat transfer coefficient: {h_final:.2f} W/m²K")
    
    # Plot results
    plt.figure(figsize=(12, 8))
    
    plt.subplot(2, 2, 1)
    plt.plot(t, T, 'b-', linewidth=2)
    plt.xlabel('Time (s)')
    plt.ylabel('Bulk Temperature (K)')
    plt.title('Temperature vs Time')
    plt.grid(True)
    
    plt.subplot(2, 2, 2)
    heating = I**2 * R * np.ones_like(t)
    cooling = np.array([calculate_h(Ti) * A_s * (Ti - T_c_in) for Ti in T])
    plt.plot(t, heating, 'r-', label='Heating')
    plt.plot(t, cooling, 'b-', label='Cooling')
    plt.plot(t, heating - cooling, 'g-', label='Net Power')
    plt.xlabel('Time (s)')
    plt.ylabel('Power (W)')
    plt.title('Power Balance')
    plt.legend()
    plt.grid(True)
    
    plt.subplot(2, 2, 3)
    h_vals = np.array([calculate_h(Ti) for Ti in T])
    plt.plot(t, h_vals, 'purple', linewidth=2)
    plt.xlabel('Time (s)')
    plt.ylabel('h (W/m²K)')
    plt.title('Heat Transfer Coefficient vs Time')
    plt.grid(True)
    
    plt.subplot(2, 2, 4)
    dTdt = np.gradient(T, t)
    plt.plot(t, dTdt, 'orange', linewidth=2)
    plt.xlabel('Time (s)')
    plt.ylabel('dT/dt (K/s)')
    plt.title('Rate of Temperature Change')
    plt.grid(True)
    
    plt.tight_layout()
    plt.show()
    
    return heating, cooling

# Run the model
if __name__ == "__main__":
    # Try simple method first
    try:
        t_simple, T_simple = run_cooling_model(method='simple')
        analyze_results(t_simple, T_simple)
    except Exception as e:
        print(f"Error in simple method: {e}")
        print("Trying alternative approach...")
        
        # Fallback: Use scipy's solver
        try:
            solution = solve_ivp(
                dTb_dt_simple, 
                [0, 1000], 
                [T_in], 
                args=(params_initial,),
                method='RK45',
                t_eval=np.linspace(0, 1000, 1000)
            )
            analyze_results(solution.t, solution.y[0])
        except Exception as e2:
            print(f"Fallback also failed: {e2}")
            print("Please check your helper functions and parameters.")

def analyze_system_behavior():
    """Analyze why the temperature profile isn't exponential"""
    T_range = np.linspace(300, 600, 100)
    
    # Calculate key parameters across temperature range
    heating_power = I_0**2 * R_0 * np.ones_like(T_range)
    cooling_power = np.array([400 * params_initial[4] * (T - params_initial[5]) 
                             for T in T_range])
    net_power = heating_power - cooling_power
    
    plt.figure(figsize=(12, 8))
    
    # Plot 1: Power balance
    plt.subplot(2, 2, 1)
    plt.plot(T_range, heating_power, 'r-', label='Heating', linewidth=2)
    plt.plot(T_range, cooling_power, 'b-', label='Cooling', linewidth=2)
    plt.plot(T_range, net_power, 'g-', label='Net Power', linewidth=2)
    plt.axhline(0, color='k', linestyle='--', alpha=0.3)
    plt.xlabel('Temperature (K)')
    plt.ylabel('Power (W)')
    plt.legend()
    plt.title('Power Balance vs Temperature')
    plt.grid(True)
    
    # Plot 2: Rate of temperature change
    m, cp_b, I, R, A_s, T_c_in, m_dot_c = params_initial
    dTdt = net_power / (m * cp_b)
    plt.subplot(2, 2, 2)
    plt.plot(T_range, dTdt, 'purple', linewidth=2)
    plt.xlabel('Temperature (K)')
    plt.ylabel('dT/dt (K/s)')
    plt.title('Rate of Temperature Change vs Temperature')
    plt.grid(True)
    
    # Plot 3: Heat transfer coefficient
    h_vals = np.array([calculate_h(T) for T in T_range])
    plt.subplot(2, 2, 3)
    plt.plot(T_range, h_vals, 'orange', linewidth=2)
    plt.xlabel('Temperature (K)')
    plt.ylabel('h (W/m²K)')
    plt.title('Heat Transfer Coefficient vs Temperature')
    plt.grid(True)
    
    # Plot 4: When cooling dominates (exponential case)
    cooling_only_power = -cooling_power  # Negative for cooling
    dTdt_cooling_only = cooling_only_power / (m * cp_b)
    plt.subplot(2, 2, 4)
    plt.plot(T_range, dTdt_cooling_only, 'brown', linewidth=2)
    plt.xlabel('Temperature (K)')
    plt.ylabel('dT/dt (K/s)')
    plt.title('Cooling Only (Would be Exponential)')
    plt.grid(True)
    
    plt.tight_layout()
    plt.show()
    
    # Find equilibrium temperature (where heating = cooling)
    equilibrium_idx = np.argmin(np.abs(net_power))
    equilibrium_temp = T_range[equilibrium_idx]
    print(f"Predicted equilibrium temperature: {equilibrium_temp:.2f} K")
    print(f"Heating power: {heating_power[0]:.2f} W")
    print(f"Cooling power at equilibrium: {cooling_power[equilibrium_idx]:.2f} W")

# Run the analysis
analyze_system_behavior()