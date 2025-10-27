import math
import numpy as np
import matplotlib.pyplot as plt

# Simplified parameters - fixed values
h_fixed = 400  # W/m²K - fixed heat transfer coefficient
cp_fixed = 1000  # J/(kg·K) - fixed heat capacity (typical for batteries)

# Parameters
I_0 = 50
R_0 = 0.1  # More realistic resistance for EV battery

params_simple = (
    1.0,      # m [kg]
    cp_fixed, # cp_b [J/(kg·K)] - fixed value
    I_0,      # I [A]
    R_0,      # R [Ω]
    0.1,      # A_s [m²] - more realistic surface area
    298.15,   # T_c_in [K] - 25°C
    0.001      # m_dot_c [kg/s]
)

def dTb_dt_simple(Tb, t, params):
    """
    Simplified differential equation with fixed h and cp
    """
    # Unpack parameters
    m, cp_b, I, R, A_s, T_c_in, m_dot_c = params
    
    # Fixed heat transfer coefficient
    h = h_fixed
    
    # Electrical heating term
    electrical_heating = I**2 * R
    
    # Simple cooling term (Newton's law of cooling)
    cooling = h * A_s * (Tb - T_c_in)
    
    # Rate of change of temperature
    dTb_dt_val = (electrical_heating - cooling) / (m * cp_b)
    
    return dTb_dt_val

def run_simple_cooling_model():
    """
    Run the simplified cooling model
    """
    # Initial conditions
    t0 = 0
    T0 = 298.15  # Start at 25°C
    
    # Total solution interval
    t_final = 1000  # seconds
    
    # Step size
    H = 1.0  # Larger step size since system is simpler
    
    # Number of steps
    n_step = math.ceil(t_final / H)
    
    # Initialize arrays
    T_rk = np.zeros(n_step + 1)
    t_rk = np.zeros(n_step + 1)
    
    # Set initial conditions
    T_rk[0] = T0
    t_rk[0] = t0
    
    print("Running simplified cooling model")
    print(f"Fixed parameters: h = {h_fixed} W/m²K, cp = {cp_fixed} J/(kg·K)")
    print(f"Initial temperature: {T0} K")
    print(f"Final time: {t_final} s")
    print("-" * 50)
    
    # Runge-Kutta integration
    for i in range(n_step):
        t_rk[i + 1] = t_rk[i] + H
        
        # Compute the four slopes for RK4
        k1 = dTb_dt_simple(T_rk[i], t_rk[i], params_simple)
        k2 = dTb_dt_simple(T_rk[i] + k1 * H / 2, t_rk[i] + H / 2, params_simple)
        k3 = dTb_dt_simple(T_rk[i] + k2 * H / 2, t_rk[i] + H / 2, params_simple)
        k4 = dTb_dt_simple(T_rk[i] + k3 * H, t_rk[i] + H, params_simple)
        
        # Weighted average slope
        slope = (k1 + 2 * k2 + 2 * k3 + k4) / 6
        
        # Update temperature
        T_rk[i + 1] = T_rk[i] + H * slope
    
    return t_rk, T_rk

def plot_temperature(t, T):
    """
    Simple plot of temperature vs time
    """
    plt.figure(figsize=(10, 6))
    
    # Convert to Celsius for more intuitive reading
    T_C = T - 273.15
    
    plt.plot(t, T_C, 'b-', linewidth=2.5, label='Battery Temperature')
    plt.xlabel('Time (s)', fontsize=12)
    plt.ylabel('Temperature (°C)', fontsize=12)
    plt.title('Battery Temperature vs Time\n(h = 400 W/m²K, Fixed Properties)', fontsize=14)
    plt.grid(True, alpha=0.3)
    
    # Add some reference lines
    plt.axhline(y=45, color='orange', linestyle='--', alpha=0.7, label='Optimal Max (45°C)')
    plt.axhline(y=60, color='red', linestyle='--', alpha=0.7, label='Safety Limit (60°C)')
    
    plt.legend(fontsize=10)
    plt.tight_layout()
    plt.show()
    
    # Print final results
    print("\n" + "="*50)
    print("FINAL RESULTS")
    print("="*50)
    print(f"Initial temperature: {T[0]-273.15:.1f} °C")
    print(f"Final temperature: {T[-1]-273.15:.1f} °C")
    print(f"Temperature change: {T[-1]-T[0]:.1f} K")
    print(f"Maximum temperature: {np.max(T)-273.15:.1f} °C")
    
    # Calculate equilibrium information
    m, cp_b, I, R, A_s, T_c_in, m_dot_c = params_simple
    heating = I**2 * R
    cooling_at_final = h_fixed * A_s * (T[-1] - T_c_in)
    
    print(f"\nPower Balance:")
    print(f"Heating power: {heating:.1f} W")
    print(f"Cooling power at final state: {cooling_at_final:.1f} W")
    print(f"Net power: {heating - cooling_at_final:.1f} W")

# Run the model
if __name__ == "__main__":
    t, T = run_simple_cooling_model()
    plot_temperature(t, T)