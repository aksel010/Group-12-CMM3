import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
import math

from config import *
from heptane_itpl import lambda_func, Cp_func, mu_func 
from ODE import dTb_dt 

def calculate_h(T: float | np.ndarray, m_dot: float) -> float | np.ndarray:
    
    C_RE_for_h = 4 * m_dot / (np.pi * D) 
    lam = lambda_func(T)
    Cp = Cp_func(T)
    mu = mu_func(T)
    
    Re = C_RE_for_h / mu
    Pr = (Cp * mu) / lam
    Nu = 0.023 * (Re**0.8) * (Pr**DITTUS_BOELTER_EXPONENT)
    
    h = (Nu * lam) / D
    
    return h


def dTb_dt_sensitivity (Tb, t, params):
    """
    Differential equation for bulk temperature Tb, adapted for sensitivity analysis.
    The last parameter is the total mass flow rate (m_dot_c).
    """
    m, cp_b, I, R, A_s, T_c_in, m_dot_c = params

    T_c_avg = (Tb + T_c_in) / 2
    h = calculate_h(T_c_avg, m_dot_c)
    cp_c = Cp_func(T_c_avg)
    
    heating = I**2 * R
    cooling_denom = 1 + (h * A_s) / (2 * (m_dot_c/4) * cp_c) 
    cooling = (h * A_s * (Tb - T_c_in)) / cooling_denom
    dTb_dt = (heating - cooling) / (m * cp_b)
    
    return dTb_dt


def solve_ode_for_mdot(m_dot_test: float) -> float:
    """
    Runs the ODE solver for a given total mass flow rate and returns the final temperature.
    """
    params_i = (
        m_b,      # m [kg]
        C_b,      # cp_b [J/(kg·K)]
        I_0,      # I [A] (Fixed current from config.py)
        DC_IR * 240, # R [Ω] (Using 240 cells from config.py)
        A_s,      # A_s [m²]
        T_in,     # T_c_in [K]
        m_dot_test # m_dot_c [kg/s] (The test variable)
    )
    
    t0 = 0
    T0 = T_in
    t_final = q_b / I_0

    # Use scipy solver for robustness
    sol = solve_ivp(
        fun=lambda t, T, *_: dTb_dt_sensitivity(T, t, params_i),  
        t_span=[t0, t_final],
        y0=[T0],
        method='LSODA', 
        rtol=1e-6,        
        atol=1e-8         
    )
    
    return sol.y[0][-1]

if __name__ == "__main__":
    def run_sensitivity_analysis():
        """Performs the sweep and plots the results."""
        
        rho_avg = 680 # density of n-heptane at room temperature
        m_dot_test_kg_s = np.linspace(0.005, 0.1, 30) 
        Q_L_min = m_dot_test_kg_s / rho_avg * 60000 
        
        T_max_K = []
        
        print(f"--- Running Sensitivity Analysis (I = {I_0} A) ---")

        for m_dot in m_dot_test_kg_s:
            try:
                T_final = solve_ode_for_mdot(m_dot)
                T_max_K.append(T_final)
                print(f"Flow: {m_dot:.5f} kg/s -> T_max: {T_final:.2f} K")
            except Exception as e:
                T_max_K.append(np.nan)
                print(f"Flow: {m_dot:.5f} kg/s -> ERROR: {e}")
                
        T_max_C = np.array(T_max_K) - 273.15 # Convert to Celsius
        T_max_constraint = T_b_max - 273.15 # 40 C

        # --- Plotting ---
        plt.figure(figsize=(10, 6))
        
        plt.plot(Q_L_min, T_max_C, 
                label=f'Max Battery Temp. at $I={I_0}$ A', 
                color='red', 
                linewidth=3)
                
        # Add constraint line (40 C)
        plt.axhline(y=T_max_constraint, 
                    color='blue', 
                    linestyle='--', 
                    label=f'Thermal Constraint ({T_max_constraint:.0f} °C)')
        
        plt.xlabel('Total Mass Flow Rate, $\\dot{m}$ (L/min)', fontsize=14)
        plt.ylabel('Maximum Battery Temperature, $T_{b, \\text{max}}$ ($\\text{°C}$)', fontsize=14)
        plt.title('Sensitivity of Maximum Battery Temperature to Coolant Flow Rate', fontsize=16)
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.legend(fontsize=12)
        plt.ylim(T_max_C.min() - 2, T_max_C.max() + 2)
        plt.tight_layout()
        plt.show()
        
        print("\n--- Analysis Complete ---")
        print("Plot saved as T_max_vs_MassFlowRate_Sensitivity.png")
        
    if __name__ == "__main__":
        run_sensitivity_analysis()
