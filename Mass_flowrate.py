import numpy as np
from scipy.optimize import fsolve
from matplotlib import pyplot as plt
from cooling_analysis import get_head_loss
from config import *
from heptane_itpl import calculate_h, Cp_func, rho_func, mu_func
from root_finders import newton

Q_heat = 0.0  # Will be set by calculate_steady_state_mass_flow

# HYDRAULIC BALANCE FUNCTIONS

m_dot_limit =  2000 * mu_func(T_b_max) * S_b /D
Q_limit = m_dot_limit * Cp_func(T_b_max) * (T_b_max -T_in)

def pump_head_curve(m_dot,T_c_avg_K):
    # currently placeholder but can use manufacture spec
    H_max = 80.0  # Max head (m)
    P_max = H_max * rho_func(T_c_avg_K)*9.81 # Pa (at max head)
    V_dot_max = 9.0 / 60000  # m³/s (9 l/min)
    K = H_max / (V_dot_max** 2)
    return P_max - K * m_dot**2

def system_head_loss_calc(m_dot,T_c_avg_K):
    V_dot = m_dot / rho_func(T_c_avg_K)  # m³/s
    return get_head_loss(V_dot) * rho_func(T_c_avg_K) * g  # Pa

# root finding of coupled system
def pressure_balance_couple(m_dot):
    """
    Function whose root represents the hydraulic balance (P_pump - P_loss = 0).
    """
    # 1. Thermal Balance (to find T_c_avg)
    Cp_c_in = Cp_func(T_in)
    # calculate outlet temp with energy balance
    T_c_out_K = T_in + Q_heat / (m_dot * Cp_c_in)
    # average Coolant Temperature
    T_c_avg_K = (T_in + T_c_out_K) / 2
    
    # 2. Hydraulic Balance
    
    # calculate supplied head
    P_supplied_pump = pump_head_curve(m_dot, T_c_avg_K )

    # calculate head loss in system
    P_loss_system = system_head_loss_calc(m_dot, T_c_avg_K)
    
    # 4. Return the residual head
    return P_supplied_pump - P_loss_system

def df(m_dot):
    """
    Analytical derivative of pressure_balance_couple function.
    Uses finite differences for complex dependencies.
    """
    H = 1e-8  # step size for numerical derivative
    f_x = pressure_balance_couple(m_dot)
    f_x_plus_h = pressure_balance_couple(m_dot + H)
    
    return (f_x_plus_h - f_x) / H

def calculate_steady_state_mass_flow(Q_gen, guess_m_dot):
    """
    Finds the steady-state mass flow rate (m_dot_ss) and calculates
    the resulting thermal properties (h, Tc_avg).
    """
    global Q_heat
    Q_heat = Q_gen

    # Check for cooling limits 
    if Q_gen >= Q_limit:
        print( "Warning: the Current is oustside of safefty limits for these dimesions")
    
 # Newton-Raphson parameters
    x0 = guess_m_dot
    epsilon = 1e-6
    max_iter = 100
    
    # 1. Solve for m_dot_ss using Newton-Raphson
    m_dot = newton(pressure_balance_couple, df, x0, epsilon, max_iter, args=()) / (2*n)
    
    if m_dot is None:
        print("Warning: Newton-Raphson failed to converge")
        m_dot = 1e-6  # Return small positive value as fallback

    # 2. Calculate resulting thermal parameters at steady state
    Cp_c_in = Cp_func(T_in)
    T_c_out_K = T_in + Q_heat / (m_dot * Cp_c_in)
    T_c_avg_K = (T_in + T_c_out_K) / 2

    #Check that the m_dot value makes sense
    while m_dot <= m_dot_limit and T_c_out_K > T_b_max:
        print("adjusting cooling to match additional heating")
        m_dot += 0.00001

    
    # Calculate h at steady state
    h_ss = calculate_h(T_c_avg_K)
    
    return m_dot, T_c_avg_K, h_ss


def run():
    if I_store[-1] == 0:
        I_store.append(50)
    Q_gen = I_store[-1]**2 * R_b      
    m_dot_ss, T_c_avg_K, h_ss = calculate_steady_state_mass_flow(
        Q_gen, M_DOT
    )

    print("\n" + "="*50)
    print("      SINGLE-CELL STEADY-STATE FLOW & THERMAL SOLVER")
    print("="*50)
    print(f"Cell Heat Generation (Q_gen): {Q_gen} W")
    print(f"Coolant Inlet Temperature (T_c,in): {T_in:.2f} K")
    print("-----------------------------------------")
    
    if m_dot_ss > 0:
        print(" Steady-State Operating Point Found")
        print("-----------------------------------------")
        print(f"Mass Flow Rate (m_dot_ss): {m_dot_ss:.5f} kg/s")
        print(f"Average Coolant Temperature (T_c,avg): {T_c_avg_K:.2f} K ({T_c_avg_K - 273.15:.2f} °C)")
        print(f"Heat Transfer Coefficient (h): {h_ss:.2f} W/(m²·K)")
        
        # Calculate the pressure values for comparison
        P_pump_ss = pump_head_curve(m_dot_ss, T_c_avg_K)
        P_system_ss = system_head_loss_calc(m_dot_ss, T_c_avg_K)
        print("\nHydraulic Balance Check:")
        print(f"Supplied Pressure Drop (dP_pump): {P_pump_ss:.2f} Pa")
        print(f"Channel Head Loss (dP_system): {P_system_ss:.2f} Pa")
        print(f"Residual (dP_pump - dP_system): {P_pump_ss - P_system_ss:.2e} Pa")
        
    else:
        print(" Calculation failed. Solver did not converge or found non-physical flow.")

    m = np.linspace(M_DOT, 0.02 , 1000)
    plt.plot(m, [pressure_balance_couple(mi) for mi in m])
    plt.axhline(0, color='k', linestyle='--')
    plt.xlabel('Mass Flow Rate (kg/s)')     
    plt.ylabel('Pressure Balance Residual (Pa)')
    plt.title('Pressure Balance Residual vs Mass Flow Rate')
    plt.show()

def get_steady_state_values():
    """
    Returns the steady-state mass flow rate and related values.
    Call this instead of importing m_dot_ss directly to avoid circular imports.
    """
    if I_store[-1] == 0:
        I_store.append(I_0)
    Q_gen = I_store[-1]**2 * R_b
    m_dot_ss, T_c_avg_K, h_ss = calculate_steady_state_mass_flow(
        Q_gen, M_DOT
    )
    return m_dot_ss, T_c_avg_K, h_ss

if __name__ == "__main__":
    run()