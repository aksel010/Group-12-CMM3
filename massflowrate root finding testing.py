import numpy as np
from scipy.optimize import fsolve
from pathlib import Path

from heptane_itpl_mdottest import (
    calculate_h, 
    Cp_func, 
    rho_func, 
    mu_func,
 )

#SYSTEM PARAMETERS (IDK WHAT THEY ARE RN)

class SystemParams:
    # --- Thermal & Operational Inputs ---
    I_sq_R = 100.0     # Steady-state generated heat, I^2 * R (Watts) - REPLACE with MAX THERMAL POWER
    T_c_in = 293.15    # Coolant inlet temperature (K) - REPLACE with measured inlet T (~20°C)
    
    # --- Geometry & Fluid Inputs ---
    D_pipe = 0.005     # Channel diameter (m) - REPLACE with inner pipe diameter
    L_pipe = 2.0       # Total equivalent pipe length (m) - REPLACE with TOTAL system equivalent length
    K_minor = 5.0      # Total minor loss coefficient (fittings, valves, etc.) - Sum of all K values
    epsilon = 1.5e-5   # Pipe roughness (m) - REPLACE with pipe material roughness (e.g., copper, steel)
    A_s = 0.001        # Heat transfer surface area (m^2) - REPLACE with As from your ODE derivation

# HYDRAULIC BALANCE FUNCTIONS

def pump_head_curve(mass_flow_rate_kg_s):
    # currently placeholder but can use manufacture spec
    P_max = 50000.0  # Max pressure (Pa)
    K = 1.0e6        # Head-flow coefficient
    return P_max - K * mass_flow_rate_kg_s**2

def system_head_loss_calc(m_dot_kg_s, T_c_avg_K, D, L, K_minor, epsilon):
    # calculates the total pressure loss across the cooling channel assuming steady state ie. values are averaged
    rho = rho_func(T_c_avg_K)
    mu = mu_func(T_c_avg_K)
    
    # volume flow rate
    Q_vol = m_dot_kg_s / rho
    A_pipe = np.pi * D**2 / 4
    V = Q_vol / A_pipe
    
    Re = rho * V * D / mu
    
    # Darcy friction factor
    if Re < 2300: # Laminar flow
        f = 64 / Re
    else: # Turbulent flow ( i think we avoid this region (AIED it))
        f = 0.25 / (np.log10((epsilon / (3.7 * D)) + (5.74 / Re**0.9)))**2
        
    # total head Loss
    P_loss = (f * (L / D) + K_minor) * (rho * V**2 / 2)
    
    return P_loss

# root finding of coupled system
def pressure_balance_couple(mass_flow_rate_kg_s, params):
    m_dot = mass_flow_rate_kg_s
    
    # 1. Thermal Balance (to find T_c_avg)
    # Use Cp at inlet temperature for bulk energy calculation (simplification)
    Cp_c_in = Cp_func(params.T_c_in)
    # calculate outlet temp with energy balance
    T_c_out_K = params.T_c_in + params.I_sq_R / (m_dot * Cp_c_in)
    #average Coolant Temperature
    T_c_avg_K = (params.T_c_in + T_c_out_K) / 2
    
    # calculate the head loss in the system
    P_loss_system = system_head_loss_calc(
        m_dot, T_c_avg_K, params.D_pipe, 
        params.L_pipe, params.K_minor, params.epsilon
    )
       # calculate supplied head
    P_supplied_pump = pump_head_curve(m_dot)
    
    # 4. Return the residual head
    return P_supplied_pump - P_loss_system

def calculate_steady_state_mass_flow(Q_gen, T_c_in, guess_m_dot):
    params = SystemParams()
    params.I_sq_R = Q_gen
    params.T_c_in = T_c_in
    
    m_dot_ss, info, ier, msg = fsolve(
        pressure_balance_couple, 
        guess_m_dot, 
        args=(params,), 
        full_output=True,
        xtol=1e-6 
    )
    
    # fsolve returns an array, extract the value
    m_dot_ss = m_dot_ss[0]

if __name__ == '__main__':
    # these will overwrite the defaults in SystemParams for the solver run
    Q_gen = 100.0      # Watts (I^2*R)
    T_c_in_K = 293.15  # Kelvin (20.0 °C)
    m_dot_guess = 0.01 # Initial guess for the solver (e.g., 10 g/s)

    m_dot_ss_kg_s, T_c_avg_K, h_ss, T_b_ss, status = calculate_steady_state_mass_flow(Q_gen, T_c_in_K, m_dot_guess)

    print("\n" + "="*50)
    print("      STEADY-STATE FLOW SOLVER RESULTS")
    print("="*50)
    
    if status == "Success":
        print("✅ Steady-State Operating Point Found")
        print("-----------------------------------------")
        print(f"Empirical Mass Flow Rate (m_dot_ss): {m_dot_ss_kg_s:.5f} kg/s")
        print(f"Average Coolant Temperature (T_c,avg): {T_c_avg_K:.2f} K ({T_c_avg_K - 273.15:.2f} °C)")
        print(f"Heat Transfer Coefficient (h): {h_ss:.2f} W/(m²·K)")
        print(f"Estimated Body Temperature (T_b,ss): {T_b_ss:.2f} K ({T_b_ss - 273.15:.2f} °C)")
        
        # Calculate the pressure values for comparison
        P_pump_ss = pump_head_curve(m_dot_ss_kg_s)
        params = SystemParams() # Need to instantiate or access defaults for reporting
        P_system_ss = system_head_loss_calc(m_dot_ss_kg_s, T_c_avg_K, 
                                            params.D_pipe, params.L_pipe, 
                                            params.K_minor, params.epsilon)
        print("\nHydraulic Balance Check:")
        print(f"Pump Head Supplied (dP_pump): {P_pump_ss:.2f} Pa")
        print(f"System Head Loss (dP_system): {P_system_ss:.2f} Pa")
        print(f"Residual (dP_pump - dP_system): {P_pump_ss - P_system_ss:.2e} Pa")
        
    else:
        print(f"❌ Calculation failed: {status}")
