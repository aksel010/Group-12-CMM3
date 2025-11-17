import numpy as np
import matplotlib.pyplot as plt
from config import * 
from ODE import Tb, dTb_dt 
from heptane_itpl import Cp_func, mu_func, rho_func, lambda_func # lambda_func for thermal conductivity
from Mass_flowrate import get_steady_state_values

# --- Analysis Settings ---
# Fetch fixed parameters from the steady-state solver
M_DOT_SS, T_C_AVG_K, H_SS = get_steady_state_values() 
I_FIXED = I_0 # Fixed current from config.py
W_C_FIXED = w_branch  # Fixed channel width from config.py

# Define the sweep range for Channel Height (h_branch) in meters
# Sweep from 1 mm to 10 mm (Adjust this range based on your physical limits)
H_BRANCH_MIN_M = 1e-3
H_BRANCH_MAX_M = 10e-3
N_POINTS = 20

h_branch_sweep_m = np.linspace(H_BRANCH_MIN_M, H_BRANCH_MAX_M, N_POINTS)

# --- Core Heat Transfer Function (Re-implemented with Geometry dependence) ---
def calculate_h_branch(h_branch_m, T_c_avg_K, m_dot_c):
    """
    Calculates h using laminar flow correlation, incorporating changing channel geometry.
    Uses the fixed steady-state mass flow rate (m_dot_c).
    """
    # Geometric parameters based on current h_branch
    S_b = W_C_FIXED * h_branch_m  # Cross-sectional area
    P_b = 2 * (W_C_FIXED + h_branch_m) # Wetted perimeter
    d_H = 4 * S_b / P_b # Hydraulic diameter
    
    # Fluid properties at average temperature
    mu = mu_func(T_c_avg_K)
    rho = rho_func(T_c_avg_K)
    k_therm = lambda_func(T_c_avg_K)
    
    # Calculate average velocity in the channel branch
    # Assuming m_dot_c is the flow rate for the entire module (divided over n channels)
    m_dot_branch = m_dot_c / n 
    V_branch = m_dot_branch / (rho * S_b) 
    
    # Reynolds Number
    Re = rho * V_branch * d_H / mu
    
    # --- Nusselt Number Correlation ---
    # For laminar, fully developed flow in a rectangular channel (aspect ratio near 1.0)
    # Nu is constant. For a typical aspect ratio, Nu ~ 4.86. 
    # If Re is high enough (e.g., > 2000), a turbulent correlation would be needed, 
    # but we assume laminar based on your cooling analysis structure.
    
    if Re < 2000:
        # Nu for laminar flow in rectangular duct (constant T wall)
        Nu = 4.86  
    else:
        # Use a turbulent approximation (e.g., Dittus-Boelter), though unlikely here
        Nu = 0.023 * Re**0.8 * (Cp_func(T_c_avg_K) * mu / k_therm)**(0.4) 

    h_ss = Nu * k_therm / d_H
    return h_ss, S_b, d_H

def run_sensitivity_analysis():
    # Results storage
    h_branch_final = []
    T_b_max_results = []
    
    print(f"--- Running Geometry Sensitivity Analysis (I = {I_FIXED} A, m_dot = {M_DOT_SS:.4f} kg/s) ---")
    
    # Loop through the channel heights
    for h_branch_m in h_branch_sweep_m:
        
        # 1. Calculate new h and geometric parameters for this height
        h_ss, S_b_new, d_H_new = calculate_h_branch(h_branch_m, T_C_AVG_K, M_DOT_SS)
        
        # Total wetted area for one module (n channels, length L_b)
        P_b_new = 2 * (W_C_FIXED + h_branch_m)
        A_s_new = P_b_new * L_b * n 
        
        # 2. Configure ODE parameters (params tuple structure from ODE.py)
        # Assuming the ODE is for the module (24 cells)
        params_mod = (
            m_cell * 24, # m [kg] - Mass of one module (24 cells)
            C_b,         # cp_b [J/(kg·K)]
            I_FIXED,     # I [A]
            DC_IR * 24,  # R_module [Ω]
            A_s_new,     # A_s [m²] - *NEW* Total wetted area for one module
            T_in,        # T_c_in [K] - Using fixed T_in
            M_DOT_SS     # m_dot_c [kg/s] - *FIXED* total flow rate for the module
        )
        
        # The dTb_dt function in ODE.py will use the h calculated here, 
        # but the ODE function needs to be slightly modified to use the new A_s and h
        # Let's run the ODE, assuming dTb_dt is:
        # dTb_dt (Tb, t, params) -> ... cooling_term = h * A_s * (Tb - T_c_avg) ...

        # We need a custom dTb_dt function to inject the calculated h_ss here, 
        # as the original dTb_dt recalculates h internally, which we want to override 
        # for this sensitivity study.
        
        # Since we cannot modify the original ODE.py, we must fix the h and A_s 
        # in the parameter list and ensure the ODE solver uses it correctly.
        # This requires a slightly different ODE parameter definition than your original:
        
        # Rerunning the ODE requires the full time-dependent dTb_dt
        # The easiest approach is to run the ODE using the *fixed* h_ss calculated above
        
        # To bypass the h recalculation in ODE.py, we run the ODE with a constant h:
        def dTb_dt_fixed_h(Tb, t, params):
            m, cp_b, I, R, A_s_geom, T_c_in, m_dot_c, h_fixed = params
            
            T_c_avg = (Tb + T_c_in) / 2
            cp_c = Cp_func(T_c_avg)
            
            heating = I**2 * R
            
            cooling_denom = 1 + (h_fixed * A_s_geom) / (2 * m_dot_c * cp_c)
            cooling_term = (h_fixed * A_s_geom * (Tb - T_c_in)) / cooling_denom
            
            return (heating - cooling_term) / (m * cp_b)

        # New params tuple to include the fixed h
        params_mod_fixed_h = (
            m_cell * 24, C_b, I_FIXED, DC_IR * 24, A_s_new, T_in, M_DOT_SS, h_ss
        )
        
        # 3. Solve the ODE (Custom RK4)
        t_i, T_i = Tb(dTb_dt_fixed_h, params_mod_fixed_h, stepsize=H)
        
        # 4. Store results
        T_b_max = T_i[-1]
        h_branch_final.append(h_branch_m * 1000) # Store in mm
        T_b_max_results.append(T_b_max)

        if (np.where(h_branch_sweep_m == h_branch_m)[0][0] + 1) % (N_POINTS // 5) == 0:
            print(f"Height: {h_branch_m*1000:.1f} mm -> h_ss: {h_ss:.0f} W/(m²K) -> T_b_max: {T_b_max-273.15:.2f} °C")


    # --- Plotting ---
    T_b_max_C = np.array(T_b_max_results) - 273.15
    
    plt.figure(figsize=(10, 6))
    plt.plot(h_branch_final, T_b_max_C, 'rD-', 
             linewidth=2, markersize=6, label=f'Max $T_b$ at $I={I_FIXED}$ A')
    
    # Add constraint
    T_b_max_C_constraint = T_b_max - 273.15 # Use the max constraint from config.py
    plt.axhline(T_b_max_C_constraint, color='red', linestyle='--', label=f'Max Constraint (${T_b_max_C_constraint:.0f}^\circ$C)')
    
    plt.xlabel('Channel Height, $h_{\\text{branch}}$ (mm)', fontsize=14)
    plt.ylabel('Maximum Battery Temperature, $T_{b, \mathrm{max}}$ ($\mathrm{^\circ C}$)', fontsize=14)
    plt.title(f'Sensitivity of Max Battery Temperature to Channel Height\n(Fixed Current $I={I_FIXED}$ A, Fixed Flow Rate)', fontsize=16)
    
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend(fontsize=12)
    plt.tight_layout()
    plt.show()
    
    print("\n--- Sensitivity Results Summary ---")
    print(f"Lowest T_b_max ({T_b_max_C.min():.2f} °C) occurred at height: {h_branch_final[np.argmin(T_b_max_C)]:.1f} mm")
    print(f"Highest T_b_max ({T_b_max_C.max():.2f} °C) occurred at height: {h_branch_final[np.argmax(T_b_max_C)]:.1f} mm")

if __name__ == "__main__":
    run_sensitivity_analysis()

#Since the maximum temperature depends on how effectively heat is removed, this analysis models two counteracting effects of increasing h branch
#Larger $h_{\text{branch}}$ leads to a larger wetted area ($A_s$) (if channel width $w_c$ is fixed), which improves cooling.
#Negative Effect: Larger $h_{\text{branch}}$ leads to a lower flow velocity ($V_{\text{branch}}$) 
# (since $\dot{m}_{\text{ss}}$ is fixed), which reduces the heat transfer coefficient ($\mathbf{h}$) and worsens cooling.
