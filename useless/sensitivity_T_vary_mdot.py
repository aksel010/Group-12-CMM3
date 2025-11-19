import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from math import ceil

# ===================================================================
# 1. CONSTANTS (Extracted from config.py)
# ===================================================================
m_cell = 0.0158       # kg (Cell mass)
c_b = 2788            # J/(kg·K) (Battery specific heat capacity)
q_b = 21600           # As (Total charge capacity: 6 Ah * 3600 s/h)
a_s = 0.0655          # m² (Branch wetted area)
t_in = 15 + 273.13    # K (Inlet coolant temp: 288.13 K)
current_0 = 15.0      # A (Nominal current used in ODE.py's params_initial)
R_FIXED = 0.06        # Ω (Fixed resistance used in ODE.py params_initial)
mass_flow_initial = 0.0001 # kg/s (Nominal mass flow rate)

# ===================================================================
# 2. MOCKED/SIMPLIFIED COOLANT PROPERTIES (Replacing heptane_itpl.py)
#    These are assumed constants for illustrative analysis, as the
#    original interpolation relies on an external data file.
# ===================================================================
CP_C_MOCK = 2000.0 # J/(kg K) (Coolant specific heat)
H_MOCK = 500.0     # W/(m² K) (Convective heat transfer coefficient)

def cp_func(T):
    # Mock for coolant specific heat
    return CP_C_MOCK

def calculate_h(T):
    # Mock for convective heat transfer coefficient
    return H_MOCK

# ===================================================================
# 3. ODE DERIVATIVE FUNCTION (from ODE.py)
# ===================================================================
def d_tb_dt(tb, t, params):
    """
    Calculate the rate of change of bulk temperature Tb.
    Note: The original ODE structure is preserved, where the mass flow rate
    'm' is used in the denominator (m * cp_b), which is unusual for a
    battery bulk temperature ODE.
    """
    # Unpack parameters: mb, cp_b, current, r, a_s, t_c_in, m (mass_flow)
    mb, cp_b, current, r, a_s, t_c_in, m = params

    # Calculate h and cp_c based on average temperature
    t_c_avg = (tb + t_c_in) / 2
    h = calculate_h(t_c_avg)
    cp_c = cp_func(t_c_avg)

    # Electrical heating term (using the fixed resistance R_FIXED)
    heating = current**2 * r

    # Cooling term
    cooling_denom = 1 + (h * a_s) / (2 * m * cp_c)
    cooling = (h * a_s * (tb - t_c_in)) / cooling_denom

    # Rate of change of temperature: (Heat - Cooling) / (mass_flow * battery_Cp)
    d_tb_dt = (heating - cooling) / (m * cp_b)

    return d_tb_dt

# ===================================================================
# 4. RK4 INTEGRATOR (from ODE.py)
# ===================================================================
def get_tb(d_tb_dt, params, stepsize):
    """Solve the ODE using RK4, returning the time and temperature arrays."""
    initial_time = 0.0
    initial_temp = t_in
    final_time = q_b / params[2] # Total charge / current
    num_steps = ceil(final_time / stepsize)

    temp_battery = np.zeros(num_steps + 1)
    time_points = np.zeros(num_steps + 1)
    temp_battery[0] = initial_temp
    time_points[0] = initial_time
    time_points[1:] = np.arange(1, num_steps + 1) * stepsize

    # RK4 integration loop
    for i in range(num_steps):
        t_current = time_points[i]
        temp_current = temp_battery[i]

        k1 = d_tb_dt(temp_current, t_current, params)
        k2 = d_tb_dt(temp_current + 0.5 * stepsize * k1,
                     t_current + 0.5 * stepsize, params)
        k3 = d_tb_dt(temp_current + 0.5 * stepsize * k2,
                     t_current + 0.5 * stepsize, params)
        k4 = d_tb_dt(temp_current + stepsize * k3,
                     t_current + stepsize, params)

        slope = (k1 + 2 * k2 + 2 * k3 + k4) / 6
        temp_battery[i + 1] = temp_current + stepsize * slope

    return time_points, temp_battery

# ===================================================================
# 5. SENSITIVITY ANALYSIS DRIVER
# ===================================================================
# Set nominal and perturbation range for mass flow rate
M_flow_nominal = mass_flow_initial
# 10 points between 90% and 110% of nominal flow
M_perturb_range = np.linspace(0.9 * M_flow_nominal, 1.1 * M_flow_nominal, 10)

results = []
stepsize = 0.2 # Integration step size

# 1. Nominal Calculation
params_nominal = (m_cell, c_b, current_0, R_FIXED, a_s, t_in, M_flow_nominal)
_, T_nominal_arr = get_tb(d_tb_dt, params_nominal, stepsize)
T_final_nominal = T_nominal_arr[-1]

results.append({
    'M_flow': M_flow_nominal,
    'T_final': T_final_nominal,
    'dT': 0,
    'dM': 0
})

# 2. Perturbation Calculations
for M_flow in M_perturb_range:
    if M_flow == M_flow_nominal:
        continue

    params_perturbed = (m_cell, c_b, current_0, R_FIXED, a_s, t_in, M_flow)
    _, T_perturbed_arr = get_tb(d_tb_dt, params_perturbed, stepsize)
    T_final_perturbed = T_perturbed_arr[-1]

    # Calculate absolute differences
    delta_T = T_final_perturbed - T_final_nominal
    delta_M = M_flow - M_flow_nominal

    results.append({
        'M_flow': M_flow,
        'T_final': T_final_perturbed,
        'dT': delta_T,
        'dM': delta_M
    })

df_results = pd.DataFrame(results)

# Calculate Dimensionless Sensitivity Coefficient: S = (dT/T_nom) / (dM/M_nom)
df_results['S_dimless'] = (df_results['dT'] / T_final_nominal) / (df_results['dM'] / M_flow_nominal)

# Filter out the nominal point for plotting sensitivity
df_sensitivity = df_results[df_results['dM'] != 0].copy()

# ===================================================================
# 6. PLOTTING AND EXPORTING
# ===================================================================

# Plot 1: Final Temperature vs Mass Flow Rate
plt.figure(figsize=(10, 5))
plt.plot(df_results['M_flow'] * 1000, df_results['T_final'], 'bo-', label='$T_{b,final}$ vs $\dot{M}$')
plt.axhline(y=T_final_nominal, color='r', linestyle='--', label='$T_{b,final, \text{nominal}}$')
plt.xlabel('Mass Flow Rate, $\dot{M}$ (g/s)')
plt.ylabel('Final Battery Temperature, $T_{b,final}$ (K)')
plt.title('Final Temperature Sensitivity to Mass Flow Rate')
plt.grid(True, alpha=0.3)
plt.legend()
plt.show()

# Plot 2: Dimensionless Sensitivity vs Mass Flow Rate
plt.figure(figsize=(10, 5))
plt.plot(df_sensitivity['M_flow'] * 1000, df_sensitivity['S_dimless'], 'g^-', label='Dimensionless Sensitivity ($S$)')
plt.axhline(y=0, color='k', linestyle='-')
plt.xlabel('Mass Flow Rate, $\dot{M}$ (g/s)')
plt.ylabel('Dimensionless Sensitivity Coefficient ($S_{\dot{M}}^{T_{b,final}}$)')
plt.title('Dimensionless Sensitivity of $T_{b,final}$ to $\dot{M}$')
plt.grid(True, alpha=0.3)
plt.legend()
plt.show()

