"""
This module simulates the final battery temperature's sensitivity to the
coolant mass flow rate using an ODE solver and a cubic spline interpolation.
"""

import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Related third party imports
from scipy.interpolate import CubicSpline

# Local application/library specific imports
from src.config import *
from src.models.battery_temperature_ode import d_tb_dt, get_tb
from src.models.mass_flowrate import get_steady_state_values

# --- 1. Define and create output directories ---
VALIDATION_DIR = os.path.join(os.path.dirname(__file__), '..', '..', 'results', 'validation')
FIGURES_DIR = os.path.join(VALIDATION_DIR, 'figures')
TABLES_DIR = os.path.join(VALIDATION_DIR, 'tables')
os.makedirs(FIGURES_DIR, exist_ok=True)
os.makedirs(TABLES_DIR, exist_ok=True)

# ---

# Retrieve steady state values.
mass_flow_ss, t_c_avg_nominal_coolant, h_ss_nominal = get_steady_state_values()

# Use 'lower_case_with_underscores' for all variables (PEP 8 naming convention).
mass_flow_nominal = mass_flow_ss

# Define the range for mass flow rate perturbations.
mass_perturb_range = np.linspace(0.9 * mass_flow_nominal, 1.1 * mass_flow_nominal, 10)
results = []
step_size = 0.2

# Define parameters for the nominal simulation.
# Operators are surrounded by a single space (e.g., * and =).
params_nominal = (m_cell, c_b, 7.2, dc_ir * 24, a_s, t_in, mass_flow_nominal)
time_nom, t_nominal_arr = get_tb(d_tb_dt, params_nominal, step_size)
t_final_nominal = t_nominal_arr[-1]

# Plot the nominal temperature evolution to check the ODE solution.
fig_ode = plt.figure()
plt.plot(time_nom, t_nominal_arr, label='Nominal Temperature Evolution')
plt.xlabel("Time (s)")
plt.ylabel("Temperature (K)")
plt.title("Nominal ODE Solution Check")
plt.grid(True, alpha=0.3)

# Store nominal results, which serve as the baseline.
results.append({
    'M_flow': mass_flow_nominal,
    'T_final': t_final_nominal,
    'dT': 0,
    'dM': 0
})

# Simulate for each perturbed mass flow rate.
for mass_flow in mass_perturb_range:
    # Define parameters for the perturbed simulation.
    params_perturbed = (m_cell, c_b, 7.2, dc_ir * 24, a_s, t_in, mass_flow)
    # The underscore (_) is used to explicitly ignore the time array from get_tb, 
    # as we only need the final temperature array.
    _, t_perturbed_arr = get_tb(d_tb_dt, params_perturbed, step_size)
    t_final_perturbed = t_perturbed_arr[-1]

    # Calculate the delta T and delta M for sensitivity analysis.
    delta_t = t_final_perturbed - t_final_nominal
    delta_m = mass_flow - mass_flow_nominal

    # Store perturbed results.
    results.append({
        'M_flow': mass_flow,
        'T_final': t_final_perturbed,
        'dT': delta_t,
        'dM': delta_m
    })

# Convert results to a DataFrame for analysis.
df_results = pd.DataFrame(results)

# Calculate the dimensionless sensitivity.
df_results['S_dimless'] = (df_results['dT'] / t_final_nominal) / (df_results['dM'] / mass_flow_nominal)
print(params_perturbed)

# --- 2. Save results table ---
table_path = os.path.join(TABLES_DIR, 'massflow_sensitivity.csv')
df_results.to_csv(table_path, index=False)
print(f"✓ Data saved to {os.path.relpath(table_path)}")

# Sort results by mass flow rate, which is useful before interpolation.
df_results_sorted = df_results.sort_values(by='M_flow').reset_index(drop=True)

# Prepare data for cubic spline interpolation.
x_data = df_results_sorted['M_flow'].values
y_data = df_results_sorted['T_final'].values

# Create the cubic spline interpolator.
cs = CubicSpline(x_data, y_data)

# Generate smooth data for plotting the spline.
mass_flow_smooth = np.linspace(x_data.min(), x_data.max(), 500)
t_final_spline = cs(mass_flow_smooth)

# ---

# Plotting Results
fig_sensitivity = plt.figure(figsize=(10, 5))

# Scatter plot of the simulated data points.
plt.scatter(
    df_results_sorted['M_flow'] * 1000,
    df_results['T_final'],
    color='blue',
    marker='o',
    label=r'$T_{b,final}$ Data Points'
)

# Plot the cubic spline interpolation.
plt.plot(
    mass_flow_smooth * 1000,
    t_final_spline,
    color='orange',
    linestyle='-',
    linewidth=2
)

# Horizontal line for the nominal final temperature.
plt.axhline(
    y=t_final_nominal,
    color='r',
    linestyle='--',
    label=r'$T_{b,final, \text{nominal}}$'
)

# Set labels and title.
plt.xlabel(r'Mass Flow Rate, $\dot{M}$ (g/s)')
plt.ylabel('Final Battery Temperature, $T_{b,final}$ (K)')
plt.title('Final Temperature Sensitivity to Mass Flow Rate with Cubic Spline')
plt.grid(True, alpha=0.3)
plt.legend()

# --- 3. Save plots ---
plot_path_ode = os.path.join(FIGURES_DIR, 'massflow_sensitivity_ode_check.png')
fig_ode.savefig(plot_path_ode, dpi=300)
print(f"✓ ODE check plot saved to {os.path.relpath(plot_path_ode)}")
plot_path_sensitivity = os.path.join(FIGURES_DIR, 'massflow_sensitivity_analysis.png')
fig_sensitivity.savefig(plot_path_sensitivity, dpi=300)
print(f"✓ Sensitivity plot saved to {os.path.relpath(plot_path_sensitivity)}")
plt.show()