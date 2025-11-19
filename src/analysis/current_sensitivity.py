"""
Plot steady-state mass flow rate versus electrical current.

This script:
    1. Defines a range of electrical currents.
    2. Computes the resulting heat generation using generated_heat = I² * R_b.
    3. Calls the steady-state mass-flow model to obtain mass_flow.
    4. Plots the steady-state mass-flow rate as a function of current.

Requires:
    - calculate_steady_state_mass_flow from Mass_flowrate.py
    - parameters from config.py
"""
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from src.models.mass_flowrate import calculate_steady_state_mass_flow
from src.config import *

# Simulation parameters
current_values = np.linspace(2.0, 20.0, 19)  # Range of currents (A)
mass_flow_values = []

for current in current_values:
    # Update last value of current_store if required
    current_store.append(current)
    # Electrical heat generation (W)
    generated_heat = current ** 2 * r_b
    # Compute steady-state mass flow rate
    mass_flow_ss, t_c_avg_k, h_ss = calculate_steady_state_mass_flow(generated_heat, mass_flow_initial)
    # Store mass-flow result
    mass_flow_values.append(mass_flow_ss)

mass_flow_values = np.array(mass_flow_values)

# --- 1. Define and create output directories ---
VALIDATION_DIR = os.path.join(os.path.dirname(__file__), '..', '..', 'results', 'validation')
FIGURES_DIR = os.path.join(VALIDATION_DIR, 'figures')
TABLES_DIR = os.path.join(VALIDATION_DIR, 'tables')
os.makedirs(FIGURES_DIR, exist_ok=True)
os.makedirs(TABLES_DIR, exist_ok=True)

# Plotting block
fig = plt.figure(figsize=(8, 5))
plt.plot(current_values, mass_flow_values, "o-", color="blue", markersize=5)
plt.xlabel("Current (A)")
plt.ylabel("Steady-State Mass Flow Rate (kg/s)")
plt.title("Steady-State Mass Flow Rate vs Current")
plt.grid(True, alpha=0.3)
plt.tight_layout()

# --- 2. Save plot and results summary ---
plot_path = os.path.join(FIGURES_DIR, 'current_sensitivity.png')
fig.savefig(plot_path, dpi=300)
print(f"✓ Plot saved to {os.path.relpath(plot_path)}")

df_out = pd.DataFrame({'current_A': current_values, 'mass_flow_kg_s': mass_flow_values})
table_path = os.path.join(TABLES_DIR, 'current_sensitivity.csv')
df_out.to_csv(table_path, index=False)
print(f"✓ Data saved to {os.path.relpath(table_path)}")
plt.show()
