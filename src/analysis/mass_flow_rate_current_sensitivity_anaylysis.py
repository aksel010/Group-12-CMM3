"""
Plot steady-state mass flow rate versus electrical current.

This script:
    1. Defines a range of electrical currents.
    2. Computes the resulting heat generation Q_gen = I² * R_b.
    3. Calls the steady-state mass-flow model to obtain ṁ.
    4. Plots the steady-state mass-flow rate as a function of current.

Requires:
    - calculate_steady_state_mass_flow from Mass_flowrate.py
    - parameters from config.py
"""

import numpy as np
import matplotlib.pyplot as plt

from src.models.Mass_flowrate import calculate_steady_state_mass_flow
from config import *


# --------------------------------------------------------------------------- #
# Simulation parameters
# --------------------------------------------------------------------------- #

# Current range (A)
I_values = np.linspace(2.0, 20.0, 19)     # 2 A → 20 A in 1-A increments

# Storage array
m_dot_values = []


# --------------------------------------------------------------------------- #
# Compute steady-state mass-flow rates
# --------------------------------------------------------------------------- #
for I in I_values:
    # Optional: update last value of I_store if used by your model
    I_store[-1] = I

    # Electrical heat generation (W)
    Q_gen = I**2 * R_b

    # Compute steady-state mass flow rate
    m_dot_ss, T_c_avg_K, h_ss = calculate_steady_state_mass_flow(Q_gen, M_DOT)

    # Store the mass-flow result
    m_dot_values.append(m_dot_ss)


# Convert to numpy array
m_dot_values = np.array(m_dot_values)


# --------------------------------------------------------------------------- #
# Plot results
# --------------------------------------------------------------------------- #
plt.figure(figsize=(8, 5))
plt.plot(I_values, m_dot_values, "o-", color="blue", markersize=5)

plt.xlabel("Current (A)")
plt.ylabel("Steady-State Mass Flow Rate (kg/s)")
plt.title("Steady-State Mass Flow Rate vs Current")

plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()
