import matplotlib.pyplot as plt
import numpy as np
from Mass_flowrate import calculate_steady_state_mass_flow
from config import *

  # Range of currents to simulate (A)
I_values = np.linspace(2, 20, 19)  # 2A to 20A in 1A steps

# Arrays to store results
m_dot_values = []

for I in I_values:
    # Update the last I_store value (optional, depends on your setup)
    I_store[-1] = I
    
    # Compute heat generation
    Q_gen = I**2 * R_b
    
    # Compute steady-state mass flow rate
    m_dot_ss, T_c_avg_K, h_ss = calculate_steady_state_mass_flow(Q_gen, M_DOT)
    
    # Store
    m_dot_values.append(m_dot_ss)

# Convert to numpy arrays for plotting
m_dot_values = np.array(m_dot_values)

# Plot
plt.figure(figsize=(8,5))
plt.plot(I_values, m_dot_values, 'o-', color='blue', markersize=5)
plt.xlabel("Current (A)")
plt.ylabel("Steady-State Mass Flow Rate (kg/s)")
plt.title("Steady-State Mass Flow Rate vs Current")
plt.grid(True)
plt.show()