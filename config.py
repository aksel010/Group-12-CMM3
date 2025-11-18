# config.py
# Constants that might be shared across multiple files
import numpy as np

g = 9.81

# NIMH battery properties
"""Battery pack is made up of 48 L5 modules, such that one channel passes through 24 cells."""
n_cell = 240
v_cell = 1.2  # V per cell,
capacity_cell = 6  # Ah per cell
energy_cell = 7.2  # Wh per cell
dc_ir = 2.5e-3  # Ohm per cell
m_cell = 0.158 # test cell mass

# Battery pack properties
m_b = m_cell * n_cell  # kg total battery pack
C_b = 2788  #J/kgK  https://doi.org/10.1016/S0378-7753(98)00064-0
capacity_battery = capacity_cell
q_b =  capacity_cell * 3600  #As 
v_b = v_cell * n_cell
R_b = dc_ir * n_cell  # Ohm total battery pack
I_0 = 15 # A initial current
I_store = [0]
I_Threshold = 1e-6


T_b_max = 40 + 273.13  # K 
T_in = 15 + 273.13  # K

# Channel properties
n = 5  # number of branches
D = 17e-03  # main pipe diameter (m)
w_branch = 30e-03    # branch width (m)
h_branch = 2.75e-03 # branch height (m)
L_a = 0.5  # main pipe length per segment (m)
L_b = 0.5  # branch length (m)
L_c = 0.2   # branch outlet length (m)
            
S_a = np.pi * (D/2)**2
S_b = w_branch * h_branch
A_s = L_b *(2*w_branch + 2*h_branch)  # wetted area of branch
d_H = 2 * w_branch * h_branch / (w_branch + h_branch)

#N_heptane properties
m_dot = 0.0001    # Initial mass flow rate [kg/s]
N_HEPTANE_MOLAR_MASS = 0.100205
DITTUS_BOELTER_EXPONENT = 0.4
C_RE = 4 * m_dot / (np.pi * D) # derived constant Reynlds no.

#Pump Properties
flowrate_min = 5.0 /60000 # m^3/s

#RK4 stepsize
H = 30
