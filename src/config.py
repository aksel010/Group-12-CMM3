"""
Global physical constants, battery/coolant parameters, and system config shared by simulation modules.
PEP8-compliant module documentation, inline explanations for all fields.
"""
import numpy as np

g = 9.81  # Gravitational acceleration [m/s^2]

# ========== NiMH Battery Properties ========== #
n_cell = 240  # total number of cells
v_cell = 1.2  # [V] per cell
capacity_cell = 6  # [Ah] per cell
energy_cell = 7.2  # [Wh] per cell
dc_ir = 2.5e-3  # [Ohm] per cell
m_cell = 0.00158  # [kg] test cell mass

# ========== Battery Pack Properties ========== #
m_b = m_cell * n_cell  # [kg] total mass
c_b = 2788  # [J/kgK] see doi:10.1016/S0378-7753(98)00064-0
capacity_battery = capacity_cell
q_b = capacity_cell * 3600  # [As]
v_b = v_cell * n_cell  # [V] total
r_b = dc_ir * n_cell  # [Ohm] total
current_0 = 17  # [A] initial current
current_store = [0]  # for optimum current search
current_threshold = 1e-6  # convergence threshold
current_error=[]

t_b_max = 40 + 273.13  # [K] maximum safe (cell) temperature
t_in = 15 + 273.13  # [K] inlet coolant temp

# ========== System/Channel Properties ========== #
n = 5  # number of coolant branches
d = 17e-3  # [m] main pipe diameter
w_branch = 30e-3  # [m] branch width
h_branch = 2.75e-3  # [m] branch height
l_a = 0.5  # [m] main pipe segment length
l_b = 0.5  # [m] branch length
l_c = 0.2  # [m] branch outlet length
s_a = np.pi * (d / 2) ** 2  # [m^2] main section area
s_b = w_branch * h_branch  # [m^2] open branch area
a_s = l_b * (2 * w_branch + 2 * h_branch)  # [m^2] branch wetted area
d_h = 2 * w_branch * h_branch / (w_branch + h_branch)  # [m] hydr. diameter

# ========== n-Heptane / Fluid / Pump Properties ========== #
mass_flow_initial = 0.0001  # [kg/s] nominal mass flowrate
N_HEPTANE_MOLAR_MASS = 0.100205  # [kg/mol]
DITTUS_BOELTER_EXPONENT = 0.4
C_RE = 4 * mass_flow_initial / (np.pi * d)  # Reynolds const for flow
FLOWRATE_MIN = 5.0 / 60000  # [m^3/s] min. pump flow

# ========== RK4 Solver Parameters ========== #
H = 30  # RK4 time integration step size [s]
