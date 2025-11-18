"""
Global physical constants, battery/coolant parameters, and system config shared by simulation modules.
PEP8-compliant module documentation, inline explanations for all fields.
"""
import numpy as np

g = 9.81  # Gravitational acceleration [m/s^2]

# ========== NiMH Battery Properties ========== #
n_cell = 240  # total number of cells
V_cell = 1.2  # [V] per cell
Capacity_cell = 6  # [Ah] per cell
Energy_cell = 7.2  # [Wh] per cell
DC_IR = 2.5e-3  # [Ohm] per cell
m_cell = 0.158  # [kg] test cell mass

# ========== Battery Pack Properties ========== #
m_b = m_cell * n_cell  # [kg] total mass
C_b = 2788  # [J/kgK] see doi:10.1016/S0378-7753(98)00064-0
Capacity_battery = Capacity_cell
q_b = Capacity_cell * 3600  # [As]
V_b = V_cell * n_cell  # [V] total
R_b = DC_IR * n_cell  # [Ohm] total
I_0 = 15  # [A] initial current
I_store = [0]  # for optimum current search
I_Threshold = 1e-6  # convergence threshold

T_b_max = 40 + 273.13  # [K] maximum safe (cell) temperature
T_in = 15 + 273.13  # [K] inlet coolant temp

# ========== System/Channel Properties ========== #
n = 5  # number of coolant branches
D = 17e-3  # [m] main pipe diameter
w_branch = 30e-3  # [m] branch width
h_branch = 2.75e-3  # [m] branch height
L_a = 0.5  # [m] main pipe segment length
L_b = 0.5  # [m] branch length
L_c = 0.2  # [m] branch outlet length
S_a = np.pi * (D / 2) ** 2  # [m^2] main section area
S_b = w_branch * h_branch  # [m^2] open branch area
A_s = L_b * (2 * w_branch + 2 * h_branch)  # [m^2] branch wetted area
d_H = 2 * w_branch * h_branch / (w_branch + h_branch)  # [m] hydr. diameter

# ========== n-Heptane / Fluid / Pump Properties ========== #
M_DOT = 0.0001  # [kg/s] nominal mass flowrate
N_HEPTANE_MOLAR_MASS = 0.100205  # [kg/mol]
DITTUS_BOELTER_EXPONENT = 0.4
C_RE = 4 * M_DOT / (np.pi * D)  # Reynolds const for flow
flowrate_min = 5.0 / 60000  # [m^3/s] min. pump flow

# ========== RK4 Solver Parameters ========== #
H = 30  # RK4 time integration step size [s]
