from heptane_itpl import M_DOT, D, Cp_func,calculate_h
from config import C_b, T_in, m_b
import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

I_0 = 50
R_0 = 240/I_0

params_initial = (
    m_b,      # m [kg]
    C_b,      # cp_b [J/(kg·K)]
    I_0,       # I [A]
    R_0,        # R [Ω]
    np.pi*D**2/4,     # A_s [m²]
    T_in,      # T_c_in [K]
    M_DOT    # m_dot_c [kg/s]
)

def dTb_dt(t, Tb, params):
    """
    Differential equation for bulk temperature Tb
    """
    # Unpack parameters
    m, cp_b, I, R, A_s, T_c_in, m_dot_c= params

    # Calculate h based on current bulk temperature
    h = calculate_h(Tb)
    cp_c = Cp_func(Tb)
    
    # Electrical heating term
    electrical_heating = I**2 * R
    
    # Cooling term denominator
    cooling_denom = 1 + (h * A_s) / (2 * m_dot_c * cp_c)
    
    # Cooling term
    cooling = (h * A_s * (Tb - T_c_in)) / cooling_denom
    
    # Rate of change of temperature
    dTb_dt = (electrical_heating - cooling) / (m * cp_b)
    
    return dTb_dt

