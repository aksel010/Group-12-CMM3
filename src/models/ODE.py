import math
from src.utils.heptane_itpl import Cp_func,calculate_h
from src.models.Mass_flowrate import get_steady_state_values
from config import *
import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt


params_initial = (
    m_b,      # m [kg]
    C_b,      # cp_b [J/(kg·K)]
    I_0,       # I [A]
    DC_IR *24,        # R [Ω]
    A_s,     # A_s [m²]
    T_in,     # T_c_in [K]\
    get_steady_state_values()[0]     # m_dot_c [kg/s]
)

def dTb_dt (Tb, t, params):
    """
    Differential equation for bulk temperature Tb
    """
    # Unpack parameters
    m, cp_b, I, R, A_s, T_c_in, m= params

    # Calculate h based on current bulk temperature
    T_c_avg = (Tb + T_c_in) / 2
    h = calculate_h(T_c_avg)
    cp_c = Cp_func(T_c_avg)
    
    # Electrical heating term
    heating = I**2 * R
    
    # Cooling term denominator
    cooling_denom = 1 + (h * A_s) / (2 * m * cp_c)
    
    # Cooling term
    cooling = (h * A_s * (Tb - T_c_in)) / cooling_denom
    
    # Rate of change of temperature
    dTb_dt = (heating - cooling) / (m * cp_b)
    
    return dTb_dt

def Tb(dTdt, params,stepsize):
    # initial conditions
    t0 = 0
    T0 = T_in  # still using the external variable

    # total solution interval
    t_final = q_b / params[2]

    # number of steps
    n_step = math.ceil(t_final / stepsize)

    # allocate solution arrays
    T_rk = np.zeros(n_step + 1)
    t_rk = np.zeros(n_step + 1)
    T_rk[0] = T0
    t_rk[0] = t0

    # populate the time array in one vectorized step
    t_rk[1:] = np.arange(1, n_step + 1) * stepsize

    # apply RK4
    for i in range(n_step):
        t_i = t_rk[i]
        T_i = T_rk[i]

        # compute all slopes in a compact way
        k1 = dTdt(T_i, t_i, params)
        k2 = dTdt(T_i + 0.5 * stepsize * k1, t_i + 0.5 * stepsize, params)
        k3 = dTdt(T_i + 0.5 * stepsize * k2, t_i + 0.5 * stepsize, params)
        k4 = dTdt(T_i + stepsize * k3, t_i + stepsize, params)

        # weighted average of slopes
        slope = (k1 + 2*k2 + 2*k3 + k4) / 6

        # update solution
        T_rk[i + 1] = T_i + stepsize * slope

    return t_rk, T_rk
    # ------------------------------------------------------

# scipy solve_ivp method

def Tb_scipy(dTdt, params):
    t0 = 0
    T0 = T_in
    t_final = q_b / params[2]
    # solve_ivp call
    sol = solve_ivp(
        fun=lambda t, T, *_: dTdt(T, t, params),  
        t_span=[t0, t_final],
        y0=[T0],
        method='LSODA', 
        dense_output=True, 
        rtol=1e-6,        
        atol=1e-8         
    )
    
    # Create an array of time points for plotting (e.g., 100 points)
    t_sol = np.linspace(t0, t_final, 100)
    
    # Use the dense output to get the solution at these time points
    T_sol = sol.sol(t_sol)[0] 

    
    return t_sol, T_sol

# Add this function block anywhere in your ODE.py file, 
# ensuring it's after the definition of Tb_scipy.

import pandas as pd
import numpy as np

def export_scipy_data(filename='RK4 solution.csv'):
    t_sol, T_sol = Tb_scipy(dTb_dt, params_initial)
    data = {
        'Time (s)': t_sol,
        'Temperature (K)': T_sol
    }
    df = pd.DataFrame(data)
    df.to_csv(filename, index=False)


def run():
    t_rk, T_rk = Tb(dTb_dt, params_initial, stepsize=H)
    t_scipy, T_scipy = Tb_scipy(dTb_dt, params_initial)
    
    # REMOVE all plt commands, RETURN data
    return {
        'rk4': (t_rk, T_rk),
        'scipy': (t_scipy, T_scipy)
    }

if __name__ == "__main__":
    # Ensure H is defined in your config.py or elsewhere for stepsize
    # Example: H = 0.2 
    run()
# ------------------------------------------------------
