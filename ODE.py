import math
from heptane_itpl import Cp_func, calculate_h
from Mass_flowrate import get_steady_state_values
from config import *
import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

# Parameters for initial conditions
params_initial = (
    m_b,      # Mass of bulk, [kg]
    C_b,      # Specific heat capacity, [J/(kg·K)]
    I_0,      # Current, [A]
    DC_IR * 24, # Resistance, [Ω]
    A_s,      # Surface area, [m²]
    T_in,     # Inlet Temperature, [K]
    get_steady_state_values()[0] # Mass flow rate, [kg/s]
)

def dTb_dt(Tb, t, params):
    """
    Calculate the rate of change of bulk temperature Tb.

    Args:
        Tb (float): Bulk temperature at time t.
        t (float): Time.
        params (tuple): Physical parameters of the system.

    Returns:
        float: dTb/dt, rate of temperature change.
    """
    # Unpack parameters
    m, cp_b, I, R, A_s, T_c_in, m_dot = params

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

def Tb(dTdt, params, stepsize):
    """
    Integrates bulk temperature evolution using RK4 scheme.

    Args:
        dTdt (callable): Derivative function.
        params (tuple): Physical system parameters.
        stepsize (float): Time step size for integration.

    Returns:
        tuple: Arrays of time points and temperatures.
    """
    # Initial conditions
    t0 = 0
    T0 = T_in
    t_final = q_b / params[2]
    n_step = math.ceil(t_final / stepsize)

    T_rk = np.zeros(n_step + 1)
    t_rk = np.zeros(n_step + 1)
    T_rk[0] = T0
    t_rk[0] = t0
    t_rk[1:] = np.arange(1, n_step + 1) * stepsize

    for i in range(n_step):
        t_i = t_rk[i]
        T_i = T_rk[i]

        k1 = dTdt(T_i, t_i, params)
        k2 = dTdt(T_i + 0.5 * stepsize * k1, t_i + 0.5 * stepsize, params)
        k3 = dTdt(T_i + 0.5 * stepsize * k2, t_i + 0.5 * stepsize, params)
        k4 = dTdt(T_i + stepsize * k3, t_i + stepsize, params)
        slope = (k1 + 2 * k2 + 2 * k3 + k4) / 6
        T_rk[i + 1] = T_i + stepsize * slope

    return t_rk, T_rk

# ------------------------------------------------------

def Tb_scipy(dTdt, params):
    """
    Integrate temperature using SciPy's solve_ivp method.

    Args:
        dTdt (callable): Derivative function.
        params (tuple): System parameters.

    Returns:
        tuple: Arrays of time points and temperatures.
    """
    t0 = 0
    T0 = T_in
    t_final = q_b / params[2]
    sol = solve_ivp(
        fun=lambda t, T, *_: dTdt(T, t, params),
        t_span=[t0, t_final],
        y0=[T0],
        method='LSODA',
        dense_output=True,
        rtol=1e-6,
        atol=1e-8
    )

    t_sol = np.linspace(t0, t_final, 100)
    T_sol = sol.sol(t_sol)[0]
    return t_sol, T_sol

import pandas as pd
import numpy as np

def export_scipy_data(filename='RK4_solution.csv'):
    """
    Export SciPy solution data to CSV.

    Args:
        filename (str): Output CSV file name.
    """
    t_sol, T_sol = Tb_scipy(dTb_dt, params_initial)
    data = {
        'Time (s)': t_sol,
        'Temperature (K)': T_sol
    }
    df = pd.DataFrame(data)
    df.to_csv(filename, index=False)


def run():
    """
    Execute both RK4 and SciPy integrators, return their results.

    Returns:
        dict: 'rk4' and 'scipy' arrays with their respective time and temperature arrays.
    """
    t_rk, T_rk = Tb(dTb_dt, params_initial, stepsize=H)
    t_scipy, T_scipy = Tb_scipy(dTb_dt, params_initial)
    return {
        'rk4': (t_rk, T_rk),
        'scipy': (t_scipy, T_scipy)
    }

if __name__ == "__main__":
    # Ensure H is defined in your config.py or elsewhere for stepsize
    run()
# ------------------------------------------------------
