import math
from src.utils.heptane_itpl import cp_func,calculate_h
from src.models.Mass_flowrate import get_steady_state_values
from src.config import *
import numpy as np
from scipy.integrate import solve_ivp
from src.models.Optimum_Current import current_params
import matplotlib.pyplot as plt


# Complete params_initial tuple
params_initial = (
    m_cell,       # 0.045 kg
    c_b,       # 900 J/(kg·K)
    current_store,      # Increased current to 15A
    0.06,      # Fixed resistance of 0.08Ω (gives 18W heating)
    a_s,       # 0.0025 m²
    t_in,      # 293.15 K
    1e-100       # Very low mass flow
)

def d_tb_dt(tb, t, params):
    """
    Calculate the rate of change of bulk temperature Tb.

    Args:
        tb (float): Bulk temperature at time t.
        t (float): Time.
        params (tuple): Physical parameters of the system.

    Returns:
        float: d_tb/dt, rate of temperature change.
    """
    # Unpack parameters
    mb, cp_b, current, r, a_s, t_c_in, m = params

    # Calculate h based on current bulk temperature
    t_c_avg = (tb + t_c_in) / 2
    h = calculate_h(t_c_avg)
    cp_c = cp_func(t_c_avg)

    # Electrical heating term
    heating = current**2 * r

    # Cooling term denominator
    cooling_denom = 1 + (h * a_s) / (2 * m * cp_c)

    # Cooling term
    cooling = (h * a_s * (tb - t_c_in)) / cooling_denom

    # Rate of change of temperature
    d_tb_dt = (heating - cooling) / (m * cp_b)

    return d_tb_dt

def get_tb(d_tb_dt, params, stepsize):
    """
    Integrates battery bulk temperature evolution using RK4 scheme.

    Args:
        d_tb_dt (callable): Derivative function for battery temperature.
            Signature: d_tb_dt(temp, time, params) -> float
        params (tuple): Physical system parameters 
            (current, mass_flow, heat_capacity, etc.).
        stepsize (float): Time step size for integration (seconds).

    Returns:
        tuple: (time_array, temperature_array)
            - time_array (np.ndarray): Time points from 0 to t_final.
            - temperature_array (np.ndarray): Battery temperatures at each time.
    """
    # Initial conditions
    initial_time = 0.0
    initial_temp = t_in  # Inlet temperature from config
    final_time = q_b / params[2]  # Total charge / current
    num_steps = math.ceil(final_time / stepsize)

    # Preallocate arrays for RK4 integration
    temp_battery = np.zeros(num_steps + 1)
    time_points = np.zeros(num_steps + 1)
    temp_battery[0] = initial_temp
    time_points[0] = initial_time
    time_points[1:] = np.arange(1, num_steps + 1) * stepsize

    # RK4 integration loop
    for i in range(num_steps):
        t_current = time_points[i]
        temp_current = temp_battery[i]

        # Runge-Kutta 4th order method
        k1 = d_tb_dt(temp_current, t_current, params)
        k2 = d_tb_dt(temp_current + 0.5 * stepsize * k1, 
                     t_current + 0.5 * stepsize, params)
        k3 = d_tb_dt(temp_current + 0.5 * stepsize * k2, 
                     t_current + 0.5 * stepsize, params)
        k4 = d_tb_dt(temp_current + stepsize * k3, 
                     t_current + stepsize, params)
        
        slope = (k1 + 2 * k2 + 2 * k3 + k4) / 6
        temp_battery[i + 1] = temp_current + stepsize * slope

    return time_points, temp_battery

# ------------------------------------------------------

def get_tb_scipy(d_tb_dt, params):
    """
    Integrate battery bulk temperature evolution using SciPy's solve_ivp method.

    Args:
        d_tb_dt (callable): Derivative function for battery temperature.
            Signature: d_tb_dt(temp, time, params) -> float
        params (tuple): Physical system parameters 
            (current, mass_flow, heat_capacity, etc.).

    Returns:
        tuple: (time_array, temperature_array)
            - time_array (np.ndarray): Time points from 0 to final_time.
            - temperature_array (np.ndarray): Battery temperatures at each time.
    """
    # Initial conditions
    initial_time = 0.0
    initial_temp = t_in  # Inlet temperature from config
    final_time = q_b / params[2]  # Total charge / current
    
    # Solve ODE using LSODA adaptive method
    sol = solve_ivp(
        fun=lambda t, T, *_: d_tb_dt(T, t, params),
        t_span=[initial_time, final_time],
        y0=[initial_temp],
        method='LSODA',
        dense_output=True,
        rtol=1e-6,
        atol=1e-8
    )

    # Extract solution at uniform time points
    num_output_points = 100
    time_points = np.linspace(initial_time, final_time, num_output_points)
    temp_battery = sol.sol(time_points)[0]
    
    return time_points, temp_battery

def export_scipy_data(d_tb_dt, params, filename='scipy_solution.csv'):
    """
    Export SciPy solution data to CSV file.

    Args:
        d_tb_dt (callable): Derivative function for battery temperature.
        params (tuple): Physical system parameters.
        filename (str): Output CSV file name. Default: 'scipy_solution.csv'
    """
    time_points, temp_battery = get_tb_scipy(d_tb_dt, params)
    data = {
        'Time (s)': time_points,
        'Temperature (K)': temp_battery
    }
    df = pd.DataFrame(data)
    df.to_csv(filename, index=False)


def run():
    """
    Execute both RK4 and SciPy integrators and return their results for comparison.

    Args:
        d_tb_dt (callable): Derivative function for battery temperature.
        params (tuple): Physical system parameters.
        stepsize (float): Time step size for RK4 integration (seconds).

    Returns:
        dict: Dictionary with 'rk4' and 'scipy' keys, each containing:
            - time_array (np.ndarray): Time points
            - temp_array (np.ndarray): Battery temperatures
    """
    time_rk4, temp_rk4 = get_tb(d_tb_dt, current_params(current_store), stepsize = H)
    time_scipy, temp_scipy = get_tb_scipy(d_tb_dt, current_params(current_store))
    
    return {
        'rk4': (time_rk4, temp_rk4),
        'scipy': (time_scipy, temp_scipy)
    }

if __name__ == "__main__":
    run()
    time_rk4, temp_rk4 = get_tb(d_tb_dt, params_initial, stepsize = H)
    plt.plot(time_rk4,temp_rk4)
    print(params_initial)
    plt.show()
