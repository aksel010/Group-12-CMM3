import math
from heptane_itpl import Cp_func,calculate_h
from Mass_flowrate import m_dot_ss
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
    m_dot_ss     # m_dot_c [kg/s]
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
    T0 = T_in
    # total solution interval
    t_final= q_b / params[2]  # total time [s]
    # step size
    # Fourth Order Runge-Kutta method

    # number of steps
    n_step = math.ceil(t_final/stepsize)

    # Definition of arrays to store the solution
    T_rk = np.zeros(n_step+1)
    t_rk = np.zeros(n_step+1)

    # Initialize first element of solution arrays 
    # with initial condition
    T_rk[0] = T0
    t_rk[0] = t0 

    # Populate the x array
    for i in range(n_step):
        t_rk[i+1]  = t_rk[i]  + stepsize

    # Apply RK method n_step times
    for i in range(n_step):
    
        # Compute the four slopes
        t_dummy = t_rk[i]
        T_dummy = T_rk[i]
        k1 =  dTdt(T_dummy,t_dummy, params)
        
        t_dummy = t_rk[i]+stepsize/2
        T_dummy = T_rk[i] + k1 * stepsize/2
        k2 =  dTdt(T_dummy,t_dummy, params)

        t_dummy = t_rk[i]+stepsize/2
        T_dummy = T_rk[i] + k2 * stepsize/2
        k3 =  dTdt(T_dummy,t_dummy, params)

        t_dummy = t_rk[i]+stepsize
        T_dummy = T_rk[i] + k3 * stepsize
        k4 =  dTdt(T_dummy,t_dummy, params)

        # compute the slope as weighted average of four slopes
        slope = 1/6 * k1 + 2/6 * k2 + 2/6 * k3 + 1/6 * k4 

        # use the RK method
        T_rk[i+1] = T_rk[i] + stepsize * slope  

    return t_rk, T_rk
    # ------------------------------------------------------

# scipy solve_ivp method

def Tb_scipy(dTdt, params):
    t0 = 0
    T0 = T_in
    t_final = q_b / params[2]
    # solve_ivp call
    sol = solve_ivp(
        # FIX: Accept optional arguments (*_) from solve_ivp 
        # but only use t and T, as 'params' is hardcoded inside the lambda body.
        fun=lambda t, T, *_: dTdt(T, t, params),  
        
        t_span=[t0, t_final],
        y0=[T0],
        # Remove the 'args' tuple since 'params' is already passed explicitly in the lambda.
        # If you keep 'args', the lambda needs another fix (see alternative below).
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

# ------------------------------------------------------
# plot results
def run():
    # 1. Custom RK4 Solution (Your existing method)
    t_rk, T_rk = Tb(dTb_dt, params_initial, stepsize=H)
    
    # 2. SciPy Validation Solution (The new method)
    t_scipy, T_scipy = Tb_scipy(dTb_dt, params_initial)

    plt.figure(figsize=(10, 6))

    # Plot custom RK4
    plt.plot(t_rk, T_rk, label=f'Custom RK4 (h={H}s)', color='blue', linestyle='--')
    
    # Plot SciPy Solution
    plt.plot(t_scipy, T_scipy, label='SciPy solve_ivp (LSODA)', color='red', linewidth=3, alpha=0.6)

    plt.xlabel('t (s)')
    plt.ylabel('Temperature of the Battery $T_b$ (K)')
    plt.title('ODE Solution Validation')
    plt.legend()
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    # Ensure H is defined in your config.py or elsewhere for stepsize
    # Example: H = 0.2 
    run()
# ------------------------------------------------------