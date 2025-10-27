import math
from heptane_itpl import M_DOT, D, Cp_func,calculate_h
from config import C_b, T_in, m_b
import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

I_0 = 50
R_0 = 240/I_0

params_initial = (
    1.0,      # m [kg]
    C_b,      # cp_b [J/(kg·K)]
    I_0,       # I [A]
    R_0,        # R [Ω]
    np.pi*D**2/4,     # A_s [m²]
    T_in,      # T_c_in [K]
    1    # m_dot_c [kg/s]
)

def dTb_dt (Tb, t, params):
    """
    Differential equation for bulk temperature Tb
    """
    # Unpack parameters
    m, cp_b, I, R, A_s, T_c_in, m_dot_c= params

    # Calculate h based on current bulk temperature
    h = 400
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

# initial conditions
t0 = 0
T0 = T_in
# total solution interval
t_final = 1000
# step size
H = 0.2
# ------------------------------------------------------

# ------------------------------------------------------
# Fourth Order Runge-Kutta method

# number of steps
n_step = math.ceil(t_final/H)

# Definition of arrays to store the solution
T_rk = np.zeros(n_step+1)
t_rk = np.zeros(n_step+1)

# Initialize first element of solution arrays 
# with initial condition
T_rk[0] = T0
t_rk[0] = t0 

# Populate the x array
for i in range(n_step):
    t_rk[i+1]  = t_rk[i]  + H

# Apply RK method n_step times
for i in range(n_step):
   
    # Compute the four slopes
    t_dummy = t_rk[i]
    T_dummy = T_rk[i]
    k1 =  dTb_dt(T_dummy,t_dummy, params_initial)
    
    t_dummy = t_rk[i]+H/2
    T_dummy = T_rk[i] + k1 * H/2
    k2 =  dTb_dt(T_dummy,t_dummy, params_initial)

    t_dummy = t_rk[i]+H/2
    T_dummy = T_rk[i] + k2 * H/2
    k3 =  dTb_dt(T_dummy,t_dummy, params_initial)

    t_dummy = t_rk[i]+H
    T_dummy = T_rk[i] + k3 * H
    k4 =  dTb_dt(T_dummy,t_dummy, params_initial)

    # compute the slope as weighted average of four slopes
    slope = 1/6 * k1 + 2/6 * k2 + 2/6 * k3 + 1/6 * k4 

    # use the RK method
    T_rk[i+1] = T_rk[i] + H * slope  
# ------------------------------------------------------

# ------------------------------------------------------
# print results on screen
print ('Solution: step x y-eul y-exact error%')
for i in range(n_step+1):
    print(i,t_rk[i],T_rk[i], T0 * math.exp(-t_rk[i]),
            (T_rk[i]- T0 * math.exp(-t_rk[i]))/ 
            (T0 * math.exp(-t_rk[i])) * 100)
# -----------------------------------------------------

# ------------------------------------------------------
# plot results
plt.plot(t_rk, T_rk)
plt.xlabel('x')
plt.ylabel('y(x)')
plt.show()
# ------------------------------------------------------
print (params_initial)

