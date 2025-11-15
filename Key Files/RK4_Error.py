# I'm looking to combine the error from the interpolation and the error from the RK4 method.
import numpy as np
from config import *
from ODE import Tb , dTb_dt , params_initial

rk4_step = H

#RK4 ERROR ESTIMATION
T_fullstep, t_fullstep = Tb(dTb_dt, params_initial, stepsize = rk4_step)
T_halfstep, t_halfstep= Tb(dTb_dt, params_initial, stepsize = rk4_step / 2)

def rk4_error():
    p = 4.0 #rk order
    T_halfstep_reshaped = [x for x in T_halfstep if x % 2 != 0]
    T_fullstep_reshaped = T_fullstep[:-1]
    T_diff = max(T_halfstep_reshaped - T_fullstep_reshaped)
    estimated_error = abs(T_diff) / (2**p - 1)
    return estimated_error

rk4_error_val = rk4_error()


print(f"Interpolation Step Size (h_interp): {H:.6e}")
print(T_halfstep[-1])
print(T_fullstep[-1])
print(f"RK4 Integration Step Size (rk4_step): {rk4_step:.6e}")
print(f"RK4 Truncation Error: {rk4_error_val:.6e} K")