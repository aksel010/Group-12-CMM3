# I'm looking to combine the error from the interpolation and the error from the RK4 method.
import numpy as np
from config import stepsize as interpolation_step
from ODE import Tb , dTb_dt , params_initial
from Optimum_Current import rmse

#RK4 ERROR ESTIMATION
T_fullstep = Tb(dTb_dt, params_initial, stepsize = rk4_step)
T_halfstep = Tb(dTb_dt, params_initial, stepsize = rk4_step / 2)

def rk4_error():
    p = 4.0
    estimated_error = abs(T_halfstep - T_fullstep) / (2**p - 1)
    return estimated_error

interpolation_e_val = rmse
rk4_error_val = rk4_error()

def combined_error(interpolation_error_val, rk4_error_val):
    return (interpolation_error_val**2 + rk4_error_val**2)**0.5

total_combined_error = combined_error(interpolation_e_val, rk4_error_val)

print(f"Interpolation Step Size (h_interp): {h_interp:.6e}")
print(f"RK4 Integration Step Size (rk4_step): {rk4_step:.6e}")
print(f"Interpolation Error: {interpolation_e_val:.6e} K")
print(f"RK4 Truncation Error: {rk4_error_val:.6e} K")
print(f"Total Combined Error: {total_combined_error:.6e} K")