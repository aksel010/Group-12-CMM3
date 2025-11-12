# I'm looking to combine the error from the interpolation and the error from the RK4 method.
import Current_profile
import numpy as np
from config import stepsize as interpolation_step
from ODE import rk4_solve, params_initial, t_total

currentprofile = Current_profile.get_delta_T_vs_I_interpolator()
h_interp = interpolation_step
rk4_step =  0.2 

fourth_derivative = np.array([])
max4th = max(fourth_derivative)

C_cubicsplien = 1/384 #### need source for this value
halfstep = rk4_step / 2 

#ITPL ERROR ESTIMATION
def interpolation_error(max_fourth_derivative, h_interp):
    return ( C_cubicsplien * max_fourth_derivative * h_interp**4 )

#RK4 ERROR ESTIMATION
T_fullstep = rk4_solve(t_total, rk4_step, params_initial)
T_halfstep = rk4_solve(t_total, halfstep, params_initial)

def rk4_error():
    p = 4.0
    estimated_error = abs(T_halfstep - T_fullstep) / (2**p - 1)
    return estimated_error

interpolation_e_val = interpolation_error(max4th, h_interp)
rk4_error_val = rk4_error()

def combined_error(interpolation_error_val, rk4_error_val):
    return (interpolation_error_val**2 + rk4_error_val**2)**0.5

total_combined_error = combined_error(interpolation_e_val, rk4_error_val)

print(f"Interpolation Step Size (h_interp): {h_interp:.6e}")
print(f"RK4 Integration Step Size (rk4_step): {rk4_step:.6e}")
print(f"Interpolation Error: {interpolation_e_val:.6e} K")
print(f"RK4 Truncation Error: {rk4_error_val:.6e} K")
print(f"Total Combined Error: {total_combined_error:.6e} K")