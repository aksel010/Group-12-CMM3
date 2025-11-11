# I'm looking to combine the error from the interpolation and the error from the RK4 method.
import Current_profile
from config import stepsize as fullstep
import numpy as np

currentprofile = current_profile.get_delta_T_vs_I_interpolator()
fourth_derivative = 
max4th = max(fourth_derivative)
C_cubicsplien = 1/384
halfstep = stepsize / 2 
T_halfstep = currentprofile(halfstep)
T_fullstep = currentprofile(fullstep)

def combined_error(interpolation_error, rk4_error):
    return (interpolation_error**2 + rk4_error**2)**0.5

def interpolation_error(fourth_derivative, fullstep):
    return ( C_cubicsplien * max4th * fullstep**4 )

def rk4_error()
    p = 4.0 # Order of the RK method
    estimated_error = (T_halfstep - T_fullstep) / (2**p - 1)
    return estimated_error

total_combined_error = combined_error(interpolation_error, rk4_error)
print(f"Total Combined Error: {total_combined_error:.6e} K")