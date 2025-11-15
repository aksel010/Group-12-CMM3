# I'm looking to combine the error from the interpolation and the error from the RK4 method.
import numpy as np
from config import *
from ODE import Tb , dTb_dt , params_initial

rk4_step = H

#RK4 ERROR ESTIMATION
def rk4_error():
    p = 4.0 #rk order
    T_halfstep_reshaped = [x for x in T_halfstep if x % 2 != 0]
    T_fullstep_reshaped = T_fullstep[:-1]
    T_diff = max(T_halfstep_reshaped - T_fullstep_reshaped)
    estimated_error = abs(T_diff) / (2**p - 1)
    return estimated_error

def get_rk4_error_val():
    """Calculate and return RK4 error value on demand"""
    return rk4_error()

def run():
        rk4_error_val = get_rk4_error_val()
    print(f"Interpolation Step Size (h_interp): {H:.6e}")
    print(f"RK4 Integration Step Size (rk4_step): {rk4_step:.6e}")
    print(f"RK4 Truncation Error: {rk4_error_val:.6e} K")

if __name__ == "__main__":
    run()
