import numpy as np
from config import *
from ODE import Tb, dTb_dt, params_initial


# -------------------------------------------------------------------------
# FUNCTION: Estimate RK4 truncation error
# -------------------------------------------------------------------------
def rk4_error():
    """
    Estimate the local truncation error of RK4 by comparing
    full-step and half-step solutions.

    Returns:
        estimated_error : float
            Maximum estimated RK4 truncation error [K]
    """
    # -------------------------------------------------------------------------
    # SETUP: Define RK4 step size and compute full- and half-step solutions
    # -------------------------------------------------------------------------
    rk4_step = H  # RK4 integration step size from config
    T_fullstep, t_fullstep = Tb(dTb_dt, params_initial, stepsize=rk4_step)
    T_halfstep, t_halfstep = Tb(dTb_dt, params_initial, stepsize=rk4_step / 2)
    p = 4.0  # RK4 order

    # Reshape half-step solution to match full-step points
    T_half_reshaped = T_halfstep[::2]  # take every 2nd point
    T_full_reshaped = T_fullstep

    # Difference between half-step and full-step solutions
    T_diff = np.array(T_half_reshaped) - np.array(T_full_reshaped)
    
    # Estimated error formula for RK4
    estimated_error = np.max(np.abs(T_diff)) / (2**p - 1)
    return estimated_error


# -------------------------------------------------------------------------
# WRAPPER: Return RK4 error on demand
# -------------------------------------------------------------------------
def get_rk4_error_val():
    """Calculate and return RK4 truncation error."""
    return rk4_error()


# -------------------------------------------------------------------------
# SCRIPT RUN FUNCTION
# -------------------------------------------------------------------------
def run():
    rk4_error_val = get_rk4_error_val()
    print(f"Interpolation Step Size (H): {H}")
    print(f"RK4 Integration Step Size: {H}")
    print(f"RK4 Truncation Error: {rk4_error_val:.6e} K")


# -------------------------------------------------------------------------
# SCRIPT ENTRY POINT
# -------------------------------------------------------------------------
if __name__ == "__main__":
    run()
