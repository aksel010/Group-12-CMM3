import numpy as np
from config import *
from ODE import get_tb, d_tb_dt, initial_params


def rk4_error():
    """
    Estimate the local truncation error of RK4 by comparing
    full-step and half-step solutions.

    Returns:
        estimated_error : float
            Maximum estimated RK4 truncation error [K]
    """
    rk4_step = H  # RK4 integration step size from config
    T_fullstep, t_fullstep = get_tb(d_tb_dt, initial_params, stepsize=rk4_step)
    T_halfstep, t_halfstep = get_tb(d_tb_dt, initial_params, stepsize=rk4_step / 2)
    p = 4.0  # RK4 order

    T_half_reshaped = T_halfstep[::2]  # take every 2nd point
    T_full_reshaped = T_fullstep
    T_diff = np.array(T_half_reshaped) - np.array(T_full_reshaped)
    estimated_error = np.max(np.abs(T_diff)) / (2**p - 1)
    return estimated_error


def get_rk4_error_val():
    """Calculate and return RK4 truncation error."""
    return rk4_error()


def run():
    rk4_error_val = get_rk4_error_val()
    print(f"Interpolation Step Size (H): {H}")
    print(f"RK4 Integration Step Size: {H}")
    print(f"RK4 Truncation Error: {rk4_error_val:.6e} K")


if __name__ == "__main__":
    run()
