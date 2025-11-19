"""
Estimate and report the truncation error for the RK4 ODE solver, comparing single and half-step solutions.
PEP8-compliant docstrings/descriptions for all functions and script blocks.
"""
import numpy as np
from src.config import *
from src.models.battery_temperature_ode import get_tb, d_tb_dt, params_initial


def rk4_error():
    """
    Estimate the maximum local truncation error for RK4 integration, by comparing full step and half step solutions.

    Returns:
        float: Max RK4 truncation error [K].
    """
    rk4_step = H
    T_fullstep, t_fullstep = get_tb(d_tb_dt, params_initial, stepsize=rk4_step)
    T_halfstep, t_halfstep = get_tb(d_tb_dt, params_initial, stepsize=rk4_step / 2)
    p = 4.0  # RK4 method order
    
    # Take every 2nd element from halfstep to match fullstep time points
    T_half_reshaped = T_halfstep[::2]
    
    # Ensure both arrays have the same length
    min_len = min(len(T_half_reshaped), len(T_fullstep))
    T_half_reshaped = T_half_reshaped[:min_len]
    T_full_reshaped = T_fullstep[:min_len]
    
    # Calculate difference and error estimate
    T_diff = np.array(T_half_reshaped) - np.array(T_full_reshaped)
    estimated_error = np.max(np.abs(T_diff)) / (2**p - 1)
    
    return estimated_error

def get_rk4_error_val():
    """
    Calculate and return RK4 local truncation error estimate (helper).
    Returns:
        float: Max RK4 error [K].
    """
    return rk4_error()

def run():
    """
    Print step size and estimated RK4 truncation error using current config and ODE params.
    """
    rk4_error_val = get_rk4_error_val()
    print(f"Interpolation Step Size (H): {H}")
    print(f"RK4 Integration Step Size: {H}")
    print(f"RK4 Truncation Error: {rk4_error_val:.6e} K")

if __name__ == "__main__":
    run()
