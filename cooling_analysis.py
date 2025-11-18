"""Compute frictional and local head losses for a multi-branch system."""

import numpy as np

from heptane_itpl import rho_func, mu_func
from config import *

# --------------------------------------------------------------------------- #
# Main head-loss calculation
# --------------------------------------------------------------------------- #

def calculate_head_loss_only(q_main, avg_temp_k=300):
    """Calculate total head loss for Group A branches and main pipe.
    Args:
        q_main (float): Total volumetric flow rate entering Group A.
        avg_temp_k (float, optional): Average fluid temperature in Kelvin.
    Returns:
        float: Total head loss (m) for both cooling and heating sections.
    """
    def calculate_branch_head_loss(q_individual, nu):
        """Compute frictional head loss for all branches (equal flow)."""
        n_branches = len(q_individual)
        h_branch_vec = np.zeros(n_branches)
        for i in range(n_branches):
            main_pipe_loss = 0.0
            for j in range(i + 1):
                sum_q_m = np.sum(q_individual[j:])
                term = (32 * nu * L_a / (D ** 2 * g)) * (sum_q_m / S_a)
                main_pipe_loss += term
            main_pipe_loss *= 2
            branch_pipe_loss = (
                32 * nu * L_b / (d_H ** 2 * g)
            ) * (q_individual[i] / S_b)
            h_branch_vec[i] = main_pipe_loss + branch_pipe_loss
        return h_branch_vec
    def calculate_local_losses(q_individual, nu):
        """Compute local losses (bends, junctions) for each branch."""
        n_branches = len(q_individual)
        h_local_vec = np.zeros(n_branches)
        xi_bend = 0.2  # Bend loss coefficient
        for i in range(n_branches):
            v_main_in = np.sum(q_individual[i:]) / S_a
            v_branch = q_individual[i] / S_b
            ratio = v_branch / v_main_in
            xi_1_3 = 0.5 * ratio ** 2 + 1
            xi_3_1 = ratio ** 2 - ratio ** 2 + 0.5 * (1 - ratio)
            h_bend = 24 * (xi_bend / (2 * g)) * v_branch ** 2
            h_local_vec[i] = (
                (xi_1_3 / (2 * g)) * v_main_in ** 2
                + (xi_3_1 / (2 * g)) * v_main_in ** 2
                + h_bend
            )
        return h_local_vec
    def calculate_total_group_a_head_loss(q_individual, nu):
        """Compute full friction + local losses for entire Group A."""
        h_branch_friction = calculate_branch_head_loss(q_individual, nu)
        h_branch_local = calculate_local_losses(q_individual, nu)
        q_total = np.sum(q_individual)
        v_main = q_total / S_a
        re_main = v_main * D / nu
        lambda_main = 64 / re_main
        h_main_friction = 2 * (
            lambda_main * (L_c / D) * (v_main ** 2 / (2 * g))
        )
        h_common_branch = h_branch_friction[0] + h_branch_local[0]
        return h_common_branch + h_main_friction
    mu = mu_func(avg_temp_k)
    rho = rho_func(avg_temp_k)
    nu = mu / rho
    q_individual = np.full(n, q_main / n)
    h_total_group_a = calculate_total_group_a_head_loss(q_individual, nu)
    return 4 * h_total_group_a

def get_head_loss(q_main):
    """Return total head loss for a given main flow rate."""
    return calculate_head_loss_only(q_main)

def run():
    """Execute head-loss calculation using default flow rate."""
    head_loss = calculate_head_loss_only(flowrate_min)
    print(f"Total Group A head loss: {head_loss:.6f} m")

if __name__ == "__main__":
    run()
