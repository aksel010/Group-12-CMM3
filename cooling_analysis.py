"""
Cooling channel network head-loss model for a multi-branch battery system.
Computes frictional and local losses, exposing convenient evaluation API.
All modules, wrappers, and runners fully docstring-annotated and commented.
"""
import numpy as np
from heptane_itpl import rho_func, mu_func
from config import *

def calculate_head_loss_only(q_main, t_avg_k=300):
    """
    Compute total cooling circuit (Group A) head loss (m) for specified main branch flow q_main at average temperature T_avg_K.

    Args:
        q_main (float): Main volumetric flow [m^3/s].
        t_avg_k (float): Average fluid temperature [K] (default: 300).
    Returns:
        float: Overall head loss [m] summed over all relevant branches.
    """
    def calculate_branch_head_loss(q_individual, nu):
        """
        Frictional head loss for all identical branches.
        Args:
            q_individual (ndarray): Individual branch flows [m^3/s].
            nu (float): Kinematic viscosity of fluid [m^2/s].
        Returns:
            ndarray: Head loss for each branch [m].
        """
        n_branches = len(q_individual)
        h_branch = np.zeros(n_branches)
        for i in range(n_branches):
            main_pipe_loss = 0.0
            for j in range(i + 1):
                sum_q_m = np.sum(q_individual[j:])
                term = (32 * nu * l_a / (d ** 2 * g)) * (sum_q_m / s_a)
                main_pipe_loss += term
            main_pipe_loss *= 2
            branch_pipe_loss = (
                32 * nu * l_b / (d_h ** 2 * g)
            ) * (q_individual[i] / s_b)
            h_branch[i] = main_pipe_loss + branch_pipe_loss
        return h_branch
    def calculate_local_losses(q_individual, nu):
        """
        Local head losses due to bends and junctions.
        Args:
            q_individual (ndarray): Branch flows.
            nu (float): Kinematic viscosity [m^2/s].
        Returns:
            ndarray: Local loss per branch [m].
        """
        n_branches = len(q_individual)
        h_local = np.zeros(n_branches)
        ξ_bend = 0.2
        for i in range(n_branches):
            v_main_in = np.sum(q_individual[i:]) / s_a
            v_branch = q_individual[i] / s_b
            ratio = v_branch / v_main_in
            ξ_1_3 = 0.5 * ratio ** 2 + 1
            ξ_3_1 = ratio ** 2 - ratio ** 2 + 0.5 * (1 - ratio)
            h_bend = 24 * (ξ_bend / (2 * g)) * v_branch ** 2
            h_local[i] = (
                (ξ_1_3 / (2 * g)) * v_main_in ** 2
                + (ξ_3_1 / (2 * g)) * v_main_in ** 2
                + h_bend
            )
        return h_local
    def calculate_total_group_a_head_loss(q_individual, nu):
        """
        Full head loss for all Group A branches including main section.
        Args:
            q_individual (ndarray): Per-branch flows.
            nu (float): Kinematic viscosity [m^2/s].
        Returns:
            float: Total head loss [m].
        """
        h_branch_friction = calculate_branch_head_loss(q_individual, nu)
        h_branch_local = calculate_local_losses(q_individual, nu)
        q_total = np.sum(q_individual)
        v_main = q_total / s_a
        re_main = v_main * d / nu
        λ_main = 64 / re_main  # Laminar assumption
        h_main_friction = 2 * (
            λ_main * (l_c / d) * (v_main ** 2 / (2 * g))
        )
        h_common_branch = h_branch_friction[0] + h_branch_local[0]
        return h_common_branch + h_main_friction
    mu = mu_func(t_avg_k)
    rho = rho_func(t_avg_k)
    nu = mu / rho
    q_individual = np.full(n, q_main / n)
    h_total_group_a = calculate_total_group_a_head_loss(q_individual, nu)
    return 4 * h_total_group_a

def get_head_loss(q_main):
    """
    Convenience wrapper: compute main branch aggregate head loss from q_main [m^3/s].
    """
    return calculate_head_loss_only(q_main)

def run():
    """
    Run and print total head loss using lowest allowed flow rate.
    """
    head_loss = calculate_head_loss_only(flowrate_min)
    print(f"Total Group A head loss: {head_loss:.6f} m")

if __name__ == "__main__":
    run()
