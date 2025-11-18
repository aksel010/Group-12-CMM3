"""
Cooling channel network head-loss model for a multi-branch battery system.
Computes frictional and local losses, exposing convenient evaluation API.
All modules, wrappers, and runners fully docstring-annotated and commented.
"""
import numpy as np
from heptane_itpl import rho_func, mu_func
from config import *

def calculate_head_loss_only(Q_main, T_avg_K=300):
    """
    Compute total cooling circuit (Group A) head loss (m) for specified main branch flow Q_main at average temperature T_avg_K.

    Args:
        Q_main (float): Main volumetric flow [m^3/s].
        T_avg_K (float): Average fluid temperature [K] (default: 300).
    Returns:
        float: Overall head loss [m] summed over all relevant branches.
    """
    def calculate_branch_head_loss(Q_individual, nu):
        """
        Frictional head loss for all identical branches.
        Args:
            Q_individual (ndarray): Individual branch flows [m^3/s].
            nu (float): Kinematic viscosity of fluid [m^2/s].
        Returns:
            ndarray: Head loss for each branch [m].
        """
        n_branches = len(Q_individual)
        H_branch = np.zeros(n_branches)
        for i in range(n_branches):
            main_pipe_loss = 0.0
            for j in range(i + 1):
                sum_Q_m = np.sum(Q_individual[j:])
                term = (32 * nu * L_a / (D ** 2 * g)) * (sum_Q_m / S_a)
                main_pipe_loss += term
            main_pipe_loss *= 2
            branch_pipe_loss = (
                32 * nu * L_b / (d_H ** 2 * g)
            ) * (Q_individual[i] / S_b)
            H_branch[i] = main_pipe_loss + branch_pipe_loss
        return H_branch
    def calculate_local_losses(Q_individual, nu):
        """
        Local head losses due to bends and junctions.
        Args:
            Q_individual (ndarray): Branch flows.
            nu (float): Kinematic viscosity [m^2/s].
        Returns:
            ndarray: Local loss per branch [m].
        """
        n_branches = len(Q_individual)
        H_local = np.zeros(n_branches)
        ξ_bend = 0.2
        for i in range(n_branches):
            V_main_in = np.sum(Q_individual[i:]) / S_a
            V_branch = Q_individual[i] / S_b
            ratio = V_branch / V_main_in
            ξ_1_3 = 0.5 * ratio ** 2 + 1
            ξ_3_1 = ratio ** 2 - ratio ** 2 + 0.5 * (1 - ratio)
            H_bend = 24 * (ξ_bend / (2 * g)) * V_branch ** 2
            H_local[i] = (
                (ξ_1_3 / (2 * g)) * V_main_in ** 2
                + (ξ_3_1 / (2 * g)) * V_main_in ** 2
                + H_bend
            )
        return H_local
    def calculate_total_group_A_head_loss(Q_individual, nu):
        """
        Full head loss for all Group A branches including main section.
        Args:
            Q_individual (ndarray): Per-branch flows.
            nu (float): Kinematic viscosity [m^2/s].
        Returns:
            float: Total head loss [m].
        """
        H_branch_friction = calculate_branch_head_loss(Q_individual, nu)
        H_branch_local = calculate_local_losses(Q_individual, nu)
        Q_total = np.sum(Q_individual)
        V_main = Q_total / S_a
        Re_main = V_main * D / nu
        λ_main = 64 / Re_main  # Laminar assumption
        H_main_friction = 2 * (
            λ_main * (L_c / D) * (V_main ** 2 / (2 * g))
        )
        H_common_branch = H_branch_friction[0] + H_branch_local[0]
        return H_common_branch + H_main_friction
    mu = mu_func(T_avg_K)
    rho = rho_func(T_avg_K)
    nu = mu / rho
    Q_individual = np.full(n, Q_main / n)
    H_total_group_A = calculate_total_group_A_head_loss(Q_individual, nu)
    return 4 * H_total_group_A

def get_head_loss(Q_main):
    """
    Convenience wrapper: compute main branch aggregate head loss from Q_main [m^3/s].
    """
    return calculate_head_loss_only(Q_main)

def run():
    """
    Run and print total head loss using lowest allowed flow rate.
    """
    head_loss = calculate_head_loss_only(flowrate_min)
    print(f"Total Group A head loss: {head_loss:.6f} m")

if __name__ == "__main__":
    run()
