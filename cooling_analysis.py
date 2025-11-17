"""Compute frictional and local head losses for a multi-branch system."""

import numpy as np

from heptane_itpl import rho_func, mu_func
from config import *


# --------------------------------------------------------------------------- #
# Main head-loss calculation
# --------------------------------------------------------------------------- #

def calculate_head_loss_only(Q_main, T_avg_K=300):
    """Calculate total head loss for Group A branches and main pipe.

    Parameters
    ----------
    Q_main : float
        Total volumetric flow rate entering Group A.
    T_avg_K : float, optional
        Average fluid temperature in Kelvin. Default is 300 K.

    Returns
    -------
    float
        Total head loss (m) for both cooling and heating sections.
    """

    # ----------------------------------------------------------------------- #
    # Nested sub-functions
    # ----------------------------------------------------------------------- #

    def calculate_branch_head_loss(Q_individual, nu):
        """Compute frictional head loss for all branches (equal flow)."""
        n_branches = len(Q_individual)
        H_branch = np.zeros(n_branches)

        for i in range(n_branches):
            main_pipe_loss = 0.0

            # Friction in main pipe segments upstream of branch i
            for j in range(i + 1):
                sum_Q_m = np.sum(Q_individual[j:])
                term = (32 * nu * L_a / (D ** 2 * g)) * (sum_Q_m / S_a)
                main_pipe_loss += term

            main_pipe_loss *= 2  # Two identical sides

            # Branch pipe friction loss
            branch_pipe_loss = (
                32 * nu * L_b / (d_H ** 2 * g)
            ) * (Q_individual[i] / S_b)

            H_branch[i] = main_pipe_loss + branch_pipe_loss

        return H_branch

    def calculate_local_losses(Q_individual, nu):
        """Compute local losses (bends, junctions) for each branch."""
        n_branches = len(Q_individual)
        H_local = np.zeros(n_branches)

        ξ_bend = 0.2  # Bend loss coefficient

        for i in range(n_branches):
            V_main_in = np.sum(Q_individual[i:]) / S_a
            V_branch = Q_individual[i] / S_b

            # Junction loss coefficients
            ratio = V_branch / V_main_in
            ξ_1_3 = 0.5 * ratio ** 2 + 1
            ξ_3_1 = ratio ** 2 - ratio ** 2 + 0.5 * (1 - ratio)

            # Bend loss
            H_bend = 24 * (ξ_bend / (2 * g)) * V_branch ** 2

            H_local[i] = (
                (ξ_1_3 / (2 * g)) * V_main_in ** 2
                + (ξ_3_1 / (2 * g)) * V_main_in ** 2
                + H_bend
            )

        return H_local

    def calculate_total_group_A_head_loss(Q_individual, nu):
        """Compute full friction + local losses for entire Group A."""
        H_branch_friction = calculate_branch_head_loss(Q_individual, nu)
        H_branch_local = calculate_local_losses(Q_individual, nu)

        # Compute friction in the common main section
        Q_total = np.sum(Q_individual)
        V_main = Q_total / S_a
        Re_main = V_main * D / nu
        λ_main = 64 / Re_main  # Laminar flow

        H_main_friction = 2 * (
            λ_main * (L_c / D) * (V_main ** 2 / (2 * g))
        )

        # Branches are symmetrical, so take branch 0
        H_common_branch = H_branch_friction[0] + H_branch_local[0]

        return H_common_branch + H_main_friction

    # ----------------------------------------------------------------------- #
    # Fluid properties and flow distribution
    # ----------------------------------------------------------------------- #

    # Convert temperature to viscosity and density
    mu = mu_func(T_avg_K)
    rho = rho_func(T_avg_K)
    nu = mu / rho

    # Equal flow split into n branches
    Q_individual = np.full(n, Q_main / n)

    # ----------------------------------------------------------------------- #
    # Total head loss for both cooling & heating sections
    # ----------------------------------------------------------------------- #

    H_total_group_A = calculate_total_group_A_head_loss(Q_individual, nu)

    return 4 * H_total_group_A


# --------------------------------------------------------------------------- #
# Convenience wrapper
# --------------------------------------------------------------------------- #

def get_head_loss(Q_main):
    """Return total head loss for a given main flow rate."""
    return calculate_head_loss_only(Q_main)


# --------------------------------------------------------------------------- #
# Example runner
# --------------------------------------------------------------------------- #

def run():
    """Execute head-loss calculation using default flow rate."""
    head_loss = calculate_head_loss_only(flowrate_min)
    print(f"Total Group A head loss: {head_loss:.6f} m")


if __name__ == "__main__":
    run()
