import numpy as np
from heptane_itpl import rho_func, mu_func
from config import *


def calculate_head_loss_only(Q_main, T_avg_K=300):

    def calculate_branch_head_loss(Q_individual):
        """Calculate friction head loss for equal flow distribution"""
        n = len(Q_individual)
        H_branch = np.zeros(n)
        
        for i in range(n):
            main_pipe_loss = 0
            for j in range(i + 1):
                sum_Q_m = np.sum(Q_individual[j:])
                term = (32 * ν * L_a / (D**2 * g)) * (sum_Q_m / S_a) 
                main_pipe_loss += term
            
            main_pipe_loss *= 2
            
            branch_pipe_loss = (32 * ν * L_b / 
                               (d_H**2 * g)) * (Q_individual[i] / S_b)
            
            H_branch[i] = main_pipe_loss + branch_pipe_loss
        
        return H_branch

    def calculate_local_losses(Q_individual):
        """Calculate local head losses for equal flow distribution"""
        n = len(Q_individual)
        H_local = np.zeros(n)
        ξ_bend = 0.2
        
        for i in range(n):
            V_main_in = np.sum(Q_individual[i:]) / S_a
            V_branch = Q_individual[i] / S_b
            
            ξ_1_3 = 0.5 * (V_branch / V_main_in)**2 + 1
            ξ_3_1 = (V_branch / V_main_in)**2 - (V_branch / V_main_in)**2 + 0.5 * (1 - V_branch / V_main_in)
            
            H_bend = 24 * (ξ_bend / (2 * g)) * V_branch**2
            
            H_local[i] = (ξ_1_3 / (2 * g)) * V_main_in**2 + \
                         (ξ_3_1 / (2 * g)) * V_main_in**2 + \
                         H_bend
        
        return H_local

    def calculate_total_group_A_head_loss(Q_individual):
        """Calculate TOTAL head loss for entire Group A"""
        n = len(Q_individual)
        
        H_branch_friction = calculate_branch_head_loss(Q_individual)
        H_branch_local = calculate_local_losses(Q_individual)
        
        Q_total = np.sum(Q_individual)
        V_main = Q_total / S_a
        Re_main = V_main * D / ν
        λ_main = 64 / Re_main 
        
        H_main_friction = 2 * (λ_main * (L_c / D) * (V_main**2 / (2 * g)))
        
        H_common_branch = H_branch_friction[0] + H_branch_local[0]  # All branches equal for equal flow
        H_total_Group_A = H_common_branch + H_main_friction
        
        return H_total_Group_A


    
    # Assume equal flow distribution for each branch
    Q_individual = np.full(n, Q_main / n)
    
    # Calculate fluid properties
    μ = mu_func(T_avg_K)
    ρ = rho_func(T_avg_K)
    ν = μ / ρ
    
    # Calculate total head loss
    H_total_Group_A = calculate_total_group_A_head_loss(Q_individual)
    
    return 4 * H_total_Group_A # Assuming that both the cooling and the heating sections have the same head loss

# Simple function to get just the head loss
def get_head_loss(Q_main):
    """Get only the head loss value"""
    return calculate_head_loss_only(Q_main)

# Example usage
def run():
    head_loss = calculate_head_loss_only(flowrate_min)
    print(f"Total Group A head loss: {head_loss:.6f} m")
    
if __name__ == "__main__":
    run()