import numpy as np
from scipy.optimize import fsolve
from heptane_itpl import rho_func, mu_func
from config import *


def calculate_head_loss_only(Q_main_L_min, T_avg_K=300, D_main=D, n, 
                           w_branch , h_branch, L_a, L_b, L_c):
    """
    Calculate head loss for a given flow rate WITHOUT internal fsolve
    This assumes equal flow distribution for simplicity
    """
    
    class SystemParams:
        def __init__(self):
            self.n = n
            self.D = D_main
            self.w_branch = w_branch
            self.h_branch = h_branch
            self.L_a = L_a
            self.L_b = L_b
            self.L_c = L_c
            self.g = 9.81
            
            self.S_a = np.pi * (self.D/2)**2
            self.S_b = self.w_branch * self.h_branch
            self.d_H = 2 * self.w_branch * self.h_branch / (self.w_branch + self.h_branch)

    def calculate_branch_head_loss(Q_individual, params, ν):
        """Calculate friction head loss for equal flow distribution"""
        n = len(Q_individual)
        H_branch = np.zeros(n)
        
        for i in range(n):
            main_pipe_loss = 0
            for j in range(i + 1):
                sum_Q_m = np.sum(Q_individual[j:])
                term = (32 * ν * params.L_a / (params.D**2 * params.g)) * (sum_Q_m / params.S_a)
                main_pipe_loss += term
            
            main_pipe_loss *= 2
            
            branch_pipe_loss = (32 * ν * params.L_b / 
                               (params.d_H**2 * params.g)) * (Q_individual[i] / params.S_b)
            
            H_branch[i] = main_pipe_loss + branch_pipe_loss
        
        return H_branch

    def calculate_local_losses(Q_individual, params, ν):
        """Calculate local head losses for equal flow distribution"""
        n = len(Q_individual)
        H_local = np.zeros(n)
        ξ_bend = 0.2
        
        for i in range(n):
            V_main_in = np.sum(Q_individual[i:]) / params.S_a
            V_branch = Q_individual[i] / params.S_b
            
            ξ_1_3 = 0.5 * (V_branch / V_main_in)**2 + 1
            ξ_3_1 = (V_branch / V_main_in)**2 - (V_branch / V_main_in)**2 + 0.5 * (1 - V_branch / V_main_in)
            
            H_bend = 24 * (ξ_bend / (2 * params.g)) * V_branch**2
            
            H_local[i] = (ξ_1_3 / (2 * params.g)) * V_main_in**2 + \
                         (ξ_3_1 / (2 * params.g)) * V_main_in**2 + \
                         H_bend
        
        return H_local

    def calculate_total_group_A_head_loss(Q_individual, params, ν):
        """Calculate TOTAL head loss for entire Group A"""
        n = len(Q_individual)
        
        H_branch_friction = calculate_branch_head_loss(Q_individual, params, ν)
        H_branch_local = calculate_local_losses(Q_individual, params, ν)
        
        Q_total = np.sum(Q_individual)
        V_main = Q_total / params.S_a
        Re_main = V_main * params.D / ν
        λ_main = 64 / Re_main if Re_main < 2300 else 0.316 / Re_main**0.25
        
        H_main_friction = 2 * (λ_main * (params.L_c / params.D) * (V_main**2 / (2 * params.g)))
        
        H_common_branch = H_branch_friction[0] + H_branch_local[0]  # All branches equal for equal flow
        H_total_Group_A = H_common_branch + H_main_friction
        
        return H_total_Group_A

    # Main calculation
    params = SystemParams()
    Q_main = Q_main_L_min / 60000  # Convert to m³/s
    
    # Assume equal flow distribution for each branch
    Q_individual = np.full(params.n, Q_main / params.n)
    
    # Calculate fluid properties
    μ = mu_func(T_avg_K)
    ρ = rho_func(T_avg_K)
    ν = μ / ρ
    
    # Calculate total head loss
    H_total_Group_A = calculate_total_group_A_head_loss(Q_individual, params, ν)
    
    return 2* H_total_Group_A

# Simple function to get just the head loss
def get_head_loss(Q_main_L_min):
    """Get only the head loss value"""
    return calculate_head_loss_only(Q_main_L_min)

# Example usage
if __name__ == "__main__":
    head_loss = calculate_head_loss_only(5.0)
    print(f"Total Group A head loss: {head_loss:.6f} m")