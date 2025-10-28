import numpy as np
import matplotlib.pyplot as plt
from ODE import Tb, dTb_dt
from config import q_b, m_b, C_b, T_in, M_DOT, T_b_max

I_runs = []
delta_T = []

def I_params(I):
    return (
        m_b,      # m [kg]
        C_b,      # cp_b [J/(kg·K)]
        I,        # I [A]
        0.1,      # R [Ω]
        0.01,     # A_s [m²]
        T_in,     # T_c_in [K]
        M_DOT     # m_dot_c [kg/s]
    )

for idx, i in enumerate(range(5, 100, 5)):
    I_0 = i
    t_total = q_b / I_0  # total time [s]
    t_i, T_i = Tb(dTb_dt, t_total, I_params(I_0))
    print(T_i[-1])
    I_runs.append(I_0)
    delta_T.append(T_i[-1] - T_b_max)


plt.scatter(I_runs, delta_T)
plt.xlabel('Current (A)')
plt.ylabel('Delta T (K)')       
plt.axhline(0, color='red', linestyle='--')
plt.title('Delta T vs Current')
plt.show()
    
    