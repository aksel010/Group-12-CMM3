import numpy as np
import matplotlib.pyplot as plt
from ODE import Tb, dTb_dt
from config import q_b, m_b, C_b, T_in, M_DOT, T_b_max

valid_runs = []

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
    if T_i[-1] < T_b_max:
        valid_runs.append(I_0)

I_max = max(valid_runs) if valid_runs else None
t_opt = q_b / I_max if I_max else None
t_opt, T_opt = Tb(dTb_dt, t_opt, I_params(I_max))
print ("Valid current runs (A) that exceed max battery temperature:", valid_runs)
plt.plot(t_opt, T_opt, label=f'Optimal I = {I_max} A', color='red', linewidth=2)
plt.xlabel('t (s)')
plt.ylabel('Temperature of the Battery $T_b$ (K)')
plt.title('Optimal Current Profile to Avoid Overheating')   
plt.legend()
plt.show()  
    
    