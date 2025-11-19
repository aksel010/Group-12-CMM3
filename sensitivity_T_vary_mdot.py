import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from math import ceil
from src.config import *
from src.models.battery_temperature_ode import d_tb_dt, get_tb
from src.models.mass_flowrate import get_steady_state_values 

M_flow_ss, T_c_avg_nominal_coolant, h_ss_nominal = get_steady_state_values()
M_flow_nominal = M_flow_ss 

M_perturb_range = np.linspace(0.9 * M_flow_nominal, 1.1 * M_flow_nominal, 10)
results = []
stepsize = 0.2

params_nominal = (m_cell, c_b, current_store[-1], r_b, a_s, t_in, M_flow_nominal)
_, T_nominal_arr = get_tb(d_tb_dt, params_nominal, stepsize)
T_final_nominal = T_nominal_arr[-1]

results.append({
    'M_flow': M_flow_nominal,
    'T_final': T_final_nominal,
    'dT': 0,
    'dM': 0
})

for M_flow in M_perturb_range:

    params_perturbed = (m_cell, c_b, current_store[-1], r_b, a_s, t_in, M_flow)
    _, T_perturbed_arr = get_tb(d_tb_dt, params_perturbed, stepsize)
    T_final_perturbed = T_perturbed_arr[-1]

    delta_T = T_final_perturbed - T_final_nominal
    delta_M = M_flow - M_flow_nominal

    results.append({
        'M_flow': M_flow,
        'T_final': T_final_perturbed,
        'dT': delta_T,
        'dM': delta_M
    })

df_results = pd.DataFrame(results)

df_results['S_dimless'] = (df_results['dT'] / T_final_nominal) / (df_results['dM'] / M_flow_nominal)
print(params_perturbed)
plt.figure(figsize=(10, 5))
plt.plot(df_results['M_flow'] * 1000, df_results['T_final'], 'bo-', label='$T_{b,final}$ vs $\dot{M}$')
plt.axhline(y=T_final_nominal, color='r', linestyle='--', label='$T_{b,final, \text{nominal}}$')
plt.xlabel('Mass Flow Rate, $\dot{M}$ (g/s)')
plt.ylabel('Final Battery Temperature, $T_{b,final}$ (K)')
plt.title('Final Temperature Sensitivity to Mass Flow Rate')
plt.grid(True, alpha=0.3)
plt.legend()
plt.show()
