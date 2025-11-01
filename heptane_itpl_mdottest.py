import numpy as np
import pandas as pd
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
from pathlib import Path

# --- Constants ---
N_HEPTANE_MOLAR_MASS = 0.100205  # kg/mol
DITTUS_BOELTER_EXPONENT = 0.4    # for heating
# NOTE: Ensure 'n heptane 2.txt' is in the same directory for this path to work
NIST_DATA_FILE = Path(__file__).parent / 'n heptane 2.txt'

# --- The fixed parameter D (Channel Diameter) is now defined locally in __main__ ---

# clean NIST data
def import_nist_txt(filepath: str | Path) -> pd.DataFrame:
    """
    Imports and cleans fluid property data from a NIST text file.
    """
    df = pd.read_csv(filepath, sep=r'\s+', skiprows=1, header=None)
    
    column_map = {
        0: 'T_K',
        1: 'Pressure',
        2: 'Density',
        8: 'Cp',
        12: 'Thermal_Cond',
        11: 'Viscosity'
    }

    df = df.rename(columns=column_map)[list(column_map.values())]
    
    # Unit conversions to SI
    df['Density'] = df['Density'] * 1000 * N_HEPTANE_MOLAR_MASS  # mol/L to kg/m^3
    df['Cp'] = df['Cp'] / N_HEPTANE_MOLAR_MASS                 # J/(mol·K) to J/(kg·K)
    df['Viscosity'] = df['Viscosity'] * 1e-6 
    
    return df

# import and assignment
nist_df = import_nist_txt(NIST_DATA_FILE)

T_data = nist_df['T_K'].values
rho_data = nist_df['Density'].values
Cp_data = nist_df['Cp'].values
lambda_data = nist_df['Thermal_Cond'].values
mu_data = nist_df['Viscosity'].values

# interplate

# Using fill_value=(first, last) for extrapolation
lambda_func = interp1d(T_data, lambda_data, kind='cubic', bounds_error=False, fill_value=(lambda_data[0], lambda_data[-1]))
Cp_func = interp1d(T_data, Cp_data, kind='cubic', bounds_error=False, fill_value=(Cp_data[0], Cp_data[-1]))
mu_func = interp1d(T_data, mu_data, kind='cubic', bounds_error=False, fill_value=(mu_data[0], mu_data[-1]))
rho_func = interp1d(T_data, rho_data, kind='cubic', bounds_error=False, fill_value=(rho_data[0], rho_data[-1]))

# create H(T)
def calculate_h(T: float | np.ndarray, m_dot: float, D: float) -> float | np.ndarray:
    """
    Calculates the heat transfer coefficient (h) for n-heptane at a given temperature and mass flow rate.
    """
    lam = lambda_func(T)
    Cp = Cp_func(T)
    mu = mu_func(T)
    
    # C_RE is calculated dynamically based on the input m_dot
    C_RE = 4 * m_dot / (np.pi * D) 
    
    Re = C_RE / mu
    Pr = (Cp * mu) / lam
    # Dittus-Boelter for Nu number
    Nu = 0.023 * (Re**0.8) * (Pr**DITTUS_BOELTER_EXPONENT)
    h = (Nu * lam) / D
    
    return h

# AI plotting (ommitted from script calling)
if __name__ == '__main__':
    
    # Define local variables for plotting/testing ONLY (These are NOT used by flow_solver.py)
    M_DOT_PLOT = 0.05    # Mass flow rate [kg/s] for plot only
    D_PLOT = 0.005       # Channel diameter [m] for plot only

    # Calculate C_RE locally for the plotting context
    C_RE_LOCAL = 4 * M_DOT_PLOT / (np.pi * D_PLOT) 
    
    print(f"System Constant C_Re = {C_RE_LOCAL:.2f} kg/(m·s)\n")
    test_T = np.linspace(T_data.min(), T_data.max(), 100)
    
    # FIX: Pass the local M_DOT_PLOT and D_PLOT to the function
    h_values = calculate_h(test_T, M_DOT_PLOT, D_PLOT) # Vectorized call
    mu_values = mu_func(test_T)
    Cp_values = Cp_func(test_T)
    lambda_values = lambda_func(test_T)
    
    plt.style.use('seaborn-v0_8-darkgrid')
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10))
    
    # --- Subplot 1: Fluid Properties ---
    ax1.plot(test_T, mu_values, label=r'$\mu$ (Viscosity) [Pa·s]', color='blue')
    ax1_twin = ax1.twinx()
    ax1_twin.plot(test_T, Cp_values, label=r'$C_p$ (Specific Heat) [J/(kg·K)]', color='green', linestyle='--')
    ax1_twin.plot(test_T, lambda_values, label=r'$\lambda$ (Thermal Conductivity) [W/(m·K)]', color='red', linestyle=':')
    
    ax1.set_title('Interpolated Fluid Properties of n-Heptane vs. Temperature')
    ax1.set_xlabel('Temperature (T) [K]')
    ax1.set_ylabel(r'Viscosity ($\mu$) [Pa$\cdot$s]', color='blue')
    
    ax1_twin.set_ylabel(r'$C_p$ [J/(kg·K)] and $\lambda$ [W/(m·K)]')
    
    lines, labels = ax1.get_legend_handles_labels()
    lines2, labels2 = ax1_twin.get_legend_handles_labels()
    ax1.legend(lines + lines2, labels + labels2, loc='upper right')
    ax1.grid(True, linestyle='--', alpha=0.6)
    
    # --- Subplot 2: Heat Transfer Coefficient (h) ---
    T_test = 330.0
    # FIX: Pass the local M_DOT_PLOT and D_PLOT to the function
    h_test = calculate_h(T_test, M_DOT_PLOT, D_PLOT)
    
    # FIX: Use M_DOT_PLOT and D_PLOT in the label
    ax2.plot(test_T, h_values, label=f'h(T) at $\dot{{m}}$={M_DOT_PLOT} kg/s, D={D_PLOT*1000:.0f} mm', color='teal')
    ax2.plot(T_test, h_test, 'o', color='red', label=f'Test Point ({T_test} K, {h_test:.0f} W/m²K)')
    
    ax2.set_title('Heat Transfer Coefficient vs. Temperature for Liquid n-Heptane')
    ax2.set_xlabel('Temperature (T) [K]')
    ax2.set_ylabel('Heat Transfer Coefficient (h) [W/(m²·K)]')
    ax2.grid(True, linestyle='--', alpha=0.6)
    ax2.legend()
    
    plt.tight_layout()
    plt.show()

    # --- Print Example Calculation ---
    mu_test = mu_func(T_test)
    Re_test = C_RE_LOCAL / mu_test
    print(f"\n--- Example Calculation at T = {T_test:.1f} K ---")
    print(f"Dynamic Viscosity (mu): {mu_test:.6f} Pa·s")
    print(f"Reynolds Number (Re): {Re_test:.0f}")
    print(f"Heat Transfer Coefficient (h): {h_test:.2f} W/(m^2·K)")
