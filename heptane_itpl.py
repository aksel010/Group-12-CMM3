from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.interpolate import interp1d

from config import *

# --------------------------------------------------------------------------- #
# Load NIST thermophysical property data
# --------------------------------------------------------------------------- #

NIST_DATA_FILE = Path(__file__).parent / "n heptane 2.txt"


def import_nist_txt(filepath: str | Path) -> pd.DataFrame:
    """Import and clean NIST property data from a whitespace-delimited file.

    Extracted variables:
        - Temperature (K)
        - Density (kg/m³)
        - Heat capacity Cp (J/kg·K)
        - Thermal conductivity (W/m·K)
        - Dynamic viscosity (Pa·s)

    Parameters
    ----------
    filepath : str or pathlib.Path
        Path to the NIST data file.

    Returns
    -------
    pandas.DataFrame
        Cleaned DataFrame with SI-converted thermophysical properties.
    """
    df = pd.read_csv(filepath, sep=r"\s+", skiprows=1, header=None)

    column_map = {
        0: "T_K",
        1: "Pressure",
        2: "Density",
        8: "Cp",
        12: "Thermal_Cond",
        11: "Viscosity",
    }

    df = df.rename(columns=column_map)[list(column_map.values())]

    # Unit conversions
    df["Density"] *= 1000 * N_HEPTANE_MOLAR_MASS  # mol/L → kg/m³
    df["Cp"] /= N_HEPTANE_MOLAR_MASS              # J/mol·K → J/kg·K
    df["Viscosity"] *= 1e-6                       # μPa·s → Pa·s

    return df


# --------------------------------------------------------------------------- #
# Extract property arrays
# --------------------------------------------------------------------------- #

nist_df = import_nist_txt(NIST_DATA_FILE)

T_data = nist_df["T_K"].values
rho_data = nist_df["Density"].values
Cp_data = nist_df["Cp"].values
lambda_data = nist_df["Thermal_Cond"].values
mu_data = nist_df["Viscosity"].values

# --------------------------------------------------------------------------- #
# Create cubic interpolation functions
# --------------------------------------------------------------------------- #

lambda_func = interp1d(
    T_data,
    lambda_data,
    kind="cubic",
    bounds_error=False,
    fill_value=(lambda_data[0], lambda_data[-1]),
)

Cp_func = interp1d(
    T_data,
    Cp_data,
    kind="cubic",
    bounds_error=False,
    fill_value=(Cp_data[0], Cp_data[-1]),
)

mu_func = interp1d(
    T_data,
    mu_data,
    kind="cubic",
    bounds_error=False,
    fill_value=(mu_data[0], mu_data[-1]),
)

rho_func = interp1d(
    T_data,
    rho_data,
    kind="cubic",
    bounds_error=False,
    fill_value=(rho_data[0], rho_data[-1]),
)


def calculate_h(T: float | np.ndarray) -> float | np.ndarray:
    """Compute the convective heat-transfer coefficient using the
    Dittus–Boelter equation.

    Steps
    -----
    1. Interpolate μ, Cp, λ at temperature T.
    2. Compute Reynolds and Prandtl numbers.
    3. Compute the Nusselt number.
    4. Convert to h = (Nu · λ) / D.

    Parameters
    ----------
    T : float or numpy.ndarray
        Temperature(s) at which to evaluate the heat-transfer coefficient.

    Returns
    -------
    float or numpy.ndarray
        Heat-transfer coefficient h(T) in W/m²·K.
    """
    lam = lambda_func(T)
    Cp = Cp_func(T)
    mu = mu_func(T)

    Re = C_RE / mu
    Pr = (Cp * mu) / lam

    Nu = 0.023 * (Re ** 0.8) * (Pr ** DITTUS_BOELTER_EXPONENT)

    return (Nu * lam) / D

#plotting
def run():
    
    test_T = np.linspace(T_data.min(), T_data.max(), 100)
    
    h_values = calculate_h(test_T) # Vectorized call
    mu_values = mu_func(test_T)
    Cp_values = Cp_func(test_T)
    lambda_values = lambda_func(test_T)
    rho_values = rho_func(test_T)  # ← ADD THIS LINE!
    
    # --- Subplot 2: Heat Transfer Coefficient (h) ---
    T_test = 330.0
    h_test = calculate_h(T_test)
    
    return {
            'mu': (test_T, mu_values),
            'rho': (test_T, rho_values),
            'Cp': (test_T, Cp_values),
            'lambda': (test_T, lambda_values),
            'h': (test_T, h_values)
        }

if __name__ == "__main__":
    run()