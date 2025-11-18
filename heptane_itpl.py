"""
Heptane thermophysical property interpolation utilities (from NIST).

Parses tabular NIST data and provides SI-physical property interpolators for use in broader battery/cooling calculations.
All functions refactored for PEP8 docstring and consistent block comments.
"""
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
    """
    Import and convert SI properties from whitespace-delimited NIST file.

    Args:
        filepath (str | Path): File location of the NIST property file.
    Returns:
        pd.DataFrame: DataFrame with property columns as SI units.
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
    df["Density"] *= 1000 * N_HEPTANE_MOLAR_MASS
    df["Cp"] /= N_HEPTANE_MOLAR_MASS
    df["Viscosity"] *= 1e-6
    return df

# --------------------------------------------------------------------------- #
# Extract and interpolate all properties
# --------------------------------------------------------------------------- #
nist_df = import_nist_txt(NIST_DATA_FILE)
T_data = nist_df["T_K"].values
rho_data = nist_df["Density"].values
Cp_data = nist_df["Cp"].values
lambda_data = nist_df["Thermal_Cond"].values
mu_data = nist_df["Viscosity"].values

lambda_func = interp1d(T_data, lambda_data, kind="cubic", bounds_error=False, fill_value=(lambda_data[0], lambda_data[-1]))
Cp_func = interp1d(T_data, Cp_data, kind="cubic", bounds_error=False, fill_value=(Cp_data[0], Cp_data[-1]))
mu_func = interp1d(T_data, mu_data, kind="cubic", bounds_error=False, fill_value=(mu_data[0], mu_data[-1]))
rho_func = interp1d(T_data, rho_data, kind="cubic", bounds_error=False, fill_value=(rho_data[0], rho_data[-1]))

def calculate_h(T: float | np.ndarray) -> float | np.ndarray:
    """
    Calculate the convective heat-transfer coefficient with the Dittus–Boelter equation.

    Args:
        T (float | np.ndarray): Temperatures [K] for evaluation.
    Returns:
        float | np.ndarray: Heat-transfer coefficient [W/m²·K].
    """
    lam = lambda_func(T)
    Cp = Cp_func(T)
    mu = mu_func(T)
    Re = C_RE / mu
    Pr = (Cp * mu) / lam
    Nu = 0.023 * (Re ** 0.8) * (Pr ** DITTUS_BOELTER_EXPONENT)
    return (Nu * lam) / D

def run():
    """
    Generate property arrays as vectors for test temperature sweeps.

    Returns:
        dict: (Arrays for mu, rho, Cp, lambda, h) over T_data span.
    """
    test_T = np.linspace(T_data.min(), T_data.max(), 100)
    h_values = calculate_h(test_T)
    mu_values = mu_func(test_T)
    Cp_values = Cp_func(test_T)
    lambda_values = lambda_func(test_T)
    rho_values = rho_func(test_T)
    return {
        'mu': (test_T, mu_values),
        'rho': (test_T, rho_values),
        'Cp': (test_T, Cp_values),
        'lambda': (test_T, lambda_values),
        'h': (test_T, h_values)
    }

if __name__ == "__main__":
    run()
