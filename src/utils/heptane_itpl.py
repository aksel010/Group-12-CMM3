"""
Heptane thermophysical property interpolation utilities (from NIST).

Parses tabular NIST data and provides SI-physical property interpolators for use in broader battery/cooling calculations.
All functions refactored for PEP8 docstring and consistent block comments.
"""
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.interpolate import interp1d
from src.config import *

# --------------------------------------------------------------------------- #
# Load NIST thermophysical property data
# --------------------------------------------------------------------------- #
NIST_data_FILE = Path(__file__).parent.parent.parent / "data" / "raw" / "n heptane 2.txt"

def import_nist_txt(filepath: str | Path) -> pd.DataFrame:
    """
    Import and convert SI properties from whitespace-delimited NIST file.

    Args:
        filepath (str | Path): File location of the NIST property file.
    Returns:
        pd.DataFrame: DataFrame with property columns as SI units.
    """
    derivative_func = pd.read_csv(filepath, sep=r"\s+", skiprows=1, header=None)
    column_map = {
        0: "T_K",
        1: "Pressure",
        2: "Density",
        8: "Cp",
        12: "Thermal_Cond",
        11: "Viscosity",
    }
    derivative_func = derivative_func.rename(columns=column_map)[list(column_map.values())]
    derivative_func["Density"] *= 1000 * N_HEPTANE_MOLAR_MASS
    derivative_func["Cp"] /= N_HEPTANE_MOLAR_MASS
    derivative_func["Viscosity"] *= 1e-6
    return derivative_func

# --------------------------------------------------------------------------- #
# Extract and interpolate all properties
# --------------------------------------------------------------------------- #
nist_derivative_func = import_nist_txt(NIST_data_FILE)
t_data = nist_derivative_func["T_K"].values
rho_data = nist_derivative_func["Density"].values
cp_data = nist_derivative_func["Cp"].values
lambda_data = nist_derivative_func["Thermal_Cond"].values
mu_data = nist_derivative_func["Viscosity"].values

lambda_func = interp1d(t_data, lambda_data, kind="cubic", bounds_error=False, fill_value=(lambda_data[0], lambda_data[-1]))
cp_func = interp1d(t_data, cp_data, kind="cubic", bounds_error=False, fill_value=(cp_data[0], cp_data[-1]))
mu_func = interp1d(t_data, mu_data, kind="cubic", bounds_error=False, fill_value=(mu_data[0], mu_data[-1]))
rho_func = interp1d(t_data, rho_data, kind="cubic", bounds_error=False, fill_value=(rho_data[0], rho_data[-1]))

def calculate_h(t: float | np.ndarray) -> float | np.ndarray:
    """
    Calculate the convective heat-transfer coefficient with the Dittus–Boelter equation.

    Args:
        t (float | np.ndarray): Temperatures [K] for evaluation.
    Returns:
        float | np.ndarray: Heat-transfer coefficient [W/m²·K].
    """
    lam = lambda_func(t)
    cp = cp_func(t)
    mu = mu_func(t)
    re = C_RE / mu
    pr = (cp * mu) / lam
    nu = 0.023 * (re ** 0.8) * (pr ** DITTUS_BOELTER_EXPONENT)
    return (nu * lam) / d

def run():
    """
    Generate property arrays as vectors for test temperature sweeps.

    Returns:
        dict: (Arrays for mu, rho, Cp, lambda, h) over t_data span.
    """
    test_t_data = np.linspace(t_data.min(), t_data.max(), 100)
    h_values = calculate_h(test_t_data)
    mu_values = mu_func(test_t_data)
    cp_values = cp_func(test_t_data)
    lambda_values = lambda_func(test_t_data)
    rho_values = rho_func(test_t_data)
    return {
        'mu': (test_t_data, mu_values),
        'rho': (test_t_data, rho_values),
        'Cp': (test_t_data, cp_values),
        'lambda': (test_t_data, lambda_values),
        'h': (test_t_data, h_values)
    }

if __name__ == "__main__":
    run()
