from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.interpolate import interp1d
from config import *

NIST_DATA_FILE = Path(__file__).parent / "n heptane 2.txt"

def import_nist_txt(filepath: str | Path) -> pd.DataFrame:
    """Import NIST property data from whitespace-delimited file."""
    df = pd.read_csv(filepath, sep=r"\s+", skiprows=1, header=None)
    column_map = {
        0: "temp_k",
        1: "pressure",
        2: "density",
        8: "cp",
        12: "thermal_cond",
        11: "viscosity",
    }
    df = df.rename(columns=column_map)[list(column_map.values())]
    df["density"] *= 1000 * N_HEPTANE_MOLAR_MASS
    df["cp"] /= N_HEPTANE_MOLAR_MASS
    df["viscosity"] *= 1e-6
    return df

nist_df = import_nist_txt(NIST_DATA_FILE)

temp_data = nist_df["temp_k"].values
rho_data = nist_df["density"].values
cp_data = nist_df["cp"].values
lambda_data = nist_df["thermal_cond"].values
mu_data = nist_df["viscosity"].values

lambda_func = interp1d(
    temp_data,
    lambda_data,
    kind="cubic",
    bounds_error=False,
    fill_value=(lambda_data[0], lambda_data[-1]),
)
cp_func = interp1d(
    temp_data,
    cp_data,
    kind="cubic",
    bounds_error=False,
    fill_value=(cp_data[0], cp_data[-1]),
)
mu_func = interp1d(
    temp_data,
    mu_data,
    kind="cubic",
    bounds_error=False,
    fill_value=(mu_data[0], mu_data[-1]),
)
rho_func = interp1d(
    temp_data,
    rho_data,
    kind="cubic",
    bounds_error=False,
    fill_value=(rho_data[0], rho_data[-1]),
)

def calculate_h(temp: float | np.ndarray) -> float | np.ndarray:
    """Compute convective heat-transfer coefficient using Dittus-Boelter."""
    lam = lambda_func(temp)
    cp = cp_func(temp)
    mu = mu_func(temp)
    re = C_RE / mu
    pr = (cp * mu) / lam
    nu = 0.023 * (re ** 0.8) * (pr ** DITTUS_BOELTER_EXPONENT)
    return (nu * lam) / D

def run():
    test_temps = np.linspace(temp_data.min(), temp_data.max(), 100)
    h_values = calculate_h(test_temps)
    mu_values = mu_func(test_temps)
    cp_values = cp_func(test_temps)
    lambda_values = lambda_func(test_temps)
    rho_values = rho_func(test_temps)
    t_test = 330.0
    h_test = calculate_h(t_test)
    return {
        'mu': (test_temps, mu_values),
        'rho': (test_temps, rho_values),
        'cp': (test_temps, cp_values),
        'lambda': (test_temps, lambda_values),
        'h': (test_temps, h_values)
    }

if __name__ == "__main__":
    run()
