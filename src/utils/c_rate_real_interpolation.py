"""
Plot measured (experimental) and model-predicted relationships between C-rate and charging time.

Loads real data, sorts and plots, then overlays the analytic model for direct visual comparison.
PEP8-compliant docstring, comments, clear computation and plot separation.
"""
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from src.config import *

# --------------------------------------------------------------------------- #
# Data load and preparation
# --------------------------------------------------------------------------- #
DATA_FILE = Path("Real_Data_C_Rate.csv")
df = pd.read_csv(DATA_FILE)
df["charging_time_hours"] = df["charging_time_min"] / 60
# Sorted by C-rate for smooth plotting
df_sorted = df.sort_values("C-rate")

# --------------------------------------------------------------------------- #
# Plot: Measured charging time vs C-rate
# --------------------------------------------------------------------------- #
plt.figure(figsize=(10, 6))
plt.plot(
    df_sorted["charging_time_hours"],
    df_sorted["C-rate"],
    "bo-", markersize=4, linewidth=1, label="Real-world Data"
)

# --------------------------------------------------------------------------- #
# Model-predicted C-rate and charging time curves
# --------------------------------------------------------------------------- #
current_range = np.linspace(0.35, 7.0, 40)
c_rates = []
t_totals = []
for current in current_range:
    c_rate_val = current / capacity_battery
    t_total_val = q_b / current / 3600  # (s → h)
    c_rates.append(c_rate_val)
    t_totals.append(t_total_val)

# Overlay model
plt.plot(
    t_totals,
    c_rates,
    "r-", linewidth=2, label="Model"
)
plt.xlabel("Charging Time (hours)")
plt.ylabel("C-rate")
plt.title("C-rate vs Charging Time")
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()
plt.show()
