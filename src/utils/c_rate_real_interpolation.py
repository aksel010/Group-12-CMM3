"""Plot measured and modelled relationships between C-rate and charging time."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from config import *

# --------------------------------------------------------------------------- #
# Load and prepare the dataset
# --------------------------------------------------------------------------- #

DATA_FILE = Path("Real_Data_C_Rate.csv")

# Import experimental data and convert charging time from minutes to hours
df = pd.read_csv(DATA_FILE)
df["charging_time_hours"] = df["charging_time_min"] / 60

# Sort values for smoother plotting
df_sorted = df.sort_values("C-rate")

# --------------------------------------------------------------------------- #
# Create the figure for plotting
# --------------------------------------------------------------------------- #

plt.figure(figsize=(10, 6))

# Plot measured charging time vs. C-rate
plt.plot(
    df_sorted["charging_time_hours"],
    df_sorted["C-rate"],
    "bo-",
    markersize=4,
    linewidth=1,
    label="Real-world Data",
)

# --------------------------------------------------------------------------- #
# Compute model-predicted C-rates and charging times
# --------------------------------------------------------------------------- #

# Generate a range of charging currents (A)
current_range = np.linspace(0.35, 7.0, 40)

c_rates = []
t_totals = []

# Loop through current values and evaluate the model
for current in current_range:
    c_rate_val = current / Capacity_battery
    t_total_val = q_b / current / 3600  # Convert seconds → hours

    c_rates.append(c_rate_val)
    t_totals.append(t_total_val)

# --------------------------------------------------------------------------- #
# Plot the modelled relationship
# --------------------------------------------------------------------------- #

plt.plot(
    t_totals,
    c_rates,
    "r-",
    linewidth=2,
    label="Model",
)

# --------------------------------------------------------------------------- #
# Formatting and display options
# --------------------------------------------------------------------------- #

plt.xlabel("Charging Time (hours)")
plt.ylabel("C-rate")
plt.title("C-rate vs Charging Time")

plt.grid(True, alpha=0.3)
plt.legend()

plt.tight_layout()
plt.show()
