"""Compare experimental temperature–time data with an RK4 numerical model."""

import matplotlib.pyplot as plt
import pandas as pd

# --------------------------------------------------------------------------- #
# Load and prepare datasets
# --------------------------------------------------------------------------- #

# Import experimental and numerical data
df_exp = pd.read_csv("TempTimeCharging_LIU2.csv")
df_num = pd.read_csv("RK4 solution.csv")

# Extract experimental time and temperature
time_exp = df_exp["Time"]
temp_exp = df_exp["Temp"]

# Extract numerical time and temperature
# Convert time from seconds → minutes and temperature from K → °C
time_num = df_num["Time (s)"] / 60
temp_num = df_num["Temperature (K)"] - 273.15

# --------------------------------------------------------------------------- #
# Plot experimental and numerical temperature profiles
# --------------------------------------------------------------------------- #

plt.figure(figsize=(10, 6))

plt.plot(
    time_exp,
    temp_exp,
    label="Liu et al.",
    color="red",
    marker="o",
    linestyle="",
)

plt.plot(
    time_num,
    temp_num,
    label="RK4 Model",
    color="blue",
    linestyle="-",
)

# --------------------------------------------------------------------------- #
# Formatting and display
# --------------------------------------------------------------------------- #

plt.xlabel("Time (minutes)")
plt.ylabel("Temperature (°C)")
plt.title("Comparison of Model and Experimental Data")

plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()
