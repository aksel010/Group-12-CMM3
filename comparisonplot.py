"""
Visual comparison of experimental and RK4 model simulation temperature–time profiles for battery charging.
Loads and overlays both datasets for clear visual validation of model accuracy. Full PEP8 docstring and block comment upgrade.
"""
import matplotlib.pyplot as plt
import pandas as pd
# --------------------------------------------------------------------------- #
# Data loading: experimental and model output
# --------------------------------------------------------------------------- #
df_exp = pd.read_csv("TempTimeCharging_LIU2.csv")
df_num = pd.read_csv("RK4 solution.csv")
time_exp = df_exp["Time"]
temp_exp = df_exp["Temp"]
time_num = df_num["Time (s)"] / 60
temp_num = df_num["Temperature (K)"] - 273.15
# --------------------------------------------------------------------------- #
# Overlay measured and computed profiles
# --------------------------------------------------------------------------- #
plt.figure(figsize=(10, 6))
plt.plot(
    time_exp, temp_exp, label="Liu et al.", color="red", marker="o", linestyle="")
plt.plot(
    time_num, temp_num, label="RK4 Model", color="blue", linestyle="-")
plt.xlabel("Time (minutes)")
plt.ylabel("Temperature (°C)")
plt.title("Comparison of Model and Experimental Data")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
