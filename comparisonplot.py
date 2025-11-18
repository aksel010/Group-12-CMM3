"""
comparisonplot.py
Visual comparison between experimental Li-ion charging data and RK4 model output.
"""

import matplotlib.pyplot as plt
import pandas as pd

def plot_experimental_vs_model(exp_csv="TempTimeCharging_LIU2.csv", num_csv="RK4 solution.csv"):
    """
    Plot temperature vs. time for both experimental and numerical (RK4) results.
    Args:
        exp_csv (str): Experimental data CSV file
        num_csv (str): Numerical model CSV file
    """
    df_exp = pd.read_csv(exp_csv)
    df_num = pd.read_csv(num_csv)
    time_exp = df_exp["Time"]
    temp_exp = df_exp["Temp"]
    time_num = df_num["Time (s)"] / 60
    temp_num = df_num["Temperature (K)"] - 273.15
    plt.figure(figsize=(10, 6))
    plt.plot(time_exp, temp_exp, label="Liu et al.", color="red", marker="o", linestyle="")
    plt.plot(time_num, temp_num, label="RK4 Model", color="blue", linestyle="-")
    plt.xlabel("Time (minutes)")
    plt.ylabel("Temperature (°C)")
    plt.title("Comparison of Model and Experimental Data")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    plot_experimental_vs_model()
