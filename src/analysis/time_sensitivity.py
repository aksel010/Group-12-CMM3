"""
Regression and plot for computation time sensitivity with respect to current.

Loads simulation run-time data and fits a linear regression of calculation time vs. reciprocal current (1/I).
Standard PEP8 docstrings and clarifying in-line comments.
"""
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import numpy as np

# -------------------------------------------------------------------
# Data loading and feature engineering
# -------------------------------------------------------------------
VALIDATION_DIR = os.path.join(os.path.dirname(__file__), '..', '..', 'results', 'validation')
FIGURES_DIR = os.path.join(VALIDATION_DIR, 'figures')
TABLES_DIR = os.path.join(VALIDATION_DIR, 'tables')
os.makedirs(FIGURES_DIR, exist_ok=True)
os.makedirs(TABLES_DIR, exist_ok=True)

relative_time_file = os.path.join(os.path.dirname(__file__), '../../data/processed/time_of_simulation.csv') 
time_file = os.path.abspath(relative_time_file)
df = pd.read_csv(time_file)
I = df["Current (A)"]
t = df["Time (s)"]
recip_I = (1 / I).values.reshape(-1, 1)

# -------------------------------------------------------------------
# Fit linear regression (t = a * (1/I) + b)
# -------------------------------------------------------------------
model = LinearRegression()
model.fit(recip_I, t)
t_pred = model.predict(recip_I)
rmse = mean_squared_error(t, t_pred)
a = model.coef_[0]
b = model.intercept_

# -------------------------------------------------------------------
# Plot results (scatter and fit)
# -------------------------------------------------------------------
fig = plt.figure(figsize=(8, 5))
plt.scatter(I, t, label="Data", color="blue")
I_range = np.linspace(I.min(), I.max(), 100)
t_fit = model.predict((1 / I_range).reshape(-1, 1))
regression_label = f"$CT = {a:.2f}/I + {b:.2f},\\ RMSE = {rmse:.2f}$ s"
plt.plot(
    I_range,
    t_fit,
    color="red",
    label=regression_label
)
plt.xlabel("Current (A)")
plt.ylabel("Computation Time (s)")
plt.title("Computation Time vs Current (Linear Regression in 1/I)")
plt.grid(True)
plt.legend()
plt.tight_layout()

# --- 4. Save plot and results summary ---
plot_path = os.path.join(FIGURES_DIR, 'time_sensitivity.png')
fig.savefig(plot_path, dpi=300)
print(f"✓ Plot saved to {os.path.relpath(plot_path)}")

summary_path = os.path.join(TABLES_DIR, 'time_sensitivity_summary.txt')
with open(summary_path, 'w') as f:
    f.write("==== Time Sensitivity Analysis Summary ====\n\n")
    f.write(f"Linear Regression Model: {regression_label}\n")
    f.write(f"Coefficient (a): {a}\n")
    f.write(f"Intercept (b): {b}\n")
    f.write(f"Root Mean Squared Error (RMSE): {rmse}\n")
print(f"✓ Results summary saved to {os.path.relpath(summary_path)}")
plt.show()
