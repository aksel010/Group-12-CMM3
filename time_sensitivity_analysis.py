import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import numpy as np

# -------------------------------------------------------------------
# Load simulation data
# -------------------------------------------------------------------
df = pd.read_csv("time_of_simulation.csv")
I = df["Current (A)"]  # Current array
t = df["Time (s)"]     # Computation time array

# -------------------------------------------------------------------
# Prepare reciprocal feature for regression (1 / I)
# -------------------------------------------------------------------
recip_I = (1 / I).values.reshape(-1, 1)

# -------------------------------------------------------------------
# Fit linear regression: t = a*(1/I) + b
# -------------------------------------------------------------------
model = LinearRegression()
model.fit(recip_I, t)

# Predictions for original data
t_pred = model.predict(recip_I)

# Compute RMSE
rmse = mean_squared_error(t, t_pred)

# Extract coefficients
a = model.coef_[0]
b = model.intercept_

# -------------------------------------------------------------------
# Plot the data and linear regression fit
# -------------------------------------------------------------------
plt.figure(figsize=(8, 5))

# Scatter data points
plt.scatter(I, t, label="Data", color="blue")

# Smooth curve for fitted line
I_range = np.linspace(I.min(), I.max(), 100)
t_fit = model.predict((1 / I_range).reshape(-1, 1))
plt.plot(
    I_range,
    t_fit,
    color="red",
    label=f"$CT = {a:.2f}/I + {b:.2f},\\ RMSE = {rmse:.2f}$ s"
)

# Labels and title
plt.xlabel("Current (A)")
plt.ylabel("Computation Time (s)")
plt.title("Computation Time vs Current (Linear Regression in 1/I)")

# Grid and legend
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()
