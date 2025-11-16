import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import numpy as np

df = pd.read_csv('time_of_simulation.csv')
I = df['Current (A)']
t = df['Time (s)']
recip_I = (1/I).values.reshape(-1,1)

# Prepare reciprocal feature
recip_I = (1 / I).values.reshape(-1,1)

# Fit linear regression on 1/I
model = LinearRegression()
model.fit(recip_I, t)

# Predictions
t_pred = model.predict(recip_I)

# Compute RMSE
rmse = mean_squared_error(t, t_pred)

# Get coefficients for the equation
a = model.coef_[0]
b = model.intercept_

# Plot
plt.figure(figsize=(8,5))
plt.scatter(I, t, label='Data', color='blue')

# Smooth line for plotting
I_range = np.linspace(I.min(), I.max(), 100)
t_fit = model.predict((1/I_range).reshape(-1,1))

plt.plot(I_range, t_fit, color='red', 
         label=f'$CT = {a:.2f}/ I + {b:.2f},\\ RMSE = {rmse:.2f}$ s')

plt.xlabel("Current (A)")
plt.ylabel("Computation Time (s)")
plt.title("Computation Time vs Current (Linear Regression in 1/I)")
plt.legend()
plt.grid(True)
plt.show()