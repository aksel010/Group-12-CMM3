import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

df = pd.read_csv(r'Key Files\Real_Data_C_Rate.csv')
df['charging_time_hours'] = df['charging_time_min'] / 60
df_sorted = df.sort_values('C-rate')

x = df_sorted['C-rate'].values
y = df_sorted['charging_time_hours'].values

# Perform linear regression using numpy
slope, intercept = np.polyfit(x, y, 1)

# Generate line for plotting
x_line = np.linspace(x.min(), x.max(), 300)
y_line = slope * x_line + intercept

# Calculate R²
y_pred = slope * x + intercept
ss_res = np.sum((y - y_pred) ** 2)
ss_tot = np.sum((y - np.mean(y)) ** 2)
r2 = 1 - (ss_res / ss_tot)

plt.figure(figsize=(12, 8))
plt.semilogy(x_line, y_line, 'purple', linewidth=2.5, label='Linear Regression')
plt.semilogy(x, y, 'ko', markersize=8, label='Data Points')
plt.xlabel('C-rate')
plt.ylabel('Charging Time (hours)')
plt.title(f'Linear Regression: Charging Time vs C-rate\nR² = {r2:.4f}, y = {slope:.4f}x + {intercept:.4f}')
plt.grid(True, alpha=0.3)
plt.legend()
plt.show()

print(f"Linear equation: y = {slope:.4f}x + {intercept:.4f}")
print(f"R² score: {r2:.4f}")