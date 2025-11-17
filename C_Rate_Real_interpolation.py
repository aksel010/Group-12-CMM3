import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from config import *

# Load and prepare data
df = pd.read_csv(r'Real_Data_C_Rate.csv')
df['charging_time_hours'] = df['charging_time_min'] / 60
df_sorted = df.sort_values('C-rate')

# Create the plot
plt.figure(figsize=(10, 6))

# FLIPPED: Plot charging_time_hours vs C-rate
plt.plot(df_sorted['charging_time_hours'], df_sorted['C-rate'], 'bo-', markersize=4, linewidth=1, label='Real-world Data')

# Calculate theoretical values
current_range = np.linspace(0.35, 7.2, 40)  # From 1A to 30A, 20 points

c_rates = []
t_totals = []

for I_var in current_range:
    c_rate = I_var / Capacity_battery
    t_total = q_b / I_var / 3600  # Convert seconds to hours
    c_rates.append(c_rate)
    t_totals.append(t_total)

# Plot theoretical line with axes swapped
plt.plot(t_totals, c_rates, 'r-', linewidth=2, label='Model')
plt.xlabel('Charging Time (hours)')
plt.ylabel('C-rate')
plt.title('C-rate vs Charging Time')

# Grid and legend
plt.grid(True, alpha=0.3)
plt.legend()

plt.tight_layout()
plt.show()