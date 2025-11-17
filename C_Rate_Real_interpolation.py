import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from config import *

# Load and prepare the dataset.
df = pd.read_csv(r'Real_Data_C_Rate.csv')
df['charging_time_hours'] = df['charging_time_min'] / 60
df_sorted = df.sort_values('C-rate')

# Create the figure for plotting.
plt.figure(figsize=(10, 6))

# Plot the measured charging time versus the C-rate.
plt.plot(
    df_sorted['charging_time_hours'],
    df_sorted['C-rate'],
    'bo-',
    markersize=4,
    linewidth=1,
    label='Real-world Data'
)

# Generate a range of current values for the model.
current_range = np.linspace(0.35, 7, 40)  # 0.35 A to 7 A.

c_rates = []       
t_totals = []      

# Compute modelled C-rates and charging times.
for I_var in current_range: 
    c_rate = I_var / Capacity_battery
    t_total = q_b / I_var / 3600  # Convert from seconds to hours.
    c_rates.append(c_rate)
    t_totals.append(t_total)

# Plot the modelled relationship.
plt.plot(
    t_totals,
    c_rates,
    'r-',
    linewidth=2,
    label='Model'
)

# Add axis labels and title.
plt.xlabel('Charging Time (hours)')
plt.ylabel('C-rate')
plt.title('C-rate vs Charging Time')

# Add grid and legend.
plt.grid(True, alpha=0.3)
plt.legend()

plt.tight_layout()
plt.show()
