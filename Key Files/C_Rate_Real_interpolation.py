import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('Real_Data_C_Rate.csv')
df['charging_time_hours'] = df['charging_time_min'] / 60

plt.figure(figsize=(10, 6))
plt.semilogy(df['C-rate'], df['charging_time_hours'], 'bo-')
plt.xlabel('C-rate')
plt.ylabel('Charging Time (hours)')
plt.grid(True)
plt.show()

