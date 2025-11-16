import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('time_of_simulation.csv')
I = df['Current (A)']
t = df['Time (s)']


plt.figure()
plt.plot(I, t, 'o-', markersize=4)
plt.xlabel("Current (A)")
plt.ylabel("Time (s)")
plt.title("Computation Time vs Current")
plt.grid(True)
plt.show()