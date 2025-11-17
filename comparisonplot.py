import matplotlib.pyplot as plt
import pandas as pd

dfexp = pd.read_csv('TempTimeCharging_LIU2.csv')
dfnum = pd.read_csv('RK4 solution.csv')
time_exp = dfexp['Time']
temp_exp = dfexp['Temp'] 
time_num = dfnum['Time (s)'] / 60  # convert to minutes
temp_num = dfnum['Temperature (K)'] - 273.15  # convert


plt.plot(time_exp, temp_exp, label='Liu Et.Al', color='red', marker='o', linestyle='')
plt.plot(time_num, temp_num, label='RK4 Model', color='blue', linestyle='-')

plt.xlim(0, 50)
plt.xlabel('Time (minutes)')          
plt.ylabel('Temperature (°C)') 
plt.title('Comparison of Model and Experimental')
plt.legend()
plt.grid(True)
plt.show()

