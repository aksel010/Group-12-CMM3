from ODE import run
import matplotlib.pyplot as plt
import pandas as pd

df = pd.read_csv('TempTimeCharging_LIU2.csv')
time_exp = df['Time']
temp_exp = df['Temp'] 

plt.xlim(0, 50)  
plt.ylim(0, 75)
plt.plot(time_exp, temp_exp, label='Liu Et.Al', color='red', marker='o', linestyle='')

plt.xlabel('Time (minutes)')          
plt.ylabel('Temperature (°C)') 
plt.title('Comparison of Model and Experimental')
plt.legend()
plt.grid(True)
plt.show()