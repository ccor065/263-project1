# Authour: Charlotte Cordwell
#
# Purpose: Plot given data

from matplotlib import pyplot as plt
import numpy as np

# Year and CO2 injection rate (kg/s)
year1, co2 = np.genfromtxt('cs_c.txt',delimiter=',',skip_header=1).T
year2, p= np.genfromtxt('cs_p.txt',delimiter=',',skip_header=1).T

fig,ax = plt.subplots()
# Plot the co2 injection rate over time
lns1 = ax.plot(year1, co2, color='r', label='Net Mass Change')

# Set the x-axis label
ax.set_xlabel('Time (year)',fontsize=10)
# Set the y-axis label
ax.set_ylabel('CO2 injection rate kg/s',color="red",fontsize=10)

ax2=ax.twinx()
# Plot the presuure  values over time
lns2 = ax2.plot(year2, p, 'b--', linewidth = 1, label='Well Injection Rate')
# Set the y-axis label
ax2.set_ylabel("pressure MPa",color="blue",fontsize=10)

ax.set_title('Pressure vs CO2 Injection rate\n')
plt.show()
