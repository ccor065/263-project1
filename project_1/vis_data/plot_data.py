# Authour: Charlotte Cordwell
#
# Purpose: Plot given data

from matplotlib import pyplot as plt
import numpy as np

# Year and CO2 injection rate (kg/s)
year1, co2 = np.genfromtxt('cs_c.txt',delimiter=',',skip_header=1).T
year2, pressure= np.genfromtxt('cs_p.txt',delimiter=',',skip_header=1).T
year3, production= np.genfromtxt('cs_q.txt',delimiter=',',skip_header=1).T
year4, co2_wt= np.genfromtxt('cs_cc.txt',delimiter=',',skip_header=1).T

# Wanting to plot all data against pressure to visualise the effects of each property on the pressure
# Plot Pressure vs CO2 injection rate
fig,ax = plt.subplots()
# Plot pressure over time
ax.plot(year2, pressure, 'b--', linewidth = 1, label='Pressure in well')
# Set the y-axis label
ax.set_ylabel("Pressure MPa",color="black",fontsize=10)

#create second y-axis
ax2=ax.twinx()
ax2.plot(year1, co2, color='r', label='CO2 injection rate')
# Set the y-axis label
ax2.set_ylabel('CO2 injection rate kg/s',color="black",fontsize=10)

ax.set_xlabel('Time (year)',fontsize=10)
ax.set_title('Pressure vs CO2 Injection rate\n')
plt.savefig('co2-rate_vs_pressure.png',dpi=300)
plt.show()

# Plot pressure vs production rate over time
fig,ax = plt.subplots()
# Plot pressure over time
ax.plot(year2, pressure, 'b--', linewidth = 1, label='Pressure in well')
# Set the y-axis label
ax.set_ylabel("Pressure MPa",color="black",fontsize=10)

#create second y-axis
ax2=ax.twinx()
ax2.plot(year3, production, color='r', label='CO2 injection rate')
# Set the y-axis label
ax2.set_ylabel('Production Rate kg/s',color="red",fontsize=10)

# Set the x-axis label
ax.set_xlabel('Time (year)',fontsize=10)
ax.set_title('Pressure vs Production Rate \n')
plt.savefig('production-rate_vs_pressure.png',dpi=300)
plt.show()

# Plot pressure vs CO2 wt%
fig,ax = plt.subplots()
# Plot pressure over time
ax.plot(year2, pressure, 'b--', linewidth = 1, label='Pressure in well')
# Set the y-axis label
ax.set_ylabel("Pressure MPa",color="black",fontsize=10)

#create second y-axis
ax2=ax.twinx()
ax2.plot(year4, co2_wt, color='r', label='CO2 wt%')
# Set the y-axis label
ax2.set_ylabel('CO2wt% ',color="red",fontsize=10)
ax.set_xlabel('Time (year)',fontsize=10)
ax.set_title('Pressure vs CO2wt%\n')
plt.savefig('co2wt_vs_pressure.png',dpi=300)
plt.show()
