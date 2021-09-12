# Authour: Charlotte Cordwell
#
# Purpose: Plot given data

from matplotlib import pyplot as plt
import numpy as np

# Year and CO2 injection rate (kg/s)
year1, co2= np.genfromtxt('cs_c.txt',delimiter=',',skip_header=1).T
year2, pressure= np.genfromtxt('cs_p.txt',delimiter=',',skip_header=1).T
year3, production= np.genfromtxt('cs_q.txt',delimiter=',',skip_header=1).T
year4, co2_wt= np.genfromtxt('cs_cc.txt',delimiter=',',skip_header=1).T

fig,ax = plt.subplots()
fig.set_figwidth(4)
# Plot pressure over time
ax.plot(year2, pressure, 'black', linewidth = 1, label='Pressure in well')
# Set the y-axis label
ax.set_ylabel("Pressure MPa",color="black",fontsize=10)
#create second y-axis
ax2=ax.twinx()
ax2.plot(year1, co2, color='r', label='CO2 Injection Rate [kg/s]')
ax2.plot(year3, production,  'b--', label= "Extraction Rate [kg/s]")
# Set the y-axis label
ax2.set_ylabel('Flow rate kg/s',color="r",fontsize=10)
ax.axvline(year1[0], linestyle = ':', label = 'Injection Begins')

ax.set_xlabel('Time (year)',fontsize=10)
ax.set_title("Comparison of Pressure to Extraction rate and Injection rate \n in the Ohaaki Geothermal Reservoir")
fig.legend(bbox_to_anchor =(0.9,0.88))
plt.savefig('pressure_flow_rates.png',dpi=300)
plt.show()


fig,ax = plt.subplots()
# Plot pressure over time
# Set the y-axis label
ax.set_ylabel("CO2 Injection Rate [kg/s",color="black",fontsize=10)
#create second y-axis
ax2=ax.twinx()
ax.plot(year1, co2, color='r', label='CO2 Injection Rate [kg/s]')
ax2.plot(year4, co2_wt,  'b--', label= "C02 [wt fraction]")
# Set the y-axis label
ax2.set_ylabel('CO2wt% ',color="red",fontsize=10)
ax.set_xlabel('Time (year)',fontsize=10)
ax.axvline(year1[0], linestyle = ':', label = 'Injection Begins')

ax.set_ylim(0, 80)
ax.set_title("Comparison of C02 Concentration and CO2 injection.")
fig.legend(bbox_to_anchor =(0.5,0.88))
plt.savefig('c02wt_vs_injection.png',dpi=300)
plt.show()



fig,ax = plt.subplots()
# Plot pressure over time
ax.plot(year2, pressure, 'black', linewidth = 1, label='Pressure in well')
# Set the y-axis label
ax.set_ylabel("Pressure MPa",color="black",fontsize=10)
#create second y-axis
ax2=ax.twinx()
ax2.plot(year1, co2, color='r', label='CO2 Injection Rate [kg/s]')
ax2.plot(year3, production,  'b--', label= "Extraction Rate [kg/s]")
# Set the y-axis label
ax2.set_ylabel('Flow rate kg/s',color="r",fontsize=10)
ax.set_xlabel('Time (year)',fontsize=10)
ax.set_title("Comparison of Pressure to Extraction rate and Injection rate \n in the Ohaaki Geothermal Reservoir")
fig.legend(bbox_to_anchor =(0.9,0.9))
plt.savefig('pressure_flow_rates.png',dpi=300)
plt.show()


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
ax2.set_ylabel('CO2 injection rate kg/s',color="r",fontsize=10)
ax.set_xlabel('Time (year)',fontsize=10)
ax.set_title('Pressure vs CO2 Injection rate\n')
fig.legend()
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
ax2.plot(year3, production, color='r', label='Production Rate')
# Set the y-axis label
ax2.set_ylabel('Production Rate kg/s',color="red",fontsize=10)

# Set the x-axis label
ax.set_xlabel('Time (year)',fontsize=10)
ax.set_title('Pressure vs Production Rate \n')
fig.legend()
plt.savefig('production-rate_vs_pressure.png',dpi=300)
plt.show()

# Plot pressure vs CO2 wt%
fig,ax = plt.subplots()
# Plot pressure over time
ax.plot(year1, co2, 'b--', linewidth = 1, label='CO2 injection rate')
# Set the y-axis label
ax.set_ylabel("Pressure MPa",color="black",fontsize=10)

#create second y-axis
ax2=ax.twinx()
ax2.plot(year4, co2_wt, color='r', label='CO2 wt%')
# Set the y-axis label
ax2.set_ylabel('CO2wt% ',color="red",fontsize=10)
ax.set_xlabel('Time (year)',fontsize=10)
ax.set_title('CO2 Injection Rate vs CO2wt%\n')
fig.legend()
plt.savefig('co2wt_vs_pressure.png',dpi=300)
plt.show()
