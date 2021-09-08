import numpy as np
from matplotlib import pyplot as plt
import matplotlib.ticker as mtick
import math
from scipy.optimize import curve_fit
from scipy.interpolate import interp1d
from load_data import *
from lpm_model_functions import *

TIME_P, PRESSURE = load_pressure_data()
TIME_C, CONC = load_c02_wt_data()
STEP = 0.1

# Pressure benchmarking plotter
def plot_pressure_benchmark():
    """
    dsfds
    """
    #Configure plots
    fig, (ax1, ax2) = plt.subplots(1, 2)
    fig.set_figwidth(10)
    plt.subplots_adjust(None, None, None,None, wspace=0.2, hspace=None)

    """
    PLOT DATA vs ODE
    """

    # find correct parameters for a,b and c to fit the model well
    a, b, c, calibrationPoint = find_pars_pressure()
    pars = [a,b,c]
    # solve ode using found parameaters
    t_ode, p_ode = solve_pressure_ode(pressure_ode_model, TIME_P[0], PRESSURE[0], TIME_P[-1], STEP, pars)


    # plot the data observations
    ax1.plot(TIME_P, PRESSURE,color='k', label =' Observations best fit')
    ax1.scatter(TIME_P, PRESSURE,color='k', marker = 'x', label ='Observations')
    ax1.axvline(t_ode[calibrationPoint], linestyle = '--', label = 'Calibration Point')

    # plot the model solution
    ax1.plot(t_ode, p_ode, color = 'r', label = 'ODE')
    ax1.set_title('ODE vs Data')
    ax1.set_ylabel("Pressure(MPa)")
    ax1.set_xlabel("Year")
    ax1.legend()

    """
    PLOT BENCHMARKING!
    """

    # get average net production rate
    q = 4
    time = np.linspace(0, 50, 100)
    conc = 0
    pars_2 = [q, conc, *pars]
    t_odeA, p_odeA = solve_pressure_const_q(pressure_ode_model, time[0], PRESSURE[0], time[-1], STEP, pars_2)
    p_ana = pressure_analytical_solution(time, q, a, b, c)
    ax2.plot(t_odeA, p_odeA, color = 'r', label = 'ODE')
    ax2.scatter(time, p_ana, color = 'b', label = 'Analytical Solution')
    ax2.set_title('ODE vs Analytical solution')
    ax2.legend()
    plt.savefig('model_vs_ODE_analytical.png',dpi=300)
    plt.show()

    """
    PLOT Convergence analysis
    """

    step_nums = np.linspace(0.001, 20, 200)
    p_at1983 = np.zeros(len(step_nums))

    for i in range(len(step_nums)):
        t, p = solve_pressure_ode(pressure_ode_model, TIME_P[0], PRESSURE[0], 1983, step_nums[i], pars = [a, b, c])
        p_at1983[i] = p[-1]

    fig, (ax1, ax2) = plt.subplots(1,2)
    ax1.scatter(step_nums, p_at1983, s = 9, color = 'r')
    ax1.set_ylabel('Pressure(MPa) at year = 1983')
    ax1.yaxis.set_major_formatter(mtick.FormatStrFormatter('%.2f'))
    ax1.set_xlabel('Step Size')

    # Compute convergence analysis for step size 0-2 to obtain better visualization
    step_nums = np.linspace(0.001, 2, 100)
    p_at1983 = np.zeros(len(step_nums))

    for i in range(len(step_nums)):
        t, p = solve_pressure_ode(pressure_ode_model, TIME_P[0], PRESSURE[0], 1983, step_nums[i], pars = [a, b, c])
        p_at1983[i] = p[-1]

    # Plot data points for convergence analyaia
    ax2.scatter(step_nums, p_at1983, s = 9, color = 'r', label = "Pressure(MPa) at x = step size")
    ax2.yaxis.set_major_formatter(mtick.FormatStrFormatter('%.2f'))
    ax2.set_xlabel('Step Size')
    #Lables axis and give a tite
    plt.suptitle('Convergence analysis for step size for p(t=1983) and h = 0.001 - 20 ')
    # Display legend and graph
    plt.legend()
    plt.savefig('ConvergenceAnalysis_pressureModel.png',dpi=300)
    plt.show()

    """
    PLOT RMS misfit between data and ODE
    """

    misfit = np.zeros(len(p_ode))
    nt = int(np.ceil((TIME_P[-1]-TIME_P[0])/STEP))	# compute number of points to compare
    ts = TIME_P[0]+np.arange(nt+1)*STEP			# x array
    p_data_interp = np.interp(ts, TIME_P, PRESSURE)

    for i in range(len(p_data_interp)):
        misfit[i] = p_ode[i] - p_data_interp[i]

    plt.scatter(ts, misfit, s = 9)
    plt.axhline(y=0, color = 'black', linestyle = '--')
    plt.ylabel('Misfit',fontsize=10)
    plt.xlabel('Time',fontsize=10)

    plt.title('Misfit ODE vs interpolated data')
    plt.savefig('misfitModel_vs_data',dpi=300)
    plt.show()

# Conc Benchmarking PLotter
def plot_conc_benchmark():

    d, m0, calibrationPoint = find_pars_conc()
    print(d, m0)

    a,b, __,_ = find_pars_pressure()

    pars = [a, b, d, m0]
    step = 0.1
    #subpars = [a, b, PRESSURE[0]]
    p0 = PRESSURE[0]
    t_ode, c_ode = solve_conc_ode(conc_ODE_model, TIME_C[0], 0.03, TIME_C[-1], STEP, p0, pars)
    plt.plot(t_ode, c_ode, label = "ode model")
    plt.plot(TIME_C, CONC, 'o', label='data')
    plt.legend()
    plt.show()

    #### Benchmark numerical(ode) solution against analytical solution
    time, c_ana = conc_analytical_solution(m0, d)
    # plot analytical solution
    plt.plot(time, c_ana, color = 'b', label = "Analytical Solution")
    # plot the model solution
    t_ode, c_ode = solve_conc_ode_ana(conc_ODE_model, TIME_C[0], 0.03, TIME_C[-1], STEP, p0, pars)
    plt.plot(t_ode, c_ode, label = "ode model")
    plt.legend()
    plt.show()

    """
    PLOT Convergence analysis
    """

    step_nums = np.linspace(0.001, 20, 200)
    end_c = np.zeros(len(step_nums))

    for i in range(len(step_nums)):
        t, c = solve_conc_ode(conc_ODE_model, TIME_C[0], 0.03, 2005, step_nums[i], p0, pars)
        end_c[i] = c[-1]

    fig, (ax1, ax2) = plt.subplots(1,2)
    ax1.scatter(step_nums, end_c, s = 9, color = 'r')
    ax1.set_ylabel('Conc(%) in year 2005')
    ax1.yaxis.set_major_formatter(mtick.FormatStrFormatter('%.2f'))
    ax1.set_xlabel('Step Size')

    # Compute convergence analysis for step size 0-2 to obtain better visualization
    step_nums = np.linspace(0.001, 2, 100)
    end_c = np.zeros(len(step_nums))

    for i in range(len(step_nums)):
        t, c = solve_conc_ode(conc_ODE_model, TIME_C[0], 0.03, 2005, step_nums[i], p0, pars)
        end_c[i] = c[-1]

    # Plot data points for convergence analyaia
    ax2.scatter(step_nums, end_c, s = 9, color = 'r')
    ax2.yaxis.set_major_formatter(mtick.FormatStrFormatter('%.2f'))
    ax2.set_xlabel('Step Size')
    #Lables axis and give a tite
    plt.suptitle('Convergence analysis for step size for p(t=2005) and h = 0.001 - 20 ')
    # Display legend and graph
    plt.legend()
    plt.savefig('ConvergenceAnalysis_concModel.png',dpi=300)
    plt.show()

    """
    PLOT RMS misfit between data and ODE
    """

    misfit = np.zeros(len(c_ode))
    nt = int(np.ceil((TIME_C[-1]-TIME_C[0])/STEP))	# compute number of points to compare
    ts = TIME_C[0]+np.arange(nt+1)*STEP			# x array
    c_data_interp = np.interp(ts, TIME_C, CONC)

    for i in range(len(c_data_interp)):
        misfit[i] = c_ode[i] - c_data_interp[i]

    plt.scatter(ts, misfit, s = 9)
    plt.axhline(y=0, color = 'black', linestyle = '--')
    plt.ylabel('Misfit',fontsize=10)
    plt.xlabel('Time',fontsize=10)

    plt.title('Misfit ODE vs interpolated data for concentration')
    plt.savefig('misfitModel_vs_data_conc',dpi=300)
    plt.show()

# Plot Model predictions
def plot_model_predictions():
    fig, (ax1, ax2) = plt.subplots(1, 2)
    fig.set_figwidth(12)
    plt.subplots_adjust(None, None, None,None, wspace=None, hspace=None)
    lines = []
    d, m0, calibrationPoint = find_pars_conc()
    a,b,c,_ = find_pars_pressure()
    pars_pressure = [a,b, c]
    pars_conc = [a, b, d, m0]

    # model
    t_ode, p_ode = solve_pressure_ode(pressure_ode_model, TIME_P[0], PRESSURE[0], TIME_P[-1], STEP, pars_pressure)
    # plot the data observations
    p1, = ax1.plot(TIME_P, PRESSURE,color='k')
    # plot the model solution
    ax1.plot(t_ode, p_ode, color = 'r')
    tc_ode, c_ode = solve_conc_ode(conc_ODE_model, TIME_C[0], CONC[0], TIME_C[-1], STEP, PRESSURE[0], pars_conc)
    p2, = ax2.plot(tc_ode, c_ode, color = 'r')
    ax2.plot(TIME_C, CONC, color = 'k')

    # Set up paramters for forecast
    endTime = TIME_P[-1] + 30                     # 30 years projection
    nt = int(np.ceil((endTime-TIME_P[-1])/STEP))	# compute number of Euler steps to take
    ts = TIME_P[-1]+np.arange(nt+1)*STEP			# x array
    t1, q_raw = load_production_data()
    t2, co2_raw = load_injection_data()
    q_prod = np.interp(ts, t1, q_raw)
    q_inj = np.interp(ts, t2, co2_raw)

    # stop injection
    injRates = [0., 0.5, 1., 2., 4.] #different injection rate multipliers
    colours = ['green', 'orange', 'blue', 'cyan', 'pink'] #for graph
    labels = ['Stop, injection =', 'Halve injection', 'Same injection', 'Double injection', 'Quadruple injection'] #for graph

    for i in range(len(injRates)):
        q_net = q_prod[-1] - (q_inj[-1])*injRates[i]
        q_newInj = (q_inj[-1])*injRates[i]
        t, p, c = get_p_conc_forecast(ts, pars_conc, pars_pressure, q_net, q_newInj)
        ax1.plot(t, p, color=colours[i], label = labels[i])
        ax2.plot(t, c, color=colours[i], label = labels[i])

    ax1.legend(loc = 'upper center')
    plt.show()
    return

if __name__ == "__main__":
    #plot_pressure_benchmark()
    #plot_conc_benchmark()
    plot_model_predictions()
