import numpy as np
from matplotlib import pyplot as plt
import matplotlib.ticker as mtick
import math
from scipy.optimize import curve_fit
from scipy.interpolate import interp1d
from load_data import *
from lpm_model_functions import *
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from uncertainty import *

## Define global Variables
TIME_P, PRESSURE = load_pressure_data()
TIME_C, CONC = load_c02_wt_data()
a, b, c, calibrationPointP, covar_pressue = find_pars_pressure()
d, m0, calibrationPointC, covar_conc = find_pars_conc()
PARS_P = [a, b, c]
PARS_C = [a, b, d, m0]
STEP = 0.04
v=0.1

# Pressure benchmarking plotter
def plot_pressure_benchmark():
    """
    This function plots all of the benchmarking for the pressure ODE.
        this includes:
        data vs ODE
        Analytical solution vs ODE
        Convergence testing
        RMS misfit
    """
    #Configure plots
    fig, (ax1, ax2) = plt.subplots(1, 2)
    fig.set_figwidth(10)
    plt.subplots_adjust(None, None, None,None, wspace=0.2, hspace=None)

    """
    PLOT DATA vs ODE
    """
    # Get solution to ODE using improved Euler
    t_ode, p_ode = solve_pressure_ode(pressure_ode_model, TIME_P[0], PRESSURE[0], TIME_P[-1], STEP, PARS_P)

    # plot the data observations
    ax1.scatter(TIME_P, PRESSURE, color='r', marker = 'x', label ='Observations')
    ax1.axvline(t_ode[calibrationPointP], linestyle = ':', label = 'Calibration Point')

    # plot the model solution
    ax1.plot(t_ode, p_ode, color = 'k', label = 'ODE')
    ax1.set_title('ODE vs Data')
    ax1.set_ylabel("Pressure(MPa)")
    ax1.set_xlabel("Year")
    ax1.legend()

    """
    PLOT BENCHMARKING!
    """
    # Initalise values for benchmarking analytical solution.
    q = 4                           # constant flow rate
    time = np.linspace(0, 50, 100)  # time array
                       # concentration

    # Get values for pressure to Benchmark using the numerical solver.
    conc = 0
    t_odeA, p_odeA = solve_pressure_const_q(pressure_ode_model, time[0], PRESSURE[0], time[-1], STEP, [q, *PARS_P])
    # Get values of pressure using the analytical solution
    p_ana = pressure_analytical_solution(time, q, *PARS_P)

    ## Plot analytical solution and numerical solver solution against eachother
    ax2.plot(t_odeA, p_odeA, color = 'black', label = 'ODE')
    ax2.scatter(time, p_ana, color = 'r', marker = 'x', label = 'Analytical Solution')
    pars_formatted = []
    for i in range(len(PARS_P)):
        pars_formatted.append(np.format_float_scientific(PARS_P[i], precision = 3))
    ax2.set_title('ODE vs Analytical solution  \n a=%s b=%s c=%s q=4.00'% (pars_formatted[0],pars_formatted[1],pars_formatted[2]))
    ax2.legend()
    plt.savefig('model_vs_ODE_analytical_pressure.png',dpi=300)
    plt.show()
    """
    PLOT Convergence
    """

    step_nums = np.linspace(0.05, 0.8, 200)
    pAt_1983 = np.zeros(len(step_nums))
    h = np.zeros(len(step_nums))

    for i in range(len(step_nums)):
        t, p = solve_pressure_ode(pressure_ode_model, time[0], PRESSURE[0], 1983, step_nums[i], PARS_P)
        pAt_1983[i] = p[-1]
        h[i] = 1/step_nums[i]

    plt.scatter(h, pAt_1983, s= 9, color = 'r', label = "Pressure 1983(x) at x = Step Size")
    plt.title("Convergence Analysis for Pressure ODE")
    plt.xlabel("Step size = 1/h")
    plt.ylabel("Pressure MPa at 1983")
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

    plt.scatter(ts, misfit, s = 9, marker = 'x', color = 'r', label = 'Misfit')
    plt.axhline(y=0, color = 'black', linestyle = '--')
    plt.ylabel('Pressure Misfit (MPa)',fontsize=10)
    plt.xlabel('Time',fontsize=10)

    plt.title('Misfit ODE vs interpolated data')
    plt.savefig('misfitModel_vs_data_pressure',dpi=300)
    plt.show()

# Conc Benchmarking PLotter
def plot_conc_benchmark():
    """
    This function plots all of the benchmarking for the pressure ODE.
        this includes:
        data vs ODE
        Analytical solution vs ODE
        Convergence testing
        RMS misfit
    """
    """
    PLOT DATA vs ODE
    """
    t_ode, c_ode = solve_conc_ode(conc_ODE_model, TIME_C[0], CONC[0], TIME_C[-1], STEP, PRESSURE[0], PARS_C)
    plt.plot(t_ode, c_ode * 100,color = 'black', label = 'ODE')
    plt.scatter(TIME_C, CONC * 100,color='r', marker = 'x', label ='Observations')
    plt.legend()
    plt.title('ODE vs Data')
    plt.xlabel('Year')
    plt.ylabel('Concentration (% weight)')
    plt.savefig('model_vs_data_conc', dpi = 300)
    plt.show()

    #### Benchmark numerical(ode) solution against analytical solution
    time, c_ana = conc_analytical_solution(m0, d)
    # plot analytical solution
    plt.plot(time, c_ana,  color = 'r', marker = 'x', label = 'Analytical Solution')
    # plot the model solution
    t_ode, c_ode = solve_conc_ode_ana(conc_ODE_model, TIME_C[0], CONC[0], TIME_C[-1], STEP, PRESSURE[0], PARS_C)
    plt.plot(t_ode, c_ode,color = 'black', label = 'ODE')
    parsc_formatted = []
    for i in range(len(PARS_C)):
        parsc_formatted.append(np.format_float_scientific(PARS_C[i], precision = 3))
    plt.title('ODE vs Analytical solution  \n  d=%s m0=%s'% (parsc_formatted[2], parsc_formatted[3]))
    plt.legend()
    #plt.savefig('conc_analytical_solution.png',dpi=300)

    plt.show()
    """
    PLOT Convergence
    """
    t_ode, c_ode = solve_conc_ode(conc_ODE_model, TIME_C[0], CONC[0], TIME_C[-1], STEP, PRESSURE[0], PARS_C)

    step_nums = np.linspace(0.05, 0.8, 200)
    end_c = np.zeros(len(step_nums))
    h = np.zeros(len(step_nums))

    for i in range(len(step_nums)):
        t, c = solve_conc_ode(conc_ODE_model, TIME_C[0], CONC[0], 2005, step_nums[i], PRESSURE[0], PARS_C)
        end_c[i] = c[-1]
        h[i] = 1/step_nums[i]

    plt.scatter(h, end_c, s= 9, color = 'r', label = "C02wt% 2005(h) at h = Step Size")
    plt.title("Convergence Analysis for Concentration ODE")
    plt.xlabel("Step size = 1/h")
    plt.ylabel("C02wt% at 2005")
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

    plt.scatter(ts, misfit * 100, s = 9, color = 'r')
    plt.axhline(y=0, color = 'black', linestyle = '--')
    plt.ylabel('Concentration Misfit (% weight)',fontsize=10)
    plt.xlabel('Time',fontsize=10)

    plt.title('Misfit ODE vs interpolated data')
    plt.savefig('misfitModel_vs_data_conc',dpi=300)
    plt.show()

# Plot Model predictions
def plot_model_predictions():
    """
    This function plots the model predictions for both the pressure and
    concentration ode, without the uncertainty.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2)
    fig.set_figwidth(13)
    plt.subplots_adjust(None, None, 0.85 ,None, wspace=None, hspace=None)


    # model
    t_ode, p_ode = solve_pressure_ode(pressure_ode_model, TIME_P[0], PRESSURE[0], TIME_P[-1], STEP, PARS_P)
    # plot the data observations
    p1 = ax1.scatter(TIME_P, PRESSURE,color='k', s= 9, label = "Observations")
    # plot the model solution
    ax1.plot(t_ode, p_ode, color = 'r', label = "ODE model")
    tc_ode, c_ode = solve_conc_ode(conc_ODE_model, TIME_C[0], CONC[0], TIME_C[-1], STEP, PRESSURE[0], PARS_C)
    p2, = ax2.plot(tc_ode, c_ode, color = 'r', label = "ODE model")
    ax2.scatter(TIME_C, CONC, color = 'k', s= 9, label ="Observations")

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
    colours = ['orange', 'green', 'red', 'blue', 'steelblue'] #for graph
    labels = ['qc02 = 0.0 kg/s', 'qc02 = %.2f kg/s',  'qc02 = %.2f kg/s ','qc02 = %.2f kg/s ','qc02 = %.2f kg/s'] #for graph

    c02_groundwater_leaks = np.zeros((len(injRates), len(ts)))
    
    endValues = []
    
    for i in range(len(injRates)):
        q_net = q_prod[-1] - (q_inj[-1])*injRates[i]
        q_newInj = (q_inj[-1])*injRates[i]
        t, p, c = get_p_conc_forecast(ts, PARS_C, PARS_P, p_ode[-1], c_ode[-1], q_net, q_newInj)
        ax1.plot(t, p, color=colours[i])
        ax2.plot(t, c, color=colours[i], label = labels[i] %(q_newInj))
        endValues.append(p[-1])
        endValues.append(c[-1])

        for j in range(len(p)):
            if (p[j] > PRESSURE[0]):
                c02_groundwater_leaks[i, j] = (PARS_P[1] / PARS_P[0]) * (p[j] - PRESSURE[0]) * c[j]
        #c02_groundwater_leaks[i, :] = np.cumsum(c02_groundwater_leaks[i, :])
        print(endValues)

    ax2.axhline(0.10, linestyle = "--", color = 'grey', label = '10 wt% C02' )    #ax1.axvline(t_ode[calibrationPointP], linestyle = '--', label = 'Calibration Point')
    ax1.axhline(PRESSURE[0], linestyle = "--", color = 'grey', label = 'Ambient Pressure P0')
    ax2.set_title("Concentration C02wt%")
    ax1.set_title("Pressure MPa")
    plt.suptitle("30 Year Forecast for Ohaaki Geothermal Field")
    ax2.legend(bbox_to_anchor=(1,1), loc="upper left")
    ax1.legend()
    ax1.set_xlabel("Time(year)")
    ax2.set_xlabel

    ax1.set_ylabel("Pressure MPa")
    ax2.set_ylabel("C02 Concentration (wt proportion)")

    plt.savefig('forecast_no_uncertain',dpi=300)

    plt.show()

    fig2, ax3 = plt.subplots(1, 1)

    colours = ['black', 'black', 'black', 'black', 'steelblue'] #for graph

    for i in range(len(injRates)):
        if (i == 0):
            ax3.plot(t, c02_groundwater_leaks[i], color = colours[i], label = 'qc02 = 0.0 kg/s - 98.60 kg/s')
        elif (i == 4):
            ax3.plot(t, c02_groundwater_leaks[i], color = colours[i], label = 'qc02 = 197.20 kg/s')
        else:
            ax3.plot(t, c02_groundwater_leaks[i], color = colours[i])

    ax3.set_title("CO2 leak into groundwater")
    ax3.set_ylabel("CO2 leak (kg/s)")
    ax3.set_xlabel("Time(year)")
    ax3.legend()

    plt.show()

    return

"""
PLOTTING POSTERIOR
"""
def plot_posterior3D(a, b, c, P):
    """Plot posterior distribution for each parameter combination
    Args:
        a (numpy array): a distribution vector
        b (numpy array): b distribution vector
        P (numpy array): posterior matrix
    """

    # plotting variables
    azim = 15.		# azimuth at which surfaces are shown

    # a and b combination
    Ab, Ba = np.meshgrid(a, b, indexing='ij')
    Pab = np.zeros(Ab.shape)
    for i in range(len(a)):
        for j in range(len(b)):
            Pab[i][j] = sum([P[i][j][k] for k in range(len(c))])

    # a and c combination
    Ac, Ca = np.meshgrid(a, c, indexing='ij')
    Pac = np.zeros(Ac.shape)
    for i in range(len(a)):
        for k in range(len(c)):
            Pac[i][k] = sum([P[i][j][k] for j in range(len(b))])

    # b and c combination
    Bc, Cb = np.meshgrid(b, c, indexing='ij')
    Pbc = np.zeros(Bc.shape)
    for j in range(len(b)):
        for k in range(len(c)):
            Pbc[j][k] = sum([P[i][j][k] for i in range(len(a))])

    # plotting
    fig = plt.figure(figsize=[20.0,15.])
    ax1 = fig.add_subplot(221, projection='3d')
    ax1.plot_surface(Ab, Ba, Pab, rstride=1, cstride=1, cmap=cm.Oranges, lw = 0.5)
    ax1.set_xlabel('a')
    ax1.set_ylabel('b')
    ax1.set_zlabel('P')
    ax1.set_xlim([a[0], a[-1]])
    ax1.set_ylim([b[0], b[-1]])
    ax1.set_zlim(0., )
    ax1.view_init(40, azim)

    ax1 = fig.add_subplot(222, projection='3d')
    ax1.plot_surface(Ac, Ca, Pac, rstride=1, cstride=1,cmap=cm.Oranges, lw = 0.5)
    ax1.set_xlabel('a')
    ax1.set_ylabel('c')
    ax1.set_zlabel('P')
    ax1.set_xlim([a[0], a[-1]])
    ax1.set_ylim([c[0], c[-1]])
    ax1.set_zlim(0., )
    ax1.view_init(40, azim)

    ax1 = fig.add_subplot(223, projection='3d')
    ax1.plot_surface(Bc, Cb, Pbc, rstride=1, cstride=1,cmap=cm.Oranges, lw = 0.5)
    ax1.set_xlabel('b')
    ax1.set_ylabel('c')
    ax1.set_zlabel('P')
    ax1.set_xlim([b[0], b[-1]])
    ax1.set_ylim([c[0], c[-1]])
    ax1.set_zlim(0., )
    ax1.view_init(40, azim)

    # save and show
    plt.show()

def plot_posterior2D(a, b, P):
    """Plot posterior distribution
    Args:
        a (numpy array): a distribution vector
        b (numpy array): b distribution vector
        P (numpy array): posterior matrix
    """

    # grid of parameter values: returns every possible combination of parameters in a and b
    A, B = np.meshgrid(a, b)

    # plotting
    fig = plt.figure(figsize=[10., 7.])				# open figure
    ax1 = fig.add_subplot(111, projection='3d')		# create 3D axes
    ax1.plot_surface(A, B, P, rstride=1, cstride=1,cmap=cm.Oranges, lw = 0.5,edgecolor='k')	# show surface

    # plotting upkeep
    ax1.set_xlabel('a')
    ax1.set_ylabel('b')
    ax1.set_zlabel('P')
    ax1.set_xlim([a[0], a[-1]])
    ax1.set_ylim([b[0], b[-1]])
    ax1.set_zlim(0., )
    ax1.view_init(40, 100.)

    # save and show
    plt.show()

"""
PLOTTING SAMPLING
"""
def plot_samples3D(a, b, c, P, samples):
    # plotting variables
    azim = 15.		# azimuth at which surfaces are shown
    # a and b combination
    Ab, Ba = np.meshgrid(a, b, indexing='ij')
    Pab = np.zeros(Ab.shape)
    for i in range(len(a)):
        for j in range(len(b)):
            Pab[i][j] = sum([P[i][j][k] for k in range(len(c))])

    # a and c combination
    Ac, Ca = np.meshgrid(a, c, indexing='ij')
    Pac = np.zeros(Ac.shape)
    for i in range(len(a)):
        for k in range(len(c)):
            Pac[i][k] = sum([P[i][j][k] for j in range(len(b))])

    # b and c combination
    Bc, Cb = np.meshgrid(b, c, indexing='ij')
    Pbc = np.zeros(Bc.shape)
    for j in range(len(b)):
        for k in range(len(c)):
            Pbc[j][k] = sum([P[i][j][k] for i in range(len(a))])


    s = np.array([np.sum((np.interp(TIME_P, *solve_pressure_ode(pressure_ode_model, TIME_P[0], PRESSURE[0], TIME_P[-1], STEP, [a, b, c]))-PRESSURE)**2)/v for a,b,c in samples])
    p = np.exp(-s/2.)


    p = p/np.max(p)*np.max(P)*1.2

    # plotting
    fig = plt.figure(figsize=[20.0,15.])
    ax1 = fig.add_subplot(221, projection='3d')
    ax1.plot_surface(Ab, Ba, Pab, rstride=1, cstride=1, cmap=cm.Oranges, lw = 0.5)
    ax1.set_xlabel('a')
    ax1.set_ylabel('b')
    ax1.set_zlabel('P')
    ax1.set_xlim([a[0], a[-1]])
    ax1.set_ylim([b[0], b[-1]])
    ax1.set_zlim(0., )
    ax1.view_init(40, azim)
    ax1.plot(samples[:,0],samples[:,1],p,'k.')

    ax1 = fig.add_subplot(222, projection='3d')
    ax1.plot_surface(Ac, Ca, Pac, rstride=1, cstride=1,cmap=cm.Oranges, lw = 0.5)
    ax1.set_xlabel('a')
    ax1.set_ylabel('c')
    ax1.set_zlabel('P')
    ax1.set_xlim([a[0], a[-1]])
    ax1.set_ylim([c[0], c[-1]])
    ax1.set_zlim(0., )
    ax1.view_init(40, azim)
    ax1.plot(samples[:,0],samples[:,-1],p,'k.')

    ax1 = fig.add_subplot(223, projection='3d')
    ax1.plot_surface(Bc, Cb, Pbc, rstride=1, cstride=1,cmap=cm.Oranges, lw = 0.5)
    ax1.set_xlabel('b')
    ax1.set_ylabel('c')
    ax1.set_zlabel('P')
    ax1.set_xlim([b[0], b[-1]])
    ax1.set_ylim([c[0], c[-1]])
    ax1.set_zlim(0., )
    ax1.view_init(40, azim)
    ax1.plot(samples[:,1],samples[:,-1],p,'k.')

    # save and show
    plt.show()

def plot_samples2D(a, b, P, samples):
    # plotting
    fig = plt.figure(figsize=[10., 7.])				# open figure
    ax1 = fig.add_subplot(111, projection='3d')		# create 3D axes
    A, B = np.meshgrid(a, b, indexing='ij')
   	# show surface


    s = np.array([np.sum((np.interp(TIME_P, *solve_pressure_ode(pressure_ode_model, TIME_P[0], PRESSURE[0], TIME_P[-1], STEP, [a, b, c_best]))-PRESSURE)**2)/v for a,b,c in samples])
    p = np.exp(-s/2.)
    p = p/np.max(p)*np.max(P)*1.2

    ax1.plot(*samples.T,p,'k.')

    # plotting upkeep
    ax1.set_xlabel('a')
    ax1.set_ylabel('b')
    ax1.set_zlabel('P')
    ax1.set_zlim(0., )
    ax1.view_init(40, 100.)

    # save and show
    plt.show()

"""
plotting uncertainty
"""
def plot_conc_pressure_uncertainty(samples):
    """
    This function plots the ODE with uncertainty within the scope of the data range

    Parameters:
    Samples : array - like
            number of parameters by  number of samples
            randome sample range of parameters


    """
    fig, (ax1, ax2) = plt.subplots(1, 2)
    fig.set_figwidth(13)
    plt.subplots_adjust(None, None, 0.85 ,None, wspace=None, hspace=None)

    for d, m0, a,b,c in samples:
        tm, pm = solve_pressure_ode(pressure_ode_model, TIME_P[0], PRESSURE[0], TIME_P[-1], STEP, [a, b, c])

        ax1.plot(tm, pm, 'black', lw=0.3,alpha=0.2)

        tc, cm = solve_conc_ode(conc_ODE_model, TIME_C[0],
                             CONC[0], TIME_C[-1], STEP, PRESSURE[0], [a, b, d, m0])
        ax2.plot(tc, cm, 'black', lw=0.3,alpha=0.2)

    ax1.axvline(tm[calibrationPointP], color='b', linestyle=':', label='Calibration Point')
    ax2.axvline(tc[calibrationPointC], color='b', linestyle=':', label='Calibration Point')
    ax1.errorbar(TIME_P, PRESSURE, yerr=0.6,fmt='ro', elinewidth = 0.3, label='data')
    ax2.errorbar(TIME_C, CONC, yerr=0.005, fmt='ro',elinewidth = 0.3, label='data')

    ax1.axhline(PRESSURE[0], linestyle = "--", color = 'grey', label = 'Ambient Pressure P0')
    ax2.set_title("Concentration C02wt%")
    ax1.set_title("Pressure MPa")
    plt.suptitle("30 Year Forecast for Ohaaki Geothermal Field")
    ax2.legend(bbox_to_anchor=(1,1), loc="upper left")
    ax1.legend()
    ax1.set_xlabel("Time(year)")
    ax2.set_xlabel("Time(year)")
    ax2.set_ylim(0.02,0.08)

    ax1.set_ylabel("Pressure MPa")
    ax2.set_ylabel("C02 Concentration (wt proportion)")
    plt.savefig('uncertainty_data',dpi=300)


    plt.show()
def plot_uncertainty_forecast(samples):
    """
    This function plots the ODE and the forecasting with uncertainty

    Parameters:
    Samples : array - like
            number of parameters by  number of samples
            randome sample range of parameters
    """
    fig, (ax1, ax2) = plt.subplots(1, 2)
    fig.set_figwidth(13)
    plt.subplots_adjust(None, None, 0.85 ,None, wspace=None, hspace=None)

    for d, m0, a,b,c in samples:
        tm, pm = solve_pressure_ode(pressure_ode_model, TIME_P[0], PRESSURE[0], TIME_P[-1], STEP, [a, b, c])

        ax1.plot(tm, pm, 'black', alpha=0.2,lw=0.3)

        tc, cm = solve_conc_ode(conc_ODE_model, TIME_C[0],
                             CONC[0], TIME_C[-1], STEP, PRESSURE[0], [a, b, d, m0])
        ax2.plot(tc, cm, 'black',alpha=0.2, lw=0.3)

    ax1.axvline(tm[calibrationPointP], color='b', linestyle=':', label='Calibration Point')
    ax2.axvline(tc[calibrationPointC], color='b', linestyle=':', label='Calibration Point')

    ax1.errorbar(TIME_P, PRESSURE, yerr=0.6,fmt='ro', elinewidth = 0.3, label='data')
    ax2.errorbar(TIME_C, CONC, yerr=0.005, fmt='ro',elinewidth = 0.3, label='data')

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
    colours = ['orange', 'green', 'red', 'blue', 'steelblue'] #for graph
    labels = ['qc02 = 0.0 kg/s', 'qc02 = %.2f kg/s',  'qc02 = %.2f kg/s ','qc02 = %.2f kg/s ','qc02 = %.2f kg/s'] #for graph
    for i in range(len(injRates)):
        q_net = q_prod[-1] - (q_inj[-1])*injRates[i]
        q_newInj = (q_inj[-1])*injRates[i]
        t, p, c = get_p_conc_forecast(ts, PARS_C, PARS_P, pm[-1], cm[-1], q_net, q_newInj)
        ax1.plot(t, p, color=colours[i], linewidth = 0.3)
        ax2.plot(t, c, color=colours[i], label = labels[i] %(q_newInj), linewidth = 0.3)

    p_finals = np.zeros((len(injRates), len(samples)))
    c_finals = np.zeros((len(injRates), len(samples)))
    j = 0

    for d, m0, a, b, c in samples:
        pars_c = [a, b, d, m0]
        pars_p = [a, b,c]
        for i in range(len(injRates)):
            q_net = q_prod[-1] - (q_inj[-1])*injRates[i]
            q_newInj = (q_inj[-1])*injRates[i]
            t, p, c = get_p_conc_forecast(ts, pars_c, pars_p, pm[-1], cm[-1], q_net, q_newInj)
            p_finals[i, j] = p[-1]
            c_finals[i, j] = c[-1]
            ax1.plot(t, p, color=colours[i],alpha= 0.2, linewidth = 1)
            ax2.plot(t, c, color=colours[i], alpha= 0.2,linewidth = 1)
        j = j + 1

    np.savetxt('final pressure uncertainty.csv', p_finals)
    np.savetxt('final conc uncertainty.csv', c_finals)

    ax2.axhline(0.10, linestyle = "--", color = 'grey', label = '10 wt% C02' )    #ax1.axvline(t_ode[calibrationPointP], linestyle = '--', label = 'Calibration Point')
    ax1.axhline(PRESSURE[0], linestyle = "--", color = 'grey', label = 'Ambient Pressure P0')
    ax2.set_title("Concentration C02wt%")
    ax1.set_title("Pressure MPa")
    fig.suptitle("30 Year Forecast for Ohaaki Geothermal Field")
    leg_2 = ax2.legend()
    for lh in leg_2.legendHandles:
        leg_2.set_alpha(1)
        ax2.legend(bbox_to_anchor=(1,1), loc="upper left")
    ax1.legend()
    ax1.set_xlabel("Time(year)")
    ax2.set_xlabel("Time(year)")

    ax1.set_ylabel("Pressure MPa")
    ax2.set_ylabel("C02 Concentration (wt proportion)")
    plt.savefig('forecast_uncertain',dpi=300)
    plt.show()
    return

if __name__ == "__main__":
    plot_pressure_benchmark()
    plot_conc_benchmark()
    plot_model_predictions()
    n_samples = 100
    samples = construct_all_samples(n_samples)
    plot_conc_pressure_uncertainty(samples)
    plot_uncertainty_forecast(samples)
