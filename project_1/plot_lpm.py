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

## Define global Variables
TIME_P, PRESSURE = load_pressure_data()
TIME_C, CONC = load_c02_wt_data()
a, b, c, calibrationPointP = find_pars_pressure()
a_best, b_best, c_best, calibrationPointP = find_pars_pressure()
d, m0, calibrationPointC = find_pars_conc()
PARS_P = [a, b, c]
PARS_C = [a, b, d, m0]
STEP = 0.04
v=0.1

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
    # Get solution to ODE using improved Euler
    t_ode, p_ode = solve_pressure_ode(pressure_ode_model, TIME_P[0], PRESSURE[0], TIME_P[-1], STEP, PARS_P)

    # plot the data observations
    ax1.scatter(TIME_P, PRESSURE, color='r', marker = 'x', label ='Observations')
    #ax1.axvline(t_ode[calibrationPointP], linestyle = '--', label = 'Calibration Point')

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
    ax2.set_title('ODE vs Analytical solution')
    ax2.legend()
    plt.savefig('model_vs_ODE_analytical.png',dpi=300)
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
    plt.savefig('misfitModel_vs_data',dpi=300)
    plt.show()

# Conc Benchmarking PLotter
def plot_conc_benchmark():

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
    plt.legend()
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
    fig, (ax1, ax2) = plt.subplots(1, 2)
    fig.set_figwidth(12)
    plt.subplots_adjust(None, None, None,None, wspace=None, hspace=None)


    # model
    t_ode, p_ode = solve_pressure_ode(pressure_ode_model, TIME_P[0], PRESSURE[0], TIME_P[-1], STEP, PARS_P)
    # plot the data observations
    p1, = ax1.plot(TIME_P, PRESSURE,color='k')
    # plot the model solution
    ax1.plot(t_ode, p_ode, color = 'r')
    tc_ode, c_ode = solve_conc_ode(conc_ODE_model, TIME_C[0], CONC[0], TIME_C[-1], STEP, PRESSURE[0], PARS_C)
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
        t, p, c = get_p_conc_forecast(ts, PARS_C, PARS_P , q_net, q_newInj)
        ax1.plot(t, p, color=colours[i], label = labels[i])
        ax2.plot(t, c, color=colours[i], label = labels[i])

    ax1.legend(loc = 'upper center')
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

if __name__ == "__main__":
    plot_pressure_benchmark()
    #plot_conc_benchmark()
    #plot_model_predictions()
