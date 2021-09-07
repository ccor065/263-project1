import numpy as np
from matplotlib import pyplot as plt
import matplotlib.ticker as mtick
import math
from scipy.optimize import curve_fit
from scipy.interpolate import interp1d
from load_data import *

# Define global variables
TIME_P, PRESSURE = load_pressure_data()
TIME_C, CONC = load_c02_wt_data()
STEP = 0.1


## Information collection
def save_ode_csv(t, p):
    """
    Saves 2-dimensional dataset to CSV file
    Parameters
    ----------
    t : array-like
        independent variable of data set.
    p : array-like
        dependent variable of data set.
    """

    modelToSave = np.array([t, p])
    modelToSave = modelToSave.T
    np.savetxt("pressureOdeModel.csv", modelToSave, fmt='%.2f,%.4f', header = 't_ode, p_ode')
def getT_P_ode():
     """
#     Solves pressure ode model with calirbated parameters for best fit
#     Returns
#     -------
#     t_ode : array of ints/double
#             independent variable values for which ode solution is numerically solved for
#     p_ode : array of ints/doubles
#             array of solutiions for pressure ODE model
#     """
     # find correct parameters for a,b and c to fit the model well
     a, b, c, calibrationPoint = find_pars_pressure()
     pars = [a,b,c]
     # solve ode using found parameaters
     t_ode, p_ode = solve_pressure_ode(pressure_ode_model, TIME_P[0], PRESSURE[0], TIME_P[-1], STEP, pars)

     return t_ode, p_ode
def get_q_dq_conc(t):
    '''
    Returns net flow rate q, and dq/dt, for number of points as in vector input, t.
    Parameters:
    -----------
    t : array-like
        vector of time points
    Returns:
    --------
    q : array-like
        net flow rate
    dq/dt: : array-like
        derivative of net flow rate
    Notes:
    -------
    q_production and q_c02 are interploated and then subtracted to find net flow rate.
    '''
    tc, conc_raw = load_c02_wt_data()
    conc = np.interp(t, tc, conc_raw)
    # load in flow rate data
    t1, q_raw = load_production_data()
    t2, co2_raw = load_injection_data()

    # Interpolate co2 injection and production vectors to have same amount of points
    # as vector t.
    q = np.interp(t, t1, q_raw)
    q_co2 = np.interp(t, t2, co2_raw)
    # compute net q
    for i in range(len(t)):
        if t[i] >= 1998.51:
            q[i] =  q[i] - q_co2[i]

    # numerically differniate q
    dqdt = 0.*q
    dqdt[1:-1] = (q[2:]-q[:-2])/(t[2:]-t[:-2])  #central differences
    dqdt[0] = (q[1]-q[0])/(t[1]-t[0])           #foward differences
    dqdt[-1] = (q[-1]-q[-2])/(t[-1]-t[-2])      #backward differences

    return q, dqdt, conc
def curve_fit_pressure(t, a, b, c):
    '''
    Solve pressure ode with the given paramters
    Parameters:
    -----------
    t : array-like
        array of time values
    a : float
         value of parameter a
    b : float
        value of parameter b
    c : float
        value of parameter c
    Returns:
    --------
    pressureODE : array-like
                numerical solution to the pressure ODE given the specific parameters
    '''

    pars = [a, b, c]
    timeODE, pressureODE = solve_pressure_ode(pressure_ode_model, t[0], PRESSURE[0], t[-1], STEP, pars)

    return pressureODE
def find_pars_pressure():
    nt = int(np.ceil((TIME_P[-1]-TIME_P[0])/STEP))		# compute number of Euler STEPs to take
    ts = TIME_P[0]+np.arange(nt+1)*STEP			    # x array

    # initial guesses
    a = 0.001 #0.001
    b = 0.09 #0.09
    c = 0.003 #0.005
    pars = [a, b, c]

    # make input pressure same length as the output from the solver
    pi = np.interp(ts, TIME_P, PRESSURE)

    # find parameters
    trainingSize = math.ceil(0.8*len(ts))
    parameters, covariance = curve_fit(curve_fit_pressure, ts[0:trainingSize], pi[0:trainingSize], pars)

    return parameters[0], parameters[1], parameters[2], trainingSize
def analytical_solution(t, q, a, b, c):
    """
    Computes analytical solution for simplified version of pressure ODE model.
    Used for bench marking.
    Returns
    -------
    p_ana : array of double
            array of analytical solutions for simplified version of pressure ODE model
    """

    p_ana = np.zeros(len(t))        # initalise analytical pressure array
    for i in range(len(t)):         # compute analtical solution
        p_ana[i] = PRESSURE[0] - ((a * q)/b)*(1 - math.exp(-b*t[i]))

    return p_ana

# Solvers
def pressure_ode_model(t, p, p0, dq, q, c_alt, a, b, c):
    """
    dgfgdfg
    """
    if p > p0:
        qloss = b/a *(p-p0)*c_alt*t
        q = q + qloss

    dpdt =  -a*q - b*(p-p0) - c*dq
    return dpdt
def improved_euler_step(f, tk, yk, h, y0, pars):
	# y+1 = y0 + h/2 (f(t0, y0) + f(x0+1, y0+1))
	f0 = f(tk, yk, y0,  *pars)
	f1 = f(tk +h , yk + h*f0, y0, *pars)
	yk1 = yk + 0.5*h*(f0 + f1)

	return yk1
def solve_pressure_ode(f, t0, y0, t1, h, pars):
    nt = int(np.ceil((t1-t0)/h))		# compute number of Euler steps to take
    ts = t0+np.arange(nt+1)*h			# x array
    ys = 0.*ts							# array to store solution
    ys[0] = y0                          # set intial value
    q, dqdt, c_alt = get_q_dq_conc(ts)
    for i in range(nt):
        pars_2 = [dqdt[i], q[i], c_alt[i], *pars]
        ys[i+1] = improved_euler_step(f, ts[i], ys[i], h, y0, pars_2)

    return  ts, ys
def solve_pressure_const_q(f, t0, y0, t1, h, pars):
    # initialise
    nt = int(np.ceil((t1-t0)/h))		# compute number of Euler steps to take
    ts = t0+np.arange(nt+1)*h			# x array
    ys = 0.*ts							# array to store solution
    ys[0] = PRESSURE[-1]                         # set intial value
    dqdt = 0
    for i in range(nt):
        ys[i+1] = improved_euler_step(f, ts[i], ys[i], h, y0, [dqdt, *pars])

    return  ts, ys
# Plotters
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
    # save ode to file to use in concentration model
    save_ode_csv(t_ode, p_ode)

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
    conc = 0.03
    pars_2 = [conc, q, *pars]
    t_odeA, p_odeA = solve_pressure_const_q(pressure_ode_model, time[0], PRESSURE[0], time[-1], STEP, pars_2)
    p_ana = analytical_solution(time, q, a, b, c)
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
    ax1.scatter(step_nums, p_at1983, color = 'r')
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
    ax2.scatter(step_nums, p_at1983, color = 'r', label = "Pressure(MPa) at x = step size")
    ax2.yaxis.set_major_formatter(mtick.FormatStrFormatter('%.2f'))
    ax2.set_xlabel('Step Size')
    #Lables axis and give a tite
    plt.suptitle('Convergence analysis for step size for p(t=1983) and h = 0.001 - 12 ')
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

    plt.scatter(ts, misfit)
    plt.axhline(y=0, color = 'black', linestyle = '--')
    plt.ylabel('Misfit',fontsize=10)
    plt.xlabel('Time',fontsize=10)

    plt.title('Misfit ODE vs interpolated data')
    plt.savefig('misfitModel_vs_data',dpi=300)
    plt.show()
def plot_individual_injRate(t, injRate, color,  description, plot):
        # load in flow rate data
        t1, q_raw = load_production_data()
        t2, co2_raw = load_injection_data()
        t3, conc_raw = load_c02_wt_data()
        q_prod = np.interp(t, t1, q_raw)
        q_inj = np.interp(t, t2, co2_raw)
        conc_interp = np.interp(t, t3, conc_raw)

        # get net flow at final time.

        q = q_prod[-1] - (q_inj[-1])*injRate

        # old model
        #t, p = solve_pressure_const_q(pressure_ode_model, t[0], PRESSURE[0], t[-1], STEP, [q, conc, *pars])
        d, m0 = find_pars_conc()
        a,b,c,_ = find_pars_pressure()
        dq = 0
        pars_conc = [a, b, d, m0]
        p = np.zeros(len(t))
        conc = np.zeros(len(t))

        p[0] = PRESSURE[-1]
        conc[0] = conc_interp[-1]
        for i in range(len(t) - 1):
            p[i+1] = improved_euler_step(pressure_ode_model, t[i], p[i], STEP, PRESSURE[0], [dq, q, conc[i], a, b, c])
            conc[i+1] = improved_euler_step_conc(conc_ODE_model, t[i],conc[i], STEP, 0.03, q, p[i], PRESSURE[0], pars_conc)

        #if plot true, plots p vs t, if false, returns pressures
        if (plot):
            plt.plot(t, p, color=color,  label =description)
            return
        else:
            return p
        
def plot_model_predictions():

    a, b, c, _ = find_pars_pressure()
    pars = [a,b,c]

    # model
    t_ode, p_ode = solve_pressure_ode(pressure_ode_model, TIME_P[0], PRESSURE[0], TIME_P[-1], STEP, pars)
    # plot the data observations
    plt.plot(TIME_P, PRESSURE,color='k', label ='Pressure Observations')
    # plot the model solution
    plt.plot(t_ode, p_ode, color = 'r', label = 'ODE')

    # Set up paramters for forecast
    a, b, c, _ = find_pars_pressure()
    pars = [a,b,c]
    endTime = TIME_P[-1] + 30                     # 30 years projection
    nt = int(np.ceil((endTime-TIME_P[-1])/STEP))	# compute number of Euler steps to take
    ts = TIME_P[-1]+np.arange(nt+1)*STEP			# x array
    print(pars)
    ##### CHANGES
    # stop injection
    plot_individual_injRate(ts, 0,  'orange',  'Stop injection', plot = True)
    # halve injection
    plot_individual_injRate(ts, 0.5,  'b',  'Halve injection', plot = True)
    # Stay at same production / injection

    plot_individual_injRate(ts, 1, 'g',  'Same injection', plot = True)
    # double injection
    plot_individual_injRate(ts, 2, 'm',  'Double injection', plot = True)
    # quadruple injection
    plot_individual_injRate(ts, 4,  'cyan',  'Quadruple injection', plot = True)
    #plt.legend()
    plt.show()
    return

def conc_ODE_model(t, c, c0, q, p, p0, a, b, d, m0):

    ''' Return the derivative dc/dt at time, t, for given parameters.
        Parameters:
        -----------
        t : float
            Independent variable.
        c : float
            Dependent variable.
        q : float
            CO2 Source/sink rate.
        p : float
            pressure value within reservoir
        a : float
            Source/sink strength parameter.
        b : float
            Recharge strength parameter.
        d : float
            C02 reaction strength parameter.
        m0 : float
            mass of reservoir.
        p0 : float
            ambient pressure of reservoir
        c0 : float
            ambient value of dependant variable
        Returns:
        --------
        dcdt : float
            Derivative of dependent variable with respect to independent variable.
    '''

    # computes value of c' term (variable name is c_alt)
    if (p > p0):
        c_alt = c
        c_alt2 = c
    else:
        c_alt = c0
        c_alt2 = 0

    qloss = (b/a)*(p-p0)*c_alt2*t # calculating CO2 loss to groundwater
    q = q - qloss # qCO2 after the loss

    return (1-c)*(q/m0) - (b/(a*m0))*(p-p0)*(c_alt-c) - d*(c-c0)
def improved_euler_step_conc(f, tk, yk, h, y0,q, p, p0, pars=[]):
	f0 = f(tk, yk, y0, q, p, p0,  *pars)
	f1 = f(tk +h , yk + h*f0, y0, q, p, p0,  *pars)
	yk1 = yk + 0.5*h*(f0 + f1)

	return yk1
def solve_conc_ode(f, t0, y0, t1, h, p0, pars=[]):
    # initialise
    nt = int(np.ceil((t1-t0)/h))		# compute number of Euler steps to take
    ts = t0+np.arange(nt+1)*h			# x array
    ys = 0.*ts							# array to store solution
    ys[0] = y0                          # set intial value

    t1, p_raw = load_pressure_data()
    t2, c02_raw = load_injection_data()

    p = np.interp(ts, t1, p_raw)
    q_arr = np.interp(ts, t2, c02_raw)


    for i in range(nt):
        if ts[i] >= 1998.51:
            q = q_arr[i]
        else:
            q = 0
        ys[i+1] = improved_euler_step_conc(f, ts[i], ys[i], h, y0, q, p[i], p0, pars)
    return  ts, ys
def solve_conc_ode_ana(f, t0, y0, t1, h, p0, pars=[]):
    # initialise
    nt = int(np.ceil((t1-t0)/h))		# compute number of Euler steps to take
    ts = t0+np.arange(nt+1)*h			# x array
    ys = 0.*ts							# array to store solution
    ys[0] = y0                          # set intial value

    t1, p_raw = load_pressure_data()
    t2, c02_raw = load_injection_data()

    p = p_raw[0]
    q = c02_raw[0]
    for i in range(nt):
        ys[i+1] = improved_euler_step_conc(f, ts[i], ys[i], h, y0, q, p, p0, pars)
    return  ts, ys
def curve_fit_conc(t, d, m0):
    a,b, __,_ = find_pars_pressure()
    pars = [a, b, d, m0]
    p0 = PRESSURE[0]
    time, conc = solve_conc_ode(conc_ODE_model, t[0], CONC[0], t[-1], STEP, p0, pars)
    return conc
def find_pars_conc():

    nt = int(np.ceil((TIME_C[-1]-TIME_C[0])/STEP))		# compute number of Euler STEPs to take
    ts = TIME_C[0]+np.arange(nt+1)*STEP			    # x array
    d = 0.1774
    m0 = 11000

    pars = [d, m0]
    ci = np.interp(ts, TIME_C, CONC)
    trainingSize = math.ceil(0.8*len(ts))
    parameters, covar = curve_fit(curve_fit_conc, ts[0:trainingSize], ci[0:trainingSize+1], pars)
    return parameters[0], parameters[1]

    return parameters[0], parameters[1], parameters[2], trainingSize
def plot_conc_model():

    d, m0 = find_pars_conc()
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
    _, q_co2_raw = load_injection_data()
    time = np.linspace(TIME_C[0], TIME_C[-1], 200)
    time = time - TIME_C[0]
    c0 = CONC[0]
    c_ana = np.zeros(len(time))        # initalise analytical pressure array
    for i in range(len(time)):         # compute analtical solution
        ki = q_co2_raw[0] / m0
        Li = (ki * c0 - ki) / (ki + d)
        c_ana[i] = (ki + d * c0) / (ki + d) + Li / math.exp((ki + d) * time[i])
    time = time + TIME_C[0]
    # plot analytical solution
    plt.plot(time, c_ana, color = 'b', label = "Analytical Solution")
    # plot the model solution
    t_ode, c_ode = solve_conc_ode_ana(conc_ODE_model, TIME_C[0], 0.03, TIME_C[-1], STEP, p0, pars)
    plt.plot(t_ode, c_ode, label = "ode model")
    plt.legend()
    plt.show()

'''
MODEL FORECASTING
'''
def plot_conc_predictions():

    d, m0 = find_pars_conc()

    a,b, c,_ = find_pars_pressure()
    pars_p = [a, b, c]
    pars_conc = [a, b, d, m0]

    t_ode, c_ode = solve_conc_ode(conc_ODE_model, TIME_C[0], 0.03, TIME_C[-1], STEP, PRESSURE[0], pars_conc)
    plt.plot(t_ode, c_ode, label = "ode model")
    plt.plot(TIME_C, CONC, 'o', label='data')

    # Set up paramters for forecast
    endTime = TIME_C[-1] + 30
    nt = int(np.ceil((endTime-TIME_C[-1])/STEP))	# compute number of Euler steps to take
    ts = TIME_C[-1]+np.arange(nt+1)*STEP			# x array


    ##### CHANGES

    injRates = [0., 0.5, 1., 2., 4.] #different injection rate multipliers
    colours = ['g', 'orange', 'b', 'cyan', 'pink'] #for graph
    labels = ['Stop injection', 'Halve injection', 'Same injection', 'Double injection', 'Quadruple injection'] #for graph
    for i in range(len(injRates)): #loops and plots forcasts for different injection rates
        q = multiply_inj_rate(ts, injRates[i])
        p = plot_individual_injRate(ts, injRates[i], None, None, False)
        t, c = forecast_solve_conc_ode(conc_ODE_model, injRates[i], ts[0], CONC[-1], ts[-1], STEP, p, pars_conc)
        plt.plot(t, c, color = colours[i], label = labels[i])
    plt.legend()
    plt.show()
    return

def multiply_inj_rate(t, injRate):

    # load in flow rate data
    t2, co2_raw = load_injection_data()

    q = np.ones(len(t)) * co2_raw[-1] * injRate
    return q

def forecast_solve_conc_ode(f, injRate, t0, y0, t1, h, p, pars=[]):
    """
    Compute solution of the coupled ODE problem using Improved Euler method.
	Parameters
	----------
	f : callable
		Derivative function.
	t0 : float
		Initial value of independent variable.
	y0 : float
		Initial value of solution.
	t1 : float
		Final value of independent variable.
	h : float
		Step size.
    p : array-like
        array of pressure values for time range
	pars : iterable
		Optional parameters to pass into derivative function.
	Returns
	-------
	xs : array-like
		Independent variable at solution.
	ys : array-like
		Solution.
	Notes
	-----
	Assumes that order of inputs to f is f(x,y,*pars).
    """
    nt = int(np.ceil((t1-t0)/h))		# compute number of Euler steps to take
    ts = t0+np.arange(nt+1)*h			# x array
    ys = 0.*ts							# array to store solution
    ys[0] = y0                          # set intial value

    q = multiply_inj_rate(ts, injRate)

    for i in range(nt):
        ys[i+1] = improved_euler_step_conc(f, ts[i], ys[i], h, y0, q[i], p[i], PRESSURE[0], pars)
    return  ts, ys

if __name__ == "__main__":
    #plot_pressure_benchmark()

    #plot_model_predictions()

    #plot_conc_model()
    plot_conc_predictions()