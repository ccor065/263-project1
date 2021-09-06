
import numpy as np
from matplotlib import pyplot as plt
import matplotlib.ticker as mtick
import math
from scipy.optimize import curve_fit
from scipy.interpolate import interp1d
from load_data import *

# Define global variables
TIME, PRESSURE = load_pressure_data()
STEP = 0.1

def get_net_flow(t):
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
            q[i] -= q_co2[i]

    # numerically differniate q
    dqdt = 0.*q
    dqdt[1:-1] = (q[2:]-q[:-2])/(t[2:]-t[:-2])  #central differences
    dqdt[0] = (q[1]-q[0])/(t[1]-t[0])           #foward differences
    dqdt[-1] = (q[-1]-q[-2])/(t[-1]-t[-2])      #backward differences

    return q, dqdt

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
    '''
    Calibrates pressure ODE model to find parameters that provide the best fit

    Returns:
    --------
    parameters : array of ints
                Best fit paramteres for Pressure ODE model
    trainingSize : int
                   Data points from data set used to calibrate model
                
    Notes:
    --------
    Elements of parameters are individually returned in index order
    
    '''
    nt = int(np.ceil((TIME[-1]-TIME[0])/STEP))		# compute number of Euler STEPs to take
    ts = TIME[0]+np.arange(nt+1)*STEP			    # x array
    
    # initial guesses
    a = 0.001 #0.001
    b = 0.09 #0.09
    c = 0.003 #0.005
    pars = [a, b, c]
   
    # make input pressure same length as the output from the solver
    pi = np.interp(ts, TIME, PRESSURE)
    
    # find parameters
    trainingSize = math.ceil(0.8*len(ts))
    parameters, covariance = curve_fit(curve_fit_pressure, ts[0:trainingSize], pi[0:trainingSize], pars)
   
    return parameters[0], parameters[1], parameters[2], trainingSize

def improved_euler_step(f, tk, yk, h, y0, q, dq, pars):
	""" Implements the improved euler method for a single step
		
        Parameters
		----------
		f : callable
			Derivative function.
		tk : float
			Independent variable at beginning of step.
		yk : float
			Solution at beginning of step.
		h : float
			Step size.
        q : float
            value of net flow at a given time, t
        dq : float
            value of the derivative of net flow dq/dt at given time, t.
		pars : iterable
			Optional parameters to pass to derivative function.
		
        Returns
		-------
		yk1 : float
			Solution at end of the Euler step.
	"""
	# y+1 = y0 + h/2 (f(t0, y0) + f(x0+1, y0+1))
	f0 = f(tk, yk, y0, q, dq, *pars)
	f1 = f(tk +h , yk + h*f0, y0, q, dq, *pars)
	yk1 = yk + 0.5*h*(f0 + f1)

	return yk1

def pressure_ode_model(t, p, p0, q, dq, a, b, c):
    ''' Return the derivative dq/dt at time, t, for given parameters.
        Parameters:
        -----------
        t : float
            Independent variable.
        p : float
            Dependent variable.
        p0 : float
            initial value of dependent variable, pressure.
        q : float
            Flow rate at time, t.
        dq : float
            value of the derivative of net flow dq/dt at given time, t.
        a : float
            parameter into ode
        b : float
            parameter into ode
        c : float
            parameter into ode
        Returns:
        --------
        dqdt : float
            Derivative of dependent variable with respect to independent variable.
    '''
   
    return -a*q - b*(p-p0) - c*dq

def solve_pressure_ode(f, t0, y0, t1, h, pars=[]):
    """
    Compute solution of pressure ODE function using improved euler method.
	
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
    q, dqdt = get_net_flow(ts)          

    for i in range(nt):
        ys[i+1] = improved_euler_step(f, ts[i], ys[i], h, y0, q[i], dqdt[i], pars)
    
    return  ts, ys

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

def solve_pressure_benchmark(f, t0, y0, t1, h, q, pars=[]):
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
   
    # initialise
    nt = int(np.ceil((t1-t0)/h))		# compute number of Euler steps to take
    ts = t0+np.arange(nt+1)*h			# x array
    ys = 0.*ts							# array to store solution
    ys[0] = y0                          # set intial value
    dqdt = 0                            # ignore dqdt for becnchmark

    for i in range(nt):
        ys[i+1] = improved_euler_step(f, ts[i], ys[i], h, y0, 4, dqdt, pars)

    return  ts, ys

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

# def getT_P_ode():
#     """
#     Solves pressure ode model with calirbated parameters for best fit
    
#     Returns
#     -------
#     t_ode : array of ints/double
#             independent variable values for which ode solution is numerically solved for
#     p_ode : array of ints/doubles
#             array of solutiions for pressure ODE model
#     """
#     # find correct parameters for a,b and c to fit the model well
#     a, b, c, calibrationPoint = find_pars_pressure()
#     pars = [a,b,c]
#     # solve ode using found parameaters
#     t_ode, p_ode = solve_pressure_ode(pressure_ode_model, TIME[0], PRESSURE[0], TIME[-1], STEP, pars)
    
#     return t_ode, p_ode

def plot_pressure_benchmark():
    '''
    Compare analytical and numerical solutions.
    
    Parameters:
    -----------
    none
    
    Returns:
    --------
    none
    
    Notes:
    ------
    This function called within if __name__ == "__main__":
    It should contain commands to obtain analytical and numerical solutions,
    plot these, and either display the plot to the screen or save it to the disk.
    '''

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
    t_ode, p_ode = solve_pressure_ode(pressure_ode_model, TIME[0], PRESSURE[0], TIME[-1], STEP, pars)
    # save ode to file to use in concentration model
    save_ode_csv(t_ode, p_ode)
   
    # plot the data observations
    ax1.plot(TIME, PRESSURE,color='k', label =' Observations best fit')
    ax1.scatter(TIME, PRESSURE,color='k', marker = 'x', label ='Observations')
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
    t_odeA, p_odeA = solve_pressure_benchmark(pressure_ode_model, time[0], PRESSURE[0], time[-1], STEP, q, pars)
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
        t, p = solve_pressure_ode(pressure_ode_model, TIME[0], PRESSURE[0], 1983, step_nums[i], pars = [a, b, c])
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
        t, p = solve_pressure_ode(pressure_ode_model, TIME[0], PRESSURE[0], 1983, step_nums[i], pars = [a, b, c])
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
    nt = int(np.ceil((TIME[-1]-TIME[0])/0.1))	# compute number of points to compare
    ts = TIME[0]+np.arange(nt+1)*0.1			# x array
    p_data_interp = np.interp(ts, TIME, PRESSURE)

    for i in range(len(p_data_interp)):
        misfit[i] = p_ode[i] - p_data_interp[i]

    plt.scatter(ts, misfit)
    plt.axhline(y=0, color = 'black', linestyle = '--')
    plt.ylabel('Misfit',fontsize=10)
    plt.xlabel('Time',fontsize=10)

    plt.title('Misfit ODE vs interploated data')
    plt.savefig('misfitModel_vs_data',dpi=300)
    plt.show()

    return

'''
MODEL FORECASTING
'''
def plot_model_predictions():
    a, b, c, _ = find_pars_pressure()
    pars = [a,b,c]
    step = 0.1
    print(pars)

    # model
    t_ode, p_ode = solve_pressure_ode(pressure_ode_model, TIME[0], PRESSURE[0], TIME[-1], step, pars)
    # plot the data observations
    plt.plot(TIME, PRESSURE,color='k', label ='Pressure Observations')
    # plot the model solution
    plt.plot(t_ode, p_ode, color = 'r', label = 'ODE')

    # Set up paramters for forecast
    a, b, c, _ = find_pars_pressure()
    pars = [a,b,c]
    endTime = TIME[-1] + 30
    nt = int(np.ceil((endTime-TIME[-1])/step))	# compute number of Euler steps to take
    ts = TIME[-1]+np.arange(nt+1)*step			# x array

    ##### CHANGES
    # Stay at same production / injection
    injRate = 1
    t_1, p_1 = forecast_solve_pressure_ode(pressure_ode_model, injRate, ts[0], PRESSURE[-1], ts[-1], step, pars)
    plt.plot(t_1, p_1, color = 'g', label = 'Same injection')

    # double injection
    injRate = 2
    t_2, p_2 = forecast_solve_pressure_ode(pressure_ode_model, injRate, ts[0], PRESSURE[-1], ts[-1], step, pars)
    plt.plot(t_2, p_2, color = 'orange', label = 'Double injection')

    # halve injection
    injRate = 0.5
    t_05, p_05 =forecast_solve_pressure_ode(pressure_ode_model, injRate, ts[0], PRESSURE[-1], ts[-1], step, pars)
    plt.plot(t_05, p_05, color = 'b', label = 'Halve injection')

    # quadruple injection
    injRate = 4
    t_4, p_4 = forecast_solve_pressure_ode(pressure_ode_model, injRate, ts[0], PRESSURE[-1], ts[-1], step, pars)
    plt.plot(t_4, p_4, color = 'cyan', label = 'Quadruple injection')

    # stop injection
    injRate = 0
    t_0, p_0 = forecast_solve_pressure_ode(pressure_ode_model, injRate, ts[0], PRESSURE[-1], ts[-1], step, pars)
    plt.plot(t_0, p_0, color = 'pink', label = 'Stop injection')
    plt.legend()
    plt.show()
    return

def get_q_different_injection_rate(t, injRate):

    # load in flow rate data
    t1, q_raw = load_production_data()
    t2, co2_raw = load_injection_data()


    # Interpolate co2 injection and production vectors to have same amount of points
    # as vector t.
    production_av = 0
    for i in range(1,10):
        production_av += q_raw[-i]
    production_av = production_av/9

    injection_av = 0
    for i in range(1,4):
        injection_av += co2_raw[-i]
    injection_av = (injection_av/3)*injRate

    q = np.zeros(len(t))

    # compute net q
    for i in range(len(t)):
        q[i] = production_av - injection_av

    # numerically differniate q
    dqdt = (np.diff(q)) / (np.diff(t))
   
    return q, dqdt

def forecast_solve_pressure_ode(f, injRate, t0, y0, t1, h, pars=[]):
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
    # initialise
    nt = int(np.ceil((t1-t0)/h))		# compute number of Euler steps to take
    ts = t0+np.arange(nt+1)*h			# x array
    ys = 0.*ts							# array to store solution
    ys[0] = y0                          # set intial value

    q, dqdt = get_q_different_injection_rate(ts, injRate)

    for i in range(nt):
        ys[i+1] = improved_euler_step(f, ts[i], ys[i], h, y0, q[i], dqdt[i], pars)
   
    return  ts, ys


if __name__ == "__main__":
    plot_pressure_benchmark()
    #plot_model_predictions()
