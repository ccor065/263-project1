
import numpy as np
from matplotlib import pyplot as plt
import math
from scipy.optimize import curve_fit

# Define global variables
TIME, PRESSURE = np.genfromtxt('cs_p.txt',delimiter=',',skip_header=1).T
STEP = 1


def load_production_data():
    ''' Returns time and production measurements from the Ohaaki geothermal field.
        Parameters:
        -----------
        none
        Returns:
        --------
        time : array-like
            Vector of time (years) at which measurements were taken.
        q: : array-like
            Vector of production measurements kg/s.
        '''

    time, q= np.genfromtxt('cs_q.txt',delimiter=',',skip_header=1).T
    return time, q
def load_injection_data():
    ''' Returns time and C02 injection measurements from the t Ohaaki geothermal field.
        Parameters:
        -----------
        none
        Returns:
        --------
        time : array-like
            Vector of time (years) at which measurements were taken.
        q: : array-like
            Vector of production measurements kg/s.
        '''

    time, q_co2 = np.genfromtxt('cs_c.txt',delimiter=',',skip_header=1).T
    return time, q_co2
def get_vals(t):
    '''
    Returns net flow rate (q) and dq/dt for number of points as in vector input, t.
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
    q_co2 = np.interp(t, t2, co2_raw)
    q = np.interp(t, t1, q_raw)

    # compute net q
    for i in range(len(t)):
        if t[i] >= 1998.51:
            q[i] -= q_co2[i]

    # numerically differniate q
    dqdt = (np.diff(q)) / (np.diff(t))
    return q, dqdt
def curve_fit_pressure(t, a, b, c):
    '''
    Returns time and C02 injection measurements from the t Ohaaki geothermal field.
    Parameters:
    -----------
    t : array-like
        array of time values
    a : float
         value of parameter a to check
    b : float
        value of parameter a to check
    c : float
        value of parameter a to check
    Returns:
    --------
    pressureODE : array-like
                numerical solution to the pressure ODE given the specfic parameters
    '''

    pars = [a, b, c]
    timeODE, pressureODE = solve_pressure_ode(pressure_ode_model, t[0], PRESSURE[0], t[-1], STEP, pars)

    return pressureODE
def find_pars():
    '''
    Returns time and C02 injection measurements from the t Ohaaki geothermal field.
    Parameters:
    -----------
    t : array-like
        array of time values
    a : float
         value of parameter a to check
    b : float
        value of parameter a to check
    c : float
        value of parameter a to check
    Returns:
    --------
    pressureODE : array-like
                numerical solution to the pressure ODE given the specfic parameters
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
    parameters, covar = curve_fit(curve_fit_pressure, ts, pi, pars)
    return parameters[0], parameters[1], parameters[2]
def improved_euler_step(f, tk, yk, h, y0, q, dq, pars):
	""" Compute an Improved euler step
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

    q, dqdt = get_vals(ts)

    for i in range(nt):
        ys[i+1] = improved_euler_step(f, ts[i], ys[i], h, y0, q[i], dqdt[i], pars)
    return  ts, ys
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
    # find correct parameters for a,b and c to fit the model well
    a, b, c = find_pars()
    step = 0.1
    # solve ode using found parameaters
    t_ode, p_ode = solve_pressure_ode(pressure_ode_model, TIME[0], PRESSURE[0], TIME[-1], step, pars = [a, b, c])

    #saves ODE model data to a file to be used elsewhere (eg in concentration ode model)
    modelToSave = np.array([t_ode, p_ode])
    modelToSave = modelToSave.T
    np.savetxt("pressureOdeModel.csv", modelToSave, fmt='%.2f,%.4f', header = 't_ode, p_ode')


    #### Benchmark numerical(ode) solution against analytical solution
    q, _ = get_vals(t_ode)           # load new flow array
    p_ana = np.zeros(len(q))        # initalise analytical pressure array
    for i in range(len(q)):         # compute analtical solution
        p_ana[i] = PRESSURE[0] - ((a * q[i])/b)*(1 - math.exp(-b*t_ode[i]))
    # plot analytical solution
    #plt.plot(t_ode, p_ana, color = 'b', label = "Analytical Solution")
    # plot the data observations
    plt.plot(TIME, PRESSURE,color='k', label ='Pressure Observations')
    # plot the model solution
    plt.plot(t_ode, p_ode, color = 'r', label = 'ODE')
    plt.legend()
    plt.show()

    ######### Convergence Analysis
    step_nums = np.linspace(0.001, 5, 100)
    p_at2000 = np.zeros(len(step_nums))

    for i in range(len(step_nums)):
        t, p = solve_pressure_ode(pressure_ode_model, TIME[0], PRESSURE[0], 2000, step_nums[i], pars = [a, b, c])
        p_at2000[i] = p[-1]

    plt.scatter(step_nums, p_at2000, color = 'r', label = "Pressure at time = 2000")
    plt.xlim (0, max(step_nums))
    #Lables axis and give a tite
    plt.xlabel('Step Size')
    plt.ylabel('pressure value at time = 2000')
    plt.title('Convergence analysis for step size for p(t=2000) and h = 0.1 - 30 ')
    # Display legend and graph
    plt.legend()
    plt.show()


    ##### Misfit
    misfit = np.zeros(len(p_ode))
    nt = int(np.ceil((TIME[-1]-TIME[0])/0.1))		# compute number of Euler steps to take
    ts = TIME[0]+np.arange(nt+1)*0.1			# x array
    p_data_interp = np.interp(ts, TIME, PRESSURE)

    for i in range(len(p_data_interp)):
        misfit[i] = math.sqrt((p_ode[i] - p_data_interp[i])**2)

    plt.scatter(ts, misfit)
    plt.ylabel('RMS Misfit',fontsize=10)
    plt.xlabel('Time',fontsize=10)

    plt.title('Root Mean Sqaured Misfit')
    plt.show()
    return

if __name__ == "__main__":
    plot_pressure_benchmark()
