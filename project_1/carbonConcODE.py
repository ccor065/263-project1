import numpy as np
from matplotlib import pyplot as plt
import math
from scipy.optimize import curve_fit

TIME, CONC = np.genfromtxt('cs_cc.txt',delimiter=',',skip_header=1).T
TIME_P, PRESSURE = np.genfromtxt('pressureOdeModel.csv',delimiter=',',skip_header=1).T
STEP = 0.1

def load_pressure_data():
    ''' Returns time and temperature measurements from kettle experiment.
        Parameters:
        -----------
        none
        Returns:
        --------
        time : array-like
            Vector of time (years) at which measurements were taken.
        pA: : array-like
            Vector of pressure measurements MPa.
        '''

    time, pA= np.genfromtxt('cs_p.txt',delimiter=',',skip_header=1).T
    return time, pA

def load_production_data():
    ''' Returns time and temperature measurements from kettle experiment.
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
    ''' Returns time and temperature measurements from kettle experiment.
        Parameters:
        -----------
        none
        Returns:
        --------
        time : array-like
            Vector of time (years) at which measurements were taken.
        q_co2: : array-like
            Vector of production measurements kg/s.
        '''

    time, q_co2 = np.genfromtxt('cs_c.txt',delimiter=',',skip_header=1).T
    return time, q_co2

def load_c02_wt_data():
    ''' Returns time and temperature measurements from kettle experiment.
        Parameters:
        -----------
        none
        Returns:
        --------
        time : array-like
            Vector of time (years) at which measurements were taken.
        wt_co2: : array-like
            Vector of co2 concentrations (%).
        '''

    time, wt_co2 = np.genfromtxt('cs_cc.txt',delimiter=',',skip_header=1).T
    return time, wt_co2

def conc_ODE_model(t, c, q, p, a, d, m0, p0, c0):

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
    else:
        c_alt = c0

    #calculates numerical derivative
    dcdt = (1 - c) * (q / m0) - (a/ m0) * (p - p0) * (c_alt - c) - d * (c - c0)

    return dcdt

def improved_euler_step_conc(f, tk, yk, h, q, p, pars):
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
		pars : iterable
			Optional parameters to pass to derivative function.
		Returns
		-------
		yk1 : float
			Solution at end of the Euler step.
	"""
	# y+1 = y0 + h/2 (f(t0, y0) + f(x0+1, y0+1))
	f0 = f(tk, yk, q, p, *pars)
	f1 = f(tk +h , yk + h*f0, q, p, *pars)
	yk1 = yk + 0.5*h*(f0 + f1)

	return yk1

def solve_conc_ode(f, t0, y0, t1, h, pars):
    """
    Compute solution of initial value ODE problem using Improved Euler method.
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
    ys[0] = y0                          # set initial value

    # Import data
    time3, q_co2_raw = load_injection_data()
    time4, pressure_raw = load_pressure_data()
    # Interpolate co2 injection and production vectors

    q_co2 = np.interp(ts, time3, q_co2_raw)
    p = np.interp(ts, time4, pressure_raw)

    for i in range(nt):
        ys[i+1] = improved_euler_step_conc(f, ts[i], ys[i], h, q_co2[i], p[i], pars)

    return  ts, ys

def curve_fit_conc(t, a, d, m0, c0):
    '''
    Returns numerical solution to the concentration ODE.
    Parameters:
    -----------
    t : array-like
        array of time values
    a : float
         value of parameter a to check
    d : float
        value of parameter d to check
    m0 : float
        value of parameter m0 to check
    p0 : float
        value of parameter p0 to check
    c0 : float
        value of parameter c0 to check
    Returns:
    --------
    concODE : array-like
                numerical solution to the concentration ODE given the specific parameters
    '''

    pars = [a, d, m0, PRESSURE[0], c0]
    timeODE, concODE = solve_conc_ode(conc_ODE_model, t[0], CONC[0], t[-1], STEP, pars)

    return concODE

def find_pars():
    '''
    Returns parameters suitable to fit concentration ODE.

    Parameters:
    -----------
    Returns:
    --------
    a : float
         value of parameter a to check
    d : float
         value of parameter d to check
    m0 : float
        value of parameter m0 to check
    c0 : float
        value of parameter c0 to check
    '''
    nt = int(np.ceil((TIME[-1]-TIME[0])/STEP))		# compute number of Euler STEPs to take
    ts = TIME[0]+np.arange(nt+1)*STEP			    # x array
    # initial guesses
    a = 0.012012 #73.95
    d = 0.0917764 #10
    m0 = 11216 #200000
    c0 = 6.17 #0.02
    pars = [a, d, m0, c0]
    # make input pressure same length as the output from the solver
    ci = np.interp(ts, TIME, CONC)
    # find parameters
    trainingSize = math.ceil(0.8*len(ts))
    parameters, covar = curve_fit(curve_fit_conc, ts[0:trainingSize], ci[0:trainingSize + 1], pars)
    return parameters[0], parameters[1], parameters[2], parameters[3]

def plot_conc_model():
    '''
    Plots Concentration Data against ODE model for it
    '''
    a, d, m0, c0 = find_pars()
    print(a, d, m0, c0)
    print('a={:2.1f}, d={:2.1f}, m0={:2.1f}, c0={:2.1f}'.format(a, d, m0, c0))
    step = 0.1

    t_model, c = solve_conc_ode(conc_ODE_model, TIME[0], CONC[0], TIME[-1], step, [a, d, m0, PRESSURE[0], c0])

    modelToSave = np.array([t_model, c])
    modelToSave = modelToSave.T
    np.savetxt("concOdeModel.csv", modelToSave, fmt='%.2f,%.4f', header = 't_ode, c_ode')

    plt.plot(t_model, c, label = "ode model")
    plt.plot(TIME, CONC, 'o', label='data')
    plt.legend()
    plt.ylim((0, 0.075))
    plt.show()

if __name__ == "__main__":
    plot_conc_model()
