
import numpy as np
from matplotlib import pyplot as plt
import math


def improved_euler_step(f, tk, yk, h, pars):
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
	f0 = f(tk, yk, *pars)
	f1 = f(tk +h , yk + h*f0, *pars)
	yk1 = yk + 0.5*h*(f0 + f1)

	return yk1

def pressure_ode_model(t, p, p0, a, q, b, c,dq):
    #print(p0, p, aq, b, cDq)
    return -a*q - b*(p-p0) - c*dq
def interplolate_pressure(t):
    return
def solve_pressure_ode(f, t0, y0, t1, h, pars=[]):
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
    # initialise
    nt = int(np.ceil((t1-t0)/h))		# compute number of Euler steps to take
    ts = t0+np.arange(nt+1)*h			# x array
    ys = 0.*ts							# array to store solution
    ys[0] = y0                          # set initial value

    # Import data
    time2, q_raw = load_production_data()
    time3, co2_raw = load_injection_data()
    # Interpolate co2 injection and production vectors

    q_co2 = np.interp(ts, time3, co2_raw)
    q = np.interp(ts, time2, q_raw)

    # compute net q
    for i in range(nt+1):
        if ts[i] >= 1998.51:
            q[i] -= q_co2[i]

    # numerically differniate q
    dqdt = (np.diff(q)) / (np.diff(ts))

    for i in range(nt):
        pars[2] = q[i]
        pars[-1] = dqdt[i]
        ys[i+1] = improved_euler_step(f, ts[i], ys[i], h, pars)

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
    # ODE solver
    time, pressure = load_pressure_data()

    q = 0
    a = 0.001
    b = 0.09 #0.09
    p0 = pressure[0]
    c = 0.003 #0.005
    dq = 0
    pars = [p0, a, q, b, c, dq]
    step = 0.25

    # find numerical solution
    timeODE, pressureODE = solve_pressure_ode(pressure_ode_model, time[0], pressure[0], time[-1], step, pars)

    # Import data
    time2, q_raw = load_production_data()
    time3, co2_raw = load_injection_data()

    # Interpolate co2 injection and production vectors
    q_co2 = np.interp(timeODE, time3, co2_raw)
    q = np.interp(timeODE, time2, q_raw)

    # compute net q
    for i in range(len(q)):
        if timeODE[i] >= 1998.51:
            q[i] -= q_co2[i]

    p_ana = np.zeros(len(pressureODE))
    for i in range(len(p_ana)):
        p_ana[i] = p0 - ((a * q[i])/b)*(1 - math.exp(-b*timeODE[i])) 

    #
    plt.plot(timeODE, p_ana, color = 'b', label = "Analytical Solution")
    plt.plot(time, pressure ,color='k', label ='Pressure Observations')
    plt.plot(timeODE, pressureODE, color = 'r', label = 'ODE')
    plt.legend()
    plt.show()
    return

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
        q: : array-like
            Vector of production measurements kg/s.
        '''

    time, q_co2 = np.genfromtxt('cs_c.txt',delimiter=',',skip_header=1).T
    return time, q_co2

if __name__ == "__main__":
    plot_pressure_benchmark()
