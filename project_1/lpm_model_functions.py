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
"""
### PRESSURE FUNCTIONs
"""
### Curve fitting functions
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

## Analytical Solution Solvers
def pressure_analytical_solution(t, q, a, b, c):
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
#### MODEL FUNCTIONS
def pressure_ode_model(t, p, p0, dq, q, conc, a, b, c):
    """
    dgfgdfg
    """
    if p > p0:
        qloss = (b/a) *(p-p0)*conc*t
        q = q + qloss
    dpdt =  -a*q - b*(p-p0) - c*dq
    return dpdt
#### numerical Solvers
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
    ys[0] = y0                        # set intial value
    dqdt = 0
    for i in range(nt):
        ys[i+1] = improved_euler_step(f, ts[i], ys[i], h, y0, [dqdt, *pars])

    return  ts, ys
'''
### Concentration FUNCTIONs
'''
### Curve fitting functions
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

## Analytical Solution Solver
def conc_analytical_solution(m0, d):
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
    return time, c_ana

#### MODEL FUNCTION
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

#### noteumerical Solvers
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
        pars_2 = [q, p[i], p0, *pars]
        ys[i+1] = improved_euler_step(f, ts[i], ys[i], h, y0, pars_2)
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
        ys[i+1] = improved_euler_step(f, ts[i], ys[i], h, y0, [q, p, p0, *pars])
    return  ts, ys

"""
## Forecasting function
"""
def get_p_conc_forecast(t, pars_conc, pars_pressure, q, q_newInj, type):
    dq = 0
    p = np.zeros(len(t))
    conc = np.zeros(len(t))
    p[0] = PRESSURE[-1]
    conc[0] = CONC[-1]
    for i in range(len(t) - 1):
        conc[i+1] = improved_euler_step(conc_ODE_model, t[i], conc[i], STEP, CONC[0], [q_newInj, p[i], PRESSURE[0], *pars_conc])
        p[i+1] = improved_euler_step(pressure_ode_model, t[i], p[i], STEP, PRESSURE[0], [dq, q, conc[i], *pars_pressure])

    return t, p, conc
