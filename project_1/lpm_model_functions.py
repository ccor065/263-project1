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
    # get numerical solution to ODE using specific pars.
    timeODE, pressureODE = solve_pressure_ode(pressure_ode_model, t[0], PRESSURE[0], t[-1], STEP, pars)

    return pressureODE
def find_pars_pressure():
    '''
    Finds the parameters for the pressure ODE which gives best fit to the data
    using scipy curve_fit.

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
    parameters[0]: float
                   value of parameter a which gives best fit.
    parameters[1]: float
                   value of parameter b which gives best fit.
    parameters[2]: float
                   value of parameter c which gives best fit.
    trainingSize: float
                 gives the index at where claibration point is in the time array.
    '''
    # initalise time array
    nt = int(np.ceil((TIME_P[-1]-TIME_P[0])/STEP))	# get number of time points
    ts = TIME_P[0]+np.arange(nt+1)*STEP			    # initial time array

    # initial parameter guesses
    a = 0.001
    b = 0.09
    c = 0.003
    pars = [a, b, c]

    # make input pressure same length as the output from the solver
    pi = np.interp(ts, TIME_P, PRESSURE)


    trainingSize = math.ceil(0.8*len(ts)) # get length training array

    # use curve_fit to find pars that give best fit ODE to data.
    parameters, covariance = curve_fit(curve_fit_pressure, ts[0:trainingSize], pi[0:trainingSize], pars)
    # return pars, a, b, c and also calibration point.
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
    q:     array-like
           net flow rate
    dqdt:  array-like
           derivative of net flow rate
    conc:  array-like
           interpolated concentration for time array, t.

    Notes:
    -------
    q_production and q_c02 are interploated and then subtracted to find net flow rate.
    '''
    # load in flow rate and concentration data
    tc, conc_raw = load_c02_wt_data()
    t1, q_raw = load_production_data()
    t2, co2_raw = load_injection_data()

    #interpolate concentration
    conc = np.interp(t, tc, conc_raw)

    # Interpolate co2 injection and production vectors to have same amount of points
    # as vector t.
    q = np.interp(t, t1, q_raw)
    q_co2 = np.interp(t, t2, co2_raw)

    # compute net flow rate
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
    Parameters:
    -----------
    t : array-like
        array of time values
    q : float
        value of constant net #flow rate
    a : float
         value of parameter a
    b : float
        value of parameter b
    c : float
        value of parameter c

    Returns
    -------
    p_ana : array of double
            array of analytical solutions for simplified version of pressure ODE model
    """

    p_ana = np.zeros(len(t))        # initalise analytical pressure array
    for i in range(len(t)):         # compute analtical solution
        p_ana[i] = PRESSURE[0] - ((a * q)/b)*(1 - math.exp(-b*t[i]))

    return p_ana# comment done
#### MODEL FUNCTIONS
def pressure_ode_model(t, p, p0, dq, q, conc, a, b, c):
    '''
    Computes dp/dt at a specific time given parameters

    Parameters:
    -----------
    t : float
        time at which dp/dt is computed
    p : float
         value of pressure at time, t.
    p0 : float
        initial pressure
    dq : float
        vlalue of the derivative of net flow rate dq/dt at time t.
    q : float
         net flow at time t.
    conc : float
        concentration at time t.
    a : float
         value of parameter a
    b : float
        value of parameter b
    c : float
        value of parameter c

    Returns:
    --------
    dpdt : float
           value of the derivate dp/dt at time t.
    '''
    # calculate ground water loss if pressure is greater than inital pressure.
    if p > p0:
        qloss = (b/a) *(p-p0)*conc*t
        q = q + qloss #q loss reduces qc02 therefore add to net flow.
    dpdt =  -a*q - b*(p-p0) - c*dq  # calculate derivative# #
    return dpdt#comment done
#### numerical Solvers
def improved_euler_step(f, tk, yk, h, y0, pars):
    '''
    Computes dp/dt at a specific time given parameters

    Parameters:
    -----------
    f : function
        ODE model function to compute derivative.
    tk : float
         value of time one increment before estimation time.
    yk : float
        value of dependent variable one increment beofe solver.
    h : float
        step-size, indcates value to incremnt independent variable
    y0 : float
        inital value of dependent variable.
    pars : array-like
        individual paramters to be passed into ode function.


    Returns:
    --------
    yk1 : float
          Value of the dependent variable at time = tk + 1(increment).
    Notes:
    --------
    Improved euler step uses this equation to compute its prediction.
     y+1 = y0 + h/2 (f(t0, y0) + f(x0+1, y0+1))
    '''
    # Compute improved euler # # step
    f0 = f(tk, yk, y0,  *pars)
    f1 = f(tk +h , yk + h*f0, y0, *pars)
    yk1 = yk + 0.5*h*(f0 + f1)

    return yk1

def solve_pressure_ode(f, t0, y0, t1, h, pars):
    '''
    Computes solution to pressure ODE using improved euler.

    Parameters:
    -----------
    f : function
        ODE model function to compute derivative.
    t0 : float
         Inital value of time to solve ode from
    y0 : float
        Inital value of dependent variable.
    t1 : float
        final value of independent variable.
    h : float
        step-size, indcates value to increment independent variable
    pars : array-like
        individual paramters that are solved using the curve_fit function.
        pars should contain: [a, b, c] only, where a,b,c are solved using
        curve_fit.

    Returns:
    --------
    ts : array-like
         Corresponding times at which the dependent variables are solved.
    ys : array-like
         Solution to ODE for given time range.

    Notes:
    --------
    The parameter 'pars' does not contain the same pars as what is passed into
    Improved euler step function nor the ODE model function. These are only the
    parameters which need to be found using curve_fit.
    '''
    nt = int(np.ceil((t1-t0)/h))		# compute number of Euler steps to take
    ts = t0+np.arange(nt+1)*h			# x array
    ys = 0.*ts							# array to store solution
    ys[0] = y0                          # set intial value

    # get q, dqdt and concentration arrays (same length as ts)
    q, dqdt, conc = get_q_dq_conc(ts)
    for i in range(nt):
        # compute improved euler step to solve the ODE numerically
        ys[i+1] = improved_euler_step(f, ts[i], ys[i], h, y0, [dqdt[i], q[i], conc[i], *pars])

    # return arays containing the time and solved dependent variables.
    return  ts, ys
def solve_pressure_const_q(f, t0, y0, t1, h, pars):
    '''
    Computes solution to pressure ODE for a contstant flow rate and Concentration
     using improved euler.

    Parameters:
    -----------
    f : function
        ODE model function to compute derivative.
    t0 : float
         Inital value of time to solve ode from
    y0 : float
        Inital value of dependent variable.
    t1 : float
        final value of independent variable.
    h : float
        step-size, indcates value to increment independent variable
    pars : array-like
        individual paramters that are solved using the curve_fit function.
        pars should contain = [conc, q, a, b, c].
        where a,b,c are solved using curve_fit and conc and q are contstant values
        of concentration wt%C02 and net flow rate respectively.

    Returns:
    --------
    ts : array-like
         Corresponding times at which the dependent variables are solved.
    ys : array-like
         Solution to ODE for given time range.

    Notes:
    --------
    The parameter 'pars' does not contain the same pars as what is passed into
    Improved euler step function nor the ODE model function. These are only the
    parameters which need to be found using curve_fit.
    '''
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
### Concentration Functions
'''
### Curve fitting functions
def curve_fit_conc(t, d, m0):
    '''
    Solve pressure ode with the given paramters

    Parameters:
    -----------
    t : array-like
        array of time values
    d : float
         value of parameter d
    m0 : float
        value of parameter initial mass of system
    Returns:
    --------
    concODE : array-like
            numerical solution to the concentration ODE given the specific parameters
    '''

    a,b, __,_ = find_pars_pressure()
    pars = [a, b, d, m0]
    p0 = PRESSURE[0]
    time, concODE = solve_conc_ode(conc_ODE_model, t[0], CONC[0], t[-1], STEP, p0, pars)
    return concODE
def find_pars_conc():
    '''
    Finds the parameters for the pressure ODE which gives best fit to the data
    using scipy curve_fit.

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
    parameters[0]: float
                   value of parameter d which gives best fit.
    parameters[1]: float
                   value of parameter m0 which gives best fit.

    trainingSize: float
                 gives the index at where claibration point is in the time array.
    '''

    nt = int(np.ceil((TIME_C[-1]-TIME_C[0])/STEP))		# compute number of Euler STEPs to take
    ts = TIME_C[0]+np.arange(nt+1)*STEP			    # x array
    d = 0.1774
    m0 = 11000

    pars = [d, m0]
    ci = np.interp(ts, TIME_C, CONC)
    trainingSize = math.ceil(0.8*len(ts))
    parameters, covar = curve_fit(curve_fit_conc, ts[0:trainingSize], ci[0:trainingSize+1], pars)
    return parameters[0], parameters[1], trainingSize

## Analytical Solution Solver
def conc_analytical_solution(m0, d):
    """
    Computes analytical solution for simplified version of pressure ODE model.
    Used for bench marking.
    Parameters:
    -----------
    m0 : float
        value of parameter m0
    d : float
        value of parameter d

    Returns
    -------
    time : array-like
           array of time values at which the concentration analytical solution
           is solved
    c_ana : array-like
            array of analytical solutions for simplified version of pressure
            ODE model.
    """

    _, q_co2_raw = load_injection_data()                # get injection data
    time = np.linspace(TIME_C[0], TIME_C[-1], 200)      # set up time array
    time = time - TIME_C[0]

    c0 = CONC[0]                       # set up inital concentration
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

    # computes value of c' thats used in ODE (variable name is c_alt) and
    # computes c' c_alt2 that is used for qC02 loss to groundwater
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
    '''
    Computes solution to concentration ODE using improved euler

    Parameters:
    -----------
    f : function
        ODE model function to compute derivative.
    t0 : float
         Inital value of time to solve ode from
    y0 : float
        Inital value of dependent variable.
    t1 : float
        final value of independent variable.
    h : float
        step-size, indcates value to increment independent variable
    pars : array-like
        individual paramters that are solved using the curve_fit function.
        pars should contain: [a, b, d, m0] only, where a,b are solved using
        curve_fit for pressure and d & m0 the concentration one.

    Returns:
    --------
    ts : array-like
         Corresponding times at which the dependent variables are solved.
    ys : array-like
         Solution to ODE for given time range.

    Notes:
    --------
    The parameter 'pars' does not contain the same pars as what is passed into
    Improved euler step function nor the ODE model function. These are only the
    parameters which need to be found using curve_fit.
    '''
    # initialise
    nt = int(np.ceil((t1-t0)/h))		# compute number of Euler steps to take
    ts = t0+np.arange(nt+1)*h			# x array
    ys = 0.*ts							# array to store solution
    ys[0] = y0                          # set intial value
    t1, p_raw = load_pressure_data()
    t2, c02_raw = load_injection_data()
    p = np.interp(ts, t1, p_raw)
    q_arr = np.interp(ts, t2, c02_raw)

    # Get injection flow rate data
    for i in range(nt):
        if ts[i] >= 1998.51:
            q = q_arr[i]
        else:           # Injection started in 1998.51 so is zero until then.
            q = 0
        ys[i+1] = improved_euler_step(f, ts[i], ys[i], h, y0, [q, p[i], p0, *pars])
    return  ts, ys
def solve_conc_ode_ana(f, t0, y0, t1, h, p0, pars=[]):
    '''
    Computes solution to concentration ODE for the benchmarking of the
    analytical solution.

    Parameters:
    -----------
    f : function
        ODE model function to compute derivative.
    t0 : float
         Inital value of time to solve ode from
    y0 : float
        Inital value of dependent variable.
    t1 : float
        final value of independent variable.
    h : float
        step-size, indcates value to increment independent variable
    p0 : float
        inital value of pressure.
    pars : array-like
        individual paramters that are solved using the curve_fit function.
        pars should contain: [a, b, d, m0] only, where a,b are solved using
        curve_fit for pressure and d & m0 the concentration one.

    Returns:
    --------
    ts : array-like
         Corresponding times at which the dependent variables are solved.
    ys : array-like
         Solution to ODE for given time range.

    Notes:
    --------
    The parameter 'pars' does not contain the same pars as what is passed into
    Improved euler step function nor the ODE model function. These are only the
    parameters which need to be found using curve_fit.
    '''
    # initialise
    nt = int(np.ceil((t1-t0)/h))		# compute number of Euler steps to take
    ts = t0+np.arange(nt+1)*h			# x array
    ys = 0.*ts							# array to store solution
    ys[0] = y0                          # set intial value


    t2, c02_raw = load_injection_data() # load in injection data

    p = p0                              # set up pressure
    q = c02_raw[0]                      # set up flow rate

    for i in range(nt):
        ys[i+1] = improved_euler_step(f, ts[i], ys[i], h, y0, [q, p0, p0, *pars])
    return  ts, ys

"""
## Forecasting function
"""
def get_p_conc_forecast(t, pars_conc, pars_pressure, q, q_newInj):
    dq = 0
    p = np.zeros(len(t))
    conc = np.zeros(len(t))
    p[0] = PRESSURE[-1]
    conc[0] = CONC[-1]
    for i in range(len(t) - 1):
        conc[i+1] = improved_euler_step(conc_ODE_model, t[i], conc[i], STEP, CONC[0], [q_newInj, p[i], PRESSURE[0], *pars_conc])
        p[i+1] = improved_euler_step(pressure_ode_model, t[i], p[i], STEP, PRESSURE[0], [dq, q, conc[i], *pars_pressure])

    return t, p, conc
