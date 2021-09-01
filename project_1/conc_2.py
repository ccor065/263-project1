import numpy as np
from matplotlib import pyplot as plt
import math
from scipy.optimize import curve_fit
from project_functions import *
from load_data import *

TIME, CONC = load_c02_wt_data()
TIME_P, PRESSURE = getT_P_ode()
STEP = 0.1


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
def curve_fit_conc(t, d, m0):
    a,b, __,_ = find_pars_pressure()
    pars = [a, b, d, m0]
    p0 = PRESSURE[0]
    time, conc = solve_conc_ode(conc_ODE_model, t[0], CONC[0], t[-1], STEP, p0, pars)
    return conc
def find_pars_conc():

    nt = int(np.ceil((TIME[-1]-TIME[0])/STEP))		# compute number of Euler STEPs to take
    ts = TIME[0]+np.arange(nt+1)*STEP			    # x array
    d = 0.1774
    m0 = 11000

    pars = [d, m0]
    ci = np.interp(ts, TIME, CONC)
    trainingSize = math.ceil(0.8*len(ts))
    parameters, covar = curve_fit(curve_fit_conc, ts[0:trainingSize], ci[0:trainingSize+1], pars)
    return parameters[0], parameters[1]
def plot_conc_model():

    d, m0 = find_pars_conc()
    print(d, m0)

    a,b, __,_ = find_pars_pressure()

    pars = [a, b, d, m0]
    step = 0.1
    #subpars = [a, b, PRESSURE[0]]
    p0 = PRESSURE[0]
    t_ode, c_ode = solve_conc_ode(conc_ODE_model, TIME[0], 0.03, TIME[-1], STEP, p0, pars)
    plt.plot(t_ode, c_ode, label = "ode model")
    plt.plot(TIME, CONC, 'o', label='data')
    plt.legend()
    plt.ylim((0, 0.075))
    plt.show()

if __name__ == "__main__":
    plot_conc_model()
