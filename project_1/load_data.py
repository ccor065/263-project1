import numpy as np

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
