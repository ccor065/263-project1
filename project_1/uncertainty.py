from plot_lpm import *


#global variables
STEP = 0.1
tp, pA = load_pressure_data()
a_best,b_best,c_best,covariance = find_pars_pressure_covariance()
covariance = np.multiply(covariance, 80)
mean = [a_best, b_best, c_best]
d, m0, calibrationPointC = find_pars_conc()
PARS_P = [a_best, b_best, c_best]
PARS_C = [a_best, b_best, d, m0]


v = 0.1


def grid_search():
    
	# number of values considered for each parameter within a given interval
    # larger N provides better visualization of s(theta) and posterior density function
    N = 5

    a = np.linspace(a_best/4,a_best*1.75, N)
    b = np.linspace(b_best/4,b_best*1.75, N)
    c = np.linspace(c_best/4, c_best*1.75, N)
    
    # grid of paramter values
    A, B, C = np.meshgrid(a, b, c, indexing='ij')
    
    # empty 3D matrix for objective function
    S = np.zeros(A.shape)
    
    for i in range(len(a)):
        for j in range(len(b)):
            for k in range(len(c)):
                tm, pm = solve_pressure_ode(pressure_ode_model, tp[0], 
                                        pA[0], tp[-1], STEP, [a[i], b[j], c[k]])
                pm = np.interp(tp, tm, pm)
                S[i, j, k] = np.sum((pA-pm)**2)/v
    
    #compute posterior
    P = np.exp(-S/2.)
    Pint = np.sum(P)*(a[1]-a[0])*(b[1]-b[0])*(c[1]-c[0])
    P = P/Pint    
        
    #plot_posterior3D(a, b, c, P=P)

    return a,b,c,P        
        
def construct_samples(a,b,c,P,N_samples):
    
    
    samples = np.random.multivariate_normal(mean, covariance, N_samples)
    #plot_samples3D(a,b,c,P,samples)

    return samples


def model_emsemble(samples):
    
    f, ax  = plt.subplots(1,1)
    
    for a,b,c in samples:
         tm, pm = solve_pressure_ode(pressure_ode_model, tp[0], 
                                        pA[0], tp[-1], STEP, [a, b, c])
         pm = np.interp(tp, tm, pm)
         ax.plot(tp, pm, 'b', lw=0.25,alpha=0.2)
    ax.plot([],[],'k-', lw=0.5,alpha=0.4, label='model ensemble')
    ax.axvline(2004, color='b', linestyle=':', label='calibration/forecast')
    ax.errorbar(tp,pA,yerr=v,fmt='ro', label='data')         
    ax.set_xlabel('time')
    ax.set_ylabel('pressure')
    ax.legend()
    plt.show()

def forecast_ensemble(samples):
    fig, (ax1) = plt.subplots(1)
    fig.set_figwidth(13)
    plt.subplots_adjust(None, None, 0.85 ,None, wspace=None, hspace=None)


    # model
    t_ode, p_ode = solve_pressure_ode(pressure_ode_model, TIME_P[0], PRESSURE[0], TIME_P[-1], STEP, PARS_P)
    # plot the data observations
    p1 = ax1.scatter(TIME_P, PRESSURE,color='k', s= 9, label = "Observations")
    # plot the model solution
    ax1.plot(t_ode, p_ode, color = 'r', label = "ODE model")
    tc_ode, c_ode = solve_conc_ode(conc_ODE_model, TIME_C[0], CONC[0], TIME_C[-1], STEP, PRESSURE[0], PARS_C)
    #p2, = ax2.plot(tc_ode, c_ode, color = 'r', label = "ODE model")
    #ax2.scatter(TIME_C, CONC, color = 'k', s= 9, label ="Observations")
    
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
    colours = ['orange', 'green', 'red', 'blue', 'slategrey'] #for graph
    labels = ['qc02 = 0.0 kg/s', 'qc02 = %.2f kg/s',  'qc02 = %.2f kg/s ','qc02 = %.2f kg/s ','qc02 = %.2f kg/s'] #for graph
    
    for i in range(len(injRates)):
        for a, b, c in samples:
            q_net = q_prod[-1] - (q_inj[-1])*injRates[i]
            q_newInj = (q_inj[-1])*injRates[i]
            t, p, co2 = get_p_conc_forecast(ts, [a,b,d,m0], [a,b,c], p_ode[-1], c_ode[-1], q_net, q_newInj)
            ax1.plot(t, p, color=colours[i], alpha=0.1)
            #ax2.plot(t, co2, color=colours[i], alpha=0.1)

    #ax2.axhline(0.10, linestyle = "--", color = 'crimson', label = '10 wt% C02' )    #ax1.axvline(t_ode[calibrationPointP], linestyle = '--', label = 'Calibration Point')
    ax1.axhline(PRESSURE[0], linestyle = "--", color = 'orange', label = 'Ambient Pressure P0')
    #ax2.set_title("Concentration C02wt%")
    ax1.set_title("Pressure MPa")
    plt.suptitle("30 Year Forecast for Ohaaki Geothermal Field")
    #ax2.legend(bbox_to_anchor=(1,1), loc="upper left")
    ax1.legend()
    plt.show()


if __name__ == "__main__":
    a,b,c,P = grid_search()
    samples = construct_samples(a,b,c,P,50)
    model_emsemble(samples)
    forecast_ensemble(samples)
    


