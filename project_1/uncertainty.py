from plot_lpm import *
from lpm_model_functions import *


#global variables
STEP = 0.1
tp, pA = load_pressure_data()
a_best,b_best,c_best,covariance = find_pars_pressure_covariance()


d_best, m0_best, conc_covar = find_pars_conc_covar()
tc, cA = load_c02_wt_data()
covariance = np.multiply(covariance, 80)
conc_covar = np.multiply(conc_covar, 80)

mean_conc = [d_best, m0_best]
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
def grid_search_conc():

	# number of values considered for each parameter within a given interval
    # larger N provides better visualization of s(theta) and posterior density function
    N = 5

    d = np.linspace(d_best/4,d_best*1.75, N)
    m0 = np.linspace(m0_best/4,m0_best*1.75, N)

    # grid of paramter values
    D, M0 = np.meshgrid(d, m0, indexing='ij')

    # empty 3D matrix for objective function
    S = np.zeros(D.shape)

    for i in range(len(d)):
        for j in range(len(m0)):
            tm, cm = solve_conc_ode(conc_ODE_model, tc[0],
                                    cA[0], tc[-1], STEP, pA[0], [a_best, b_best, d[i], m0[j]])
            cm = np.interp(tc, tm, cm)
            S[i, j] = np.sum((cA-cm)**2)/v

    #compute posterior
    C = np.exp(-S/2.)
    Cint = np.sum(C)*(d[1]-d[0])*(m0[1]-m0[0])
    C = C/Cint

    #plot_posterior3D(a, b, c, P=P)

    return d, m0, C

def construct_samples_pressure(a,b,c,P,N_samples):


    samples = np.random.multivariate_normal(mean, covariance, N_samples)
    #plot_samples3D(a,b,c,P,samples)

    return samples
def construct_samples_conc(d,m0, C, N_samples):


    samples = np.random.multivariate_normal(mean_conc, conc_covar, N_samples)

    #plot_samples3D(a,b,c,P,samples)

    return samples
def construct_all_samples(N_samples):
    samples_conc = np.random.multivariate_normal(mean_conc, conc_covar, N_samples)
    samples_p = np.random.multivariate_normal(mean, covariance, N_samples)
    samples = np.append(samples_conc, samples_p, 1)
    return samples

def model_emsemble_pressure(samples):

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
def model_emsemble_concentration(samples):

    f, ax  = plt.subplots(1,1)
    for d, m0, a, b ,c in samples:
         tm, cm = solve_conc_ode(conc_ODE_model, tc[0],
                                        cA[0], tc[-1], STEP, pA[0], [a_best, b_best, d, m0])
         cm = np.interp(tc, tm, cm)
         ax.plot(tc, cm, 'b', lw=0.25,alpha=0.2)
    ax.plot([],[],'k-', lw=0.5,alpha=0.4, label='model ensemble')
    ax.axvline(2004, color='b', linestyle=':', label='calibration/forecast')
    ax.errorbar(tc,cA, yerr=0.003,fmt='ro', label='data')
    ax.set_xlabel('time')
    ax.set_ylim(0, 0.08)
    ax.set_ylabel('conc')
    ax.legend()
    plt.show()
def model_emsemble_forecast(samples):
        fig, (ax1, ax2) = plt.subplots(1, 2)
        fig.set_figwidth(13)
        plt.subplots_adjust(None, None, 0.85 ,None, wspace=None, hspace=None)


        # model
        t_ode, p_ode = solve_pressure_ode(pressure_ode_model, TIME_P[0], PRESSURE[0], TIME_P[-1], STEP, PARS_P)
        # plot the data observations
        p1 = ax1.scatter(TIME_P, PRESSURE,color='k', s= 9, label = "Observations")
        # plot the model solution
        ax1.plot(t_ode, p_ode, color = 'r', label = "ODE model")
        tc_ode, c_ode = solve_conc_ode(conc_ODE_model, TIME_C[0], CONC[0], TIME_C[-1], STEP, PRESSURE[0], PARS_C)
        p2, = ax2.plot(tc_ode, c_ode, color = 'r', label = "ODE model")
        ax2.scatter(TIME_C, CONC, color = 'k', s= 9, label ="Observations")

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
        colours = ['orange', 'green', 'red', 'blue', 'steelblue'] #for graph
        labels = ['qc02 = 0.0 kg/s', 'qc02 = %.2f kg/s',  'qc02 = %.2f kg/s ','qc02 = %.2f kg/s ','qc02 = %.2f kg/s'] #for graph
        for i in range(len(injRates)):
            q_net = q_prod[-1] - (q_inj[-1])*injRates[i]
            q_newInj = (q_inj[-1])*injRates[i]
            t, p, c = get_p_conc_forecast(ts, PARS_C, PARS_P, p_ode[-1], c_ode[-1], q_net, q_newInj)
            ax1.plot(t, p, color=colours[i])
            ax2.plot(t, c, color=colours[i])
        '''
        for a, b, c, d, m0 in samples:
            pars_c =[a, b, d, m0]
            pars_p = [a,b,c]
            for i in range(len(injRates)):
                q_net = q_prod[-1] - (q_inj[-1])*injRates[i]
                q_newInj = (q_inj[-1])*injRates[i]
                t, p, c = get_p_conc_forecast(ts, pars_c, pars_p, p_ode[-1], c_ode[-1], q_net, q_newInj)
                ax1.plot(t, p, color=colours[i])
                ax2.plot(t, c, color=colours[i])
        '''
        ax2.axhline(0.10, linestyle = "--", color = 'grey', label = '10 wt% C02' )    #ax1.axvline(t_ode[calibrationPointP], linestyle = '--', label = 'Calibration Point')
        ax1.axhline(PRESSURE[0], linestyle = "--", color = 'grey', label = 'Ambient Pressure P0')
        ax2.set_title("Concentration C02wt%")
        ax1.set_title("Pressure MPa")
        plt.suptitle("30 Year Forecast for Ohaaki Geothermal Field")
        ax2.legend(bbox_to_anchor=(1,1), loc="upper left")
        ax1.legend()
        ax1.set_xlabel("Time(year)")
        ax2.set_xlabel("Time(year)")

        ax1.set_ylabel("Pressure MPa")
        ax2.set_ylabel("C02 Concentration (wt proportion)")
        plt.savefig('forecast_no_uncertain',dpi=300)

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
<<<<<<< HEAD
    p2, = ax2.plot(tc_ode, c_ode, color = 'r', label = "ODE model")
    ax2.scatter(TIME_C, CONC, color = 'k', s= 9, label ="Observations")

=======
    #p2, = ax2.plot(tc_ode, c_ode, color = 'r', label = "ODE model")
    #ax2.scatter(TIME_C, CONC, color = 'k', s= 9, label ="Observations")
    
>>>>>>> da0f820cbd3310273e00084ee64ee7f841dd89bc
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
<<<<<<< HEAD
    for d, m0, a, b, c in samples:
        pars_c = [a,b,d,m0]
        pars_p = [a,b, c]
        for i in range(len(injRates)):
                q_net = q_prod[-1] - (q_inj[-1])*injRates[i]
                q_newInj = (q_inj[-1])*injRates[i]
                t, p, co2 = get_p_conc_forecast(ts, [a,b,d,m0], [a,b,c], p_ode[-1], c_ode[-1], q_net, q_newInj)
                ax1.plot(t, p, color=colours[i], alpha=0.1)
                ax2.plot(t, co2, color=colours[i], alpha=0.1)
=======
    
    for i in range(len(injRates)):
        for a, b, c in samples:
            q_net = q_prod[-1] - (q_inj[-1])*injRates[i]
            q_newInj = (q_inj[-1])*injRates[i]
            t, p, co2 = get_p_conc_forecast(ts, [a,b,d,m0], [a,b,c], p_ode[-1], c_ode[-1], q_net, q_newInj)
            ax1.plot(t, p, color=colours[i], alpha=0.1)
            #ax2.plot(t, co2, color=colours[i], alpha=0.1)
>>>>>>> da0f820cbd3310273e00084ee64ee7f841dd89bc

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
    d, m0, C = grid_search_conc()
    samples_pressure = construct_samples_pressure(a,b,c,P,100)
    #model_emsemble_pressure(samples_pressure)
    samples_conc = construct_samples_conc(d,m0,C,100)




    samples = np.append(samples_conc, samples_pressure, 1)

    #model_emsemble_concentration(samples)
    model_emsemble_forecast(samples)
