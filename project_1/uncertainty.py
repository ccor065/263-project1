from plot_lpm import *
from lpm_model_functions import *


#global variables
STEP = 0.1
tp, pA = load_pressure_data()
a_best,b_best,c_best,cp, covariance = find_pars_pressure()


d_best, m0_best, cp_c, conc_covar = find_pars_conc()
tc, cA = load_c02_wt_data()
increase_var = 1
covariance = np.multiply(covariance, increase_var)
conc_covar = np.multiply(conc_covar, increase_var)

mean_conc = [d_best, m0_best]
mean = [a_best, b_best, c_best]

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
def construct_all_samples(N_samples):
    samples_conc = np.random.multivariate_normal(mean_conc, conc_covar, N_samples)
    samples_p = np.random.multivariate_normal(mean, covariance, N_samples)
    samples = np.append(samples_conc, samples_p, 1)
    return samples



if __name__ == "__main__":
    a,b,c,P = grid_search()
    d, m0, C = grid_search_conc()
