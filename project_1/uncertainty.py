from conc_2 import *
from plot_lpm import *


#global variables
STEP = 0.1
tp, pA = load_pressure_data()
a_best,b_best,c_best,covariance = find_pars_pressure_covariance()
covariance = np.multiply(covariance, 80)
mean = [a_best, b_best, c_best]
v = 0.1


def grid_search():
    
	# number of values considered for each parameter within a given interval
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
        
    plot_posterior3D(a, b, c, P=P)

    return a,b,c,P        
        
def construct_samples(a,b,c,P,N_samples):
    
    
    samples = np.random.multivariate_normal(mean, covariance, N_samples)
    plot_samples3D(a,b,c,P,samples)

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


if __name__ == "__main__":
    a,b,c,P = grid_search()
    samples = construct_samples(a,b,c,P,100)
    model_emsemble(samples)


