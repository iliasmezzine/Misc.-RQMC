import numpy as np
import matplotlib.pyplot as plt

def times(T,steps):
    return [T*i/steps for i in range(1,steps+1)]

def covmat(t):
    x = np.zeros([len(t),len(t)])
    for i in range(len(t)):
        for j in range(len(t)):
            x[i,j] = min(t[i],t[j])
    return np.array(x)

def chol(M):
    try :
        return np.array(np.linalg.cholesky(M))
    except:
        return "Non DP-Matrix"

def brw_path(T,steps): # brownian path with steps [t[0],...,t[-1]]
    chol_dec = chol(covmat(times(T,steps)))
    path = np.dot(chol_dec,np.random.normal(0,1,steps))
    return path


def gbm_path(s0,T,steps,r,v): # geometric brownian path with steps [t[0],...,t[-1]] and chosen vol, drift
    bm = brw_path(T,steps)
    gbm = [s0*np.exp((r-0.5*v**2)*times(T,steps)[i] + v*bm[i]) for i in range(len(bm))]
    return np.array(gbm)

def asian(s0,T,steps,r,v,K):
    return np.exp(-r*T)*max(np.mean(gbm_path(s0,T,steps,r,v))-K,0)

def asianMC(s0,T,steps,r,v,K,nSim):
    return np.mean([asian(s0,T,steps,r,v,K) for i in range(nSim)])

def bm_path_drifted(drift,T,steps):
    
    path = brw_path(T,steps) 
    drifted_path = [path[i]+times(T,steps)[i]*drift for i in range(len(path))]  
    return np.array(drifted_path)

def gbm_path_drifted(drift,s0,T,steps,r,v):
    bm = bm_path_drifted(drift,T,steps)
    gbm = [s0*np.exp((r-0.5*v**2)*times(T,steps)[i] + v*bm[i]) for i in range(len(bm))]
    return np.array(gbm)

def asianGirsanov(drift,s0,T,steps,r,v,K):
    
    path = brw_path(T,steps)
    
    mult_drift = np.exp(-drift*path[-1] - 0.5*(drift**2)*times(T,steps)[-1])
    pathD = [path[i]+drift*times(T,steps)[i] for i in range(len(path))]
    gbmD = [s0*np.exp((r-0.5*v**2)*times(T,steps)[i] + v*pathD[i]) for i in range(len(pathD))]
    
    opt_price = np.exp(-r*T)*mult_drift*max(np.mean(gbmD)-K,0)

    return opt_price

def opt(drift,s0,T,steps,r,v,K):
    
    t = times(T,steps)
    path = [s0*np.exp((r-0.5*v**2 + v*drift)*t[i]) for i in range(len(t))]
    return np.mean(path) - K


def plotter(rng): #plots the mean spot against the drift
    points = np.linspace(-rng,rng,10000)
    y = [opt(x,s0,T,steps,r,v,K) for x in points]
    plt.plot(points,y)
    plt.show()
### parameters ###
    

s0 = 100
T = 1
steps = 5
r = 0.05
v = 0.3
K = 150
nSim = 10000
drift = 2.5

### tests ###

drifted_samples = [asianGirsanov(drift,s0,T,steps,r,v,K) for i in range(1,10001)]
regular_samples = [asian(s0,T,steps,r,v,K) for i in range (1,10001)]

drifted_means = [np.mean(drifted_samples[:i]) for i in range(len(drifted_samples))]
regular_means = [np.mean(regular_samples[:i]) for i in range(len(regular_samples))]

plt.plot(drifted_means,label="drifted means")
plt.plot(regular_means,label = "regular means")
plt.legend()
plt.show()


#plt.plot(means)

