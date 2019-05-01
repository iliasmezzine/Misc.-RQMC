import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt
import sobol_seq as sob
from numpy.linalg import eigh

### Basket QMC ####

def PCA(cm): #computes the PCA*sqrt(D)
    pcadec = eigh(cm)

    #flip eigval/eigvect to descending order
    eigv = np.flip(pcadec[0],0)
    for x in range(pcadec[1].shape[0]):
        pcadec[1][x,:] = np.flip(pcadec[1][x,:],0)
    #compute Psqrt(D)
    return np.dot(pcadec[1],np.diag(np.sqrt(eigv)))

def ninv(x): #Returns the inverse cumulative standard function of some point x in ]0,1[
    return norm.ppf(x)

def sobol(dim,seed): # Generates the seed-th term of the p-dimensional Sobol Sequence (WARNING P MUST BE <=40)
    return list(sob.i4_sobol(dim,seed)[0])

def means_of(x):
    return [np.mean(x[:i]) for i in range(len(x))]

def std_of(x):
    return [np.std(x[:i])/np.sqrt(i+1) for i in range(len(x))]

def N(x):
    return norm.cdf(x)

def vols(coefs): #Vols matrix for the bsk
    variances = [np.sum((coefs**2)[i]) for i in range(np.shape(coefs)[0])]
    return np.array([np.sqrt(variances[i]) for i in range(len(variances))])

def covars(coefs): #Correl matrix of end multidim. BM Sample
    
    d = np.shape(coefs)[0]
    v = vols(coefs)
    temp = np.zeros([d,d])
    
    for i in range(d):
        for j in range(d):
            temp[i,j] = np.dot(coefs[i],coefs[j])/(v[i]*v[j])
    return temp

def GBMSample(spots,coefs,r,T):
    BM = np.dot(np.linalg.cholesky(covars(coefs)*T),np.random.normal(0,1,np.shape(coefs)[0]))
    v = vols(coefs)
    return [spots[i]*np.exp((r-0.5*v[i]**2)*T + v[i]*BM[i]) for i in range(len(spots))]

def GBMSobol(spots,coefs,r,T,seed):
    BM = np.dot(np.linalg.cholesky(covars(coefs)*T),ninv(sobol(len(spots),seed)))
    v = vols(coefs)
    return [spots[i]*np.exp((r-0.5*v[i]**2)*T + v[i]*BM[i]) for i in range(len(spots))]

def GBMPCA(spots,coefs,r,T,seed):
    BM = np.dot(PCA(covars(coefs)*T),ninv(sobol(len(spots),seed)))
    v = vols(coefs)
    return [spots[i]*np.exp((r-0.5*v[i]**2)*T + v[i]*BM[i]) for i in range(len(spots))]
    

def BskSample(spots,weights,coefs,r,T,K):
    wEnds = np.dot(GBMSample(spots,coefs,r,T),weights)
    return np.exp(-r*T)*max(wEnds - K,0)

def BskSobolSample(spots,weights,coefs,r,T,K,seed): #Basket Sample with Cholesky decomposition + Sobol at given seed
    wEnds = np.dot(GBMSobol(spots,coefs,r,T,seed),weights)
    return np.exp(-r*T)*max(wEnds - K,0)

def BskPCASample(spots,weights,coefs,r,T,K,seed): #Basket Sample with Cholesky decomposition + Sobol at given seed
    wEnds = np.dot(GBMPCA(spots,coefs,r,T,seed),weights)
    return np.exp(-r*T)*max(wEnds - K,0)

def BskMCRegular(spots,weights,coefs,r,T,K,nSim):
    x = [BskSample(spots,weights,coefs,r,T,K) for i in range(nSim)]
    return np.mean(x)


def BskMC(spots,weights,coefs,r,T,K,nSim):
    return means_of([BskSample(spots,weights,coefs,r,T,K) for i in range(nSim)])

def BskQMC(spots,weights,coefs,r,T,K,nSim):
    return means_of([BskSobolSample(spots,weights,coefs,r,T,K,i+300) for i in range(nSim)])

def BskPCA(spots,weights,coefs,r,T,K,nSim):
    return means_of([BskPCASample(spots,weights,coefs,r,T,K,i+300) for i in range(nSim)])

### Adaptation of the distance matrix ###


def distBskQMC(max_time,disc_time,max_strike,disc_strike,nSim,spots,T,r,coefs):
    #Pour chaque strike et maturité, on génère d'abord les nSim échantillons, ensuite on remplit toutes les matrices puis on calcule la distance
    strikes = np.linspace(0,max_strike,disc_strike)
    times = np.linspace(0.01,max_time,disc_time)
    
    sims_Sobol = np.ndarray([len(strikes),len(times),nSim])
    sims_MC = np.ndarray([len(strikes),len(times),nSim])
    sims_PCA = np.ndarray([len(strikes),len(times),nSim])
    
    #First, fill the sims matrix with mean prices
    # BskSample(spots,weights,coefs,r,T,K)
    # BskSobolSample(spots,weights,coefs,r,T,K,seed)
    # BskPCASample(spots,weights,coefs,r,T,K,seed)
    
    for i in range(len(strikes)):
        print("strike : " + str(strikes[i]))
        for j in range(len(times)):
           print("times : " + str(times[j]))
           means_Sobol = means_of([BskSobolSample(spots,weights,coefs,r,times[j],strikes[i],n+300) for n in range(nSim)])
           means_MC = means_of([BskSample(spots,weights,coefs,r,times[j],strikes[i]) for n in range(nSim)])
           means_PCA = means_of([BskPCASample(spots,weights,coefs,r,times[j],strikes[i],n+300) for n in range(nSim)])
           for k in range(nSim):
               sims_Sobol[i,j,k] = means_Sobol[k]
               sims_MC[i,j,k] = means_MC[k]
               sims_PCA[i,j,k] = means_PCA[k]
               
               
    #Then, return the distance matrix
    dist_Sobol = [np.sqrt(np.sum((sims_Sobol[:,:,i]-sims_Sobol[:,:,-1])**2)) for i in range(nSim)] #distance with Sobol & Chol
    dist_MC = [np.sqrt(np.sum((sims_MC[:,:,i]-sims_MC[:,:,-1])**2)) for i in range(nSim)] #distance with CV
    dist_PCA = [np.sqrt(np.sum((sims_PCA[:,:,i]-sims_PCA[:,:,-1])**2)) for i in range(nSim)] #distance with Sobol & PCA

    plt.plot(dist_Sobol, dashes=[1,1], color ="black" ,label="QMC Basket + Cholesky & Sobol ")    
    plt.plot(dist_PCA, dashes=[1,1] , color = "green", label="QMC Basket + PCA & Sobol")
    plt.plot(dist_MC, color = "blue",label="MC Basket + CV")
                
    plt.legend()
    plt.show() 
    return [dist_MC,dist_Sobol ,dist_PCA ]
    

### parameters ###

coefs = np.array([[0.3,0.5,0.20],[0.29,0.15,0.25],[0.21,0.23,0.2]])
spots = [100,100,100]
weights = [1/3,1/3,1/3]
nSim = 10000
T = 0.5
r = 0.25
K = 86

max_time = 3
disc_time = 5
max_strike = 150
disc_strike = 5

x = distBskQMC(max_time,disc_time,max_strike,disc_strike,nSim,spots,T,r,coefs)








