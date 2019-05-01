import numpy as np
import sobol_seq as sb

#############################################################
################### RQMC FUNCTIONS ##########################
#############################################################

def sobol(dim,seed):
    return np.array(sb.i4_sobol(dim,seed)[0])

def rndunif(dim):
    return np.random.uniform(0,1,dim)

def fb(x): #float to binary floating point list. precision is kept from :20 bin. floating points
    binl = []
    s=x
    for i in range(20):
        binl.append(int(2*s))
        s = 2*s - int(2*s)
    return np.array(binl)

def rev(listx): #reverse function of fb
    x = np.array([2**-(i+1) for i in range(len(listx))])
    return np.dot(x,listx)
def xor1(x,y):
    if x!=y:
        return 1
    return 0
def xorlist(lst1,lst2): #XOR a list bitwise
    return np.array([xor1(lst1[i],lst2[i]) for i in range(len(lst1))])

def fXOR(x,y):#XOR bitwise 2 floats 
    return rev(xorlist(fb(x),fb(y)))

vXOR2 = np.vectorize(fXOR)

#RQMC pour SOBOL : generate separate sobol batches and digitally shift each batch with a random uniform vector

def dShift_sep(dim,batch_size,nsamples): 
    batch = [] 
    for n in range(nsamples):
        U = rndunif(dim)
        batch+=[[vXOR2(sobol(dim,s),U) for s in range(n*batch_size,(n+1)*batch_size)]] 
        print("done shifting batch : " + str(n+1))
    return batch

def dShift(dim,batch_size,nsamples): 
    batch = []
    unshifted = [sobol(dim,s) for s in range(batch_size,2*batch_size)]
    for n in range(nsamples):
        U = rndunif(dim)
        batch+=[[vXOR2(S,U) for S in unshifted]] 
        print("done shifting batch : " + str(n+1))
    return batch

dim = 10
batch_size = 1000
nsamples = 10

x = dShift(dim,batch_size,nsamples)
