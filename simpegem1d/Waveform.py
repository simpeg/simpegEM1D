import numpy as np
from scipy.interpolate import interp1d

def TriangleFun(time, ta, tb):
    out = np.zeros(time.size)
    out[time<=ta] = 1/ta*time[time<=ta]
    out[(time>ta)&(time<tb)] = -1/(tb-ta)*(time[(time>ta)&(time<tb)]-tb)
    return out

def TriangleFunDeriv(time, ta, tb):
    """    
    """    
    out = np.zeros(time.size)
    out[time<=ta] = 1/ta
    out[(time>ta)&(time<tb)] = -1/(tb-ta)
    return out   

def VTEMFun(time, ta, tb, a):
    out = np.zeros(time.size)
    out[time<=ta] = (1-np.exp(-a*time[time<=ta]/ta))/(1-np.exp(-a))
    out[(time>ta)&(time<tb)] = -1/(tb-ta)*(time[(time>ta)&(time<tb)]-tb)
    return out

def CausalConv(array1, array2, time):
    """
        Evaluate convolution for two causal functions.
        >>Input
        array1: array for fun1(t) 
        array2: array for fun2(t)
        time: array for time
        
        .. math::
        
            Out(t) = \int_{0}^{t} fun1(a) fun2(t-a) da
            
    """
    
    if array1.shape == array2.shape == time.shape:
        out = np.convolve(array1, array2)
        # print time[1]-time[0]
        return out[0:np.size(time)]*(time[1]-time[0])
    else:
        print "Give me same size of 1D arrays!!"       

def RectFun(time, ta, tb):
    """
        .. math::

        I(t) = 1, 0 < t \le t_a

        I(t) = -1, t_a < t < t_b

        I(t) = 0, t \le t_a or t \ge t_b
    """    
    out = np.zeros(time.size)
    out[time<=ta] = 1
    out[(time>ta)&(time<tb)] = -1
    return out  
          
def RectFun(time, ta, tb):
    """
        .. math::

        I(t) = 1, 0 < t \le t_a

        I(t) = -1, t_a < t < t_b

        I(t) = 0, t \le t_a or t \ge t_b
    """    
    out = np.zeros(time.size)
    out[time<=ta] = 1
    out[(time>ta)&(time<tb)] = -1
    return out  

def CenDiff(val, tin, tout):
    """
       
    """
    dbdtm = mu_0*np.diff(val, n=1)/np.diff(tin, n=1)
    tm = np.diff(time_trial, n=1)*0.5 + time_trial[:-1]

    return out, tint
