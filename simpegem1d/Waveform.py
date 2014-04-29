import numpy as np
from scipy.interpolate import interp1d

def TriangleFun(time, ta, tb):
    """
        Triangular wave function
        
        * time: input time array
        * ta: starting time for up-linear ramp
        * tb: ending time for down-linear ramp

        .. math::

            I(t) = \\frac{1}{t_a}t, \ 0 \ge t \le t_a

            I(t) = \\frac{-1}{t_b-t_a}(t-t_b), t_a < t < t_b

            I(t) = 0, \ t \ge t_b

    """      
    
    out = np.zeros(time.size)
    out[time<=ta] = 1/ta*time[time<=ta]
    out[(time>ta)&(time<tb)] = -1/(tb-ta)*(time[(time>ta)&(time<tb)]-tb)
    return out

def TriangleFunDeriv(time, ta, tb):
    """
        Triangular wave function
        
        * time: input time array
        * ta: starting time for up-linear ramp
        * tb: ending time for down-linear ramp

        .. math::

            I(t) = \\frac{1}{t_a}, \ 0 \ge t \le t_a

            I(t) = \\frac{-1}{t_b-t_a}, t_a < t < t_b

            I(t) = 0, \ t \ge t_b

    """        
    out = np.zeros(time.size)
    out[time<=ta] = 1/ta
    out[(time>ta)&(time<tb)] = -1/(tb-ta)
    return out   

def VTEMFun(time, ta, tb, a):
    """
        VTEM wave function
        
        * time: input time array
        * ta: starting time for linear ramp
        * tb: ending time for linear ramp
        * a: slope of exponential function

        .. math::

            I(t) = \\frac{1-e^{-at}}{(1-e^{-a})}, \ 0 \ge t \le t_a

            I(t) = \\frac{-1}{t_b-t_a}(t-t_b), \t_a < t < t_b

            I(t) = 0, \ t \ge t_b

    """    
    out = np.zeros(time.size)
    out[time<=ta] = (1-np.exp(-a*time[time<=ta]/ta))/(1-np.exp(-a))
    out[(time>ta)&(time<tb)] = -1/(tb-ta)*(time[(time>ta)&(time<tb)]-tb)
    return out

def CausalConv(array1, array2, time):
    """
        Evaluate convolution for two causal functions.
        
        * array1: array for \\\\(\\\\ f_1(t)\\\\)
        * array2: array for \\\\(\\\\ f_1(t)\\\\)
        * time: array for time
        
        .. math::
        
            Out(t) = (f_1 \otimes f_2) (t) = \int_{0}^{t} f_1(\\tau) f_2(t-\\tau) d\\tau
            
    """
    
    if array1.shape == array2.shape == time.shape:
        out = np.convolve(array1, array2)
        # print time[1]-time[0]
        return out[0:np.size(time)]*(time[1]-time[0])
    else:
        print "Give me same size of 1D arrays!!"       

def RectFun(time, ta, tb):
    """
        Rectangular wave function
        
        * time: input time array
        * ta: starting time for Rectangular wave
        * tb: ending time for Rectangular wave

        .. math::

            I(t) = 1, 0 < t \le t_a

            I(t) = -1, t_a < t < t_b

            I(t) = 0, t \le t_a  \ \\text{or} \ t \ge t_b

    """    
    out = np.zeros(time.size)
    out[time<=ta] = 1
    out[(time>ta)&(time<tb)] = -1
    return out  

def CenDiff(val, tin, tout):
    """
       Compute time derivative of given function using 
       central difference

       * val: function value
       * tin: time of function
       * TODO: need to be fixed ... for general types of wave form
    """
    dbdtm = np.diff(val, n=1)/np.diff(tin, n=1)
    tm = np.diff(time_trial, n=1)*0.5 + time_trial[:-1]

    return tint
