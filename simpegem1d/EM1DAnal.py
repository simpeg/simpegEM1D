import numpy as np
from SimPEG import Utils
from scipy.constants import mu_0, pi
from scipy.special import erf
import matplotlib.pyplot as plt
from DigFilter import transFiltImpulse, transFilt, setFrequency

def Hzanal(sig, f, r, flag):

    """

        Analytic solution for half-space (VMD source)
        Tx and Rx are on the surface

    """
    mu0 = 4*np.pi*1e-7
    w = 2*np.pi*f
    k = np.sqrt(-1j*w*mu0*sig)
    Hz = 1./(2*np.pi*k**2*r**5)*(9-(9+9*1j*k*r-4*k**2*r**2-1j*k**3*r**3)*np.exp(-1j*k*r))
    if flag == 'secondary':
        Hzp = -1/(4*np.pi*r**3)
        Hz = Hz-Hzp
    return Hz

def HzanalCirc(sig, f, I, a, flag):

    """

        Analytic solution for half-space (Circular-loop source)
        Tx and Rx are on the surface and receiver is located at the center of the loop.

    """
    mu_0 = 4*np.pi*1e-7
    w = 2*np.pi*f
    k = np.sqrt(-1j*w*mu_0*sig)
    Hz = -I/(k**2*a**3)*(3-(3+3*1j*k*a-k**2*a**2)*np.exp(-1j*k*a))
    if flag == 'secondary':
        Hzp = I/2./a
        Hz = Hz-Hzp
    return Hz

def dHzdsiganalCirc(sig, f, I, a, flag):

    """

        Analytic solution for half-space (Circular-loop source)
        Tx and Rx are on the surface and receiver is located at the center of the loop.

    """
    mu_0 = 4*np.pi*1e-7
    w = 2*np.pi*f
    k = np.sqrt(-1j*w*mu_0*sig)
    perc = 0.001
    Hzfun = lambda m: HzanalCirc(m, f, I, a, flag)
    dHzdsig = (Hzfun(sig+perc*sig)-Hzfun(sig-perc*sig))/(2*perc*sig)
    return dHzdsig


def ColeCole(f, sig_inf=1e-2, eta=0.1, tau=0.1, c=1):
    """
        Computing Cole-Cole model in frequency domain
    """
    if np.isscalar(sig_inf):
        w = 2*np.pi*f
        sigma = sig_inf - sig_inf*eta/(1+(1-eta)*(1j*w*tau)**c)        
    else:
        sigma = np.zeros((f.size,sig_inf.size), dtype=complex)
        for i in range(f.size):
            w = 2*np.pi*f[i]
            sigma[i,:] = Utils.mkvc(sig_inf - sig_inf*eta/(1+(1-eta)*(1j*w*tau)**c))
    return sigma

def BzAnalT(r, t, sigma):
    theta = np.sqrt((sigma*mu_0)/(4*t))
    tr = theta*r
    etr = erf(tr)
    t1 = (9/(2*tr**2) - 1)*etr
    t2 = (1/np.sqrt(pi))*(9/tr + 4*tr)*np.exp(-tr**2)
    hz = (t1 - t2)/(4*pi*r**3)
    return mu_0*hz

def BzAnalCircT(a, t, sigma):
    theta = np.sqrt((sigma*mu_0)/(4*t))
    ta = theta*a
    eta = erf(ta)
    t1 = (3/(np.sqrt(pi)*ta))*np.exp(-ta**2)
    t2 = (1 - (3/(2*ta**2)))*eta
    hz = (t1 + t2)/(2*a)
    return mu_0*hz

def dBzdtAnalCircT(a, t, sigma):
    theta = np.sqrt((sigma*mu_0)/(4*t))
    const = -1/(mu_0*sigma*a**3)
    ta = theta*a
    eta = erf(ta)
    t1 = 3*eta
    t2 = -2/(np.pi**0.5)*ta*(3+2*ta**2)*np.exp(-ta**2)
    dhzdt = const*(t1+t2)
    return mu_0*dhzdt    

def BzAnalCircTCole(a, t, sigma):

    wt, tbase, omega_int = setFrequency(t)
    hz = HzanalCirc(sigma, omega_int/2/np.pi, 1., a, 'secondary')
    # Treatment for inaccuracy in analytic solutions
    ind = omega_int < 0.2
    hz[ind] = 0.
    hzTD, f0 = transFilt(hz, wt, tbase, omega_int, t)
    return hzTD*mu_0

def dBzdtAnalCircTCole(a, t, sigma):
    
    wt, tbase, omega_int = setFrequency(t)
    hz = HzanalCirc(sigma, omega_int/2/np.pi, 1., a, 'secondary')    
    # Treatment for inaccuracy in analytic solutions
    ind = omega_int < 0.2
    hz[ind] = 0.

    dhzdtTD = -transFiltImpulse(hz, wt, tbase, omega_int, t)

    return dhzdtTD*mu_0    
