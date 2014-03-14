import numpy as np

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

def ColeCole(f, sig_inf=1e-2, eta=0.1, tau=0.1, c=1):
    """
        Computing Cole-Cole model in frequency domain
    """
    sigma = []
    for i in range(f.size):

        w = 2*np.pi*f[i]
        sigma_temp = sig_inf + sig_inf*eta/(1+(1-eta)*(1j*w*tau)**c)
        sigma.append(sigma_temp)

    return sigma
