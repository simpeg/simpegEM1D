import numpy as np
import matplotlib.pyplot as plt

def plotLayer(sig, LocSigZ, xscale='log', ax=None, **kwargs):
    """
        Plot Conductivity model for the layered earth model
    """
    sigma = np.repeat(sig, 2, axis=0)
    z = np.repeat(LocSigZ[1:], 2, axis=0)
    z = np.r_[LocSigZ[0], z, LocSigZ[-1]]
    sig_min = sig.min()*0.5
    sig_max = sig.max()*2

    if xscale == 'linear' and sig.min() == 0.:
        sig_min = -sig.max()*0.5
        sig_max = sig.max()*2

    if ax==None:
        plt.xscale(xscale)
        ax.set_xlim(sig_min, sig_max)
        plt.ylim(z.min(), z.max())
        plt.xlabel('Conductivity (S/m)', fontsize = 14)
        plt.ylabel('Depth (m)', fontsize = 14)        
        for locz in LocSigZ:
            plt.plot(np.linspace(sig_min, sig_max, 100), np.ones(100)*locz, 'b--', lw = 0.5)
        return plt.plot(sigma, z, 'k-', **kwargs)        

    else:
        ax.set_xscale(xscale)
        ax.set_xlim(sig_min, sig_max)
        ax.set_ylim(z.min(), z.max())
        ax.set_xlabel('Conductivity (S/m)', fontsize = 14)
        ax.set_ylabel('Depth (m)', fontsize = 14)
        for locz in LocSigZ:
            ax.plot(np.linspace(sig_min, sig_max, 100), np.ones(100)*locz, 'b--', lw = 0.5)
        return ax.plot(sigma, z, 'k-', **kwargs)

def plotComplexData(frequency, val, xscale='log', ax=None, **kwargs):
    """
        Plot Complex EM responses
        * Complex value val should be sorted as: 
            val = [val0.real, val1.real, val2.real ..., val0.imag, val1.imag, ...]
    """
    Nfreq = frequency.size
    if ax==None:    

        plt.semilogx(frequency, val[:Nfreq], 'b', **kwargs)
        plt.xlabel('Frequency (Hz)', fontsize = 14)
        plt.grid(color='k', alpha=0.5, linestyle='dashed', linewidth=0.5)        
        return plt.semilogx(frequency, val[Nfreq:], 'r', **kwargs)
    else:

        ax.semilogx(frequency, val[:Nfreq], 'b', **kwargs)        
        ax.set_xlabel('Frequency (Hz)', fontsize = 14)
        ax.grid(color='k', alpha=0.5, linestyle='dashed', linewidth=0.5)        

        return ax.semilogx(frequency, val[Nfreq:], 'r', **kwargs)
