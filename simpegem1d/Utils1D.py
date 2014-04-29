import numpy as np
import matplotlib.pyplot as plt

def plotLayer(sig, LocSigZ, xscale, ax=None, **kwargs):
    """
        Plot Conductivity model for the layered earth model
    """
    sigma = np.repeat(sig, 2, axis=0)
    z = np.repeat(LocSigZ[1:], 2, axis=0)
    z = np.r_[LocSigZ[0], z, LocSigZ[-1]]
    plt.xscale(xscale)
    plt.xlim(sig.min()*0.5, sig.max()*2)
    plt.ylim(z.min(), z.max())
    plt.grid(color='k', alpha=0.5, linestyle='dashed', linewidth=0.5)
    plt.xlabel('Conductivity (S/m)', fontsize = 14)
    plt.ylabel('Depth (m)', fontsize = 14)
    return plt.plot(sigma, z, 'k-', **kwargs)
