import numpy as np
from scipy.constants import mu_0
def rTEfun(nlay, f, lamda, sig, chi, depth):
    """
        Compute reflection coefficients for Transverse Electric (TE) mode.
        Only one for loop for multiple layers. Do not use for loop for lambda,
        which has 801 times of loops (actually, this makes the code really slow).
    """

    thick = -np.diff(depth)
    w = 2*np.pi*f

    rTE = np.zeros(lamda.size, dtype=complex)
    utemp0 = np.zeros(lamda.size, dtype=complex)
    utemp1 = np.zeros(lamda.size, dtype=complex)
    const = np.zeros(lamda.size, dtype=complex)

    utemp0 = lamda
    utemp1 = np.sqrt(lamda**2+1j*w*mu_0*(1+chi[0])*sig[0])
    const = mu_0*utemp1/(mu_0*(1+chi[0])*utemp0)

    Mtemp00 = np.zeros(lamda.size, dtype=complex)
    Mtemp10 = np.zeros(lamda.size, dtype=complex)
    Mtemp01 = np.zeros(lamda.size, dtype=complex)
    Mtemp11 = np.zeros(lamda.size, dtype=complex)

    Mtemp00 = 0.5*(1+const)
    Mtemp10 = 0.5*(1-const)
    Mtemp01 = 0.5*(1-const)
    Mtemp11 = 0.5*(1+const)

    # TODO: for computing Jacobian
    # M00 = []
    # M10 = []
    # M01 = []
    # M11 = []

    # M00.append(Mtemp00)
    # M01.append(Mtemp01)
    # M10.append(Mtemp10)
    # M11.append(Mtemp11)

    M0sum00 = Mtemp00
    M0sum10 = Mtemp10
    M0sum01 = Mtemp01
    M0sum11 = Mtemp11

    for j in range (nlay-1):

        Mtemp00 = np.zeros(lamda.size, dtype=complex)
        Mtemp10 = np.zeros(lamda.size, dtype=complex)
        Mtemp01 = np.zeros(lamda.size, dtype=complex)
        Mtemp11 = np.zeros(lamda.size, dtype=complex)
        # This in necessary... I am not quite sure why though
        M1sum00 = np.zeros(lamda.size, dtype=complex)
        M1sum10 = np.zeros(lamda.size, dtype=complex)
        M1sum01 = np.zeros(lamda.size, dtype=complex)
        M1sum11 = np.zeros(lamda.size, dtype=complex)

        utemp0 = np.sqrt(lamda**2+1j*w*mu_0*(1+chi[j])*sig[j])
        utemp1 = np.sqrt(lamda**2+1j*w*mu_0*(1+chi[j+1])*sig[j+1])
        const = mu_0*(1+chi[j])*utemp1/(mu_0*(1+chi[j+1])*utemp0)

        h0 = thick[j]

        Mtemp00 = 0.5*(1.+ const)*np.exp(-2.*utemp0*h0)
        Mtemp10 = 0.5*(1.- const)
        Mtemp01 = 0.5*(1.- const)*np.exp(-2.*utemp0*h0)
        Mtemp11 = 0.5*(1.+ const)

        M1sum00 = M0sum00*Mtemp00 + M0sum01*Mtemp10
        M1sum10 = M0sum10*Mtemp00 + M0sum11*Mtemp10
        M1sum01 = M0sum00*Mtemp01 + M0sum01*Mtemp11
        M1sum11 = M0sum10*Mtemp01 + M0sum11*Mtemp11

        M0sum00 = M1sum00
        M0sum10 = M1sum10
        M0sum01 = M1sum01
        M0sum11 = M1sum11

        # TODO: for Computing Jacobian
        # M00.append(M1temp00)
        # M01.append(M1temp01)
        # M10.append(M1temp10)
        # M11.append(M1temp11)


    rTE = M1sum01/M1sum11
    return rTE
