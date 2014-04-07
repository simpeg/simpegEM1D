import numpy as np
from scipy.constants import mu_0
from SimPEG import Utils
# from numba import jit


# def rTEfun(nlay, f, lamda, sig, chi, depth):
#     """
#         Compute reflection coefficients for Transverse Electric (TE) mode.
#         Only one for loop for multiple layers. Do not use for loop for lambda,
#         which has 801 times of loops (actually, this makes the code really slow).
#     """`

#     thick = -np.diff(depth)
#     w = 2*np.pi*f

#     rTE = np.zeros(lamda.size, dtype=complex)
#     utemp0 = np.zeros(lamda.size, dtype=complex)
#     utemp1 = np.zeros(lamda.size, dtype=complex)
#     const = np.zeros(lamda.size, dtype=complex)

#     utemp0 = lamda
#     utemp1 = np.sqrt(lamda**2+1j*w*mu_0*(1+chi[0])*sig[0])
#     const = mu_0*utemp1/(mu_0*(1+chi[0])*utemp0)

#     Mtemp00 = np.zeros(lamda.size, dtype=complex)
#     Mtemp10 = np.zeros(lamda.size, dtype=complex)
#     Mtemp01 = np.zeros(lamda.size, dtype=complex)
#     Mtemp11 = np.zeros(lamda.size, dtype=complex)

#     Mtemp00 = 0.5*(1+const)
#     Mtemp10 = 0.5*(1-const)
#     Mtemp01 = 0.5*(1-const)
#     Mtemp11 = 0.5*(1+const)

#     # TODO: for computing Jacobian
#     M00 = []
#     M10 = []
#     M01 = []
#     M11 = []

#     M00.append(Mtemp00)
#     M01.append(Mtemp01)
#     M10.append(Mtemp10)
#     M11.append(Mtemp11)

#     M0sum00 = Mtemp00
#     M0sum10 = Mtemp10
#     M0sum01 = Mtemp01
#     M0sum11 = Mtemp11

#     for j in range (nlay-1):

#         Mtemp00 = np.zeros(lamda.size, dtype=complex)
#         Mtemp10 = np.zeros(lamda.size, dtype=complex)
#         Mtemp01 = np.zeros(lamda.size, dtype=complex)
#         Mtemp11 = np.zeros(lamda.size, dtype=complex)
#         # This in necessary... I am not quite sure why though
#         M1sum00 = np.zeros(lamda.size, dtype=complex)
#         M1sum10 = np.zeros(lamda.size, dtype=complex)
#         M1sum01 = np.zeros(lamda.size, dtype=complex)
#         M1sum11 = np.zeros(lamda.size, dtype=complex)

#         utemp0 = np.sqrt(lamda**2+1j*w*mu_0*(1+chi[j])*sig[j])
#         utemp1 = np.sqrt(lamda**2+1j*w*mu_0*(1+chi[j+1])*sig[j+1])
#         const = mu_0*(1+chi[j])*utemp1/(mu_0*(1+chi[j+1])*utemp0)

#         h0 = thick[j]

#         Mtemp00 = 0.5*(1.+ const)*np.exp(-2.*utemp0*h0)
#         Mtemp10 = 0.5*(1.- const)
#         Mtemp01 = 0.5*(1.- const)*np.exp(-2.*utemp0*h0)
#         Mtemp11 = 0.5*(1.+ const)

#         M1sum00 = M0sum00*Mtemp00 + M0sum01*Mtemp10
#         M1sum10 = M0sum10*Mtemp00 + M0sum11*Mtemp10
#         M1sum01 = M0sum00*Mtemp01 + M0sum01*Mtemp11
#         M1sum11 = M0sum10*Mtemp01 + M0sum11*Mtemp11

#         M0sum00 = M1sum00
#         M0sum10 = M1sum10
#         M0sum01 = M1sum01
#         M0sum11 = M1sum11

#         # TODO: for Computing Jacobian
#         M00.append(Mtemp00)
#         M01.append(Mtemp01)
#         M10.append(Mtemp10)
#         M11.append(Mtemp11)

#     rTE = M1sum01/M1sum11

#     return rTE, 0.

def rTEfun(nlay, f, lamda, sig, chi, depth, HalfSwitch):
    """
        Compute reflection coefficients for Transverse Electric (TE) mode.
        Only one for loop for multiple layers. Do not use for loop for lambda,
        which has 801 times of loops (actually, this makes the code really slow).

        .. math ::

            \\frac{a}{b}
    """

    Mtemp00 = np.zeros(lamda.size, dtype=complex)
    Mtemp10 = np.zeros(lamda.size, dtype=complex)
    Mtemp01 = np.zeros(lamda.size, dtype=complex)
    Mtemp11 = np.zeros(lamda.size, dtype=complex)

    Msum00 = np.zeros(lamda.size, dtype=complex)
    Msum10 = np.zeros(lamda.size, dtype=complex)
    Msum01 = np.zeros(lamda.size, dtype=complex)
    Msum11 = np.zeros(lamda.size, dtype=complex)

    dj0Mtemp00 = np.zeros(lamda.size, dtype=complex)
    dj0Mtemp10 = np.zeros(lamda.size, dtype=complex)
    dj0Mtemp01 = np.zeros(lamda.size, dtype=complex)
    dj0Mtemp11 = np.zeros(lamda.size, dtype=complex)

    dj1Mtemp00 = np.zeros(lamda.size, dtype=complex)
    dj1Mtemp10 = np.zeros(lamda.size, dtype=complex)
    dj1Mtemp01 = np.zeros(lamda.size, dtype=complex)
    dj1Mtemp11 = np.zeros(lamda.size, dtype=complex)

    thick = -np.diff(depth)
    w = 2*np.pi*f

    rTE = np.zeros(lamda.size, dtype=complex)
    drTE = np.zeros((nlay, lamda.size) , dtype=complex)
    utemp_1 = np.zeros(lamda.size, dtype=complex)
    utemp0 = np.zeros(lamda.size, dtype=complex)
    utemp1 = np.zeros(lamda.size, dtype=complex)
    const = np.zeros(lamda.size, dtype=complex)
    const0 = np.zeros(lamda.size, dtype=complex)
    const1a = np.zeros(lamda.size, dtype=complex)
    const1b = np.zeros(lamda.size, dtype=complex)

    utemp0 = lamda
    utemp1 = np.sqrt(lamda**2+1j*w*mu_0*(1+chi[0])*sig[0])
    const = mu_0*utemp1/(mu_0*(1+chi[0])*utemp0)


    #Compute M1
    Mtemp00 = 0.5*(1+const)
    Mtemp10 = 0.5*(1-const)
    Mtemp01 = 0.5*(1-const)
    Mtemp11 = 0.5*(1+const)

    utemp0 = lamda
    utemp1 = np.sqrt(lamda**2+1j*w*mu_0*(1+chi[0])*sig[0])
    const = mu_0*utemp1/(mu_0*(1+chi[0])*utemp0)

    #Compute dM1u1
    dj0Mtemp00 =  0.5*(mu_0/(mu_0*(1+chi[0])*utemp0))
    dj0Mtemp10 = -0.5*(mu_0/(mu_0*(1+chi[0])*utemp0))
    dj0Mtemp01 = -0.5*(mu_0/(mu_0*(1+chi[0])*utemp0))
    dj0Mtemp11 =  0.5*(mu_0/(mu_0*(1+chi[0])*utemp0))

    # TODO: for computing Jacobian
    M00 = []
    M10 = []
    M01 = []
    M11 = []

    dJ00 = []
    dJ10 = []
    dJ01 = []
    dJ11 = []

    M00.append(Mtemp00)
    M01.append(Mtemp01)
    M10.append(Mtemp10)
    M11.append(Mtemp11)

    M0sum00 = Mtemp00.copy()
    M0sum10 = Mtemp10.copy()
    M0sum01 = Mtemp01.copy()
    M0sum11 = Mtemp11.copy()

    if HalfSwitch == True:

        M1sum00 = np.zeros(lamda.size, dtype=complex)
        M1sum10 = np.zeros(lamda.size, dtype=complex)
        M1sum01 = np.zeros(lamda.size, dtype=complex)
        M1sum11 = np.zeros(lamda.size, dtype=complex)

        M1sum00 = M0sum00.copy()
        M1sum10 = M0sum10.copy()
        M1sum01 = M0sum01.copy()
        M1sum11 = M0sum11.copy()

    else:

        dJ_10Mtemp00 = np.zeros(lamda.size, dtype=complex)
        dJ_10Mtemp10 = np.zeros(lamda.size, dtype=complex)
        dJ_10Mtemp01 = np.zeros(lamda.size, dtype=complex)
        dJ_10Mtemp11 = np.zeros(lamda.size, dtype=complex)

        dJ01Mtemp00 = np.zeros(lamda.size, dtype=complex)
        dJ01Mtemp10 = np.zeros(lamda.size, dtype=complex)
        dJ01Mtemp01 = np.zeros(lamda.size, dtype=complex)
        dJ01Mtemp11 = np.zeros(lamda.size, dtype=complex)


        for j in range (nlay-1):

            utemp0  = np.sqrt(lamda**2+1j*w*mu_0*(1+chi[j])*sig[j])
            utemp1  = np.sqrt(lamda**2+1j*w*mu_0*(1+chi[j+1])*sig[j+1])
            const = mu_0*(1+chi[j])*utemp1/(mu_0*(1+chi[j+1])*utemp0)

            h0 = thick[j]
            # When j = 0             j=k                             j = nlay-2
            # Compute M2      ::     Compute Mk+2      ::            Compute Mnlay
            Mtemp00 = 0.5*(1.+ const)*np.exp(-2.*utemp0*h0)
            Mtemp10 = 0.5*(1.- const)
            Mtemp01 = 0.5*(1.- const)*np.exp(-2.*utemp0*h0)
            Mtemp11 = 0.5*(1.+ const)

            # Compute M1*M2   ::     Compute (M1*M2...*Mk+1)*Mk+2 :: Compute (M1*M2...*Mnlay-1)*Mnlay
            M1sum00 = M0sum00*Mtemp00 + M0sum01*Mtemp10
            M1sum10 = M0sum10*Mtemp00 + M0sum11*Mtemp10
            M1sum01 = M0sum00*Mtemp01 + M0sum01*Mtemp11
            M1sum11 = M0sum10*Mtemp01 + M0sum11*Mtemp11

            M0sum00 = M1sum00.copy()
            M0sum10 = M1sum10.copy()
            M0sum01 = M1sum01.copy()
            M0sum11 = M1sum11.copy()

            k0 = np.sqrt(-1j*w*mu_0*(1+chi[j])*sig[j])
            dkdsig = k0/sig[j]*0.5
            dudk = -k0/utemp0
            dudsig = dkdsig*dudk

            if j==0:

                const1a = mu_0*(1+chi[j])*utemp1/(mu_0*(1+chi[j+1])*utemp0**2)
                const1b = const1a*utemp0*h0
                # Compute dM2du1
                dj1Mtemp00 = -const1b*np.exp(-2.*utemp0*h0)-0.5*const1a*np.exp(-2.*utemp0*h0)
                dj1Mtemp10 =  0.5*const1a
                dj1Mtemp01 =  const1b*np.exp(-2.*utemp0*h0)+0.5*const1a*np.exp(-2.*utemp0*h0)
                dj1Mtemp11 = -0.5*const1a

                # Compute dM1du1*M2
                dJ_10Mtemp00 = dj0Mtemp00*Mtemp00 + dj0Mtemp01*Mtemp10
                dJ_10Mtemp10 = dj0Mtemp10*Mtemp00 + dj0Mtemp11*Mtemp10
                dJ_10Mtemp01 = dj0Mtemp00*Mtemp01 + dj0Mtemp01*Mtemp11
                dJ_10Mtemp11 = dj0Mtemp10*Mtemp01 + dj0Mtemp11*Mtemp11

                # Compute M1*dM2du1
                dJ01Mtemp00  = M00[0]*dj1Mtemp00 + M01[0]*dj1Mtemp10
                dJ01Mtemp10  = M10[0]*dj1Mtemp00 + M11[0]*dj1Mtemp10
                dJ01Mtemp01  = M00[0]*dj1Mtemp01 + M01[0]*dj1Mtemp11
                dJ01Mtemp11  = M10[0]*dj1Mtemp01 + M11[0]*dj1Mtemp11

                dJ00.append(dudsig*(dJ_10Mtemp00+dJ01Mtemp00))
                dJ10.append(dudsig*(dJ_10Mtemp10+dJ01Mtemp10))
                dJ01.append(dudsig*(dJ_10Mtemp01+dJ01Mtemp01))
                dJ11.append(dudsig*(dJ_10Mtemp11+dJ01Mtemp11))

            else:

                h_1 = thick[j-1]
                utemp_1 = np.sqrt(lamda**2+1j*w*mu_0*(1+chi[j-1])*sig[j-1])
                const0 = mu_0*(1+chi[j-1])/(mu_0*(1+chi[j])*utemp_1)
                # When j = k (k>0)
                # Compute dMk+1duk+1

                dj0Mtemp00 =  0.5*(const0)*np.exp(-2.*utemp_1*h_1)
                dj0Mtemp10 = -0.5*(const0)
                dj0Mtemp01 = -0.5*(const0)*np.exp(-2.*utemp_1*h_1)
                dj0Mtemp11 =  0.5*(const0)

                const1a = mu_0*(1+chi[j])*utemp1/(mu_0*(1+chi[j+1])*utemp0**2)
                const1b = const1a*utemp0*h0
                # When j = k (k>0)
                # Compute dMk+2duk+1
                dj1Mtemp00 = -const1b*np.exp(-2.*utemp1*h0)-0.5*const1a*np.exp(-2.*utemp1*h0)
                dj1Mtemp10 =  0.5*const1a
                dj1Mtemp01 =  const1b*np.exp(-2.*utemp1*h0)+0.5*const1a*np.exp(-2.*utemp1*h0)
                dj1Mtemp11 = -0.5*const1a

                # When j = k (k>0)                                :: j = 1
                # Compute dMk+1duk+1*Mk+2                         :: Compute dM2du2*M3
                dJ_10Mtemp00 = dj0Mtemp00*Mtemp00 + dj0Mtemp01*Mtemp10
                dJ_10Mtemp10 = dj0Mtemp10*Mtemp00 + dj0Mtemp11*Mtemp10
                dJ_10Mtemp01 = dj0Mtemp00*Mtemp01 + dj0Mtemp01*Mtemp11
                dJ_10Mtemp11 = dj0Mtemp10*Mtemp01 + dj0Mtemp11*Mtemp11

                # When j = k (k>0)                                :: j = 1
                # Compute Mk+1dMk+2duk+1                          :: Compute M2*dM3du2
                dJ01Mtemp00  = M00[j]*dj1Mtemp00 + M01[j]*dj1Mtemp10
                dJ01Mtemp10  = M10[j]*dj1Mtemp00 + M11[j]*dj1Mtemp10
                dJ01Mtemp01  = M00[j]*dj1Mtemp01 + M01[j]*dj1Mtemp11
                dJ01Mtemp11  = M10[j]*dj1Mtemp01 + M11[j]*dj1Mtemp11

                # When j = k (k>0)                                :: j = 1
                # Compute dMk+1dsigk+1*Mk+2 + Mk+1dMk+2dsigk+1    :: Compute dM2dsig2*M3 + M2*dM3dsig2
                dJ00.append(dudsig*(dJ_10Mtemp00+dJ01Mtemp00))
                dJ10.append(dudsig*(dJ_10Mtemp10+dJ01Mtemp10))
                dJ01.append(dudsig*(dJ_10Mtemp01+dJ01Mtemp01))
                dJ11.append(dudsig*(dJ_10Mtemp11+dJ01Mtemp11))

            # When j = 0             :: j = k
            # Input M[0] = M1        :: M[k] = Mk+1
            M00.append(Mtemp00)
            M01.append(Mtemp01)
            M10.append(Mtemp10)
            M11.append(Mtemp11)

    rTE = M1sum01/M1sum11

    if HalfSwitch ==  True:

        utemp0 = np.sqrt(lamda**2+1j*w*mu_0*(1+chi[0])*sig[0])
        k0 = np.sqrt(-1j*w*mu_0*(1+chi[0])*sig[0])
        dkdsig = k0/sig[0]*0.5
        dudk = -k0/utemp0
        dudsig = dkdsig*dudk

        dJ1sum00 = np.zeros(lamda.size, dtype=complex)
        dJ1sum10 = np.zeros(lamda.size, dtype=complex)
        dJ1sum01 = np.zeros(lamda.size, dtype=complex)
        dJ1sum11 = np.zeros(lamda.size, dtype=complex)

        dJ1sum00 = dudsig*dj0Mtemp00
        dJ1sum10 = dudsig*dj0Mtemp10
        dJ1sum01 = dudsig*dj0Mtemp01
        dJ1sum11 = dudsig*dj0Mtemp11

        drTE = dJ1sum01/M1sum11 - M1sum01/(M1sum11**2)*dJ1sum11

    else:

        #j = nlay
        h_1 = thick[nlay-2]
        utemp_1 = np.sqrt(lamda**2+1j*w*mu_0*(1+chi[nlay-2])*sig[nlay-2])
        const0 = mu_0*(1+chi[nlay-2])/(mu_0*(1+chi[nlay-1])*utemp_1)

        dj0Mtemp00 =  0.5*(const0)*np.exp(-2.*utemp_1*h_1)
        dj0Mtemp10 = -0.5*(const0)
        dj0Mtemp01 = -0.5*(const0)*np.exp(-2.*utemp_1*h_1)
        dj0Mtemp11 =  0.5*(const0)

        dJ_10Mtemp00 = dj0Mtemp00.copy()
        dJ_10Mtemp10 = dj0Mtemp10.copy()
        dJ_10Mtemp01 = dj0Mtemp01.copy()
        dJ_10Mtemp11 = dj0Mtemp11.copy()

        utemp0 = np.sqrt(lamda**2+1j*w*mu_0*(1+chi[nlay-1])*sig[nlay-1])
        k0 = np.sqrt(-1j*w*mu_0*(1+chi[nlay-1])*sig[nlay-1])
        dkdsig = k0/sig[nlay-1]*0.5
        dudk = -k0/utemp0
        dudsig = dkdsig*dudk

        dJ00.append(dudsig*dJ_10Mtemp00)
        dJ10.append(dudsig*dJ_10Mtemp10)
        dJ01.append(dudsig*dJ_10Mtemp01)
        dJ11.append(dudsig*dJ_10Mtemp11)

        for i in range (nlay):

            dJ0sum00 = np.zeros(lamda.size, dtype=complex)
            dJ0sum10 = np.zeros(lamda.size, dtype=complex)
            dJ0sum01 = np.zeros(lamda.size, dtype=complex)
            dJ0sum11 = np.zeros(lamda.size, dtype=complex)

            dJ1sum00 = np.zeros(lamda.size, dtype=complex)
            dJ1sum10 = np.zeros(lamda.size, dtype=complex)
            dJ1sum01 = np.zeros(lamda.size, dtype=complex)
            dJ1sum11 = np.zeros(lamda.size, dtype=complex)

            if i==0:

                dJ0sum00 = dJ00[0].copy()
                dJ0sum10 = dJ10[0].copy()
                dJ0sum01 = dJ01[0].copy()
                dJ0sum11 = dJ11[0].copy()

                for j in range (nlay-2):

                    # When j = k                   ::   j = 2
                    # Compute (J1*M3*...Mk+1)*Mk+2 ::   Compute (J1*M3)*M4
                    dJ1sum00 = dJ0sum00*M00[j+2] + dJ0sum01*M10[j+2]
                    dJ1sum10 = dJ0sum10*M00[j+2] + dJ0sum11*M10[j+2]
                    dJ1sum01 = dJ0sum00*M01[j+2] + dJ0sum01*M11[j+2]
                    dJ1sum11 = dJ0sum10*M01[j+2] + dJ0sum11*M11[j+2]

                    dJ0sum00 = dJ1sum00.copy()
                    dJ0sum10 = dJ1sum10.copy()
                    dJ0sum01 = dJ1sum01.copy()
                    dJ0sum11 = dJ1sum11.copy()

            elif (i>0) & (i<nlay-1):

                dJ0sum00 = M00[0].copy()
                dJ0sum10 = M10[0].copy()
                dJ0sum01 = M01[0].copy()
                dJ0sum11 = M11[0].copy()

                for j in range (nlay-2):

                    # When i = 2, j = 1
                    # Compute (M1*M2)*J3
                    if j==i-1:

                        dJ1sum00 = dJ0sum00*dJ00[i] + dJ0sum01*dJ10[i]
                        dJ1sum10 = dJ0sum10*dJ00[i] + dJ0sum11*dJ10[i]
                        dJ1sum01 = dJ0sum00*dJ01[i] + dJ0sum01*dJ11[i]
                        dJ1sum11 = dJ0sum10*dJ01[i] + dJ0sum11*dJ11[i]

                    # When i = 2, j = 0
                    # Compute M1*M2
                    elif j < i-1:

                        dJ1sum00 = dJ0sum00*M00[j+1] + dJ0sum01*M10[j+1]
                        dJ1sum10 = dJ0sum10*M00[j+1] + dJ0sum11*M10[j+1]
                        dJ1sum01 = dJ0sum00*M01[j+1] + dJ0sum01*M11[j+1]
                        dJ1sum11 = dJ0sum10*M01[j+1] + dJ0sum11*M11[j+1]

                    # When i = 2, j = 2
                    # Compute (M1*M2*J3)*M5
                    elif j > i-1:

                        dJ1sum00 = dJ0sum00*M00[j+2] + dJ0sum01*M10[j+2]
                        dJ1sum10 = dJ0sum10*M00[j+2] + dJ0sum11*M10[j+2]
                        dJ1sum01 = dJ0sum00*M01[j+2] + dJ0sum01*M11[j+2]
                        dJ1sum11 = dJ0sum10*M01[j+2] + dJ0sum11*M11[j+2]


                    dJ0sum00 = dJ1sum00.copy()
                    dJ0sum10 = dJ1sum10.copy()
                    dJ0sum01 = dJ1sum01.copy()
                    dJ0sum11 = dJ1sum11.copy()

            elif i==nlay-1:

                dJ0sum00 = M00[0].copy()
                dJ0sum10 = M10[0].copy()
                dJ0sum01 = M01[0].copy()
                dJ0sum11 = M11[0].copy()

                for j in range (nlay-1):

                    if j < i-1:

                        dJ1sum00 = dJ0sum00*M00[j+1] + dJ0sum01*M10[j+1]
                        dJ1sum10 = dJ0sum10*M00[j+1] + dJ0sum11*M10[j+1]
                        dJ1sum01 = dJ0sum01*M01[j+1] + dJ0sum01*M11[j+1]
                        dJ1sum11 = dJ0sum11*M01[j+1] + dJ0sum11*M11[j+1]


                    elif j == i-1:

                        dJ1sum00 = dJ0sum00*dJ00[i] + dJ0sum01*dJ10[i]
                        dJ1sum10 = dJ0sum10*dJ00[i] + dJ0sum11*dJ10[i]
                        dJ1sum01 = dJ0sum01*dJ01[i] + dJ0sum01*dJ11[i]
                        dJ1sum11 = dJ0sum11*dJ01[i] + dJ0sum11*dJ11[i]

                    dJ0sum00 = dJ1sum00.copy()
                    dJ0sum10 = dJ1sum10.copy()
                    dJ0sum01 = dJ1sum01.copy()
                    dJ0sum11 = dJ1sum11.copy()

            drTE[i,:] = Utils.mkvc(dJ1sum01/M1sum11 - M1sum01/(M1sum11**2)*dJ1sum11)

    return rTE, drTE
