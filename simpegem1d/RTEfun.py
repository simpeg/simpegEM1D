import numpy as np
from scipy.constants import mu_0
from numba import jit


# def rTEfun(nlay, f, lamda, sig, chi, depth):
#     """
#         Compute reflection coefficients for Transverse Electric (TE) mode.
#         Only one for loop for multiple layers. Do not use for loop for lambda,
#         which has 801 times of loops (actually, this makes the code really slow).
#     """

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

def rTEfun(nlay, f, lamda, sig, chi, depth):
    """
        Compute reflection coefficients for Transverse Electric (TE) mode.
        Only one for loop for multiple layers. Do not use for loop for lambda,
        which has 801 times of loops (actually, this makes the code really slow).
    """

    thick = -np.diff(depth)
    w = 2*np.pi*f

    rTE = np.zeros(lamda.size, dtype=complex)
    drTE = np.zeros((nlay, lamda.size) , dtype=complex)
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

    dMtemp00 = np.zeros(lamda.size, dtype=complex)
    dMtemp10 = np.zeros(lamda.size, dtype=complex)
    dMtemp01 = np.zeros(lamda.size, dtype=complex)
    dMtemp11 = np.zeros(lamda.size, dtype=complex)


    Mtemp00 = 0.5*(1+const)
    Mtemp10 = 0.5*(1-const)
    Mtemp01 = 0.5*(1-const)
    Mtemp11 = 0.5*(1+const)

    utemp0 = lamda
    utemp1 = np.sqrt(lamda**2+1j*w*mu_0*(1+chi[0])*sig[0])

    const = mu_0*utemp1/(mu_0*(1+chi[0])*utemp0)

    dj0Mtemp00 = 0.5*(const/utemp1)
    dj0Mtemp10 = 0.5*(-const/utemp1)
    dj0Mtemp01 = 0.5*(-const/utemp1)
    dj0Mtemp11 = 0.5*(const/utemp1)

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

        utemp0  = np.sqrt(lamda**2+1j*w*mu_0*(1+chi[j])*sig[j])
        utemp1  = np.sqrt(lamda**2+1j*w*mu_0*(1+chi[j+1])*sig[j+1])
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
        dudsig =  0.5*1j*mu_0*(1+chi[j])*w/(np.sqrt(lamda**2+1j*mu_0*(1+chi[j])*sig[j]*w))
        if j<1:

            const1a = mu_0*(1+chi[j])*utemp1/(mu_0*(1+chi[j+1])*utemp0**2)
            const1b = const1a*utemp1*h0

            dj1Mtemp00 = -const1b*np.exp(-2.*utemp1*thick[0])-0.5*const1a*np.exp(-2.*utemp1*thick[0])
            dj1Mtemp10 =  0.5*const1a
            dj1Mtemp01 =  const1b*np.exp(-2.*utemp1*thick[0])+0.5*const1a*np.exp(-2.*utemp1*thick[0])
            dj1Mtemp11 = -0.5*const1a

            dJ_10Mtemp00 = dj0Mtemp00*Mtemp00 + dj0Mtemp01*Mtemp10
            dJ_10Mtemp10 = dj0Mtemp10*Mtemp00 + dj0Mtemp11*Mtemp10
            dJ_10Mtemp01 = dj0Mtemp00*Mtemp01 + dj0Mtemp01*Mtemp11
            dJ_10Mtemp11 = dj0Mtemp10*Mtemp01 + dj0Mtemp11*Mtemp11

            dJ01Mtemp00  = M00[j]*dj1Mtemp00 + M01[j]*dj1Mtemp10
            dJ01Mtemp10  = M10[j]*dj1Mtemp00 + M11[j]*dj1Mtemp10
            dJ01Mtemp01  = M00[j]*dj1Mtemp01 + M01[j]*dj1Mtemp11
            dJ01Mtemp11  = M10[j]*dj1Mtemp01 + M11[j]*dj1Mtemp11

            dJ00.append(dudsig*(dJ_10Mtemp00+dJ01Mtemp00))
            dJ10.append(dudsig*(dJ_10Mtemp10+dJ01Mtemp10))
            dJ01.append(dudsig*(dJ_10Mtemp01+dJ01Mtemp01))
            dJ11.append(dudsig*(dJ_10Mtemp11+dJ01Mtemp11))

        else:
            h_1 = thick[j-1]            

            utemp_1 = np.sqrt(lamda**2+1j*w*mu_0*(1+chi[j-1])*sig[j-1])
            const0 = mu_0*(1+chi[j-1])/(mu_0*(1+chi[j])*utemp_1)            

            dj0Mtemp00 =  0.5*(const0)*np.exp(-2.*utemp_1*h_1)
            dj0Mtemp10 = -0.5*(const0)
            dj0Mtemp01 = -0.5*(const0)*np.exp(-2.*utemp_1*h_1)
            dj0Mtemp11 =  0.5*(const0)

            const1a = mu_0*(1+chi[j])*utemp1/(mu_0*(1+chi[j+1])*utemp0**2)
            const1b = const1a*utemp1*h0

            dj1Mtemp00 = -const1b*np.exp(-2.*utemp1*thick[0])-0.5*const1a*np.exp(-2.*utemp1*thick[0])
            dj1Mtemp10 =  0.5*const1a
            dj1Mtemp01 =  const1b*np.exp(-2.*utemp1*thick[0])+0.5*const1a*np.exp(-2.*utemp1*thick[0])
            dj1Mtemp11 = -0.5*const1a

            dJ_10Mtemp00 = dj0Mtemp00*Mtemp00 + dj0Mtemp01*Mtemp10
            dJ_10Mtemp10 = dj0Mtemp10*Mtemp00 + dj0Mtemp11*Mtemp10
            dJ_10Mtemp01 = dj0Mtemp00*Mtemp01 + dj0Mtemp01*Mtemp11
            dJ_10Mtemp11 = dj0Mtemp10*Mtemp01 + dj0Mtemp11*Mtemp11

            dJ01Mtemp00  = M00[j]*dj1Mtemp00 + M01[j]*dj1Mtemp10
            dJ01Mtemp10  = M10[j]*dj1Mtemp00 + M11[j]*dj1Mtemp10
            dJ01Mtemp01  = M00[j]*dj1Mtemp01 + M01[j]*dj1Mtemp11
            dJ01Mtemp11  = M10[j]*dj1Mtemp01 + M11[j]*dj1Mtemp11
            

            dJ00.append(dudsig*(dJ_10Mtemp00+dJ01Mtemp00))
            dJ10.append(dudsig*(dJ_10Mtemp10+dJ01Mtemp10))
            dJ01.append(dudsig*(dJ_10Mtemp01+dJ01Mtemp01))
            dJ11.append(dudsig*(dJ_10Mtemp11+dJ01Mtemp11))

        M00.append(Mtemp00)
        M01.append(Mtemp01)
        M10.append(Mtemp10)
        M11.append(Mtemp11)


    rTE = M1sum01/M1sum11
    h_1 = thick[nlay-2]            

    utemp_1 = np.sqrt(lamda**2+1j*w*mu_0*(1+chi[nlay-2])*sig[nlay-2])
    const0 = mu_0*(1+chi[nlay-2])/(mu_0*(1+chi[nlay-1])*utemp_1)            

    dj0Mtemp00 =  0.5*(const0)*np.exp(-2.*utemp_1*h_1)
    dj0Mtemp10 = -0.5*(const0)
    dj0Mtemp01 = -0.5*(const0)*np.exp(-2.*utemp_1*h_1)
    dj0Mtemp11 =  0.5*(const0)

    dJ00.append(dudsig*dJ_10Mtemp00)
    dJ10.append(dudsig*dJ_10Mtemp10)
    dJ01.append(dudsig*dJ_10Mtemp01)
    dJ11.append(dudsig*dJ_10Mtemp11)    


    dJ0sum00 = np.zeros(lamda.size, dtype=complex)
    dJ0sum10 = np.zeros(lamda.size, dtype=complex)
    dJ0sum01 = np.zeros(lamda.size, dtype=complex)
    dJ0sum11 = np.zeros(lamda.size, dtype=complex)

    for i in range (nlay):

        if i==0:

            for j in range (nlay-2):

                dJ1sum00 = np.zeros(lamda.size, dtype=complex)
                dJ1sum10 = np.zeros(lamda.size, dtype=complex)
                dJ1sum01 = np.zeros(lamda.size, dtype=complex)
                dJ1sum11 = np.zeros(lamda.size, dtype=complex)

                if j==0:

                    dJ0sum00 = dJ00[i]*M00[j+2] + dJ00[i]*M10[j+2]
                    dJ0sum10 = dJ00[i]*M00[j+2] + dJ00[i]*M10[j+2]
                    dJ0sum01 = dJ01[i]*M01[j+2] + dJ01[i]*M11[j+2]
                    dJ0sum11 = dJ01[i]*M01[j+2] + dJ01[i]*M11[j+2]        

                else:

                    dJ1sum00 = dJ0sum00*M00[j+2] + dJ0sum00*M10[j+2]
                    dJ1sum10 = dJ0sum10*M00[j+2] + dJ0sum10*M10[j+2]
                    dJ1sum01 = dJ0sum01*M01[j+2] + dJ0sum01*M11[j+2]
                    dJ1sum11 = dJ0sum11*M01[j+2] + dJ0sum11*M11[j+2]        

                dJ0sum00 = dJ1sum00
                dJ0sum10 = dJ1sum10
                dJ0sum01 = dJ1sum01
                dJ0sum11 = dJ1sum11                

        elif (i>0) & (i<nlay-1):

            dJ0sum00 = M00[0]
            dJ0sum10 = M10[0]
            dJ0sum01 = M01[0]
            dJ0sum11 = M11[0]

            for j in range (nlay-2):

                dJ1sum00 = np.zeros(lamda.size, dtype=complex)
                dJ1sum10 = np.zeros(lamda.size, dtype=complex)
                dJ1sum01 = np.zeros(lamda.size, dtype=complex)
                dJ1sum11 = np.zeros(lamda.size, dtype=complex)


                if j==i-1:

                    dJ1sum00 = M00[j]*dJ00[i] + M10[j]*dJ00[i]
                    dJ1sum10 = M00[j]*dJ00[i] + M10[j]*dJ00[i]
                    dJ1sum01 = M01[j]*dJ01[i] + M11[j]*dJ01[i]
                    dJ1sum11 = M01[j]*dJ01[i] + M11[j]*dJ01[i]

                elif j==i:

                    dJ1sum00 = dJ00[i]*M00[j+2] + dJ00[i]*M10[j+2]
                    dJ1sum10 = dJ00[i]*M00[j+2] + dJ00[i]*M10[j+2]
                    dJ1sum01 = dJ01[i]*M01[j+2] + dJ01[i]*M11[j+2]
                    dJ1sum11 = dJ01[i]*M01[j+2] + dJ01[i]*M11[j+2]

                elif j < i-1:

                    dJ1sum00 = dJ0sum00*M00[j+1] + dJ0sum00*M10[j+1]
                    dJ1sum10 = dJ0sum10*M00[j+1] + dJ0sum10*M10[j+1]
                    dJ1sum01 = dJ0sum01*M01[j+1] + dJ0sum01*M11[j+1]
                    dJ1sum11 = dJ0sum11*M01[j+1] + dJ0sum11*M11[j+1]        


                else:

                    dJ1sum00 = dJ0sum00*M00[j+2] + dJ0sum00*M10[j+2]
                    dJ1sum10 = dJ0sum10*M00[j+2] + dJ0sum10*M10[j+2]
                    dJ1sum01 = dJ0sum01*M01[j+2] + dJ0sum01*M11[j+2]
                    dJ1sum11 = dJ0sum11*M01[j+2] + dJ0sum11*M11[j+2]        


                dJ0sum00 = dJ1sum00
                dJ0sum10 = dJ1sum10
                dJ0sum01 = dJ1sum01
                dJ0sum11 = dJ1sum11   

        elif i==nlay-1:

            dJ0sum00 = M00[0]
            dJ0sum10 = M10[0]
            dJ0sum01 = M01[0]
            dJ0sum11 = M11[0]

            for j in range (nlay-2):

                dJ1sum00 = np.zeros(lamda.size, dtype=complex)
                dJ1sum10 = np.zeros(lamda.size, dtype=complex)
                dJ1sum01 = np.zeros(lamda.size, dtype=complex)
                dJ1sum11 = np.zeros(lamda.size, dtype=complex)

                if j < nlay-3:

                    dJ1sum00 = dJ0sum00*M00[j+1] + dJ0sum00*M10[j+1]
                    dJ1sum10 = dJ0sum10*M00[j+1] + dJ0sum10*M10[j+1]
                    dJ1sum01 = dJ0sum01*M01[j+1] + dJ0sum01*M11[j+1]
                    dJ1sum11 = dJ0sum11*M01[j+1] + dJ0sum11*M11[j+1]     

                elif j == nlay-3:

                    dJ1sum00 = M00[j]*dJ00[i] + M10[j]*dJ00[i]
                    dJ1sum10 = M00[j]*dJ00[i] + M10[j]*dJ00[i]
                    dJ1sum01 = M01[j]*dJ01[i] + M11[j]*dJ01[i]
                    dJ1sum11 = M01[j]*dJ01[i] + M11[j]*dJ01[i]    

                dJ0sum00 = dJ1sum00
                dJ0sum10 = dJ1sum10
                dJ0sum01 = dJ1sum01
                dJ0sum11 = dJ1sum11

        M1sum01/M1sum11
        drTE[i,:] = dJ1sum01/M1sum11 - M1sum01/M1sum11**2*dJ1sum11

    return rTE, drTE