from SimPEG import *
import numpy as np
from .BaseEM1D import BaseEM1DSurvey
# from future import division
from scipy.constants import mu_0
# from Kernels import HzKernel_layer, HzkernelCirc_layer
from .DigFilter import EvalDigitalFilt, LoadWeights
from .RTEfun import rTEfunfwd, rTEfunjac
from profilehooks import profile
from scipy.interpolate import InterpolatedUnivariateSpline as iuSpline
from empymod import filters


class EM1D(Problem.BaseProblem):
    """
        Pseudo analytic solutions for frequency and time domain EM problems assuming
        Layered earth (1D).
    """
    surveyPair = BaseEM1DSurvey
    mapPair = Maps.IdentityMap
    WT1 = None
    WT0 = None
    YBASE = None
    chi = None
    jacSwitch = True
    filter_type = 'key_101'
    verbose = False

    sigma, sigmaMap, sigmaDeriv = Props.Invertible(
        "Electrical conductivity at infinite frequency(S/m)"
    )

    chi = Props.PhysicalProperty(
        "Magnetic susceptibility",
        default=0.
    )

    eta, etaMap, etaDeriv = Props.Invertible(
        "Electrical chargeability (V/V)",
        default=0.
    )

    tau, tauMap, tauDeriv = Props.Invertible(
        "Time constant (s)",
        default=1.
    )

    c, cMap, cDeriv = Props.Invertible(
        "frequency Dependency",
        default=0.5
    )

    def __init__(self, mesh, **kwargs):
        Problem.BaseProblem.__init__(self, mesh, **kwargs)

        if self.filter_type == 'key_201':
            if self.verbose:
                print (">> Use Key 201 filter for Hankel Tranform")
            fht = filters.key_201_2009()
            self.WT0 = np.empty(201, complex)
            self.WT1 = np.empty(201, complex)
            self.YBASE = np.empty(201, complex)
            self.WT0 = fht.j0
            self.WT1 = fht.j1
            self.YBASE = fht.base
        elif self.filter_type == 'key_101':
            if self.verbose:
                print (">> Use Key 101 filter for Hankel Tranform")
            fht = filters.key_101_2009()
            self.WT0 = np.empty(101, complex)
            self.WT1 = np.empty(101, complex)
            self.YBASE = np.empty(101, complex)
            self.WT0 = fht.j0
            self.WT1 = fht.j1
            self.YBASE = fht.base
        elif self.filter_type == 'anderson_801':
            if self.verbose:
                print (">> Use Anderson 801 filter for Hankel Tranform")
            fht = filters.anderson_801_1982()
            self.WT0 = np.empty(801, complex)
            self.WT1 = np.empty(801, complex)
            self.YBASE = np.empty(801, complex)
            self.WT0 = fht.j0
            self.WT1 = fht.j1
            self.YBASE = fht.base
        else:
            raise NotImplementedError()
            # WT0, WT1, YBASE = LoadWeights()
            # self.WT0 = WT0
            # self.WT1 = WT1
            # self.YBASE = YBASE

    def HzKernel_layer(self, lamda, f, nlay, sig, chi, depth, h, z, flag):

        """
            Kernel for vertical magnetic component (Hz) due to
            vertical magnetic diopole (VMD) source in (kx,ky) domain

        """
        u0 = lamda
        rTE = np.zeros(lamda.size, dtype=complex)

        if self.jacSwitch:
            drTE = np.zeros((nlay, lamda.size), dtype=complex)
            rTE, drTE = rTEfunjac(
                nlay, f, lamda, sig, chi, depth, self.survey.HalfSwitch
            )
        else:
            rTE = rTEfunfwd(
                nlay, f, lamda, sig, chi, depth, self.survey.HalfSwitch
            )

        if flag == 'secondary':
            # Note
            # Here only computes secondary field.
            # I am not sure why it does not work if we add primary term.
            # This term can be analytically evaluated, where h = 0.
            kernel = 1/(4*np.pi)*(rTE*np.exp(-u0*(z+h)))*lamda**3/u0

        else:
            kernel = (
                1./(4*np.pi) *
                (np.exp(u0*(z-h))+rTE * np.exp(-u0*(z+h)))*lamda**3/u0
            )

        if self.jacSwitch:
            jackernel = 1/(4*np.pi)*(drTE)*(np.exp(-u0*(z+h))*lamda**3/u0)
            Kernel = []
            Kernel.append(kernel)
            Kernel.append(jackernel)
        else:
            Kernel = kernel
        return  Kernel

    def HzkernelCirc_layer(self, lamda, f, nlay, sig, chi, depth, h, z, I, a, flag):

        """

                Kernel for vertical magnetic component (Hz) at the center
                due to circular loop source in (kx,ky) domain

                .. math::

                    H_z = \\frac{Ia}{2} \int_0^{\infty} [e^{-u_0|z+h|} + r_{TE}e^{u_0|z-h|}] \\frac{\lambda^2}{u_0} J_1(\lambda a)] d \lambda


        """

        w = 2*np.pi*f
        rTE = np.empty(lamda.size, dtype=complex)
        u0 = lamda
        if self.jacSwitch ==  True:
            drTE = np.empty((nlay, lamda.size), dtype=complex)
            rTE, drTE = rTEfunjac(nlay, f, lamda, sig, chi, depth, self.survey.HalfSwitch)
        else:
            rTE = rTEfunfwd(nlay, f, lamda, sig, chi, depth, self.survey.HalfSwitch)

        if flag == 'secondary':
            kernel = I*a*0.5*(rTE*np.exp(-u0*(z+h)))*lamda**2/u0
        else:
            kernel = I*a*0.5*(np.exp(u0*(z-h))+rTE*np.exp(-u0*(z+h)))*lamda**2/u0

        if self.jacSwitch == True:
            jackernel = I*a*0.5*(drTE)*(np.exp(-u0*(z+h))*lamda**2/u0)
            Kernel = []
            Kernel.append(kernel)
            Kernel.append(jackernel)
        else:
            Kernel = kernel

        return  Kernel

    def sigma_cole(self, f):
        w = 2*np.pi*f
        sigma_complex = (
            self.sigma -
            self.sigma*self.eta/(1+(1-self.eta)*(1j*w*self.tau)**self.c)
        )
        return sigma_complex

    def fields(self, m):
        """
                Return Bz or dBzdt

        """
        if self.verbose:
            print ('>> Compute fields')

        f = self.survey.frequency
        nfreq = self.survey.Nfreq
        flag = self.survey.fieldtype
        r = self.survey.offset

        self.model = m

        nlay = self.survey.nlay
        depth = self.survey.depth
        nfilt = self.YBASE.size
        h = self.survey.h
        z = self.survey.z
        HzFHT = np.empty(nfreq, dtype = complex)
        dHzFHTdsig = np.empty((nlay, nfreq), dtype = complex)
        chi = self.chi
        n_int = 31

        # for inversion
        if self.jacSwitch==True:
            hz = np.empty(nfilt, complex)
            dhz = np.empty((nfilt, nlay), complex)
            if self.survey.srcType == 'VMD':
                r = self.survey.offset
                for ifreq in range(nfreq):
                    sig = self.sigma_cole(f[ifreq])
                    hz, dhz = self.HzKernel_layer(
                        self.YBASE/r[ifreq], f[ifreq], nlay, sig, chi, depth, h, z, flag
                    )
                    HzFHT[ifreq] = np.dot(hz, self.WT0)/r[ifreq]
                    dHzFHTdsig[:, ifreq] = np.dot(dhz, self.WT0)/r[ifreq]
            elif self.survey.srcType == 'CircularLoop':
                I = self.survey.I
                a = self.survey.a
                for ifreq in range(nfreq):
                    sig = self.sigma_cole(f[ifreq])
                    hz, dhz = self.HzkernelCirc_layer(
                        self.YBASE/a, f[ifreq], nlay, sig, chi, depth, h, z, I, a, flag
                    )
                    HzFHT[ifreq] = np.dot(hz, self.WT1)/a
                    dHzFHTdsig[:, ifreq] = np.dot(dhz, self.WT1)/a
            else :
                raise Exception("Src options are only VMD or CircularLoop!!")

            return  HzFHT, dHzFHTdsig.T

        # for simulation
        else:
            hz = np.empty(nfilt, complex)
            if self.survey.srcType == 'VMD':
                r = self.survey.offset
                for ifreq in range(nfreq):
                    sig = self.sigma_cole(f[ifreq])
                    hz = self.HzKernel_layer(
                        self.YBASE/r[ifreq], f[ifreq], nlay, sig, chi, depth, h, z, flag
                    )
                    HzFHT[ifreq] = np.dot(hz, self.WT0)/r[ifreq]

            elif self.survey.srcType == 'CircularLoop':
                I = self.survey.I
                a = self.survey.a
                for ifreq in range(nfreq):
                    sig = self.sigma_cole(f[ifreq])
                    hz = self.HzkernelCirc_layer(
                        self.YBASE/a, f[ifreq], nlay, sig, chi, depth, h, z, I, a, flag
                    )
                    HzFHT[ifreq] = np.dot(hz, self.WT1)/a
            else :
                raise Exception("Src options are only VMD or CircularLoop!!")

            return  HzFHT

    # @profile
    def Jvec(self, m, v, f=None):
        """
            Computing Jacobian^T multiplied by vector.
        """
        if f is None:

            f = self.fields(m)

        u, dudsig = f[0], f[1]

        if self.survey.switchFDTD == 'FD':

            resp = self.survey.projectFields(u)
            drespdsig = self.survey.projectFields(dudsig)

        elif self.survey.switchFDTD == 'TD':
            resp = self.survey.projectFields(u)
            drespdsig = self.survey.projectFields(dudsig)
            if drespdsig.size == self.survey.Nch:
                drespdsig = np.reshape(drespdsig, (-1, 1), order='F')
            else:
                drespdsig = np.reshape(
                    drespdsig, (self.survey.Nch, drespdsig.shape[1]), order='F'
                )
        else:

            raise Exception('Not implemented!!')

        Jv = np.dot(drespdsig, self.sigmaMap.deriv(m, v))
        return Jv

    # @profile
    def Jtvec(self, m, v, f=None):
        """
            Computing Jacobian^T multiplied by vector.
        """
        if f is None:
            f = self.fields(m)

        u, dudsig = f[0], f[1]

        if self.survey.switchFDTD == 'FD':

            resp = self.survey.projectFields(u)
            drespdsig = self.survey.projectFields(dudsig)

        elif self.survey.switchFDTD == 'TD':
            resp = self.survey.projectFields(u)
            drespdsig = self.survey.projectFields(dudsig)
            if drespdsig.size == self.survey.Nch:
                drespdsig = np.reshape(drespdsig, (-1, 1), order='F')
            else:
                drespdsig = np.reshape(drespdsig, (self.survey.Nch, drespdsig.shape[1]), order='F')
        else:

            raise Exception('Not implemented!!')

        Jtv = self.sigmaMap.deriv(m, np.dot(drespdsig.T, v))
        return Jtv

if __name__ == '__main__':
    main()
