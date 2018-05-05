from SimPEG import Maps, Utils, Problem, Props
import numpy as np
from .Survey import BaseEM1DSurvey
from scipy.constants import mu_0
from .RTEfun import rTEfunfwd, rTEfunjac
from scipy.interpolate import InterpolatedUnivariateSpline as iuSpline

from empymod import filters
from empymod.transform import dlf, get_spline_values
from empymod.utils import check_hankel


class EM1D(Problem.BaseProblem):
    """
    Pseudo analytic solutions for frequency and time domain EM problems
    assumingLayered earth (1D).
    """
    surveyPair = BaseEM1DSurvey
    mapPair = Maps.IdentityMap
    chi = None
    jacSwitch = True
    hankel_filter = 'key_101_2009'  # Default: Hankel filter
    hankel_pts_per_dec = None       # Default: Standard DLF
    verbose = False
    fix_Jmatrix = False
    _Jmatrix_sigma = None
    _Jmatrix_height = None
    _pred = None

    sigma, sigmaMap, sigmaDeriv = Props.Invertible(
        "Electrical conductivity at infinite frequency(S/m)"
    )

    chi = Props.PhysicalProperty(
        "Magnetic susceptibility",
        default=0.
    )

    eta, etaMap, etaDeriv = Props.Invertible(
        "Electrical chargeability (V/V), 0 <= eta < 1",
        default=0.
    )

    tau, tauMap, tauDeriv = Props.Invertible(
        "Time constant (s)",
        default=1.
    )

    c, cMap, cDeriv = Props.Invertible(
        "Frequency Dependency, 0 < c < 1",
        default=0.5
    )

    h, hMap, hDeriv = Props.Invertible(
        "Receiver Height (m), h > 0",
    )

    def __init__(self, mesh, **kwargs):
        Problem.BaseProblem.__init__(self, mesh, **kwargs)

        # Check input arguments. If self.hankel_filter is not a valid filter,
        # it will set it to the default (key_201_2009).
        ht, htarg = check_hankel('fht', [self.hankel_filter,
                                         self.hankel_pts_per_dec], 1)

        self.fhtfilt = htarg[0]                 # Store filter
        self.hankel_filter = self.fhtfilt.name  # Store name
        self.hankel_pts_per_dec = htarg[1]      # Store pts_per_dec
        if self.verbose:
            print(">> Use "+self.hankel_filter+" filter for Hankel Transform")

        if self.hankel_pts_per_dec != 0:
            raise NotImplementedError()


    def hz_kernel_vertical_magnetic_dipole(
        self, lamda, f, n_layer, sig, chi, depth, h, z,
        flag, output_type='response'
    ):

        """
            Kernel for vertical magnetic component (Hz) due to
            vertical magnetic diopole (VMD) source in (kx,ky) domain

        """
        u0 = lamda
        rTE = np.zeros(lamda.size, dtype=complex)
        coefficient_wavenumber = 1/(4*np.pi)*lamda**3/u0

        if output_type == 'sensitivity_sigma':
            drTE = np.zeros((n_layer, lamda.size), dtype=complex)
            drTE = rTEfunjac(
                n_layer, f, lamda, sig, chi, depth, self.survey.half_switch
            )
            kernel = drTE * np.exp(-u0*(z+h)) * coefficient_wavenumber
        else:
            rTE = rTEfunfwd(
                n_layer, f, lamda, sig, chi, depth, self.survey.half_switch
            )
            kernel = rTE * np.exp(-u0*(z+h)) * coefficient_wavenumber
            if output_type == 'sensitivity_height':
                kernel *= -2*u0

        return kernel

        # Note
        # Here only computes secondary field.
        # I am not sure why it does not work if we add primary term.
        # This term can be analytically evaluated, where h = 0.
        #     kernel = (
        #         1./(4*np.pi) *
        #         (np.exp(u0*(z-h))+rTE * np.exp(-u0*(z+h)))*lamda**3/u0
        #     )

    def hz_kernel_circular_loop(
        self, lamda, f, n_layer, sig, chi, depth, h, z, I, a,
        flag,  output_type='response'
    ):

        """

        Kernel for vertical magnetic component (Hz) at the center
        due to circular loop source in (kx,ky) domain

        .. math::

            H_z = \\frac{Ia}{2} \int_0^{\infty} [e^{-u_0|z+h|} + \\r_{TE}e^{u_0|z-h|}] \\frac{\lambda^2}{u_0} J_1(\lambda a)] d \lambda

        """

        w = 2*np.pi*f
        rTE = np.empty(lamda.size, dtype=complex)
        u0 = lamda
        coefficient_wavenumber = I*a*0.5*lamda**2/u0

        if output_type == 'sensitivity_sigma':
            drTE = np.empty((n_layer, lamda.size), dtype=complex)
            drTE = rTEfunjac(
                n_layer, f, lamda, sig, chi, depth, self.survey.half_switch
            )
            kernel = drTE * np.exp(-u0*(z+h)) * coefficient_wavenumber
        else:
            rTE = rTEfunfwd(
                n_layer, f, lamda, sig, chi, depth, self.survey.half_switch
            )

            if flag == 'secondary':
                kernel = rTE * np.exp(-u0*(z+h)) * coefficient_wavenumber
            else:
                kernel = rTE * (
                    np.exp(-u0*(z+h)) + np.exp(u0*(z-h))
                ) * coefficient_wavenumber

            if output_type == 'sensitivity_height':
                kernel *= -2*u0

        return kernel

    def hz_kernel_horizontal_electric_dipole(
        self, lamda, f, n_layer, sig, chi, depth, h, z,
        flag, output_type='response'
    ):

        """
            Kernel for vertical magnetic field (Hz) due to
            horizontal electric diopole (HED) source in (kx,ky) domain

        """
        u0 = lamda
        rTE = np.zeros(lamda.size, dtype=complex)
        coefficient_wavenumber = 1/(4*np.pi)*lamda**2/u0

        if output_type == 'sensitivity_sigma':
            drTE = np.zeros((n_layer, lamda.size), dtype=complex)
            drTE = rTEfunjac(
                n_layer, f, lamda, sig, chi, depth, self.survey.half_switch
            )
            kernel = drTE * np.exp(-u0*(z+h)) * coefficient_wavenumber
        else:
            rTE = rTEfunfwd(
                n_layer, f, lamda, sig, chi, depth, self.survey.half_switch
            )
            kernel = rTE * np.exp(-u0*(z+h)) * coefficient_wavenumber
            if output_type == 'sensitivity_height':
                kernel *= -2*u0

        return kernel

    def sigma_cole(self, f):
        """
        Computes Pelton's Cole-Cole conductivity model
        in frequency domain.

        Parameter
        ---------

        f: ndarray
            frequency (Hz)

        Return
        ------

        sigma_complex: ndarray
            Cole-Cole conductivity values at given frequencies.

        """
        w = 2*np.pi*f
        sigma_complex = (
            self.sigma -
            self.sigma*self.eta/(1+(1-self.eta)*(1j*w*self.tau)**self.c)
        )
        return sigma_complex

    def forward(self, m, output_type='response'):
        """
            Return Bz or dBzdt
        """

        f = self.survey.frequency
        n_frequency = self.survey.n_frequency
        flag = self.survey.field_type

        # Get lambd and offset, will depend on pts_per_dec
        r = self.survey.offset
        lambd, _ = get_spline_values(self.fhtfilt, r, self.hankel_pts_per_dec)

        self.model = m

        n_layer = self.survey.n_layer
        depth = self.survey.depth
        nfilt = self.fhtfilt.base.size

        # h is an inversion parameter
        if self.hMap is not None:
            h = self.h
        else:
            h = self.survey.h
        z = h + self.survey.dz
        chi = self.chi

        if np.isscalar(self.chi):
            chi = np.ones_like(self.sigma) * self.chi

        if output_type == 'response':
            if self.verbose:
                print ('>> Compute response')

            # for simulation
            hz = np.empty((n_frequency, nfilt), complex)
            hz0 = np.zeros((n_frequency, nfilt), complex)  # For zero kernels

            if self.survey.src_type == 'VMD':
                for ifreq in range(n_frequency):
                    sig = self.sigma_cole(f[ifreq])
                    hz[ifreq, :] = self.hz_kernel_vertical_magnetic_dipole(
                        lambd[ifreq, :], f[ifreq], n_layer,
                        sig, chi, depth, h, z,
                        flag, output_type=output_type
                    )
                # ??? D.W. General question: Why does the wavenumber depend
                #     on frequency (lambd[ifreq, :], or before
                #     self.YBASE/r[ifreq])? Shouldn't it only depend on offset?
                #     Because this might be an issue if we want to use Lagged
                #     Convolution or Splined DLF, as those alter the used
                #     wavenumbers...

                PJ = (hz, hz0, hz0)  # PJ0

            elif self.survey.src_type == 'CircularLoop':
                I = self.survey.I
                a = self.survey.a
                for ifreq in range(n_frequency):
                    sig = self.sigma_cole(f[ifreq])
                    hz[ifreq, :] = self.hz_kernel_circular_loop(
                        lambd[ifreq, :], f[ifreq], n_layer,
                        sig, chi, depth, h, z, I, a,
                        flag, output_type=output_type
                    )
                # ??? D.W. lambd here was self.YBASE/a?
                # ??? D.W. The result after DLF was divided by a?

                PJ = (hz0, hz, hz0)  # PJ1

            elif self.survey.src_type == "piecewise_line":
                for ifreq in range(n_frequency):
                    sig = self.sigma_cole(f[ifreq])
                    # Need to compute y
                    hz[ifreq, :] = self.hz_kernel_horizontal_electric_dipole(
                        lambd[ifreq, :], f[ifreq], n_layer,
                        sig, chi, depth, h, z, I, a,
                        flag, output_type=output_type
                    )
                # ??? D.W. lambd here was self.YBASE/r[ifreq]*y?
                # ??? D.W. The result after DLF was divided by a?

                PJ = (hz0, hz, hz0)  # PJ1

            else:
                raise Exception("Src options are only VMD or CircularLoop!!")

        elif output_type == 'sensitivity_sigma':

            # for simulation
            hz = np.empty((n_frequency, n_layer, nfilt), complex)
            hz0 = np.zeros((n_frequency, n_layer, nfilt), complex)  # For 0 krn

            if self.survey.src_type == 'VMD':
                r = self.survey.offset
                for ifreq in range(n_frequency):
                    sig = self.sigma_cole(f[ifreq])
                    hz[ifreq, :, :] = self.hz_kernel_vertical_magnetic_dipole(
                        lambd[ifreq, :], f[ifreq], n_layer,
                        sig, chi, depth, h, z,
                        flag, output_type=output_type
                    )

                PJ = (hz, hz0, hz0)  # PJ0

            elif self.survey.src_type == 'CircularLoop':
                I = self.survey.I
                a = self.survey.a
                for ifreq in range(n_frequency):
                    sig = self.sigma_cole(f[ifreq])
                    hz[ifreq, :, :] = self.hz_kernel_circular_loop(
                        lambd[ifreq, :], f[ifreq], n_layer,
                        sig, chi, depth, h, z, I, a,
                        flag, output_type=output_type
                    )
                # ??? D.W. lambd here was self.YBASE/a?
                # ??? D.W. The result after DLF was divided by a?

                PJ = (hz0, hz, hz0)  # PJ1

            else:
                raise Exception("Src options are only VMD or CircularLoop!!")

        elif output_type == 'sensitivity_height':

            # for simulation
            hz = np.empty((n_frequency, nfilt), complex)
            hz0 = np.zeros((n_frequency, nfilt), complex)  # For zero kernels

            if self.survey.src_type == 'VMD':
                for ifreq in range(n_frequency):
                    sig = self.sigma_cole(f[ifreq])
                    hz[ifreq, :] = self.hz_kernel_vertical_magnetic_dipole(
                        lamb[ifreq, :], f[ifreq], n_layer,
                        sig, chi, depth, h, z,
                        flag, output_type=output_type
                    )

                PJ = (hz, hz0, hz0)  # PJ0

            elif self.survey.src_type == 'CircularLoop':
                I = self.survey.I
                a = self.survey.a
                for ifreq in range(n_frequency):
                    sig = self.sigma_cole(f[ifreq])
                    hz = self.hz_kernel_circular_loop(
                        lambd[ifreq, :], f[ifreq], n_layer,
                        sig, chi, depth, h, z, I, a,
                        flag, output_type=output_type
                    )
                # ??? D.W. lambd here was self.YBASE/a?
                # ??? D.W. The result after DLF was divided by a?

                PJ = (hz0, hz, hz0)  # PJ1

            else:
                raise Exception("Src options are only VMD or CircularLoop!!")

        # Carry out Hankel DLF
        # ab=66 => 33 (vertical magnetic src and rec)
        HzFHT = dlf(PJ, lambd, r, self.fhtfilt, self.hankel_pts_per_dec,
                    factAng=np.full_like(r, 1.), ab=33)

        return HzFHT


    # @profile
    def fields(self, m):
        f = self.forward(m, output_type='response')
        self.survey._pred = Utils.mkvc(self.survey.projectFields(f))
        return f

    def getJ_height(self, m, f=None):
        """

        """
        if self.hMap is None:
            return Utils.Zero()

        if self._Jmatrix_height is not None:
            return self._Jmatrix_height
        else:

            if self.verbose:
                print (">> Compute J height ")

            dudz = self.forward(m, output_type="sensitivity_height")

            self._Jmatrix_height = (
                self.survey.projectFields(dudz)
            ).reshape([-1, 1])

            return self._Jmatrix_height

    # @profile
    def getJ_sigma(self, m, f=None):

        if self.sigmaMap is None:
            return Utils.Zero()

        if self._Jmatrix_sigma is not None:
            return self._Jmatrix_sigma
        else:

            if self.verbose:
                print (">> Compute J sigma")

            dudsig = self.forward(m, output_type="sensitivity_sigma")

            self._Jmatrix_sigma = self.survey.projectFields(dudsig)
            if self._Jmatrix_sigma.ndim == 1:
                self._Jmatrix_sigma = self._Jmatrix_sigma.reshape([-1, 1])
            return self._Jmatrix_sigma

    def getJ(self, m, f=None):
        return (
            self.getJ_sigma(m, f=f) * self.sigmaDeriv +
            self.getJ_height(m, f=f) * self.hDeriv
        )

    def Jvec(self, m, v, f=None):
        """
            Computing Jacobian^T multiplied by vector.
        """

        J_sigma = self.getJ_sigma(m, f=f)
        J_height = self.getJ_height(m, f=f)

        if self.hMap is None:
            Jv = np.dot(J_sigma, self.sigmaMap.deriv(m, v))
        else:
            Jv = np.dot(J_sigma, self.sigmaMap.deriv(m, v))
            Jv += np.dot(J_height, self.hMap.deriv(m, v))
        return Jv

    def Jtvec(self, m, v, f=None):
        """
            Computing Jacobian^T multiplied by vector.
        """

        J_sigma = self.getJ_sigma(m, f=f)
        J_height = self.getJ_height(m, f=f)
        if self.hMap is None:
            Jtv = self.sigmaMap.deriv(m, np.dot(J_sigma.T, v))
        else:
            Jtv = (self.sigmaDeriv.T*np.dot(J_sigma.T, v))
            Jtv += self.hDeriv.T*np.dot(J_height.T, v)
        return Jtv

    @property
    def deleteTheseOnModelUpdate(self):
        toDelete = []
        if self.fix_Jmatrix is False:
            if self._Jmatrix_sigma is not None:
                toDelete += ['_Jmatrix_sigma']
            if self._Jmatrix_height is not None:
                toDelete += ['_Jmatrix_height']
        return toDelete

    def depth_of_investigation(self, uncert, thres_hold=0.8):
        thres_hold = 0.8
        J = self.getJ(self.model)
        S = np.cumsum(abs(np.dot(J.T, 1./uncert))[::-1])[::-1]
        active = S-0.8 > 0.
        doi = abs(self.survey.depth[active]).max()
        return doi, active

    def get_threshold(self, uncert):
        _, active = self.depth_of_investigation(uncert)
        JtJdiag = self.get_JtJdiag(uncert)
        delta = JtJdiag[active].min()
        return delta

    def get_JtJdiag(self, uncert):
        J = self.getJ(self.model)
        JtJdiag = ((Utils.sdiag(1./uncert)*J)**2).sum(axis=0)
        return JtJdiag

if __name__ == '__main__':
    main()
