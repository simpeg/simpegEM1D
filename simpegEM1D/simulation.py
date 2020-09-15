from SimPEG import maps, utils, props
from SimPEG.simulation import BaseSimulation
import numpy as np
from .sources import *
from .survey import BaseEM1DSurvey, EM1DSurveyTD
from .supporting_functions.kernels import *
from scipy.constants import mu_0
from scipy.interpolate import InterpolatedUnivariateSpline as iuSpline
from scipy.linalg import block_diag
import properties

from empymod.utils import check_time
from empymod import filters
from empymod.transform import dlf, fourier_dlf, get_dlf_points
from empymod.utils import check_hankel

from .KnownWaveforms import (
    piecewise_pulse_fast,
    butterworth_type_filter, butter_lowpass_filter
)




class BaseEM1DSimulation(BaseSimulation):
    """
    Pseudo analytic solutions for frequency and time domain EM problems
    assumingLayered earth (1D).
    """
    surveyPair = BaseEM1DSurvey
    mapPair = maps.IdentityMap
    chi = None
    hankel_filter = 'key_101_2009'  # Default: Hankel filter
    hankel_pts_per_dec = None       # Default: Standard DLF
    verbose = False
    fix_Jmatrix = False
    _Jmatrix_sigma = None
    _Jmatrix_height = None
    _pred = None

    sigma, sigmaMap, sigmaDeriv = props.Invertible(
        "Electrical conductivity at infinite frequency(S/m)"
    )

    h, hMap, hDeriv = props.Invertible(
        "Receiver Height (m), h > 0",
    )

    rho, rhoMap, rhoDeriv = props.Invertible(
        "Electrical resistivity (Ohm m)"
    )

    props.Reciprocal(sigma, rho)

    chi = props.PhysicalProperty(
        "Magnetic susceptibility",
        default=0.
    )

    eta, etaMap, etaDeriv = props.Invertible(
        "Electrical chargeability (V/V), 0 <= eta < 1",
        default=0.
    )

    tau, tauMap, tauDeriv = props.Invertible(
        "Time constant (s)",
        default=1.
    )

    c, cMap, cDeriv = props.Invertible(
        "Frequency Dependency, 0 < c < 1",
        default=0.5
    )

    survey = properties.Instance(
        "a survey object", BaseEM1DSurvey, required=True
    )

    topo = properties.Array("Topography (x, y, z)", dtype=float)

    half_switch = properties.Bool("Switch for half-space", default=False)

    # depth = properties.Array("Depth of the layers", dtype=float, required=True)
    # Add layer thickness as invertible property
    thicknesses, thicknessesMap, thicknessesDeriv = props.Invertible(
        "thicknesses of the layers"
    )

    def __init__(self, **kwargs):
        BaseSimulation.__init__(self, **kwargs)

        # Check input arguments. If self.hankel_filter is not a valid filter,
        # it will set it to the default (key_201_2009).

        ht, htarg = check_hankel(
            'dlf',
            {
                'dlf': self.hankel_filter,
                'pts_per_dec': 0
            },
            1
        )

        self.fhtfilt = htarg['dlf']                 # Store filter
        self.hankel_pts_per_dec = htarg['pts_per_dec']      # Store pts_per_dec
        if self.verbose:
            print(">> Use "+self.hankel_filter+" filter for Hankel Transform")


    # @property
    # def h(self):
    #     """
    #         Source height
    #     """

    #     if getattr(self, '_h', None) is None:
    #         self._h = np.array([src.location[2] for src in self.survey.source_list])

    #     return self._h


    @property
    def n_layer(self):
        """
            number of layers
        """
        if self.half_switch is False:
            return int(self.thicknesses.size + 1)
        elif self.half_switch is True:
            return int(1)

    def sigma_cole(self, frequencies):
        """
        Computes Pelton's Cole-Cole conductivity model
        in frequency domain.

        Parameter
        ---------

        n_filter: int
            the number of filter values
        f: ndarray
            frequency (Hz)

        Return
        ------

        sigma_complex: ndarray (n_layer x n_frequency x n_filter)
            Cole-Cole conductivity values at given frequencies

        """
        n_layer = self.n_layer
        n_frequency = len(frequencies)
        n_filter = self.n_filter

        sigma = np.tile(self.sigma.reshape([-1, 1]), (1, n_frequency))
        if np.isscalar(self.eta):
            eta = self.eta
            tau = self.tau
            c = self.c
        else:
            eta = np.tile(self.eta.reshape([-1, 1]), (1, n_frequency))
            tau = np.tile(self.tau.reshape([-1, 1]), (1, n_frequency))
            c = np.tile(self.c.reshape([-1, 1]), (1, n_frequency))

        w = np.tile(
            2*np.pi*frequencies,
            (n_layer, 1)
        )

        sigma_complex = np.empty(
            [n_layer, n_frequency], dtype=np.complex128, order='F'
        )
        sigma_complex[:, :] = (
            sigma -
            sigma*eta/(1+(1-eta)*(1j*w*tau)**c)
        )

        sigma_complex_tensor = np.empty(
            [n_layer, n_frequency, n_filter], dtype=np.complex128, order='F'
        )
        sigma_complex_tensor[:, :, :] = np.tile(sigma_complex.reshape(
            (n_layer, n_frequency, 1)), (1, 1, n_filter)
        )

        return sigma_complex_tensor

    @property
    def n_filter(self):
        """ Length of filter """
        return self.fhtfilt.base.size

    def depth(self):
        if self.thicknesses is not None:
            return np.r_[0., -np.cumsum(self.thicknesses)]
    

    def compute_integral(self, m, output_type='response'):
        """
            
        """

        
        # Set evaluation frequencies for time domain
        if isinstance(self.survey, EM1DSurveyTD):
            # self.set_time_intervals()  # SOMETHING IS UP WITH THIS
            if self.frequencies_are_set is False:
                self.set_frequencies()

        # Physical Properties
        self.model = m

        chi = self.chi
        if np.isscalar(self.chi):
            chi = np.ones_like(self.sigma) * self.chi

        n_layer = self.n_layer

        # Source heights
        if self.hMap is not None:
            h_vector = self.h
        else:
            if self.topo is None:
                h_vector = np.array([src.location[2] for src in self.survey.source_list])
            else:
                h_vector = np.array([src.location[2]-self.topo[-1] for src in self.survey.source_list])

        n_filter = self.n_filter

        fields_list = []
        
        for ii, src in enumerate(self.survey.source_list):

            I = src.I

            for jj, rx in enumerate(src.receiver_list):

                n_frequency = len(rx.frequencies)
                # TODO: potentially store
                
                f = np.empty([n_frequency, n_filter], order='F')
                f[:, :] = np.tile(
                    rx.frequencies.reshape([-1, 1]), (1, n_filter)
                )

                # Create globally, not for each receiver
                sig = self.sigma_cole(rx.frequencies)

                # Compute receiver height
                h = h_vector[ii]
                z = h + src.location[2] - rx.locations[2]

                if isinstance(src, HarmonicMagneticDipoleSource) | isinstance(src, TimeDomainMagneticDipoleSource):

                    # Radial distance
                    r = src.location[0:2] - rx.locations[0:2]
                    r = np.sqrt(np.sum(r**2))
                    
#                    if r > 0.01:

                    r_vec = r * np.ones(n_frequency)

                    # Use function from empymod
                    # size of lambd is (n_frequency x n_filter)
                    lambd = np.empty([n_frequency, n_filter], order='F')
                    lambd[:, :], _ = get_dlf_points(
                        self.fhtfilt, r_vec, self.hankel_pts_per_dec
                    )

                    # Get kernel function at all lambda and frequencies
                    PJ = magnetic_dipole_kernel(
                        self, lambd, f, n_layer, sig, chi, I, h, z, r,
                        src, rx, output_type
                    )

                    PJ = tuple(PJ)

                    if output_type=="sensitivity_sigma":
                        r_vec = np.tile(r_vec, (n_layer, 1))

                    integral_output = dlf(
                        PJ, lambd, r_vec, self.fhtfilt, self.hankel_pts_per_dec, ang_fact=None, ab=33
                    )

                    # elif src.orientation == "z":

                    #     z_vec = -1j *z * np.ones(n_frequency)

                    #     # Use function from empymod
                    #     # size of lambd is (n_frequency x n_filter)
                    #     lambd = np.empty([n_frequency, n_filter], order='F')
                    #     lambd[:, :], _ = get_dlf_points(
                    #         self.fhtfilt, z_vec, self.hankel_pts_per_dec
                    #     )

                    #     PJ = magnetic_dipole_fourier(
                    #         self, lambd, f, n_layer, sig, chi, I, h, z, r,
                    #         src, rx, output_type
                    #     )

                    #     integral_output = fourier_dlf(
                    #         PJ, lambd, z_vec, filters.key_201_2009(kind='sin'), self.hankel_pts_per_dec
                    #     )



                elif isinstance(src, HarmonicHorizontalLoopSource) | isinstance(src, TimeDomainHorizontalLoopSource):
                    
                    # radial distance and loop radius
                    r = src.location[0:2] - rx.locations[0:2]
                    r_vec = np.sqrt(np.sum(r**2)) * np.ones(n_frequency)
                    a_vec = src.a * np.ones(n_frequency)

                    # Use function from empymod
                    # size of lambd is (n_frequency x n_filter)
                    lambd = np.empty([n_frequency, n_filter], order='F')
                    lambd[:, :], _ = get_dlf_points(
                        self.fhtfilt, a_vec, self.hankel_pts_per_dec
                    )

                    hz = horizontal_loop_kernel(
                        self, lambd, f, n_layer,
                        sig, chi, I, a_vec, h, z, r,
                        rx.orientation, output_type
                    )

                    # kernels for each bessel function
                    # (j0, j1, j2)
                    PJ = (None, hz, None)  # PJ1

                    if output_type == "sensitivity_sigma":
                        a_vec = np.tile(a_vec, (n_layer, 1))

                    integral_output = dlf(
                        PJ, lambd, a_vec, self.fhtfilt, self.hankel_pts_per_dec, ang_fact=None, ab=33
                    )

                if output_type == "sensitivity_sigma":
                    fields_list.append(integral_output.T)
                else:
                    fields_list.append(integral_output)

        return fields_list


    def fields(self, m):
        f = self.compute_integral(m, output_type='response')
        f = self.projectFields(f)
        return np.hstack(f)

    def dpred(self, m, f=None):
        """
            Computes predicted data.
            Here we do not store predicted data
            because projection (`d = P(f)`) is cheap.
        """

        # if f is None:
        #     f = self.fields(m)
        # return utils.mkvc(self.projectFields(f))

        if f is None:
            if m is None:
                m = self.model
            f = self.fields(m)

        return f

    def getJ_height(self, m, f=None):
        """

        """
        if self.hMap is None:
            return utils.Zero()

        if self._Jmatrix_height is not None:
            return self._Jmatrix_height
        else:

            if self.verbose:
                print(">> Compute J height ")

            dudz = self.compute_integral(m, output_type="sensitivity_height")
            dudz = self.projectFields(dudz)

            if self.survey.nSrc == 1:
                self._Jmatrix_height = np.hstack(dudz).reshape([-1, 1])
            else:
                COUNT = 0
                dudz_by_source = []
                for ii, src in enumerate(self.survey.source_list):
                    temp = np.array([])
                    for jj, rx in enumerate(src.receiver_list):
                        temp = np.r_[temp, dudz[COUNT]]
                        COUNT += 1
                    dudz_by_source.append(temp.reshape([-1, 1]))
                
                self._Jmatrix_height= block_diag(*dudz_by_source)
            return self._Jmatrix_height

    # @profile
    def getJ_sigma(self, m, f=None):

        if self.sigmaMap is None:
            return utils.Zero()

        if self._Jmatrix_sigma is not None:
            return self._Jmatrix_sigma
        else:

            if self.verbose:
                print(">> Compute J sigma")

            dudsig = self.compute_integral(m, output_type="sensitivity_sigma")
            # print("SIGMA SENSITIVITIES LIST")
            # print(np.shape(dudsig))

            self._Jmatrix_sigma = np.vstack(self.projectFields(dudsig))
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
        Jv = np.dot(J_sigma, self.sigmaMap.deriv(m, v))
        if self.hMap is not None:
            Jv += np.dot(J_height, self.hMap.deriv(m, v))
        return Jv

    def Jtvec(self, m, v, f=None):
        """
            Computing Jacobian^T multiplied by vector.
        """

        J_sigma = self.getJ_sigma(m, f=f)
        J_height = self.getJ_height(m, f=f)
        Jtv = self.sigmaDeriv.T*np.dot(J_sigma.T, v)
        if self.hMap is not None:
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

    def depth_of_investigation_christiansen_2012(self, std, thres_hold=0.8):
        pred = self.survey._pred.copy()
        delta_d = std * np.log(abs(self.survey.dobs))
        J = self.getJ(self.model)
        J_sum = abs(utils.sdiag(1/delta_d/pred) * J).sum(axis=0)
        S = np.cumsum(J_sum[::-1])[::-1]
        active = S-thres_hold > 0.
        doi = abs(self.depth[active]).max()
        return doi, active

    def get_threshold(self, uncert):
        _, active = self.depth_of_investigation(uncert)
        JtJdiag = self.get_JtJdiag(uncert)
        delta = JtJdiag[active].min()
        return delta

    def get_JtJdiag(self, uncert):
        J = self.getJ(self.model)
        JtJdiag = (np.power((utils.sdiag(1./uncert)*J), 2)).sum(axis=0)
        return JtJdiag


    



class EM1DFMSimulation(BaseEM1DSimulation):

    def __init__(self, **kwargs):
        BaseEM1DSimulation.__init__(self, **kwargs)

    
    def projectFields(self, u):
        """
            Decompose frequency domain EM responses as real and imaginary
            components
        """

        COUNT = 0
        for ii, src in enumerate(self.survey.source_list):
            for jj, rx in enumerate(src.receiver_list):

                u_temp = u[COUNT]

                if rx.component == 'real':
                    u_temp = np.real(u_temp)
                else:
                    u_temp = np.imag(u_temp)

                if rx.field_type != "secondary":

                    u_primary = src.PrimaryField(rx.locations)

                    if rx.field_type == "ppm":
                        k = [comp == rx.orientation for comp in ["x", "y", "z"]]
                        u_temp = 1e6 * u_temp/u_primary[0, k]
                    else:
                        u_temp =+ u_primary

                u[COUNT] = u_temp
                COUNT = COUNT + 1

        return u


class EM1DTMSimulation(BaseEM1DSimulation):


    time_intervals_are_set = False
    frequencies_are_set = False


    def __init__(self, **kwargs):
        BaseEM1DSimulation.__init__(self, **kwargs)

        self.fftfilt = filters.key_81_CosSin_2009()


    def set_time_intervals(self):
        """
        Set time interval for particular receiver
        """

        for src in self.survey.source_list:
            if src.wave_type == "general":
                for rx in src.receiver_list:

                    if src.moment_type == "single":
                        time = rx.times
                        pulse_period = src.pulse_period
                        period = src.period
                    # Dual moment
                    else:
                        time = np.unique(np.r_[rx.times, src.time_dual_moment])
                        pulse_period = np.maximum(
                            src.pulse_period, src.pulse_period_dual_moment
                        )
                        period = np.maximum(src.period, src.period_dual_moment)
                    tmin = time[time>0.].min()
                    if src.n_pulse == 1:
                        tmax = time.max() + pulse_period
                    elif src.n_pulse == 2:
                        tmax = time.max() + pulse_period + period/2.
                    else:
                        raise NotImplementedError("n_pulse must be either 1 or 2")
                    n_time = int((np.log10(tmax)-np.log10(tmin))*10+1)
                    
                    rx.time_interval = np.logspace(
                        np.log10(tmin), np.log10(tmax), n_time
                    )

        self.time_intervals_are_set = True
            # print (tmin, tmax)


    def set_frequencies(self, pts_per_dec=-1):
        """
        Compute Frequency reqired for frequency to time transform
        """

        if self.time_intervals_are_set == False:
            self.set_time_intervals()
        
        for src in self.survey.source_list:
            for rx in src.receiver_list:
                
                if src.wave_type == "general":
                    _, freq, ft, ftarg = check_time(
                        rx.time_interval, -1, 'dlf',
                        {'pts_per_dec': pts_per_dec, 'dlf': self.fftfilt}, 0
                    )
                elif src.wave_type == "stepoff":
                    _, freq, ft, ftarg = check_time(
                        rx.times, -1, 'dlf',
                        {'pts_per_dec': pts_per_dec, 'dlf': self.fftfilt}, 0,
                    )
                else:
                    raise Exception("wave_type must be either general or stepoff")

                rx.frequencies = freq
                rx.ftarg = ftarg

        self.frequencies_are_set = True


    def projectFields(self, u):
        """
            Transform frequency domain responses to time domain responses
        """
        # Compute frequency domain reponses right at filter coefficient values
        # Src waveform: Step-off

        COUNT = 0
        for ii, src in enumerate(self.survey.source_list):

            for jj, rx in enumerate(src.receiver_list):

                u_temp = u[COUNT]

                if src.use_lowpass_filter:
                    factor = src.lowpass_filter.copy()
                else:
                    factor = np.ones_like(rx.frequencies, dtype=complex)

                if rx.component in ["b", "h"]:
                    factor *= 1./(2j*np.pi*rx.frequencies)

                if rx.component in ["b", "dbdt"]:
                    factor *= mu_0

                if src.wave_type == 'stepoff':
                    # Compute EM responses
                    if u_temp.size == rx.n_frequency:
                        resp, _ = fourier_dlf(
                            u_temp.flatten()*factor, rx.times, rx.frequencies, rx.ftarg
                        )
                    # Compute EM sensitivities
                    else:
                        resp = np.zeros(
                            (rx.n_time, self.n_layer), dtype=np.float64, order='F')
                        # )
                        # TODO: remove for loop
                        for i in range(self.n_layer):
                            resp_i, _ = fourier_dlf(
                                u_temp[:, i]*factor, rx.times, rx.frequencies, rx.ftarg
                            )
                            resp[:, i] = resp_i

                # Evaluate piecewise linear input current waveforms
                # Using Fittermann's approach (19XX) with Gaussian Quadrature
                elif src.wave_type == 'general':
                    # Compute EM responses
                    if u_temp.size == rx.n_frequency:
                        resp_int, _ = fourier_dlf(
                            u_temp.flatten()*factor, rx.time_interval, rx.frequencies, rx.ftarg
                        )
                        # step_func = interp1d(
                        #     self.time_int, resp_int
                        # )
                        step_func = iuSpline(
                            np.log10(rx.time_interval), resp_int
                        )

                        resp = piecewise_pulse_fast(
                            step_func, rx.times,
                            src.time_input_currents,
                            src.input_currents,
                            src.period,
                            n_pulse=src.n_pulse
                        )

                        # Compute response for the dual moment
                        if src.moment_type == "dual":
                            resp_dual_moment = piecewise_pulse_fast(
                                step_func, src.time_dual_moment,
                                src.time_input_currents_dual_moment,
                                src.input_currents_dual_moment,
                                src.period_dual_moment,
                                n_pulse=src.n_pulse
                            )
                            # concatenate dual moment response
                            # so, ordering is the first moment data
                            # then the second moment data.
                            resp = np.r_[resp, resp_dual_moment]

                    # Compute EM sensitivities
                    else:
                        if src.moment_type == "single":
                            resp = np.zeros(
                                (rx.n_time, self.n_layer), dtype=np.float64, order='F'
                            )
                        else:
                            # For dual moment
                            resp = np.zeros(
                                (rx.n_time+src.n_time_dual_moment, self.n_layer),
                                dtype=np.float64, order='F'
                            )

                        # TODO: remove for loop (?)
                        for i in range(self.n_layer):
                            resp_int_i, _ = fourier_dlf(
                                u_temp[:, i]*factor, rx.time_interval, rx.frequencies, rx.ftarg
                            )
                            # step_func = interp1d(
                            #     self.time_int, resp_int_i
                            # )

                            step_func = iuSpline(
                                np.log10(rx.time_interval), resp_int_i
                            )

                            resp_i = piecewise_pulse_fast(
                                step_func, rx.times,
                                src.time_input_currents, src.input_currents,
                                src.period, n_pulse=src.n_pulse
                            )

                            if src.moment_type == "single":
                                resp[:, i] = resp_i
                            else:
                                resp_dual_moment_i = piecewise_pulse_fast(
                                    step_func,
                                    src.time_dual_moment,
                                    src.time_input_currents_dual_moment,
                                    src.input_currents_dual_moment,
                                    src.period_dual_moment,
                                    n_pulse=src.n_pulse
                                )
                                resp[:, i] = np.r_[resp_i, resp_dual_moment_i]
                
                u[COUNT] = resp * (-2.0/np.pi)
                COUNT = COUNT + 1

        return u
