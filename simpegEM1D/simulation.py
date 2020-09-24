from SimPEG import maps, utils, props
from SimPEG.simulation import BaseSimulation
import numpy as np
from .sources import *
from .survey import BaseEM1DSurvey, EM1DSurveyFD, EM1DSurveyTD
from .supporting_functions.kernels import *
from scipy import sparse as sp
from scipy.constants import mu_0
from scipy.interpolate import InterpolatedUnivariateSpline as iuSpline
from scipy.linalg import block_diag
import properties

from empymod.utils import check_time
from empymod import filters
from empymod.transform import dlf, fourier_dlf, get_dlf_points
from empymod.utils import check_hankel

from .known_waveforms import (
    piecewise_pulse_fast,
    butterworth_type_filter, butter_lowpass_filter
)

try:
    from multiprocessing import Pool
except ImportError:
    print("multiprocessing is not available")
    PARALLEL = False
else:
    PARALLEL = True
    import multiprocessing


#######################################################################
#               SIMULATION FOR A SINGLE SOUNDING
#######################################################################


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

    half_switch = properties.Bool("Switch for half-space")

    # depth = properties.Array("Depth of the layers", dtype=float, required=True)
    # Add layer thickness as invertible property
    thicknesses, thicknessesMap, thicknessesDeriv = props.Invertible(
        "thicknesses of the layers",
        default=np.array([])
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

        # Assign flag if halfspace
        if self.half_switch is None:
            if len(self.thicknesses)==0:
                self.half_switch=True
            else:
                self.half_switch=False
        
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

        # Source height above topography
        if self.hMap is not None:
            h_vector = np.array(self.h)
        else:
            if self.topo is None:
                h_vector = np.array([src.location[2] for src in self.survey.source_list])
            else:
                h_vector = np.array([src.location[2]-self.topo[-1] for src in self.survey.source_list])

        n_filter = self.n_filter

        fields_list = []
        
        for ii, src in enumerate(self.survey.source_list):

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
                if rx.use_source_receiver_offset:
                    z = h + rx.locations[2]
                else:
                    z = h + rx.locations[2] - src.location[2]


                if isinstance(src, HarmonicMagneticDipoleSource) | isinstance(src, TimeDomainMagneticDipoleSource):

                    # Radial distance
                    if rx.use_source_receiver_offset:
                        r = rx.locations[0:2]
                    else:
                        r = rx.locations[0:2] - src.location[0:2]

                    r = np.sqrt(np.sum(r**2))
                    r_vec = r * np.ones(n_frequency)

                    # Use function from empymod
                    # size of lambd is (n_frequency x n_filter)
                    lambd = np.empty([n_frequency, n_filter], order='F')
                    lambd[:, :], _ = get_dlf_points(
                        self.fhtfilt, r_vec, self.hankel_pts_per_dec
                    )

                    # Get kernel function at all lambda and frequencies
                    PJ = magnetic_dipole_kernel(
                        self, lambd, f, n_layer, sig, chi, h, z, r,
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
                    if rx.use_source_receiver_offset:
                        r = rx.locations[0:2]
                    else:
                        r = rx.locations[0:2] - src.location[0:2]

                    r_vec = np.sqrt(np.sum(r**2)) * np.ones(n_frequency)
                    a_vec = src.a * np.ones(n_frequency)

                    # Use function from empymod
                    # size of lambd is (n_frequency x n_filter)
                    lambd = np.empty([n_frequency, n_filter], order='F')
                    lambd[:, :], _ = get_dlf_points(
                        self.fhtfilt, a_vec, self.hankel_pts_per_dec
                    )

                    hz = horizontal_loop_kernel(
                        self, lambd, f, n_layer, sig, chi, a_vec, h, z, r,
                        src, rx, output_type
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

                    u_primary = src.PrimaryField(rx.locations, rx.use_source_receiver_offset)

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



#######################################################################
#               STITCHED 1D SIMULATION CLASS
#######################################################################

def dot(args):
    return np.dot(args[0], args[1])


def run_simulation_FD(args):
    """
        args

        src: source object
        topo: Topographic location (x, y, z)
        hz: Thickeness of the vertical layers
        sigma: conductivities
        eta
        tau
        c
        chi
        h
        jac_switch
        invert_height
        half_switch :
    """

    src, topo, thicknesses, sigma, eta, tau, c, chi, h, jac_switch, invert_height, half_switch = args

    n_layer = len(thicknesses) + 1
    local_survey = EM1DSurveyFD([src])
    expmap = maps.ExpMap(nP=n_layer)

    if not invert_height:
        # Use Exponential Map
        # This is hard-wired at the moment
        
        sim = EM1DFMSimulation(
            survey=local_survey, thicknesses=thicknesses,
            sigmaMap=expmap, chi=chi, eta=eta, tau=tau, c=c, topo=topo,
            half_switch=half_switch, hankel_filter='key_101_2009'
        )
        
        if jac_switch == 'sensitivity_sigma':
            drespdsig = sim.getJ_sigma(np.log(sigma))
            return utils.mkvc(drespdsig * sim.sigmaDeriv)
        else:
            resp = sim.dpred(np.log(sigma))
            return resp
    else:
        
        wires = maps.Wires(('sigma', n_layer), ('h', 1))
        sigmaMap = expmap * wires.sigma
        
        sim = EM1DFMSimulation(
            survey=local_survey, thicknesses=thicknesses,
            sigmaMap=sigmaMap, hMap=wires.h, topo=topo,
            chi=chi, eta=eta, tau=tau, c=c,
            half_switch=half_switch, hankel_filter='key_101_2009'
        )
        
        m = np.r_[np.log(sigma), h]
        if jac_switch == 'sensitivity_sigma':
            drespdsig = sim.getJ_sigma(m)
            return utils.mkvc(drespdsig * utils.sdiag(sigma))
            # return utils.mkvc(drespdsig)
        elif jac_switch == 'sensitivity_height':
            drespdh = sim.getJ_height(m)
            return utils.mkvc(drespdh)
        else:
            resp = sim.dpred(m)
            return resp


def run_simulation_TD(args):
    """
        args

        src: source object
        topo: Topographic location (x, y, z)
        hz: Thickeness of the vertical layers
        sigma: conductivities
        eta
        tau
        c
        chi
        h
        jac_switch
        invert_height
        half_switch :
    """

    src, topo, thicknesses, sigma, eta, tau, c, chi, h, jac_switch, invert_height, half_switch = args

    n_layer = len(thicknesses) + 1
    local_survey = EM1DSurveyTD([src])
    expmap = maps.ExpMap(nP=n_layer)

    if not invert_height:
        # Use Exponential Map
        # This is hard-wired at the moment
        sim = EM1DTMSimulation(
            survey=local_survey, thicknesses=thicknesses,
            sigmaMap=expmap, chi=chi, eta=eta, tau=tau, c=c, topo=topo,
            half_switch=half_switch, hankel_filter='key_101_2009'
        )
        
        if jac_switch == 'sensitivity_sigma':
            drespdsig = sim.getJ_sigma(np.log(sigma))
            return utils.mkvc(drespdsig * sim.sigmaDeriv)
        else:
            resp = sim.dpred(np.log(sigma))
            return resp
    else:
        
        wires = maps.Wires(('sigma', n_layer), ('h', 1))
        sigmaMap = expmap * wires.sigma
        sim = EM1DTMSimulation(
            survey=local_survey, thicknesses=thicknesses,
            sigmaMap=sigmaMap, hMap=wires.h, topo=topo,
            chi=chi, eta=eta, tau=tau, c=c,
            half_switch=half_switch, hankel_filter='key_101_2009'
        )
        
        m = np.r_[np.log(sigma), h]
        if jac_switch == 'sensitivity_sigma':
            drespdsig = sim.getJ_sigma(m)
            return utils.mkvc(drespdsig * utils.sdiag(sigma))
        elif jac_switch == 'sensitivity_height':
            drespdh = sim.getJ_height(m)
            return utils.mkvc(drespdh)
        else:
            resp = sim.dpred(m)
            return resp


class BaseStitchedEM1DSimulation(BaseSimulation):
    """
        The GlobalProblem allows you to run a whole bunch of SubProblems,
        potentially in parallel, potentially of different meshes.
        This is handy for working with lots of sources,
    """
    
    _Jmatrix_sigma = None
    _Jmatrix_height = None
    run_simulation = None
    n_cpu = None
    parallel = False
    parallel_jvec_jtvec = False
    verbose = False
    fix_Jmatrix = False
    invert_height = None

    thicknesses, thicknessesMap, thicknessesDeriv = props.Invertible(
        "thicknesses of the layers",
        default=np.array([])
    )

    sigma, sigmaMap, sigmaDeriv = props.Invertible(
        "Electrical conductivity (S/m)"
    )

    h, hMap, hDeriv = props.Invertible(
        "Receiver Height (m), h > 0",
    )

    chi = props.PhysicalProperty(
        "Magnetic susceptibility (H/m)",
    )

    eta = props.PhysicalProperty(
        "Electrical chargeability (V/V), 0 <= eta < 1"
    )

    tau = props.PhysicalProperty(
        "Time constant (s)"
    )

    c = props.PhysicalProperty(
        "Frequency Dependency, 0 < c < 1"
    )

    topo = properties.Array("Topography (x, y, z)", dtype=float, shape=('*', 3))

    survey = properties.Instance(
        "a survey object", BaseEM1DSurvey, required=True
    )

    half_switch = properties.Bool("Switch for half-space", default=False)

    def __init__(self, **kwargs):
        utils.setKwargs(self, **kwargs)
        
        if PARALLEL:
            if self.parallel:
                print(">> Use multiprocessing for parallelization")
                if self.n_cpu is None:
                    self.n_cpu = multiprocessing.cpu_count()
                print((">> n_cpu: %i") % (self.n_cpu))
            else:
                print(">> Serial version is used")
        else:
            print(">> Serial version is used")

        if self.hMap is None:
            self.invert_height = False
        else:
            self.invert_height = True

    # ------------- For survey ------------- #
    # @property
    # def dz(self):
    #     if self.mesh.dim==2:
    #         return self.mesh.dy
    #     elif self.mesh.dim==3:
    #         return self.mesh.dz
   
    @property
    def n_layer(self):
        if self.thicknesses is None:
            return 1
        else:
            return len(self.thicknesses) + 1

    @property
    def n_sounding(self):
        return len(self.survey.source_list)


    @property
    def data_index(self):
        return self.survey.data_index


    # ------------- For physical properties ------------- #
    @property
    def Sigma(self):
        if getattr(self, '_Sigma', None) is None:
            # Ordering: first z then x
            self._Sigma = self.sigma.reshape((self.n_sounding, self.n_layer))
        return self._Sigma

    @property
    def Chi(self):
        if getattr(self, '_Chi', None) is None:
            # Ordering: first z then x
            if self.chi is None:
                self._Chi = np.zeros(
                    (self.n_sounding, self.n_layer), dtype=float, order='C'
                )
            else:
                self._Chi = self.chi.reshape((self.n_sounding, self.n_layer))
        return self._Chi

    @property
    def Eta(self):
        if getattr(self, '_Eta', None) is None:
            # Ordering: first z then x
            if self.eta is None:
                self._Eta = np.zeros(
                    (self.n_sounding, self.n_layer), dtype=float, order='C'
                )
            else:
                self._Eta = self.eta.reshape((self.n_sounding, self.n_layer))
        return self._Eta

    @property
    def Tau(self):
        if getattr(self, '_Tau', None) is None:
            # Ordering: first z then x
            if self.tau is None:
                self._Tau = 1e-3*np.ones(
                    (self.n_sounding, self.n_layer), dtype=float, order='C'
                )
            else:
                self._Tau = self.tau.reshape((self.n_sounding, self.n_layer))
        return self._Tau

    @property
    def C(self):
        if getattr(self, '_C', None) is None:
            # Ordering: first z then x
            if self.c is None:
                self._C = np.ones(
                    (self.n_sounding, self.n_layer), dtype=float, order='C'
                )
            else:
                self._C = self.c.reshape((self.n_sounding, self.n_layer))
        return self._C

    @property
    def JtJ_sigma(self):
        return self._JtJ_sigma

    def JtJ_height(self):
        return self._JtJ_height

    @property
    def H(self):
        if self.hMap is None:
            return np.ones(self.n_sounding)
        else:
            return self.h


    # ------------- Etcetra .... ------------- #
    @property
    def IJLayers(self):
        if getattr(self, '_IJLayers', None) is None:
            # Ordering: first z then x
            self._IJLayers = self.set_ij_n_layer()
        return self._IJLayers

    @property
    def IJHeight(self):
        if getattr(self, '_IJHeight', None) is None:
            # Ordering: first z then x
            self._IJHeight = self.set_ij_n_layer(n_layer=1)
        return self._IJHeight

    # ------------- For physics ------------- #

    def input_args(self, i_sounding, jac_switch='forward'):
        output = (
            self.survey.source_list[i_sounding],
            self.topo[i_sounding, :],
            self.thicknesses,
            self.Sigma[i_sounding, :],
            self.Eta[i_sounding, :],
            self.Tau[i_sounding, :],
            self.C[i_sounding, :],
            self.Chi[i_sounding, :],
            self.H[i_sounding],
            jac_switch,
            self.invert_height,
            self.half_switch
        )
        return output

    def fields(self, m):
        if self.verbose:
            print("Compute fields")

        return self.forward(m)

    def dpred(self, m, f=None):
        """
            Return predicted data.
            Predicted data, (`_pred`) are computed when
            self.fields is called.
        """
        if f is None:
            f = self.fields(m)

        return f

    def forward(self, m):
        self.model = m

        if self.verbose:
            print(">> Compute response")

        # Set flat topo at zero
        if self.topo is None:
            self.set_null_topography()

        if self.survey.__class__ == EM1DSurveyFD:
            print("Correct Run Simulation")
            run_simulation = run_simulation_FD
        else:
            run_simulation = run_simulation_TD

        if self.parallel:
            pool = Pool(self.n_cpu)
            # This assumes the same # of layers for each of sounding
            result = pool.map(
                run_simulation,
                [
                    self.input_args(i, jac_switch='forward') for i in range(self.n_sounding)
                ]
            )
            pool.close()
            pool.join()
        else:
            result = [
                run_simulation(self.input_args(i, jac_switch='forward')) for i in range(self.n_sounding)
            ]
        return np.hstack(result)


    def set_null_topography(self):
        self.topo = np.vstack(
            [np.c_[src.location[0], src.location[1], 0.] for i, src in enumerate(self.survey.source_list)]
        )


    def set_ij_n_layer(self, n_layer=None):
        """
        Compute (I, J) indicies to form sparse sensitivity matrix
        This will be used in GlobalEM1DSimulation when after sensitivity matrix
        for each sounding is computed
        """
        I = []
        J = []
        shift_for_J = 0
        shift_for_I = 0
        if n_layer is None:
            m = self.n_layer
        else:
            m = n_layer

        for i in range(self.n_sounding):
            n = self.survey.vnD_by_sounding[i]
            J_temp = np.tile(np.arange(m), (n, 1)) + shift_for_J
            I_temp = (
                np.tile(np.arange(n), (1, m)).reshape((n, m), order='F') +
                shift_for_I
            )
            J.append(utils.mkvc(J_temp))
            I.append(utils.mkvc(I_temp))
            shift_for_J += m
            shift_for_I = I_temp[-1, -1] + 1
        J = np.hstack(J).astype(int)
        I = np.hstack(I).astype(int)
        return (I, J)

    def set_ij_height(self):
        """
        Compute (I, J) indicies to form sparse sensitivity matrix
        This will be used in GlobalEM1DSimulation when after sensitivity matrix
        for each sounding is computed
        """
        I = []
        J = []
        shift_for_J = 0
        shift_for_I = 0
        m = self.n_layer
        for i in range(self.n_sounding):
            n = self.survey.vnD[i]
            J_temp = np.tile(np.arange(m), (n, 1)) + shift_for_J
            I_temp = (
                np.tile(np.arange(n), (1, m)).reshape((n, m), order='F') +
                shift_for_I
            )
            J.append(utils.mkvc(J_temp))
            I.append(utils.mkvc(I_temp))
            shift_for_J += m
            shift_for_I = I_temp[-1, -1] + 1
        J = np.hstack(J).astype(int)
        I = np.hstack(I).astype(int)
        return (I, J)


    def getJ_sigma(self, m):
        """
             Compute d F / d sigma
        """
        if self._Jmatrix_sigma is not None:
            return self._Jmatrix_sigma
        if self.verbose:
            print(">> Compute J sigma")
        self.model = m

        if self.survey.__class__ == EM1DSurveyFD:
            run_simulation = run_simulation_FD
        else:
            run_simulation = run_simulation_TD

        if self.parallel:
            pool = Pool(self.n_cpu)
            self._Jmatrix_sigma = pool.map(
                run_simulation,
                [
                    self.input_args(i, jac_switch='sensitivity_sigma') for i in range(self.n_sounding)
                ]
            )
            pool.close()
            pool.join()
            if self.parallel_jvec_jtvec is False:
                # self._Jmatrix_sigma = sp.block_diag(self._Jmatrix_sigma).tocsr()
                self._Jmatrix_sigma = np.hstack(self._Jmatrix_sigma)
                # self._JtJ_sigma_diag =
                self._Jmatrix_sigma = sp.coo_matrix(
                    (self._Jmatrix_sigma, self.IJLayers), dtype=float
                ).tocsr()
        else:
            # _Jmatrix_sigma is block diagnoal matrix (sparse)
            # self._Jmatrix_sigma = sp.block_diag(
            #     [
            #         run_simulation(self.input_args(i, jac_switch='sensitivity_sigma')) for i in range(self.n_sounding)
            #     ]
            # ).tocsr()
            self._Jmatrix_sigma = [
                    run_simulation(self.input_args(i, jac_switch='sensitivity_sigma')) for i in range(self.n_sounding)
            ]
            self._Jmatrix_sigma = np.hstack(self._Jmatrix_sigma)
            self._Jmatrix_sigma = sp.coo_matrix(
                (self._Jmatrix_sigma, self.IJLayers), dtype=float
            ).tocsr()

        return self._Jmatrix_sigma

    def getJ_height(self, m):
        """
             Compute d F / d height
        """
        if self.hMap is None:
            return utils.Zero()

        if self._Jmatrix_height is not None:
            return self._Jmatrix_height
        if self.verbose:
            print(">> Compute J height")

        self.model = m

        if self.survey.__class__ == EM1DSurveyFD:
            run_simulation = run_simulation_FD
        else:
            run_simulation = run_simulation_TD

        if self.parallel:
            pool = Pool(self.n_cpu)
            self._Jmatrix_height = pool.map(
                run_simulation,
                [
                    self.input_args(i, jac_switch="sensitivity_height") for i in range(self.n_sounding)
                ]
            )
            pool.close()
            pool.join()
            if self.parallel_jvec_jtvec is False:
                # self._Jmatrix_height = sp.block_diag(self._Jmatrix_height).tocsr()
                self._Jmatrix_height = np.hstack(self._Jmatrix_height)
                self._Jmatrix_height = sp.coo_matrix(
                    (self._Jmatrix_height, self.IJHeight), dtype=float
                ).tocsr()
        else:
            # self._Jmatrix_height = sp.block_diag(
            #     [
            #         run_simulation(self.input_args(i, jac_switch='sensitivity_height')) for i in range(self.n_sounding)
            #     ]
            # ).tocsr()
            self._Jmatrix_height = [
                    run_simulation(self.input_args(i, jac_switch='sensitivity_height')) for i in range(self.n_sounding)
            ]
            self._Jmatrix_height = np.hstack(self._Jmatrix_height)
            self._Jmatrix_height = sp.coo_matrix(
                (self._Jmatrix_height, self.IJHeight), dtype=float
            ).tocsr()

        return self._Jmatrix_height

    def Jvec(self, m, v, f=None):
        J_sigma = self.getJ_sigma(m)
        J_height = self.getJ_height(m)
        # This is deprecated at the moment
        # if self.parallel and self.parallel_jvec_jtvec:
        #     # Extra division of sigma is because:
        #     # J_sigma = dF/dlog(sigma)
        #     # And here sigmaMap also includes ExpMap
        #     v_sigma = utils.sdiag(1./self.sigma) * self.sigmaMap.deriv(m, v)
        #     V_sigma = v_sigma.reshape((self.n_sounding, self.n_layer))

        #     pool = Pool(self.n_cpu)
        #     Jv = np.hstack(
        #         pool.map(
        #             dot,
        #             [(J_sigma[i], V_sigma[i, :]) for i in range(self.n_sounding)]
        #         )
        #     )
        #     if self.hMap is not None:
        #         v_height = self.hMap.deriv(m, v)
        #         V_height = v_height.reshape((self.n_sounding, self.n_layer))
        #         Jv += np.hstack(
        #             pool.map(
        #                 dot,
        #                 [(J_height[i], V_height[i, :]) for i in range(self.n_sounding)]
        #             )
        #         )
        #     pool.close()
        #     pool.join()
        # else:
        Jv = J_sigma*(utils.sdiag(1./self.sigma)*(self.sigmaDeriv * v))
        if self.hMap is not None:
            Jv += J_height*(self.hDeriv * v)
        return Jv

    def Jtvec(self, m, v, f=None):
        J_sigma = self.getJ_sigma(m)
        J_height = self.getJ_height(m)
        # This is deprecated at the moment
        # if self.parallel and self.parallel_jvec_jtvec:
        #     pool = Pool(self.n_cpu)
        #     Jtv = np.hstack(
        #         pool.map(
        #             dot,
        #             [(J_sigma[i].T, v[self.data_index[i]]) for i in range(self.n_sounding)]
        #         )
        #     )
        #     if self.hMap is not None:
        #         Jtv_height = np.hstack(
        #             pool.map(
        #                 dot,
        #                 [(J_sigma[i].T, v[self.data_index[i]]) for i in range(self.n_sounding)]
        #             )
        #         )
        #         # This assumes certain order for model, m = (sigma, height)
        #         Jtv = np.hstack((Jtv, Jtv_height))
        #     pool.close()
        #     pool.join()
        #     return Jtv
        # else:
        # Extra division of sigma is because:
        # J_sigma = dF/dlog(sigma)
        # And here sigmaMap also includes ExpMap
        Jtv = self.sigmaDeriv.T * (utils.sdiag(1./self.sigma) * (J_sigma.T*v))
        if self.hMap is not None:
            Jtv += self.hDeriv.T*(J_height.T*v)
        return Jtv

    def getJtJdiag(self, m, W=None, threshold=1e-8):
        """
        Compute diagonal component of JtJ or
        trace of sensitivity matrix (J)
        """
        J_sigma = self.getJ_sigma(m)
        J_matrix = J_sigma*(utils.sdiag(1./self.sigma)*(self.sigmaDeriv))

        if self.hMap is not None:
            J_height = self.getJ_height(m)
            J_matrix += J_height*self.hDeriv

        if W is None:
            W = utils.speye(J_matrix.shape[0])

        J_matrix = W*J_matrix
        JtJ_diag = (J_matrix.T*J_matrix).diagonal()
        JtJ_diag /= JtJ_diag.max()
        JtJ_diag += threshold
        return JtJ_diag

    @property
    def deleteTheseOnModelUpdate(self):
        toDelete = []
        if self.sigmaMap is not None:
            toDelete += ['_Sigma']
        if self.fix_Jmatrix is False:
            if self._Jmatrix_sigma is not None:
                toDelete += ['_Jmatrix_sigma']
            if self._Jmatrix_height is not None:
                toDelete += ['_Jmatrix_height']
        return toDelete


class StitchedEM1DFMSimulation(BaseStitchedEM1DSimulation):

    def run_simulation(self, args):
        if self.verbose:
            print(">> Frequency-domain")
        return run_simulation_FD(args)

    # @property
    # def frequency(self):
    #     return self.survey.frequency

    # @property
    # def switch_real_imag(self):
    #     return self.survey.switch_real_imag


class StitchedEM1DTMSimulation(BaseStitchedEM1DSimulation):

    # @property
    # def wave_type(self):
    #     return self.survey.wave_type

    # @property
    # def input_currents(self):
    #     return self.survey.input_currents

    # @property
    # def time_input_currents(self):
    #     return self.survey.time_input_currents

    # @property
    # def n_pulse(self):
    #     return self.survey.n_pulse

    # @property
    # def base_frequency(self):
    #     return self.survey.base_frequency

    # @property
    # def time(self):
    #     return self.survey.time

    # @property
    # def use_lowpass_filter(self):
    #     return self.survey.use_lowpass_filter

    # @property
    # def high_cut_frequency(self):
    #     return self.survey.high_cut_frequency

    # @property
    # def moment_type(self):
    #     return self.survey.moment_type

    # @property
    # def time_dual_moment(self):
    #     return self.survey.time_dual_moment

    # @property
    # def time_input_currents_dual_moment(self):
    #     return self.survey.time_input_currents_dual_moment

    # @property
    # def input_currents_dual_moment(self):
    #     return self.survey.input_currents_dual_moment

    # @property
    # def base_frequency_dual_moment(self):
    #     return self.survey.base_frequency_dual_moment

    def run_simulation(self, args):
        if self.verbose:
            print(">> Time-domain")
        return run_simulation_TD(args)

    # def forward(self, m, f=None):
    #     self.model = m

    #     if self.parallel:
    #         pool = Pool(self.n_cpu)
    #         # This assumes the same # of layer for each of soundings
    #         result = pool.map(
    #             run_simulation_TD,
    #             [
    #                 self.input_args(i, jac_switch=False) for i in range(self.n_sounding)
    #             ]
    #         )
    #         pool.close()
    #         pool.join()
    #     else:
    #         result = [
    #             run_simulation_TD(self.input_args(i, jac_switch=False)) for i in range(self.n_sounding)
    #         ]
    #     return np.hstack(result)

    # def getJ(self, m):
    #     """
    #          Compute d F / d sigma
    #     """
    #     if self._Jmatrix is not None:
    #         return self._Jmatrix
    #     if self.verbose:
    #         print(">> Compute J")
    #     self.model = m
    #     if self.parallel:
    #         pool = Pool(self.n_cpu)
    #         self._Jmatrix = pool.map(
    #             run_simulation_TD,
    #             [
    #                 self.input_args(i, jac_switch=True) for i in range(self.n_sounding)
    #             ]
    #         )
    #         pool.close()
    #         pool.join()
    #         if self.parallel_jvec_jtvec is False:
    #             self._Jmatrix = sp.block_diag(self._Jmatrix).tocsr()
    #     else:
    #         # _Jmatrix is block diagnoal matrix (sparse)
    #         self._Jmatrix = sp.block_diag(
    #             [
    #                 run_simulation_TD(self.input_args(i, jac_switch=True)) for i in range(self.n_sounding)
    #             ]
    #         ).tocsr()
    #     return self._Jmatrix








