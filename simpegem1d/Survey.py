from SimPEG import Maps, Survey, Utils
import numpy as np
import scipy.sparse as sp
from scipy.constants import mu_0
from .EM1DAnalytics import ColeCole
from .DigFilter import (
    transFilt, transFiltImpulse, transFiltInterp, transFiltImpulseInterp
)
from .Waveform import CausalConv
from scipy.interpolate import interp1d
import properties
from empymod import filters
from empymod.utils import check_time
from empymod.transform import ffht
from .Waveforms import piecewise_pulse, butter_lowpass_filter
from profilehooks import profile


class BaseEM1DSurvey(Survey.BaseSurvey, properties.HasProperties):
    """
        Base EM1D Survey

    """

    frequency = properties.Array("Frequency (Hz)", dtype=float)

    rx_location = properties.Array("Receiver location (x, y, z)", dtype=float)
    src_location = properties.Array("Source location (x, y, z)", dtype=float)
    rx_type = properties.StringChoice(
        "Source type",
        default="Bz",
        choices=["Bz", "dBzdt"]
    )
    src_type = properties.StringChoice(
        "Source type",
        default="VMD",
        choices=[
            "VMD", "CircularLoop"
        ]
    )
    offset = properties.Array("Src-Rx offsets", dtype=float)
    rx_type = properties.StringChoice(
        "Source location",
        default="Hz",
        choices=["Hz", "Bz", "dBzdt"]
    )
    field_type = properties.StringChoice(
        "Field type",
        default="secondary",
        choices=["total", "secondary"]
    )
    depth = properties.Array("Depth of the layers", dtype=float)
    topo = properties.Array("Topography (x, y, z)", dtype=float)
    I = properties.Float("Src loop current", default=1.)
    a = properties.Float("Src loop radius", default=1.)
    half_switch = properties.Bool("Switch for half-space", default=False)

    def __init__(self, **kwargs):
        Survey.BaseSurvey.__init__(self, **kwargs)

    @property
    def h(self):
        """
            Source height
        """
        return self.src_location[2]-self.topo[2]

    @property
    def z(self):
        """
            Receiver height
        """
        return self.rx_location[2]-self.topo[2]

    @property
    def dz(self):
        """
            Source height - Rx height
        """
        return self.z - self.h

    @property
    def n_layer(self):
        """
            Srource height
        """
        if self.half_switch is False:
            return self.depth.size
        elif self.half_switch is True:
            return int(1)

    @property
    def n_frequency(self):
        """
            # of frequency
        """

        return int(self.frequency.size)

    @Utils.requires('prob')
    def dpred(self, m, f=None):
        """
            Computes predicted data.
            Here we do not store predicted data
            because projection (`d = P(f)`) is cheap.
        """

        if f is None:
            f = self.prob.fields(m)
        return Utils.mkvc(self.projectFields(f))


class EM1DSurveyFD(BaseEM1DSurvey):
    """
        Freqency-domain EM1D survey
    """
    # Nfreq = None
    switch_real_imag = properties.StringChoice(
        "Switch for real and imaginary part of the data",
        default="all",
        choices=["all", "real", "imag"]
    )

    def __init__(self, **kwargs):
        BaseEM1DSurvey.__init__(self, **kwargs)

        if self.src_type == "VMD":
            if self.offset is None:
                raise Exception("offset is required!")

            if self.offset.size == 1:
                self.offset = self.offset * np.ones(self.n_frequency)

    @property
    def nD(self):
        """
            # of data
        """

        if self.switch_real_imag == "all":
            return int(self.frequency.size * 2)
        elif (
            self.switch_real_imag == "imag" or self.switch_real_imag == "real"
        ):
            return int(self.n_frequency)

    def projectFields(self, u):
        """
            Decompose frequency domain EM responses as real and imaginary
            components
        """

        ureal = (u.real).copy()
        uimag = (u.imag).copy()

        if self.rx_type == 'Hz':
            if self.switch_real_imag == 'all':
                ureal = (u.real).copy()
                uimag = (u.imag).copy()
                if ureal.ndim == 1 or 0:
                    resp = np.r_[ureal, uimag]
                elif ureal.ndim == 2:
                    resp = np.vstack((ureal, uimag))
                else:
                    raise NotImplementedError()
            elif self.switch_real_imag == 'real':
                resp = (u.real).copy()
            elif self.switch_real_imag == 'imag':
                resp = (u.imag).copy()
            else:
                raise NotImplementedError()
        else:
            raise NotImplementedError()
        return mu_0*resp


class EM1DSurveyTD(BaseEM1DSurvey):
    """docstring for EM1DSurveyTD"""

    time = properties.Array(
        "Time channels (s) at current off-time", dtype=float
    )

    wave_type = properties.StringChoice(
        "Source location",
        default="stepoff",
        choices=["stepoff", "general"]
    )

    moment_type = properties.StringChoice(
        "Source moment type",
        default="single",
        choices=["single", "dual"]
    )

    n_pulse = properties.Integer(
        "The number of pulses",
    )

    base_frequency = properties.Float(
        "Base frequency (Hz)"
    )

    time_input_currents = properties.Array(
        "Time for input currents", dtype=float
    )

    input_currents = properties.Array(
        "Input currents", dtype=float
    )

    use_lowpass_filter = properties.Bool(
        "Switch for low pass filter", default=False
    )

    high_cut_frequency = properties.Float(
        "High cut frequency for low pass filter (Hz)", default=1e5
    )

    # Predicted data
    _pred = None

    # ------------- For dual moment ------------- #

    time_dual_moment = properties.Array(
        "Off-time channels (s) for the dual moment", dtype=float
    )

    time_input_currents_dual_moment = properties.Array(
        "Time for input currents (dual moment)", dtype=float
    )

    input_currents_dual_moment = properties.Array(
        "Input currents (dual moment)", dtype=float
    )

    base_frequency_dual_moment = properties.Float(
        "Base frequency for the dual moment (Hz)"
    )

    def __init__(self, **kwargs):
        BaseEM1DSurvey.__init__(self, **kwargs)
        if self.time is None:
            raise Exception("time is required!")

        # Use Sin filter for frequency to time transform
        self.fftfilt = filters.key_81_CosSin_2009()
        self.set_frequency()

        if self.src_type == "VMD":
            if self.offset is None:
                raise Exception("offset is required!")

            if self.offset.size == 1:
                self.offset = self.offset * np.ones(self.n_frequency)

    @property
    def time_int(self):
        """
        Time channels (s) for interpolation"
        """
        if getattr(self, '_time_int', None) is None:

            if self.moment_type == "single":
                time = self.time

            # Dual moment
            else:
                time = np.unique(np.r_[self.time, self.time_dual_moment])

            tmin = time.min()
            if self.n_pulse == 1:
                tmax = time.max() + self.pulse_period
            elif self.n_pulse == 2:
                tmax = time.max() + self.pulse_period + self.period/2.
            else:
                raise NotImplementedError("n_pulse must be either 1 or 2")
            n_time = int((np.log10(tmax)-np.log10(tmin))*10+1)
            self._time_int = np.logspace(
                np.log10(tmin), np.log10(tmax), n_time
            )

        return self._time_int

    @property
    def n_time(self):
        return int(self.time.size)

    @property
    def period(self):
        return 1./self.base_frequency

    @property
    def pulse_period(self):
        Tp = (
            self.time_input_currents.max()-self.time_input_currents.min()
        )
        return Tp

    # ------------- For dual moment ------------- #
    @property
    def n_time_dual_moment(self):
        return int(self.time_dual_moment.size)

    @property
    def period_dual_moment(self):
        return 1./self.base_frequency_dual_moment

    @property
    def pulse_period_dual_moment(self):
        Tp = (
            self.time_input_currents_dual_moment.max() -
            self.time_input_currents._dual_moment.min()
        )
        return Tp

    @property
    def nD(self):
        """
            # of data
        """
        if self.moment_type == "single":
            return self.n_time
        else:
            return self.n_time + self.n_time_dual_moment

    @property
    def lowpass_filter(self):
        """
            Low pass filter values
        """
        if getattr(self, '_lowpass_filter', None) is None:
            filter_frequency, values = butter_lowpass_filter(
                self.high_cut_frequency
            )
            lowpass_func = interp1d(
                filter_frequency, values, fill_value='extrapolate'
            )
            self._lowpass_filter = lowpass_func(self.frequency)

        return self._lowpass_filter

    def set_frequency(self):
        """
        Compute Frequency reqired for frequency to time transform
        """
        if self.wave_type == "general":
            _, frequency, ft, ftarg = check_time(
                self.time_int, 0, 'sin',
                {'pts_per_dec': 5, 'fftfilt': self.fftfilt}, 0
            )
        elif self.wave_type == "stepoff":
            _, frequency, ft, ftarg = check_time(
                self.time, 0, 'sin',
                {'pts_per_dec': 5, 'fftfilt': self.fftfilt}, 0
            )
        else:
            raise Exception("wave_type must be either general or stepoff")

        self.frequency = frequency
        self.ftarg = ftarg

    def projectFields(self, u):
        """
            Transform frequency domain responses to time domain responses
        """
        # Compute frequency domain reponses right at filter coefficient values
        # Src waveform: Step-off

        if self.use_lowpass_filter:
            factor = self.lowpass_filter.copy()
        else:
            factor = np.ones_like(self.frequency, dtype=complex)

        if self.rx_type == 'Bz':
            factor *= 1./(2j*np.pi*self.frequency)

        if self.wave_type == 'stepoff':
            # Compute EM responses
            if u.size == self.n_frequency:
                resp = np.empty(self.n_time, dtype=float)
                resp, _ = ffht(
                    u.flatten()*factor, self.time,
                    self.frequency, self.ftarg
                )
            # Compute EM sensitivities
            else:
                resp = np.zeros(
                    (self.n_time, self.n_layer), dtype=float, order='F'
                )
                resp_i = np.empty(self.n_time, dtype=float)
                for i in range(self.n_layer):
                    resp_i, _ = ffht(
                        u[:, i]*factor, self.time,
                        self.frequency, self.ftarg
                    )
                    resp[:, i] = resp_i

        # Evaluate piecewise linear input current waveforms
        # Using Fittermann's approach (19XX) with Gaussian Quadrature
        elif self.wave_type == 'general':
            # Compute EM responses
            if u.size == self.n_frequency:
                resp = np.empty(self.n_time, dtype=float)
                resp_int = np.empty(self.time_int.size, dtype=float)
                resp_int, _ = ffht(
                    u.flatten()*factor, self.time_int,
                    self.frequency, self.ftarg
                )
                step_func = interp1d(
                    self.time_int, resp_int
                )
                resp = piecewise_pulse(
                    step_func, self.time,
                    self.time_input_currents, self.input_currents,
                    self.period, n_pulse=self.n_pulse
                )

                # Compute response for the dual moment
                if self.moment_type == "dual":
                    resp_dual_moment = np.empty(
                        self.n_time_dual_moment, dtype=float
                    )
                    resp_dual_moment = piecewise_pulse(
                        step_func, self.time_dual_moment,
                        self.time_input_currents_dual_moment,
                        self.input_currents_dual_moment,
                        self.period_dual_moment,
                        n_pulse=self.n_pulse
                    )
                    # concatenate dual moment response
                    # so, ordering is the first moment data
                    # then the second moment data.
                    resp = np.r_[resp, resp_dual_moment]

            # Compute EM sensitivities
            else:

                if self.moment_type == "single":
                    resp = np.zeros(
                        (self.n_time, self.n_layer), dtype=float, order='F'
                    )
                    resp_i = np.empty(self.time.size, dtype=float)
                else:
                    # For dual moment
                    resp = np.zeros(
                        (self.n_time+self.n_time_dual_moment, self.n_layer),
                        dtype=float, order='F'
                    )
                    resp_i = np.empty(self.n_time, dtype=float)
                    resp_dual_moment_i = np.empty(
                        self.time_dual_moment.size, dtype=float
                    )

                resp_int_i = np.empty(self.time_int.size, dtype=float)

                for i in range(self.n_layer):
                    resp_int_i, _ = ffht(
                        u[:, i]*factor, self.time_int,
                        self.frequency, self.ftarg
                    )
                    step_func = interp1d(
                        self.time_int, resp_int_i
                    )
                    resp_i = piecewise_pulse(
                        step_func, self.time,
                        self.time_input_currents, self.input_currents,
                        self.period, n_pulse=self.n_pulse
                    )

                    if self.moment_type == "single":
                        resp[:, i] = resp_i
                    else:
                        resp_dual_moment_i = piecewise_pulse(
                            step_func,
                            self.time_dual_moment,
                            self.time_input_currents_dual_moment,
                            self.input_currents_dual_moment,
                            self.period_dual_moment,
                            n_pulse=self.n_pulse
                        )
                        resp[:, i] = np.r_[resp_i, resp_dual_moment_i]

        return resp * (-2/np.pi) * mu_0

    @Utils.requires('prob')
    def dpred(self, m, f=None):
        """
            Computes predicted data.
            Predicted data (`_pred`) are computed and stored
            when self.prob.fields(m) is called.
        """
        if f is None:
            f = self.prob.fields(m)

        return self._pred

    # ------------ Dummy codes for older version ------------ #
    # This can be used in later use for handling on-time data.
    # ------------------------------------------------------- #

    # def setWaveform(self, **kwargs):
    #     """
    #         Set parameters for Src Waveform
    #     """
    #     # TODO: this is hp is only valid for Circular loop system
    #     self.hp = self.I/self.a*0.5

    #     self.toff = kwargs['toff']
    #     self.waveform = kwargs['waveform']
    #     self.waveformDeriv = kwargs['waveformDeriv']
    #     self.tconv = kwargs['tconv']

    # def projectFields(self, u):
    #     """
    #         Transform frequency domain responses to time domain responses
    #     """
    #     # Case1: Compute frequency domain reponses right at filter coefficient values
    #     if self.switchInterp == False:
    #         # Src waveform: Step-off
    #         if self.wave_type == 'stepoff':
    #             if self.rx_type == 'Bz':
    #                 # Compute EM responses
    #                 if u.size == self.n_frequency:
    #                     resp, f0 = transFilt(Utils.mkvc(u), self.wt,self.tbase, self.frequency*2*np.pi, self.time)
    #                 # Compute EM sensitivities
    #                 else:
    #                     resp = np.zeros((self.n_time, self.n_layer))
    #                     for i in range (self.n_layer):
    #                         resp[:,i], f0 = transFilt(u[:,i], self.wt,self.tbase, self.frequency*2*np.pi, self.time)

    #             elif self.rx_type == 'dBzdt':
    #                 # Compute EM responses
    #                 if u.size == self.n_frequency:
    #                     resp = -transFiltImpulse(u, self.wt,self.tbase, self.frequency*2*np.pi, self.time)
    #                 # Compute EM sensitivities
    #                 else:
    #                     resp = np.zeros((self.n_time, self.n_layer))
    #                     for i in range (self.n_layer):
    #                         resp[:,i] = -transFiltImpulse(u[:,i], self.wt,self.tbase, self.frequency*2*np.pi, self.time)

    #         # Src waveform: General (it can be any waveform)
    #         # We evaluate this with time convolution
    #         elif self.wave_type == 'general':
    #             # Compute EM responses
    #             if u.size == self.n_frequency:
    #                 # TODO: write small code which compute f at t = 0
    #                 f, f0 = transFilt(Utils.mkvc(u), self.wt, self.tbase, self.frequency*2*np.pi, self.tconv)
    #                 fDeriv = -transFiltImpulse(Utils.mkvc(u), self.wt,self.tbase, self.frequency*2*np.pi, self.tconv)

    #                 if self.rx_type == 'Bz':

    #                     waveConvfDeriv = CausalConv(self.waveform, fDeriv, self.tconv)
    #                     resp1 = (self.waveform*self.hp*(1-f0[1]/self.hp)) - waveConvfDeriv
    #                     respint = interp1d(self.tconv, resp1, 'linear')

    #                     # TODO: make it as an opition #2
    #                     # waveDerivConvf = CausalConv(self.waveformDeriv, f, self.tconv)
    #                     # resp2 = (self.waveform*self.hp) - waveDerivConvf
    #                     # respint = interp1d(self.tconv, resp2, 'linear')

    #                     resp = respint(self.time)

    #                 if self.rx_type == 'dBzdt':
    #                     waveDerivConvfDeriv = CausalConv(self.waveformDeriv, fDeriv, self.tconv)
    #                     resp1 = self.hp*self.waveformDeriv*(1-f0[1]/self.hp) - waveDerivConvfDeriv
    #                     respint = interp1d(self.tconv, resp1, 'linear')
    #                     resp = respint(self.time)

    #             # Compute EM sensitivities
    #             else:

    #                 resp = np.zeros((self.n_time, self.n_layer))
    #                 for i in range (self.n_layer):

    #                     f, f0 = transFilt(u[:,i], self.wt, self.tbase, self.frequency*2*np.pi, self.tconv)
    #                     fDeriv = -transFiltImpulse(u[:,i], self.wt,self.tbase, self.frequency*2*np.pi, self.tconv)

    #                     if self.rx_type == 'Bz':

    #                         waveConvfDeriv = CausalConv(self.waveform, fDeriv, self.tconv)
    #                         resp1 = (self.waveform*self.hp*(1-f0[1]/self.hp)) - waveConvfDeriv
    #                         respint = interp1d(self.tconv, resp1, 'linear')

    #                         # TODO: make it as an opition #2
    #                         # waveDerivConvf = CausalConv(self.waveformDeriv, f, self.tconv)
    #                         # resp2 = (self.waveform*self.hp) - waveDerivConvf
    #                         # respint = interp1d(self.tconv, resp2, 'linear')

    #                         resp[:,i] = respint(self.time)

    #                     if self.rx_type == 'dBzdt':
    #                         waveDerivConvfDeriv = CausalConv(self.waveformDeriv, fDeriv, self.tconv)
    #                         resp1 = self.hp*self.waveformDeriv*(1-f0[1]/self.hp) - waveDerivConvfDeriv
    #                         respint = interp1d(self.tconv, resp1, 'linear')
    #                         resp[:,i] = respint(self.time)

    #     # Case2: Compute frequency domain reponses in logarithmic then intepolate
    #     if self.switchInterp == True:
    #         # Src waveform: Step-off
    #         if self.wave_type == 'stepoff':
    #             if self.rx_type == 'Bz':
    #                 # Compute EM responses
    #                 if u.size == self.n_frequency:
    #                     resp, f0 = transFiltInterp(Utils.mkvc(u), self.wt,self.tbase, self.frequency*2*np.pi, self.omega_int, self.time)
    #                 # Compute EM sensitivities
    #                 else:
    #                     resp = np.zeros((self.n_time, self.n_layer))
    #                     for i in range (self.n_layer):
    #                         resp[:,i], f0 = transFiltInterp(u[:,i], self.wt,self.tbase, self.frequency*2*np.pi, self.omega_int, self.time)

    #             elif self.rx_type == 'dBzdt':
    #                 # Compute EM responses
    #                 if u.size == self.n_frequency:
    #                     resp = -transFiltImpulseInterp(Utils.mkvc(u), self.wt,self.tbase, self.frequency*2*np.pi, self.omega_int, self.time)
    #                 # Compute EM sensitivities
    #                 else:
    #                     resp = np.zeros((self.n_time, self.n_layer))
    #                     for i in range (self.n_layer):
    #                         resp[:,i] = -transFiltImpulseInterp(u[:,i], self.wt,self.tbase, self.frequency*2*np.pi, self.omega_int, self.time)

    #         # Src waveform: General (it can be any waveform)
    #         # We evaluate this with time convolution
    #         elif self.wave_type == 'general':
    #             # Compute EM responses
    #             if u.size == self.n_frequency:
    #                 # TODO: write small code which compute f at t = 0
    #                 f, f0 = transFiltInterp(Utils.mkvc(u), self.wt, self.tbase, self.frequency*2*np.pi, self.omega_int, self.tconv)
    #                 fDeriv = -transFiltImpulseInterp(Utils.mkvc(u), self.wt,self.tbase, self.frequency*2*np.pi, self.omega_int, self.tconv)

    #                 if self.rx_type == 'Bz':

    #                     waveConvfDeriv = CausalConv(self.waveform, fDeriv, self.tconv)
    #                     resp1 = (self.waveform*self.hp*(1-f0[1]/self.hp)) - waveConvfDeriv
    #                     respint = interp1d(self.tconv, resp1, 'linear')

    #                     # TODO: make it as an opition #2
    #                     # waveDerivConvf = CausalConv(self.waveformDeriv, f, self.tconv)
    #                     # resp2 = (self.waveform*self.hp) - waveDerivConvf
    #                     # respint = interp1d(self.tconv, resp2, 'linear')

    #                     resp = respint(self.time)

    #                 if self.rx_type == 'dBzdt':
    #                     waveDerivConvfDeriv = CausalConv(self.waveformDeriv, fDeriv, self.tconv)
    #                     resp1 = self.hp*self.waveformDeriv*(1-f0[1]/self.hp) - waveDerivConvfDeriv
    #                     respint = interp1d(self.tconv, resp1, 'linear')
    #                     resp = respint(self.time)

    #             # Compute EM sensitivities
    #             else:

    #                 resp = np.zeros((self.n_time, self.n_layer))
    #                 for i in range (self.n_layer):

    #                     f, f0 = transFiltInterp(u[:,i], self.wt, self.tbase, self.frequency*2*np.pi, self.omega_int, self.tconv)
    #                     fDeriv = -transFiltImpulseInterp(u[:,i], self.wt,self.tbase, self.frequency*2*np.pi, self.omega_int, self.tconv)

    #                     if self.rx_type == 'Bz':

    #                         waveConvfDeriv = CausalConv(self.waveform, fDeriv, self.tconv)
    #                         resp1 = (self.waveform*self.hp*(1-f0[1]/self.hp)) - waveConvfDeriv
    #                         respint = interp1d(self.tconv, resp1, 'linear')

    #                         # TODO: make it as an opition #2
    #                         # waveDerivConvf = CausalConv(self.waveformDeriv, f, self.tconv)
    #                         # resp2 = (self.waveform*self.hp) - waveDerivConvf
    #                         # respint = interp1d(self.tconv, resp2, 'linear')

    #                         resp[:,i] = respint(self.time)

    #                     if self.rx_type == 'dBzdt':
    #                         waveDerivConvfDeriv = CausalConv(self.waveformDeriv, fDeriv, self.tconv)
    #                         resp1 = self.hp*self.waveformDeriv*(1-f0[1]/self.hp) - waveDerivConvfDeriv
    #                         respint = interp1d(self.tconv, resp1, 'linear')
    #                         resp[:,i] = respint(self.time)

    #     return mu_0*resp
