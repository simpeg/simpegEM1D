from SimPEG import maps, utils
from SimPEG.survey import BaseSurvey
import numpy as np
import scipy.sparse as sp
from scipy.constants import mu_0
from .analytics import ColeCole
from .supporting_functions.digital_filter import (
    transFilt, transFiltImpulse, transFiltInterp, transFiltImpulseInterp
)
from .waveforms import CausalConv
from scipy.interpolate import interp1d
from scipy.interpolate import InterpolatedUnivariateSpline as iuSpline
import properties
from empymod import filters
from empymod.utils import check_time
from empymod.transform import fourier_dlf
from .KnownWaveforms import (
    piecewise_pulse_fast,
    butterworth_type_filter, butter_lowpass_filter
)


class BaseEM1DSurvey(BaseSurvey, properties.HasProperties):
    """
        Base EM1D Survey

    """

    def __init__(self, source_list=None, **kwargs):
        BaseSurvey.__init__(self, source_list, **kwargs)


class EM1DSurveyFD(BaseEM1DSurvey):
    """
        Freqency-domain EM1D survey
    """

    def __init__(self, source_list=None, **kwargs):
        BaseEM1DSurvey.__init__(self, source_list, **kwargs)

    @property
    def nD(self):
        """
            # of data

        """

        nD = 0

        for src in self.source_list:
            for rx in src.receiver_list:
                nD += len(rx.frequencies)

        return int(nD)

    # @property
    # def hz_primary(self):
    #     # Assumes HCP only at the moment
    #     if self.src_type == 'VMD':
    #         return -1./(4*np.pi*self.offset**3)
    #     elif self.src_type == 'CircularLoop':
    #         return self.I/(2*self.a) * np.ones_like(self.frequency)
    #     else:
    #         raise NotImplementedError()


class EM1DSurveyTD(BaseEM1DSurvey):
    """docstring for EM1DSurveyTD"""

    

    def __init__(self, source_list=None, **kwargs):
        BaseEM1DSurvey.__init__(self, source_list, **kwargs)

        # Use Sin filter for frequency to time transform
        self.fftfilt = filters.key_81_CosSin_2009()


    @property
    def nD(self):
        """
            # of data

        """

        nD = 0

        for src in self.source_list:
            for rx in src.receiver_list:
                nD += len(rx.times)

        return int(nD)
    

    @property
    def lowpass_filter(self):
        """
            Low pass filter values
        """
        if getattr(self, '_lowpass_filter', None) is None:
            # self._lowpass_filter = butterworth_type_filter(
            #     self.frequency, self.high_cut_frequency
            # )

            self._lowpass_filter = (1+1j*(self.frequency/self.high_cut_frequency))**-1
            self._lowpass_filter *= (1+1j*(self.frequency/3e5))**-0.99
            # For actual butterworth filter

            # filter_frequency, values = butter_lowpass_filter(
            #     self.high_cut_frequency
            # )
            # lowpass_func = interp1d(
            #     filter_frequency, values, fill_value='extrapolate'
            # )
            # self._lowpass_filter = lowpass_func(self.frequency)

        return self._lowpass_filter
