import numpy as np
from SimPEG import survey
import properties

class BaseSrc(survey.BaseSrc):

    _offset_list = properties.List("List containing offsets") # Contains the list of xyz offsets for each source-receiver pair

    I = properties.Float("Source loop current", default=1.)

    def __init__(self, receiver_list=None, **kwargs):
        super(BaseSrc, self).__init__(receiver_list=receiver_list, **kwargs)

    @property
    def offset_list(self):
        
        if self._offset_list is not None:
            return self._offset_list

        else:
            if self.receiver_list is not None:
                temp = len(self.receiver_list)*[None]
                src_loc = np.reshape(self.location, (1, 3))
                for ii, rx in enumerate(self.receiver_list):
                    temp[ii] = rx.locations - np.repeat(src_loc, rx.nD, axis=0)

                self._offset_list = temp
                return self._offset_list

            else:
                return
    



#############################################################################
# Harmonic Sources

class HarmonicMagneticDipoleSource(BaseSrc):
    
    orientation = properties.StringChoice(
        "Dipole Orientation", default="z", choices=["z"]
    )

    def __init__(self, receiver_list=None, **kwargs):
        super(HarmonicMagneticDipoleSource, self).__init__(receiver_list=receiver_list, **kwargs)

class HarmonicHorizontalLoopSource(BaseSrc):
    
    a = properties.Float("Source loop radius", default=1.)

    def __init__(self, receiver_list=None, **kwargs):
        super(HarmonicHorizontalLoopSource, self).__init__(receiver_list=receiver_list, **kwargs)


class HarmonicLineSource(BaseSrc):
    
    src_path = properties.Array(
        "Source path (xi, yi, zi), i=0,...N",
        dtype=float
    )

    def __init__(self, receiver_list=None, **kwargs):
        super(HarmonicLineSource, self).__init__(receiver_list=receiver_list, **kwargs)



#############################################################################
# Time Sources



class BaseTimeSrc(BaseSrc):

    wave_type = properties.StringChoice(
        "Waveform",
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
        "High cut frequency for low pass filter (Hz)",
        default=210*1e3
    )


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


    def __init__(self, receiver_list=None, **kwargs):
        super(BaseTimeSrc, self).__init__(receiver_list=receiver_list, **kwargs)







    

    @property
    def period(self):
        return 1./self.base_frequency

    @property
    def pulse_period(self):
        Tp = (
            self.time_input_currents.max() -
            self.time_input_currents.min()
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
            self.time_input_currents_dual_moment.min()
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


    
    


class TimeDomainMagneticDipoleSource(BaseTimeSrc):

    orientation = properties.StringChoice(
        "Dipole Orientation", default="z", choices=["z"]
    )

    def __init__(self, receiver_list=None, **kwargs):
        super(TimeDomainMagneticDipoleSource, self).__init__(receiver_list=receiver_list, **kwargs)


class TimeDomainHorizontalLoopSource(BaseTimeSrc):

    a = properties.Float("Source loop radius", default=1.)

    def __init__(self, receiver_list=None, **kwargs):
        super(TimeDomainHorizontalLoopSource, self).__init__(receiver_list=receiver_list, **kwargs)


class TimeDomainLineSource(BaseTimeSrc):
    
    src_path = properties.Array(
        "Source path (xi, yi, zi), i=0,...N",
        dtype=float
    )

    def __init__(self, receiver_list=None, **kwargs):
        super(TimeDomainLineSource, self).__init__(receiver_list=receiver_list, **kwargs)
    
    

