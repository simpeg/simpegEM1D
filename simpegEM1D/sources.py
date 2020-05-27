import numpy as np
from SimPEG.survey import BaseSrc
import properties

class HarmonicMagneticDipoleSource(BaseSrc):
    
    frequency = properties.Array("Frequency (Hz)", dtype=float)
    
    I = properties.Float("Source loop current", default=1.)
    
    orientation = properties.StringChoice(
        "Dipole Orientation", default="Z", choices=["Z"]
    )

    def __init__(self, receiver_list=None, frequency=None, **kwargs):
        super(HarmonicMagneticDipoleSource, self).__init__(receiver_list=receiver_list, **kwargs)
        if frequency is not None:
            self.frequency = frequency

class HarmonicHorizontalLoopSource(BaseSrc):

    frequency = properties.Array("Frequency (Hz)", dtype=float)
    
    I = properties.Float("Source loop current", default=1.)
    
    a = properties.Float("Source loop radius", default=1.)

    def __init__(self, receiver_list=None, frequency=None, **kwargs):
        super(HarmonicHorizontalLoopSource, self).__init__(receiver_list=receiver_list, **kwargs)
        if frequency is not None:
            self.frequency = frequency


class HarmonicLineSource(BaseSrc):

    frequency = properties.Array("Frequency (Hz)", dtype=float)
    
    I = properties.Float("Source loop current", default=1.)
    
    src_path = properties.Array(
        "Source path (xi, yi, zi), i=0,...N",
        dtype=float
    )

    def __init__(self, receiver_list=None, frequency=None, **kwargs):
        super(HarmonicLineSource, self).__init__(receiver_list=receiver_list, **kwargs)
        if frequency is not None:
            self.frequency = frequency


class TimeDomainMagneticDipoleSource(BaseSrc):

    time = properties.Array("Time channels (s) at current off-time", dtype=float)
    
    I = properties.Float("Source loop current", default=1.)
    
    orientation = properties.StringChoice(
        "Dipole Orientation", default="z", choices=["z"]
    )
    
    wave_type = properties.StringChoice(
        "Source location",
        default="stepoff",
        choices=["stepoff", "general"]
    )

    def __init__(self, receiver_list=None, time=None, **kwargs):
        super(TimeDomainMagneticDipoleSource, self).__init__(receiver_list=receiver_list, **kwargs)
        if time is not None:
            self.time = time


class TimeDomainHorizontalLoopSource(BaseSrc):

    time = properties.Array("Time channels (s) at current off-time", dtype=float)
    
    I = properties.Float("Source loop current", default=1.)
    
    a = properties.Float("Source loop radius", default=1.)
    
    wave_type = properties.StringChoice(
        "Source location",
        default="stepoff",
        choices=["stepoff", "general"]
    )

    def __init__(self, receiver_list=None, time=None, **kwargs):
        super(TimeDomainHorizontalLoopSource, self).__init__(receiver_list=receiver_list, **kwargs)
        if time is not None:
            self.time = time

class TimeDomainLineSource(BaseSrc):

    time = properties.Array("Time channels (s) at current off-time", dtype=float)
    
    I = properties.Float("Source loop current", default=1.)
    
    src_path = properties.Array(
        "Source path (xi, yi, zi), i=0,...N",
        dtype=float
    )
    
    wave_type = properties.StringChoice(
        "Source location",
        default="stepoff",
        choices=["stepoff", "general"]
    )

    def __init__(self, receiver_list=None, time=None, **kwargs):
        super(TimeDomainLineSource, self).__init__(receiver_list=receiver_list, **kwargs)
        if time is not None:
            self.time = time
    
    

