import numpy as np
import properties
from SimPEG.survey import BaseRx, BaseTimeRx
        

class HarmonicPointReceiver(BaseRx):

    frequencies = properties.Array("Frequency (Hz)", dtype=float)
    
    orientation = properties.StringChoice(
        "Field orientation", default="z", choices=["z"]
    )

    field_type = properties.StringChoice(
        "Data type",
        default="secondary",
        choices=["total", "secondary", "ppm"]
    )

    component = properties.StringChoice(
        "component of the field (real or imag)", {
            "real": ["re", "in-phase", "in phase"],
            "imag": ["imaginary", "im", "out-of-phase", "out of phase"]
        }
    )

    def __init__(self, locations=None, frequencies=None, orientation=None, field_type=None, component=None, **kwargs):

        super(HarmonicPointReceiver, self).__init__(locations, **kwargs)
        if frequencies is not None:
            self.frequencies = frequencies
        if orientation is not None:
            self.orientation = orientation
        if component is not None:
            self.component = component
        if field_type is not None:
            self.field_type = field_type

        

class TimeDomainPointReceiver(BaseTimeRx):

    orientation = properties.StringChoice(
        "Field orientation", default="z", choices=["z"]
    )

    component = properties.StringChoice(
        "component of the field (h, b, dhdt, dbdt)",
        default="dbdt",
        choices=["h", "b", "dhdt", "dbdt"]
    )

    def __init__(self, locations=None, times=None, orientation=None, component=None, **kwargs):

        super(TDEMPointReceiver, self).__init__(locations, times, **kwargs)

        if orientation is not None:
            self.orientation = orientation
        if component is not None:
            self.component = component




