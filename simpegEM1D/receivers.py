import numpy as np
import properties
from SimPEG.survey import BaseRx, BaseTimeRx
        

class HarmonicPointReceiver(BaseRx):

    locations = properties.Array(
        "Receiver Location", dtype=float, shape=("*",), required=True
    )

    frequencies = properties.Array(
        "Frequency (Hz)", dtype=float, shape=("*",), required=True
    )
    
    orientation = properties.StringChoice(
        "Field orientation", default="z", choices=["x", "y", "z"]
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

    ftarg = None

    locations = properties.Array(
        "Receiver Location", dtype=float, shape=("*",), required=True
    )

    orientation = properties.StringChoice(
        "Field orientation", default="z", choices=["z"]
    )

    component = properties.StringChoice(
        "component of the field (h, b, dhdt, dbdt)",
        default="dbdt",
        choices=["h", "b", "dhdt", "dbdt"]
    )

    frequencies = properties.Array(
        "Frequency (Hz)", dtype=float, shape=("*",), required=True
    )

    time_interval = properties.Array(
        "Full time interval", dtype=float, shape=("*",)
    )



    def __init__(self, locations=None, times=None, orientation=None, component=None, **kwargs):

        super(TimeDomainPointReceiver, self).__init__(locations, times, **kwargs)

        if orientation is not None:
            self.orientation = orientation
        if component is not None:
            self.component = component

        # Required static property
        self.field_type = "secondary"



    @property
    def n_time(self):
        return int(self.times.size)

    @property
    def n_frequency(self):
        """
            # of frequency
        """

        return int(self.frequencies.size)


