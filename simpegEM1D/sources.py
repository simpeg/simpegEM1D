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


class TimeDomainMagneticDipoleSource(BaseSrc):

    orientation = properties.StringChoice(
        "Dipole Orientation", default="z", choices=["z"]
    )
    
    wave_type = properties.StringChoice(
        "Source location",
        default="stepoff",
        choices=["stepoff", "general"]
    )

    def __init__(self, receiver_list=None, **kwargs):
        super(TimeDomainMagneticDipoleSource, self).__init__(receiver_list=receiver_list, **kwargs)


class TimeDomainHorizontalLoopSource(BaseSrc):

    a = properties.Float("Source loop radius", default=1.)
    
    wave_type = properties.StringChoice(
        "Source location",
        default="stepoff",
        choices=["stepoff", "general"]
    )

    def __init__(self, receiver_list=None, **kwargs):
        super(TimeDomainHorizontalLoopSource, self).__init__(receiver_list=receiver_list, **kwargs)


class TimeDomainLineSource(BaseSrc):
    
    src_path = properties.Array(
        "Source path (xi, yi, zi), i=0,...N",
        dtype=float
    )
    
    wave_type = properties.StringChoice(
        "Source location",
        default="stepoff",
        choices=["stepoff", "general"]
    )

    def __init__(self, receiver_list=None, **kwargs):
        super(TimeDomainLineSource, self).__init__(receiver_list=receiver_list, **kwargs)
    
    

