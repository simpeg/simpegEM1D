import numpy as np
from SimPEG import survey
import properties
from scipy import special as spec




###############################################################################
#                                                                             #
#                                    Sources                                  #
#                                                                             #
###############################################################################

class BaseSrc(survey.BaseSrc):
    """
        Base source class for EM1D

    """

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
#############################################################################


class BaseHarmonicSrc(BaseSrc):

    def __init__(self, receiver_list=None, **kwargs):
        super(BaseHarmonicSrc, self).__init__(receiver_list=receiver_list, **kwargs)


class HarmonicMagneticDipoleSource(BaseSrc):
    
    orientation = properties.StringChoice(
        "Dipole Orientation", default="z", choices=["x", "y", "z"]
    )

    def __init__(self, receiver_list=None, **kwargs):
        super(HarmonicMagneticDipoleSource, self).__init__(receiver_list=receiver_list, **kwargs)


    def PrimaryField(self, xyz, is_offset=False):

        # xyz is np.array(N, 3) with receiver locations
        # is_offet defines whether rx positions or absolute of source-receiver offset

        I = self.I
        if is_offset:
            r0 = np.zeros(3)
        else:
            r0 = self.location

        if self.orientation == "x":
            m = np.r_[1., 0., 0.]
        elif self.orientation == "y":
            m = np.r_[0., 1., 0.]
        elif self.orientation == "z":
            m = np.r_[0., 0., 1.]

        r = np.sqrt((xyz[0]-r0[0])**2 + (xyz[1]-r0[1])**2 + (xyz[2]-r0[2])**2)
        mdotr = m[0]*(xyz[0]-r0[0]) + m[1]*(xyz[1]-r0[1]) + m[2]*(xyz[2]-r0[2])

        hx0 = (1/(4*np.pi))*(3*(xyz[0]-r0[0])*mdotr/r**5 - m[0]/r**3)
        hy0 = (1/(4*np.pi))*(3*(xyz[1]-r0[1])*mdotr/r**5 - m[1]/r**3)
        hz0 = (1/(4*np.pi))*(3*(xyz[2]-r0[2])*mdotr/r**5 - m[2]/r**3)

        return I*np.c_[hx0, hy0, hz0]


class HarmonicHorizontalLoopSource(BaseSrc):
    
    a = properties.Float("Source loop radius", default=1.)

    def __init__(self, receiver_list=None, **kwargs):
        super(HarmonicHorizontalLoopSource, self).__init__(receiver_list=receiver_list, **kwargs)


    def PrimaryField(self, is_offset=False):

        # xyz is np.array(N, 3) with receiver locations
        # is_offet defines whether rx positions or absolute of source-receiver offset

        a = self.a
        I = self.I
        if is_offset:
            r0 = np.zeros(3)
        else:
            r0 = self.location

        theta = 0.  # Azimuthal
        alpha = 0.  # Declination

        # Rotate x,y,z into coordinate axis of transmitter loop
        rot_x = np.r_[
            np.c_[1, 0, 0],
            np.c_[0, np.cos(np.pi*theta/180), -np.sin(np.pi*theta/180)],
            np.c_[0, np.sin(np.pi*theta/180), np.cos(np.pi*theta/180)]
        ]     # CCW ROTATION OF THETA AROUND X-AXIS

        rot_z = np.r_[
            np.c_[np.cos(np.pi*alpha/180), -np.sin(np.pi*alpha/180), 0],
            np.c_[np.sin(np.pi*alpha/180), np.cos(np.pi*alpha/180), 0],
            np.c_[0, 0, 1]
        ]     # CCW ROTATION OF (90-ALPHA) ABOUT Z-AXIS

        rot_mat = np.dot(rot_x, rot_z)            # THE ORDER MATTERS

        x1p = np.dot(np.c_[xyz[0]-r0[0], xyz[1]-r0[1], xyz[2]-r0[2]], rot_mat[0, :].T)
        x2p = np.dot(np.c_[xyz[0]-r0[0], xyz[1]-r0[1], xyz[2]-r0[2]], rot_mat[1, :].T)
        x3p = np.dot(np.c_[xyz[0]-r0[0], xyz[1]-r0[1], xyz[2]-r0[2]], rot_mat[2, :].T)

        s = np.sqrt(x1p**2 + x2p**2) + 1e-10     # Radial distance
        k = 4*a*s/(x3p**2 + (a+s)**2)

        hxp = (x1p/s)*(x3p*I/(2*np.pi*s*np.sqrt(x3p**2 + (a + s)**2)))*(((a**2 + x3p**2 + s**2)/(x3p**2 + (s-a)**2))*spec.ellipe(k) - spec.ellipk(k))
        hyp = (x2p/s)*(x3p*I/(2*np.pi*s*np.sqrt(x3p**2 + (a + s)**2)))*(((a**2 + x3p**2 + s**2)/(x3p**2 + (s-a)**2))*spec.ellipe(k) - spec.ellipk(k))
        hzp =         (    I/(2*np.pi*  np.sqrt(x3p**2 + (a + s)**2)))*(((a**2 - x3p**2 - s**2)/(x3p**2 + (s-a)**2))*spec.ellipe(k) + spec.ellipk(k))

        # Rotate the other way to get back into original coordinates
        rot_mat_t = rot_mat.T
        hx0 = np.dot(np.c_[hxp, hyp, hzp], rot_mat_t[0, :].T)
        hy0 = np.dot(np.c_[hxp, hyp, hzp], rot_mat_t[1, :].T)
        hz0 = np.dot(np.c_[hxp, hyp, hzp], rot_mat_t[2, :].T)

        return np.c_[hx0, hy0, hz0]


class HarmonicLineSource(BaseSrc):
    
    node_locations = properties.Array(
        "Source path (xi, yi, zi), i=0,...N",
        dtype=float
    )

    def PrimaryField(self, xyz):

        # This could be more efficient given online on rx loc right now

        I = self.I
        tx_nodes = self.node_locations
        x1tr = tx_nodes[:, 0]
        x2tr = tx_nodes[:, 1]
        x3tr = tx_nodes[:, 2]

        nLoc = np.shape(xyz)[0]
        nSeg = np.size(x1tr)-1

        hx0 = np.zeros(nLoc)
        hy0 = np.zeros(nLoc)
        hz0 = np.zeros(nLoc)

        for pp in range(0, nSeg):

            # Wire ends for transmitter wire pp
            x1a = x1tr[pp]
            x2a = x2tr[pp]
            x3a = x3tr[pp]
            x1b = x1tr[pp+1]
            x2b = x2tr[pp+1]
            x3b = x3tr[pp+1]

            # Vector Lengths between points
            vab = np.sqrt((x1b - x1a)**2 + (x2b - x2a)**2 + (x3b - x3a)**2)
            vap = np.sqrt((xyz[:, 0] - x1a)**2 + (xyz[:, 1] - x2a)**2 + (xyz[:, 2] - x3a)**2)
            vbp = np.sqrt((xyz[:, 0] - x1b)**2 + (xyz[:, 1] - x2b)**2 + (xyz[:, 2] - x3b)**2)

            # Cosines from cos()=<v1,v2>/(|v1||v2|)
            cos_alpha = ((xyz[:, 0]-x1a)*(x1b - x1a) + (xyz[:, 1]-x2a)*(x2b - x2a) + (xyz[:, 2]-x3a)*(x3b - x3a))/(vap*vab)
            cos_beta  = ((xyz[:, 0]-x1b)*(x1a - x1b) + (xyz[:, 1]-x2b)*(x2a - x2b) + (xyz[:, 2]-x3b)*(x3a - x3b))/(vbp*vab)

            # Determining Radial Vector From Wire
            dot_temp = (
                (x1a - xyz[:, 0])*(x1b - x1a) +
                (x2a - xyz[:, 1])*(x2b - x2a) +
                (x3a - xyz[:, 2])*(x3b - x3a)
            )

            rx1 = (x1a - xyz[:, 0]) - dot_temp*(x1b - x1a)/vab**2
            rx2 = (x2a - xyz[:, 1]) - dot_temp*(x2b - x2a)/vab**2
            rx3 = (x3a - xyz[:, 2]) - dot_temp*(x3b - x3a)/vab**2

            r = np.sqrt(rx1**2 + rx2**2 + rx3**2)

            phi = (cos_alpha + cos_beta)/r

            # I/4*pi in each direction
            ix1 = I*(x1b - x1a)/(4*np.pi*vab)
            ix2 = I*(x2b - x2a)/(4*np.pi*vab)
            ix3 = I*(x3b - x3a)/(4*np.pi*vab)

            # Add contribution from wire pp into array
            hx0 = hx0 + phi*(-ix2*rx3 + ix3*rx2)/r
            hy0 = hy0 + phi*( ix1*rx3 - ix3*rx1)/r
            hz0 = hz0 + phi*(-ix1*rx2 + ix2*rx1)/r

        return np.c_[hx0, hy0, hz0]


    def __init__(self, receiver_list=None, **kwargs):
        super(HarmonicLineSource, self).__init__(receiver_list=receiver_list, **kwargs)


#############################################################################
# Time Sources
#############################################################################


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
    

