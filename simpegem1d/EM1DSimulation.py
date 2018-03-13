import numpy as np
from SimPEG import Mesh, Maps
from .EM1DAnalytics import skin_depth
from .EM1D import EM1D
from .BaseEM1D import EM1DSurveyFD

def get_vertical_discretization(
    frequency, sigma_background=0.01, factor_fmin=4, facter_fmax=1., n_layer=19,
    hz_min=None, z_max=None
):
    if hz_min is None:
        hz_min = skin_depth(frequency.max(), sigma_background) / factor_fmin
    if z_max is None:
        z_max = skin_depth(frequency.min(), sigma_background) * facter_fmax
    i = 4
    hz = np.logspace(np.log10(hz_min), np.log10(hz_min*i), n_layer)
    z_sum = hz.sum()
    while z_sum<z_max:
        i+=1
        hz = np.logspace(np.log10(hz_min), np.log10(hz_min*i), n_layer)
        z_sum = hz.sum()
    print (z_max)
    return hz

def set_mesh_1d(hz):
    return Mesh.TensorMesh([hz], x0=[0])

def run_simulation_FD(args):
    """
        args

        rx_location: Recevier location (x, y, z)
        src_location: Source location (x, y, z)
        topo: Topographic location (x, y, z)
        hz: Thickeness of the vertical layers
        offset: Source-Receiver offset
        frequency: Frequency (Hz)
        field_type:
        rx_type:
        src_type:
        sigma:
        jacSwitch :
    """

    rx_location, src_location, topo, hz, offset, frequency, field_type, rx_type, src_type, sigma, jacSwitch = args
    mesh_1d = set_mesh_1d(hz)
    depth = -mesh_1d.gridN[:-1]
    FDsurvey = EM1DSurveyFD(
        rx_location = rx_location,
        src_location = src_location,
        topo = topo,
        frequency = frequency,
        offset = offset,
        field_type = field_type,
        rx_type = rx_type,
        src_type = src_type,
        depth = depth
    )
    # Use Exponential Map
    # This is hard-wired at the moment
    expmap  = Maps.ExpMap(mesh_1d)
    prob = EM1D(
        mesh_1d, sigmaMap=expmap, filter_type='key_101',
        jacSwitch=jacSwitch
    )
    if prob.ispaired:
        prob.unpair()
    if FDsurvey.ispaired:
        FDsurvey.unpair()
    prob.pair(FDsurvey)
    if jacSwitch:
        u, dudsig = prob.fields(np.log(sigma))
        drespdsig = FDsurvey.projectFields(dudsig)
        return drespdsig * prob.sigmaDeriv
    else:
        resp = FDsurvey.dpred(np.log(sigma))
        return resp
