import numpy as np
from discretize import TensorMesh
from SimPEG import maps, utils
from .analytics import skin_depth, diffusion_distance
from .simulation import EM1DFMSimulation, EM1DTMSimulation
from .survey import EM1DSurveyFD, EM1DSurveyTD


def get_vertical_discretization_frequency(
    frequency, sigma_background=0.01,
    factor_fmax=4, factor_fmin=1., n_layer=19,
    hz_min=None, z_max=None
):
    if hz_min is None:
        hz_min = skin_depth(frequency.max(), sigma_background) / factor_fmax
    if z_max is None:
        z_max = skin_depth(frequency.min(), sigma_background) * factor_fmin
    i = 4
    hz = np.logspace(np.log10(hz_min), np.log10(hz_min*i), n_layer)
    z_sum = hz.sum()

    while z_sum < z_max:
        i += 1
        hz = np.logspace(np.log10(hz_min), np.log10(hz_min*i), n_layer)
        z_sum = hz.sum()
    return hz


def get_vertical_discretization_time(
    time, sigma_background=0.01,
    factor_tmin=4, facter_tmax=1., n_layer=19,
    hz_min=None, z_max=None
):
    if hz_min is None:
        hz_min = diffusion_distance(time.min(), sigma_background) / factor_tmin
    if z_max is None:
        z_max = diffusion_distance(time.max(), sigma_background) * facter_tmax
    i = 4
    hz = np.logspace(np.log10(hz_min), np.log10(hz_min*i), n_layer)
    z_sum = hz.sum()
    while z_sum < z_max:
        i += 1
        hz = np.logspace(np.log10(hz_min), np.log10(hz_min*i), n_layer)
        z_sum = hz.sum()
    return hz


def set_mesh_1d(hz):
    return TensorMesh([hz], x0=[0])


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

    src, topo, hz, sigma, eta, tau, c, chi, h, jac_switch, invert_height, half_switch = args

    local_survey = EM1DSurveyFD([src])
    expmap = maps.ExpMap(nP=len(hz))
    thicknesses = hz[0:-1]

    if not invert_height:
        # Use Exponential Map
        # This is hard-wired at the moment
        
        sim = EM1DFMSimulation(
            survey=local_survey, thicknesses=thicknesses,
            sigmaMap=expmap, chi=chi, eta=eta, tau=tau, c=c,
            half_switch=half_switch, hankel_filter='key_101_2009'
        )
        
        if jac_switch == 'sensitivity_sigma':
            drespdsig = sim.getJ_sigma(np.log(sigma))
            return utils.mkvc(drespdsig * sim.sigmaDeriv)
            # return utils.mkvc(drespdsig)
        else:
            resp = sim.dpred(np.log(sigma))
            return resp
    else:
        
        mesh1D = set_mesh_1d(hz)
        wires = maps.Wires(('sigma', mesh1D.nC), ('h', 1))
        sigmaMap = expmap * wires.sigma
        
        sim = EM1DFMSimulation(
            survey=local_survey, thicknesses=thicknesses,
            sigmaMap=sigmaMap, Map=wires.h,
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

    src, topo, hz, sigma, eta, tau, c, chi, h, jac_switch, invert_height, half_switch = args

    local_survey = EM1DSurveyTD([src])
    expmap = maps.ExpMap(nP=len(hz))
    thicknesses = hz[0:-1]

    if not invert_height:
        # Use Exponential Map
        # This is hard-wired at the moment
        sim = EM1DTMSimulation(
            survey=local_survey, thicknesses=thicknesses,
            sigmaMap=expmap, chi=chi, eta=eta, tau=tau, c=c,
            half_switch=half_switch, hankel_filter='key_101_2009'
        )
        
        if jac_switch == 'sensitivity_sigma':
            drespdsig = sim.getJ_sigma(np.log(sigma))
            return utils.mkvc(drespdsig * sim.sigmaDeriv)
        else:
            resp = sim.dpred(np.log(sigma))
            return resp
    else:
        
        mesh1D = set_mesh_1d(hz)
        wires = maps.Wires(('sigma', mesh1D.nC), ('h', 1))
        sigmaMap = expmap * wires.sigma
        sim = EM1DTMSimulation(
            survey=local_survey, thicknesses=thicknesses,
            sigmaMap=sigmaMap, Map=wires.h,
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

