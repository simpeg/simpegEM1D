from SimPEG import *
from simpegEM1D import (
    EM1D, EM1DSurveyTD, Utils1D, get_vertical_discretization_time, 
    set_mesh_1d, EM1DAnalytics
)
import numpy as np
from simpegEM1D import skytem_HM_2015
wave = skytem_HM_2015()
time = np.logspace(-6, -2, 21)
hz = get_vertical_discretization_time(time, facter_tmax=0.5, factor_tmin=10.)
mesh1D = set_mesh_1d(hz)
depth = -mesh1D.gridN[:-1]
LocSigZ = -mesh1D.gridCC

# time_input_currents = np.r_[0., 5.5*1e-4, 5.5*1e-4+1e-5]
# input_currents = np.r_[0., 1., 0.]
time_input_currents = wave.current_times[-7:]
input_currents = wave.currents[-7:]

TDsurvey = EM1DSurveyTD(
    rx_location = np.array([0., 0., 100.+30.]),
    src_location = np.array([0., 0., 100.+30.]),
    topo = np.r_[0., 0., 100.],
    depth = depth,
    rx_type = 'dBzdt',
    wave_type = 'general',
    src_type = 'CircularLoop',
    a = 13.,
    time = time,
    time_input_currents=time_input_currents,
    input_currents=input_currents,
    n_pulse = 2,
    base_frequency = 25.            
)

# TDsurvey = EM1DSurveyTD(
#     rx_location = np.array([0., 0., 100.+30.]),
#     src_location = np.array([0., 0., 100.+30.]),
#     topo = np.r_[0., 0., 100.],
#     depth = depth,
#     rx_type = 'dBzdt',
#     wave_type = 'stepoff',
#     src_type = 'CircularLoop',
#     a = 13.,
#     I = 1.,
#     time = time
# )

sig_half = 1e-2
sig_blk = 1e-1
chi_half = 0.
expmap = Maps.ExpMap(mesh1D)
sig  = np.ones(TDsurvey.n_layer)*sig_half
blk_ind = (0>LocSigZ) & (-30<LocSigZ)
sig[blk_ind] = sig_blk
m_true = np.log(sig)
prob = EM1D(mesh1D, sigmaMap=expmap, verbose=True)
if prob.ispaired:
    prob.unpair()
if TDsurvey.ispaired:
    TDsurvey.unpair()
prob.pair(TDsurvey)
prob.chi = np.zeros(TDsurvey.n_layer)
d_true = TDsurvey.dpred(m_true)
TDsurvey.dtrue = d_true
std = 0.1
noise = std*abs(TDsurvey.dtrue)*np.random.randn(*TDsurvey.dtrue.shape)
floor = 0.
std = 0.15
TDsurvey.dobs = TDsurvey.dtrue+noise
uncert = abs(TDsurvey.dobs)*std+floor

dmisfit = DataMisfit.l2_DataMisfit(TDsurvey)
dmisfit.W = 1./(abs(TDsurvey.dobs)*std+floor)
m0 = np.log(np.ones_like(sig)*sig_half)
reg = Regularization.Simple(
    mesh1D
)
opt = Optimization.InexactGaussNewton(maxIter = 3)
opt.maxIterLS = 5
invProb = InvProblem.BaseInvProblem(dmisfit, reg, opt)
beta = Directives.BetaSchedule(coolingFactor=2., coolingRate=1)
betaest = Directives.BetaEstimate_ByEig(beta0_ratio=1e0)
target = Directives.TargetMisfit()
inv = Inversion.BaseInversion(invProb, directiveList=[beta, betaest, target])
prob.counter = opt.counter = Utils.Counter()
opt.LSshorten = 0.5
opt.remember('xc')
inv.run(m0)