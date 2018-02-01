from mpi4py import MPI
from SimPEG import *
from simpegem1d import *
from scipy.constants import mu_0
import numpy as np
    
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

def run_simulation():
    FDsurvey = EM1DSurveyFD()
    FDsurvey.rxLoc = np.array([0., 0., 100.+30.])
    FDsurvey.srcLoc = np.array([0., 0., 100.+30.])
    FDsurvey.fieldtype = 'secondary'
    FDsurvey.rxType = 'Hz'
    FDsurvey.srcType = 'VMD'
    FDsurvey.offset = np.r_[8., 8., 8.]
    cs = 10.
    nearthick = np.logspace(-1, 1, 3)
    linthick = np.ones(15)*cs
    deepthick = np.logspace(1, 2, 3)
    hx = np.r_[nearthick, linthick, deepthick, deepthick[-1]]
    mesh1D = Mesh.TensorMesh([hx], [0.])
    depth = -mesh1D.gridN[:-1]
    LocSigZ = -mesh1D.gridCC
    nlay = depth.size
    topo = np.r_[0., 0., 100.]
    FDsurvey.depth = depth
    FDsurvey.topo = topo
    FDsurvey.LocSigZ = LocSigZ
    # FDsurvey.frequency = np.logspace(3, 5, 11)
    FDsurvey.frequency = np.r_[900., 7200., 56000]
    FDsurvey.Nfreq = FDsurvey.frequency.size
    FDsurvey.Setup1Dsystem()
    FDsurvey.SetOffset()
    sig_half = 1e-4
    sig_blk = 1e-2
    chi_half = 0.
    expmap = Maps.ExpMap(mesh1D)
    sig  = np.ones(nlay)*sig_half
    blk_ind = (-50>LocSigZ) & (-100<LocSigZ)
    sig[blk_ind] = sig_blk
    m_true = np.log(sig)

    WT0, WT1, YBASE = DigFilter.LoadWeights()
    prob = EM1D(
        mesh1D, sigmaMap=expmap, filter_type='key_101',
        jacSwitch=True,
        chi= np.zeros(FDsurvey.nlay)
    )
    if prob.ispaired:
        prob.unpair()
    if FDsurvey.ispaired:
        FDsurvey.unpair()
    prob.pair(FDsurvey)    
    u, dudsig = prob.fields(m_true)
    resp = FDsurvey.projectFields(u)
    drespdsig = FDsurvey.projectFields(dudsig)
    
#     FDsurvey.dtrue = d_true
#     std = 0.05
#     floor = 1e-16
#     np.random.seed(1)
#     uncert = std*abs(FDsurvey.dtrue)+floor
#     noise = std*FDsurvey.dtrue*np.random.randn(FDsurvey.dtrue.size)
#     FDsurvey.dobs = FDsurvey.dtrue+noise
#     dmisfit = DataMisfit.l2_DataMisfit(FDsurvey)
#     dmisfit.W = 1./(abs(FDsurvey.dobs)*std+floor)
#     m0 = np.log(np.ones_like(sig)*1e-3)
#     reg = Regularization.Tikhonov(mesh1D)
#     opt = Optimization.InexactGaussNewton(maxIter = 6)
#     opt.maxIterLS = 5
#     invProb = InvProblem.BaseInvProblem(dmisfit, reg, opt)
#     beta = Directives.BetaSchedule(coolingFactor=2, coolingRate=1)
#     betaest = Directives.BetaEstimate_ByEig(beta0_ratio=10**-1)
#     target = Directives.TargetMisfit()
#     inv = Inversion.BaseInversion(invProb, directiveList=[beta,betaest,target])
#     reg.alpha_s = 10.
#     reg.alpha_x = 1.
#     reg.alpha_y = 1.
#     reg.alpha_z = 1.

#     prob.counter = opt.counter = Utils.Counter()
#     opt.LSshorten = 0.5
#     opt.remember('xc')
#     mopt = inv.run(m0)
    return resp, drespdsig

n_sounding = 400
n_layer = 22
n_data = 6

def get_n_sounding_per_proc(n_sounding, size):    
    return n_sounding_per_proc

t_start = MPI.Wtime()
comm.Barrier()
dpred_local = np.empty((int(n_sounding/size), 6), dtype='float')
dpred_dsig_local = np.empty((int(n_sounding/size), 6, n_layer), dtype='float')
v_d = np.ones(n_data)
v_m = np.ones(n_layer)
Jv_local = np.empty((int(n_sounding/size), n_data), dtype='float')
Jtv_local = np.empty((int(n_sounding/size), n_layer), dtype='float')

for i in range(int(n_sounding/size)):
    dpred_local[i, :], dpred_dsig_local[i,:,:] = run_simulation()
    Jv_local[i, :] = np.dot(dpred_dsig_local[i,:,:], v_m)
    Jtv_local[i, :] = np.dot(dpred_dsig_local[i,:,:].T, v_d)
    
comm.Barrier()

dpred = None
Jv = None
Jtv = None
if rank == 0:
    dpred = np.empty([n_sounding, n_data], dtype='float')
    Jv = np.empty([int(n_sounding*n_data)], dtype='float')
    Jtv = np.empty([int(n_sounding*n_layer)], dtype='float')

comm.Gather(dpred_local, dpred, root=0)
comm.Gather(Jv_local, Jv, root=0)
comm.Gather(Jtv_local, Jtv, root=0)
t_end = MPI.Wtime()
if rank == 0:
    print ("Time %.1f ms" % ((t_end-t_start)*1e3))