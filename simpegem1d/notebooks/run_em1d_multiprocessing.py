from SimPEG import *
from simpegem1d import *
from scipy.constants import mu_0
import numpy as np
# from multiprocessing import Pool   
from schwimmbad import MPIPool as Pool
import time 

def set_mesh_1d():        
    cs = 10.
    nearthick = np.logspace(-1, 1, 3)
    linthick = np.ones(15)*cs
    deepthick = np.logspace(1, 2, 3)
    hx = np.r_[nearthick, linthick, deepthick, deepthick[-1]]
    return Mesh.TensorMesh([hx], [0.])

def run_simulation(tag):
    """
        rxLoc, SrcLoc, mesh_1d, offset, frequency,
        field_type = 'secondary',
        rxType = 'Hz',
        srcType = 'VMD'

    """    
    mesh_1d = set_mesh_1d(jacSwitch)
    # Todo: require input args
#     rxLoc, SrcLoc, mesh_1d, offset, frequency, field_type, rxType, srcType = args            
    FDsurvey = EM1DSurveyFD()
    FDsurvey.rxLoc = np.array([0., 0., 100.+30.])
    FDsurvey.srcLoc = np.array([0., 0., 100.+30.])
    FDsurvey.fieldtype = 'secondary'
    FDsurvey.rxType = 'Hz'
    FDsurvey.srcType = 'VMD'
    FDsurvey.offset = np.r_[8., 8., 8.]        
    depth = -mesh_1d.gridN[:-1]
    LocSigZ = -mesh_1d.gridCC
    nlay = depth.size
    topo = np.r_[0., 0., 100.]
    FDsurvey.depth = depth
    FDsurvey.topo = topo
    FDsurvey.LocSigZ = LocSigZ
    FDsurvey.Nfreq = FDsurvey.frequency.size
    FDsurvey.Setup1Dsystem()
    FDsurvey.SetOffset()
    sig_half = 1e-4
    sig_blk = 1e-2
    chi_half = 0.
    expmap = Maps.ExpMap(mesh_1d)
    sig  = np.ones(nlay)*sig_half
    blk_ind = (-50>LocSigZ) & (-100<LocSigZ)
    sig[blk_ind] = sig_blk
    m_true = np.log(sig)

    WT0, WT1, YBASE = DigFilter.LoadWeights()
    prob = EM1D(
        mesh_1d, sigmaMap=expmap, filter_type='key_101',
        jacSwitch=jacSwitch,
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
    return resp, drespdsig

class GlobalProblem(Problem.BaseProblem):
    """
        The GlobalProblem allows you to run a whole bunch of SubProblems,
        potentially in parallel, potentially of different meshes.
        This is handy for working with lots of sources,
    """
    sigma, sigmaMap, sigmaDeriv = Props.Invertible(
        "Electrical conductivity (S/m)"
    )    
    
    _Jmatrix = None
    n_sounding = 10
    
    def __init__(self, mesh,**kwargs):
        if self.n_cpu is None:
            self.n_cpu = multiprocessing.cpu

    def forward(self, m):        
        pool = Pool(self.n_cpu)
        result = pool.map(run_simulation, [True for in range(self.n_sounding)])
        return result
        
    def Jvec(self, m, v)
        pass

    def Jvec(self, m, v)
        pass    