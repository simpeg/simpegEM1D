from SimPEG import Problem, Utils, Maps
from mpi4py import MPI
from SimPEG import Mesh
from SimPEG import Props
import numpy as np
from multiprocessing import Pool

def run_simulation(
    rxLoc, SrcLoc, mesh_1d, offset, frequency,
    field_type = 'secondary',
    rxType = 'Hz',
    srcType = 'VMD'
):
    FDsurvey = EM1DSurveyFD()
    depth = -mesh1D.gridN[:-1]
    LocSigZ = -mesh1D.gridCC
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
    return resp, drespdsig

from mpi4py import MPI
class GlobalProblem(Problem.BaseProblem):
    """
        The GlobalProblem allows you to run a whole bunch of SubProblems,
        potentially in parallel, potentially of different meshes.
        This is handy for working with lots of sources,
    """
    sigma, sigmaMap, sigmaDeriv = Props.Invertible(
        "Electrical conductivity (S/m)"
    )

    def __init__(self, mesh,**kwargs):
        self.comm = MPI.COMM_WORLD
        if self.comm.Get_rank()==0:
            Utils.setKwargs(self, **kwargs)
            assert isinstance(mesh, Mesh.BaseMesh), "mesh must be a SimPEG.Mesh object."
            self.mesh = mesh  
            
        mesh_1d = self.set_mesh_1d()
        print (mesh_1d)
            
    def set_mesh_1d(self):        
        cs = 10.
        nearthick = np.logspace(-1, 1, 3)
        linthick = np.ones(15)*cs
        deepthick = np.logspace(1, 2, 3)
        hx = np.r_[nearthick, linthick, deepthick, deepthick[-1]]
        return Mesh.TensorMesh([hx], [0.])
        
if __name__ == '__main__':
    mesh = Mesh.TensorMesh([10, 10])    
    prob = GlobalProblem(mesh, sigmaMap=Maps.IdentityMap(mesh))  