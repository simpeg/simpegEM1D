from SimPEG import DC
from SimPEG import Mesh, Maps
from mpi4py import MPI
from scipy.constants import mu_0
import numpy as np
    
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()


def fields(m):
    mesh = Mesh.TensorMesh([10, 10, 10])
    prob = DC.Problem3D_N(mesh, sigmaMap=Maps.IdentityMap(mesh))
    rx = DC.Rx.Dipole(np.r_[0., 0., 0.],  np.r_[0.5, 0., 0.])
    src = DC.Src.Dipole([rx], np.r_[0.1, 0., 0.],  np.r_[0.2, 0., 0.])
    survey = DC.Survey([src])
    survey.pair(prob)
    f = prob.fields(m)
    return f

m = np.ones(1000)
f = fields(m)
if rank == 0:
    print (f)