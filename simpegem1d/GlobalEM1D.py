try:
    from multiprocessing import Pool
except ImportError:
    print ("multiprocessing is not available")
    PARALLEL = False
else:
    PARALLEL = True
    import multiprocessing

import numpy as np
import scipy.sparse as sp
from SimPEG import Problem, Props, Utils, Maps
from .BaseEM1D import EM1DSurveyFD
from .EM1DSimulation import run_simulation_FD, get_vertical_discretization
import properties

def dot(args):
    return np.dot(args[0], args[1])

class GlobalEM1DProblem(Problem.BaseProblem):
    """
        The GlobalProblem allows you to run a whole bunch of SubProblems,
        potentially in parallel, potentially of different meshes.
        This is handy for working with lots of sources,
    """
    sigma, sigmaMap, sigmaDeriv = Props.Invertible(
        "Electrical conductivity (S/m)"
    )

    _Jmatrix = None
    n_cpu = None
    hz = None
    parallel = False
    verbose = False

    def __init__(self, mesh, **kwargs):
        Utils.setKwargs(self, **kwargs)
        self.mesh = mesh
        if PARALLEL:
            if self.parallel:
                print (">> Use multiprocessing for parallelization")
                if self.n_cpu is None:
                    self.n_cpu = multiprocessing.cpu_count()
                print ((">> n_cpu: %i")%(self.n_cpu))
            else:
                print (">> Serial version is used")
        else:
            print (">> Serial version is used")
        if self.hz is None:
            raise Exception("Input vertical thickness hz !")

    @property
    def n_layer(self):
        return self.hz.size

    @property
    def n_sounding(self):
        return self.survey.n_rx

    @property
    def rx_locations(self):
        return self.survey.rx_locations

    @property
    def src_locations(self):
        return self.survey.src_locations

    @property
    def topo(self):
        return self.survey.topo

    @property
    def offset(self):
        return self.survey.offset

    @property
    def frequency(self):
        return self.survey.frequency

    @property
    def field_type(self):
        return self.survey.field_type

    @property
    def rx_type(self):
        return self.survey.rx_type

    @property
    def src_type(self):
        return self.survey.src_type

    @property
    def half_switch(self):
        return self.survey.half_switch

    @property
    def switch_real_imag(self):
        return self.survey.switch_real_imag

    def output_args(self, i_sounding, jacSwitch=False):
        output = (
            self.rx_locations[i_sounding,:], self.src_locations[i_sounding,:],
            self.topo[i_sounding,:], self.hz,
            self.offset, self.frequency,
            self.field_type, self.rx_type, self.src_type,
            self.Sigma[i_sounding, :], jacSwitch
        )
        return output
    @property
    def Sigma(self):
        if getattr(self, '_Sigma', None) is None:
            # Ordering: first z then x
            self._Sigma = self.sigma.reshape((self.n_sounding, self.n_layer))
        return self._Sigma

    def forward(self, m, f=None):
        self.model = m

        if PARALLEL:
            pool = Pool(self.n_cpu)
            # This assumes the same # of layer for each of soundings
            result = pool.map(
                run_simulation_FD,
                [
                    self.output_args(i, jacSwitch=False) for i in range(self.n_sounding)
                ]
            )
            pool.close()
            pool.join()
        else:
            result = [
                run_simulation_FD(self.output_args(i, jacSwitch=False)) for i in range(self.n_sounding)
            ]
        return np.hstack(result)

    def getJ(self, m):
        """
             Compute d F / d sigma
        """
        if self._Jmatrix is not None:
            return self._Jmatrix
        if self.verbose:
            print (">> Compute J")
        self.model = m
        if PARALLEL:
            pool = Pool(self.n_cpu)
            self._Jmatrix = pool.map(
                run_simulation_FD,
                [
                    self.output_args(i, jacSwitch=True) for i in range(self.n_sounding)
                ]
            )
            pool.close()
            pool.join()

        else:
            # _Jmatrix is block diagnoal matrix (sparse)
            self._Jmatrix = sp.block_diag(
                [
                    run_simulation_FD(self.output_args(i, jacSwitch=True)) for i in range(self.n_sounding)
                ]
            ).tocsr()
        return self._Jmatrix

    def Jvec(self, m, v, f=None):
        J = self.getJ(m)
        if PARALLEL:
            V = v.reshape((self.n_sounding, self.n_layer))

            pool = Pool(self.n_cpu)
            Jv = np.hstack(
                pool.map(
                    dot,
                    [(J[i], V[i, :]) for i in range(self.n_sounding)]
                )
            )
            pool.close()
            pool.join()
        else:
            return J * v
        return Jv

    def Jtvec(self, m, v, f=None):
        J = self.getJ(m)
        if PARALLEL:
            V = v.reshape((self.n_sounding, int(self.survey.nD/self.n_sounding)))
            pool = Pool(self.n_cpu)

            Jtv = np.hstack(
                pool.map(
                    dot,
                    [(J[i].T, V[i, :]) for i in range(self.n_sounding)]
                )
            )
            pool.close()
            pool.join()
            return Jtv
        else:

            return J.T*v

    def fields(m, f=None):
        return None

    @property
    def deleteTheseOnModelUpdate(self):
        toDelete = []
        if self.sigmaMap is not None:
            toDelete += ['_Sigma']
        if self._Jmatrix is not None:
            toDelete += ['_Jmatrix']
        return toDelete


class GlobalEM1DSurveyFD(EM1DSurveyFD):

    # This assumes a multiple sounding locations
    rx_locations = properties.Array(
        "Receiver locations ", dtype=float, shape=('*', 3)
    )
    src_locations = properties.Array(
        "Source locations ", dtype=float, shape=('*', 3)
    )
    topo = properties.Array(
        "Topography", dtype=float, shape=('*', 3)
    )
    @Utils.count
    @Utils.requires('prob')

    def dpred(self, m=None, f=None):
        """
            Compute predicted data with a given model, m
        """
        return self.prob.forward(m)

    @property
    def n_rx(self):
        """
            # of Receiver locations
        """
        return self.rx_locations.shape[0]

    @property
    def n_layer(self):
        """
            # of Receiver locations
        """
        return self.prob.n_layer

    @property
    def nD(self):
        if self.switch_real_imag == "all":
            return int(self.n_frequency * 2) * self.n_rx
        elif (
            self.switch_real_imag == "imag" or self.switch_real_imag == "real"
        ):
            return int(self.n_frequency) * self.n_rx

    def read_xyz_data(self, fname):
        """
        Read csv file format
        This is a place holder at this point
        """
        pass
