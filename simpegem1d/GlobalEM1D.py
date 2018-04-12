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
from .Survey import EM1DSurveyFD, EM1DSurveyTD
from .EM1DSimulation import run_simulation_FD, run_simulation_TD
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
    run_simulation = None
    n_cpu = None
    hz = None
    parallel = False
    verbose = False
    fix_Jmatrix=False

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
    def a(self):
        return self.survey.a

    @property
    def I(self):
        return self.survey.I

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
    def Sigma(self):
        if getattr(self, '_Sigma', None) is None:
            # Ordering: first z then x
            self._Sigma = self.sigma.reshape((self.n_sounding, self.n_layer))
        return self._Sigma

    def fields(self, m):
        if self.verbose:
            print ("Compute fields")
        self.survey._pred = self.forward(m)
        return []

    def Jvec(self, m, v, f=None):
        J = self.getJ(m)
        if self.parallel:
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
        if self.parallel:
            V = v.reshape(
                (self.n_sounding, int(self.survey.nD/self.n_sounding))
            )
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

    @property
    def deleteTheseOnModelUpdate(self):
        toDelete = []
        if self.sigmaMap is not None:
            toDelete += ['_Sigma']
        if self.fix_Jmatrix is False:
            if self._Jmatrix is not None:
                toDelete += ['_Jmatrix']
        return toDelete


class GlobalEM1DProblemFD(GlobalEM1DProblem):

    run_simulation = run_simulation_FD

    @property
    def frequency(self):
        return self.survey.frequency

    @property
    def switch_real_imag(self):
        return self.survey.switch_real_imag

    def input_args(self, i_sounding, jacSwitch=False):
        output = (
            self.rx_locations[i_sounding, :],
            self.src_locations[i_sounding, :],
            self.topo[i_sounding, :], self.hz,
            self.offset, self.frequency,
            self.field_type, self.rx_type, self.src_type,
            self.Sigma[i_sounding, :], jacSwitch
        )
        return output

    def forward(self, m):
        self.model = m

        if self.verbose:
            print (">> Compute response")

        if self.parallel:
            pool = Pool(self.n_cpu)
            # This assumes the same # of layer for each of soundings
            result = pool.map(
                run_simulation_FD,
                [
                    self.input_args(i, jacSwitch=False) for i in range(self.n_sounding)
                ]
            )
            pool.close()
            pool.join()
        else:
            result = [
                run_simulation_FD(self.input_args(i, jacSwitch=False)) for i in range(self.n_sounding)
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
        if self.parallel:
            pool = Pool(self.n_cpu)
            self._Jmatrix = pool.map(
                run_simulation_FD,
                [
                    self.input_args(i, jacSwitch=True) for i in range(self.n_sounding)
                ]
            )
            pool.close()
            pool.join()

        else:
            # _Jmatrix is block diagnoal matrix (sparse)
            self._Jmatrix = sp.block_diag(
                [
                    run_simulation_FD(self.input_args(i, jacSwitch=True)) for i in range(self.n_sounding)
                ]
            ).tocsr()
        return self._Jmatrix


class GlobalEM1DProblemTD(GlobalEM1DProblem):

    run_simulation = run_simulation_TD

    @property
    def wave_type(self):
        return self.survey.wave_type

    @property
    def input_currents(self):
        return self.survey.input_currents

    @property
    def time_input_currents(self):
        return self.survey.time_input_currents

    @property
    def n_pulse(self):
        return self.survey.n_pulse

    @property
    def base_frequency(self):
        return self.survey.base_frequency

    @property
    def time(self):
        return self.survey.time

    def input_args(self, i_sounding, jacSwitch=False):
        output = (
            self.rx_locations[i_sounding, :],
            self.src_locations[i_sounding, :],
            self.topo[i_sounding, :], self.hz,
            self.time,
            self.field_type, self.rx_type, self.src_type, self.wave_type,
            self.offset, self.a,
            self.time_input_currents, self.input_currents,
            self.n_pulse, self.base_frequency,
            self.Sigma[i_sounding, :], jacSwitch
        )
        return output

    def forward(self, m, f=None):
        self.model = m

        if self.parallel:
            pool = Pool(self.n_cpu)
            # This assumes the same # of layer for each of soundings
            result = pool.map(
                run_simulation_TD,
                [
                    self.input_args(i, jacSwitch=False) for i in range(self.n_sounding)
                ]
            )
            pool.close()
            pool.join()
        else:
            result = [
                run_simulation_TD(self.input_args(i, jacSwitch=False)) for i in range(self.n_sounding)
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
        if self.parallel:
            pool = Pool(self.n_cpu)
            self._Jmatrix = pool.map(
                run_simulation_TD,
                [
                    self.input_args(i, jacSwitch=True) for i in range(self.n_sounding)
                ]
            )
            pool.close()
            pool.join()

        else:
            # _Jmatrix is block diagnoal matrix (sparse)
            self._Jmatrix = sp.block_diag(
                [
                    run_simulation_TD(self.input_args(i, jacSwitch=True)) for i in range(self.n_sounding)
                ]
            ).tocsr()
        return self._Jmatrix


class GlobalEM1DSurvey(properties.HasProperties):

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

    _pred = None

    @Utils.requires('prob')
    def dpred(self, m, f=None):
        """
            Return predicted data.
            Predicted data, (`_pred`) are computed when
            self.prob.fields is called.
        """
        if f is None:
            f = self.prob.fields(m)

        return self._pred

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

    def read_xyz_data(self, fname):
        """
        Read csv file format
        This is a place holder at this point
        """
        pass


class GlobalEM1DSurveyFD(GlobalEM1DSurvey, EM1DSurveyFD):

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


class GlobalEM1DSurveyTD(GlobalEM1DSurvey, EM1DSurveyTD):

    @property
    def nD(self):
        return int(self.time.size) * self.n_rx


