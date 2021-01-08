from __future__ import print_function
import unittest
import numpy as np
import simpegEM1D as em1d
from simpegEM1D.utils import get_vertical_discretization_time
from simpegEM1D.waveforms import TriangleFun
from SimPEG import *
from discretize import TensorMesh
from pymatsolver import PardisoSolver


np.random.seed(41)


class GlobalEM1DTD(unittest.TestCase):

    def setUp(self, parallel=True):

        times = np.logspace(-5, -2, 31)
        n_layer = 20
        thicknesses = get_vertical_discretization_time(
            times, facter_tmax=0.5, factor_tmin=10., n_layer=n_layer-1
        )

        n_sounding = 5
        dx = 20.
        hx = np.ones(n_sounding) * dx
        hz = np.r_[thicknesses, thicknesses[-1]]
        mesh = TensorMesh([hx, hz], x0='00')
        inds = mesh.gridCC[:, 1] < 25
        inds_1 = mesh.gridCC[:, 1] < 50
        sigma = np.ones(mesh.nC) * 1./100.
        sigma[inds_1] = 1./10.
        sigma[inds] = 1./50.
        sigma_em1d = sigma.reshape(mesh.vnC, order='F').flatten()
        mSynth = np.log(sigma_em1d)

        x = mesh.vectorCCx
        y = np.zeros_like(x)
        z = np.ones_like(x) * 30.
        receiver_locations = np.c_[x, y, z]
        source_locations = np.c_[x, y, z]
        topo = np.c_[x, y, z-30.].astype(float)

        sigma_map = maps.ExpMap(mesh)

        source_list = []

        for ii in range(0, n_sounding):

            source_location = mkvc(source_locations[ii, :])
            receiver_location = mkvc(receiver_locations[ii, :])

            receiver_list = []

            receiver_list.append(
                em1d.receivers.TimeDomainPointReceiver(
                    receiver_location, times, orientation="z",
                    component="b"
                )
            )

            receiver_list.append(
                em1d.receivers.TimeDomainPointReceiver(
                    receiver_location, times, orientation="z",
                    component="dbdt"
                )
            )

            time_input_currents = np.r_[-np.logspace(-2, -5, 31), 0.]
            input_currents = TriangleFun(time_input_currents+0.01, 5e-3, 0.01)

            source_list.append(
                em1d.sources.TimeDomainHorizontalLoopSource(
                    receiver_list=receiver_list,
                    location=source_location,
                    a=5., I=1.,
                    wave_type="general",
                    time_input_currents=time_input_currents,
                    input_currents=input_currents,
                    n_pulse = 1,
                    base_frequency = 25.,
                    use_lowpass_filter=False,
                    high_cut_frequency=210*1e3
                )
            )

        survey = em1d.survey.EM1DSurveyTD(source_list)

        simulation = em1d.simulation.StitchedEM1DTMSimulation(
            survey=survey, thicknesses=thicknesses, sigmaMap=sigma_map,
            topo=topo, parallel=False, n_cpu=2, verbose=False, solver=PardisoSolver
        )

        dpred = simulation.dpred(mSynth)
        noise = 0.1*np.abs(dpred)*np.random.rand(len(dpred))
        uncertainties = 0.1*np.abs(dpred)*np.ones(np.shape(dpred))
        dobs =  dpred + noise
        data_object = data.Data(survey, dobs=dobs, noise_floor=uncertainties)

        dmis = data_misfit.L2DataMisfit(simulation=simulation, data=data_object)
        dmis.W = 1./uncertainties

        reg = regularization.Tikhonov(mesh)

        opt = optimization.InexactGaussNewton(
            maxIterLS=20, maxIter=10, tolF=1e-6,
            tolX=1e-6, tolG=1e-6, maxIterCG=6
        )

        invProb = inverse_problem.BaseInvProblem(dmis, reg, opt, beta=0.)
        inv = inversion.BaseInversion(invProb)

        self.data = data_object
        self.dmis = dmis
        self.inv = inv
        self.reg = reg
        self.sim = simulation
        self.mesh = mesh
        self.m0 = mSynth
        self.survey = survey


    def test_misfit(self):
        passed = tests.checkDerivative(
            lambda m: (
                self.sim.dpred(m),
                lambda mx: self.sim.Jvec(self.m0, mx)
            ),
            self.m0,
            plotIt=False,
            num=3
        )
        self.assertTrue(passed)

    def test_adjoint(self):
        # Adjoint Test
        v = np.random.rand(self.mesh.nC)
        w = np.random.rand(self.data.dobs.shape[0])
        wtJv = w.dot(self.sim.Jvec(self.m0, v))
        vtJtw = v.dot(self.sim.Jtvec(self.m0, w))
        passed = np.abs(wtJv - vtJtw) < 1e-10
        print('Adjoint Test', np.abs(wtJv - vtJtw), passed)
        self.assertTrue(passed)

    def test_dataObj(self):
        passed = tests.checkDerivative(
            lambda m: [self.dmis(m), self.dmis.deriv(m)],
            self.m0,
            plotIt=False,
            num=3
        )
        self.assertTrue(passed)

class GlobalEM1DTD_Height(unittest.TestCase):

    def setUp(self, parallel=True):

        times = np.logspace(-5, -2, 31)

        hz = 1.
        n_sounding = 10
        dx = 20.
        hx = np.ones(n_sounding) * dx
        e = np.ones(n_sounding)
        mSynth = np.r_[e*np.log(1./100.), e*30]
        mesh = TensorMesh([hx, hz], x0='00')

        wires = maps.Wires(('sigma', n_sounding),('height', n_sounding))
        expmap = maps.ExpMap(nP=n_sounding)
        sigma_map = expmap * wires.sigma

        x = mesh.vectorCCx
        y = np.zeros_like(x)
        z = np.ones_like(x) * 30.
        receiver_locations = np.c_[x, y, z]
        source_locations = np.c_[x, y, z]
        topo = np.c_[x, y, z-30.].astype(float)

        source_list = []

        for ii in range(0, n_sounding):

            source_location = mkvc(source_locations[ii, :])
            receiver_location = mkvc(receiver_locations[ii, :])

            receiver_list = []

            receiver_list.append(
                em1d.receivers.TimeDomainPointReceiver(
                    receiver_location, times, orientation="z",
                    component="b"
                )
            )

            receiver_list.append(
                em1d.receivers.TimeDomainPointReceiver(
                    receiver_location, times, orientation="z",
                    component="dbdt"
                )
            )

            time_input_currents = np.r_[-np.logspace(-2, -5, 31), 0.]
            input_currents = TriangleFun(time_input_currents+0.01, 5e-3, 0.01)

            source_list.append(
                em1d.sources.TimeDomainHorizontalLoopSource(
                    receiver_list=receiver_list,
                    location=source_location,
                    a=5., I=1.,
                    wave_type="general",
                    time_input_currents=time_input_currents,
                    input_currents=input_currents,
                    n_pulse = 1,
                    base_frequency = 25.,
                    use_lowpass_filter=False,
                    high_cut_frequency=210*1e3
                )
            )

        survey = em1d.survey.EM1DSurveyTD(source_list)

        simulation = em1d.simulation.StitchedEM1DTMSimulation(
            survey=survey, sigmaMap=sigma_map, hMap=wires.height,
            topo=topo, parallel=False, n_cpu=2, verbose=False, solver=PardisoSolver
        )

        dpred = simulation.dpred(mSynth)
        noise = 0.1*np.abs(dpred)*np.random.rand(len(dpred))
        uncertainties = 0.1*np.abs(dpred)*np.ones(np.shape(dpred))
        dobs =  dpred + noise
        data_object = data.Data(survey, dobs=dobs, noise_floor=uncertainties)

        dmis = data_misfit.L2DataMisfit(simulation=simulation, data=data_object)
        dmis.W = 1./uncertainties

        reg_mesh = TensorMesh([int(n_sounding)])
        reg_sigma = regularization.Tikhonov(reg_mesh, mapping=wires.sigma)
        reg_height = regularization.Tikhonov(reg_mesh, mapping=wires.height)

        reg = reg_sigma + reg_height

        opt = optimization.InexactGaussNewton(
            maxIterLS=20, maxIter=10, tolF=1e-6,
            tolX=1e-6, tolG=1e-6, maxIterCG=6
        )

        invProb = inverse_problem.BaseInvProblem(dmis, reg, opt, beta=0.)
        inv = inversion.BaseInversion(invProb)

        self.data = data_object
        self.dmis = dmis
        self.inv = inv
        self.reg = reg
        self.sim = simulation
        self.mesh = reg_mesh
        self.m0 = mSynth * 1.2
        self.survey = survey


    def test_misfit(self):
        passed = tests.checkDerivative(
            lambda m: (
                self.sim.dpred(m),
                lambda mx: self.sim.Jvec(self.m0, mx)
            ),
            self.m0,
            plotIt=False,
            num=3
        )
        self.assertTrue(passed)

    def test_adjoint(self):
        # Adjoint Test
        v = np.random.rand(2*self.mesh.nC)
        w = np.random.rand(self.data.dobs.shape[0])
        wtJv = w.dot(self.sim.Jvec(self.m0, v))
        vtJtw = v.dot(self.sim.Jtvec(self.m0, w))
        passed = np.abs(wtJv - vtJtw) < 1e-10
        print('Adjoint Test', np.abs(wtJv - vtJtw), passed)
        self.assertTrue(passed)

    def test_dataObj(self):
        passed = tests.checkDerivative(
            lambda m: [self.dmis(m), self.dmis.deriv(m)],
            self.m0,
            plotIt=False,
            num=3
        )
        self.assertTrue(passed)

if __name__ == '__main__':
    unittest.main()
