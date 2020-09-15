from __future__ import print_function
import unittest
import numpy as np
import simpegEM1D as em1d
from simpegEM1D.utils import get_vertical_discretization_frequency
from SimPEG import *
from discretize import TensorMesh
from pymatsolver import PardisoSolver

np.random.seed(41)


class GlobalEM1DFD(unittest.TestCase):

    def setUp(self, parallel=True):
        
        frequencies = np.array([900, 7200, 56000], dtype=float)
        hz = get_vertical_discretization_frequency(
            frequencies, sigma_background=0.1
        )
        
        n_sounding = 10
        dx = 20.
        hx = np.ones(n_sounding) * dx
        
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
        receiver_locations = np.c_[x+8., y, z]
        source_locations = np.c_[x, y, z]
        topo = np.c_[x, y, z-30.].astype(float)
        
        sigma_map = maps.ExpMap(mesh)
        
        source_list = []

        for ii in range(0, n_sounding):
            
            source_location = mkvc(source_locations[ii, :])
            receiver_location = mkvc(receiver_locations[ii, :])
            
            receiver_list = []
            
            receiver_list.append(
                em1d.receivers.HarmonicPointReceiver(
                    receiver_location, frequencies, orientation="z",
                    field_type="secondary", component="real"
                )
            )
            receiver_list.append(
                em1d.receivers.HarmonicPointReceiver(
                    receiver_location, frequencies, orientation="z",
                    field_type="secondary", component="imag"
                )
            )
            
            source_list.append(
                em1d.sources.HarmonicMagneticDipoleSource(
                    receiver_list=receiver_list, location=source_location,
                    orientation="z", I=1.
                )
            )
        
        survey = em1d.survey.EM1DSurveyFD(source_list)
        
        simulation = em1d.simulation_stitched1d.GlobalEM1DSimulationFD(
            mesh, survey=survey, sigmaMap=sigma_map, hz=hz, topo=topo,
            parallel=False, n_cpu=2, verbose=True, Solver=PardisoSolver
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
        # u = np.random.rand(self.mesh.nC * self.survey.nSrc)
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

class GlobalEM1DFD_Height(unittest.TestCase):

    def setUp(self, parallel=True):
        
        frequencies = np.array([900, 7200, 56000], dtype=float)
        hz = np.r_[1.]
        n_sounding = 10
        dx = 20.
        hx = np.ones(n_sounding) * dx
        e = np.ones(n_sounding)
        mSynth = np.r_[e*np.log(1./100.), e*20.]
        mesh = TensorMesh([hx, hz], x0='00')
        
        x = mesh.vectorCCx
        y = np.zeros_like(x)
        z = np.ones_like(x) * 30.
        receiver_locations = np.c_[x+8., y, z]
        source_locations = np.c_[x, y, z]
        topo = np.c_[x, y, z-30.].astype(float)

        wires = maps.Wires(('sigma', n_sounding),('h', n_sounding))
        expmap = maps.ExpMap(nP=n_sounding)
        sigma_map = expmap * wires.sigma
        
        source_list = []

        for ii in range(0, n_sounding):
            
            source_location = mkvc(source_locations[ii, :])
            receiver_location = mkvc(receiver_locations[ii, :])
            
            receiver_list = []
            
            receiver_list.append(
                em1d.receivers.HarmonicPointReceiver(
                    receiver_location, frequencies, orientation="z",
                    field_type="secondary", component="real"
                )
            )
            receiver_list.append(
                em1d.receivers.HarmonicPointReceiver(
                    receiver_location, frequencies, orientation="z",
                    field_type="secondary", component="imag"
                )
            )
            
            source_list.append(
                em1d.sources.HarmonicMagneticDipoleSource(
                    receiver_list=receiver_list, location=source_location,
                    orientation="z", I=1.
                )
            )
        
        survey = em1d.survey.EM1DSurveyFD(source_list)
        
        simulation = em1d.simulation_stitched1d.GlobalEM1DSimulationFD(
            mesh, survey=survey, sigmaMap=sigma_map, hz=hz, hMap=wires.h, topo=topo,
            parallel=False, n_cpu=2, verbose=True, Solver=PardisoSolver
        )
        
        dpred = simulation.dpred(mSynth)
        noise = 0.1*np.abs(dpred)*np.random.rand(len(dpred))
        uncertainties = 0.1*np.abs(dpred)*np.ones(np.shape(dpred))
        dobs =  dpred + noise
        data_object = data.Data(survey, dobs=dobs, noise_floor=uncertainties)
        
        
        dmis = data_misfit.L2DataMisfit(simulation=simulation, data=data_object)
        dmis.W = 1./uncertainties
        
        reg_mesh = TensorMesh([int(n_sounding * 2)])
        reg = regularization.Tikhonov(reg_mesh)
        
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
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
#         # frequency = np.array([900, 7200, 56000], dtype=float)
#         # hz = np.r_[1.]
#         # n_sounding = 10
#         # dx = 20.
#         # hx = np.ones(n_sounding) * dx
#         # e = np.ones(n_sounding)
#         # mSynth = np.r_[e*np.log(1./100.), e*20]

#         # x = np.arange(n_sounding)
#         # y = np.zeros_like(x)
#         # z = np.ones_like(x) * 30.
#         # rx_locations = np.c_[x, y, z]
#         # src_locations = np.c_[x, y, z]
#         # topo = np.c_[x, y, z-30.].astype(float)

#         # wires = Maps.Wires(('sigma', n_sounding),('h', n_sounding))
#         # expmap = Maps.ExpMap(nP=n_sounding)
#         # sigmaMap = expmap * wires.sigma

#         # survey = GlobalEM1DSurveyFD(
#         #     rx_locations=rx_locations,
#         #     src_locations=src_locations,
#         #     frequency=frequency,
#         #     offset=np.ones_like(frequency) * 8.,
#         #     src_type="VMD",
#         #     rx_type="ppm",
#         #     field_type='secondary',
#         #     topo=topo,
#         #     half_switch=True
#         # )
#         #
#         # problem = GlobalEM1DProblemFD(
#         #     [], sigmaMap=sigmaMap, hMap=wires.h, hz=hz,
#         #     parallel=parallel, n_cpu=2
#         # )
#         # problem.pair(survey)
#         # survey.makeSyntheticData(mSynth)

#         # # Now set up the problem to do some minimization
#         # mesh = Mesh.TensorMesh([int(n_sounding * 2)])
#         # dmis = DataMisfit.l2_DataMisfit(survey)
#         # reg = regularization.Tikhonov(mesh)
#         # opt = Optimization.InexactGaussNewton(
#         #     maxIterLS=20, maxIter=10, tolF=1e-6,
#         #     tolX=1e-6, tolG=1e-6, maxIterCG=6
#         # )

#         # invProb = InvProblem.BaseInvProblem(dmis, reg, opt, beta=0.)
#         # inv = Inversion.BaseInversion(invProb)
#         # self.inv = inv
#         # self.reg = reg
#         # self.p = problem
#         # self.mesh = mesh
#         # self.m0 = mSynth * 1.2
#         # self.survey = survey
#         # self.dmis = dmis

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
        # u = np.random.rand(self.mesh.nC * self.survey.nSrc)
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

if __name__ == '__main__':
    unittest.main()
