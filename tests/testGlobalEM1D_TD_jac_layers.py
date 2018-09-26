from __future__ import print_function
import unittest
import numpy as np
from simpegEM1D import (
    GlobalEM1DProblemTD, GlobalEM1DSurveyTD,
    get_vertical_discretization_time
)
from SimPEG import (
    Regularization, Inversion, InvProblem,
    DataMisfit, Utils, Mesh, Maps, Optimization,
    Tests
)

from simpegEM1D import skytem_HM_2015
wave = skytem_HM_2015()


np.random.seed(41)


class GlobalEM1DTD(unittest.TestCase):

    def setUp(self, parallel=False):
        time = np.logspace(-6, -3, 21)
        hz = get_vertical_discretization_time(
            time, facter_tmax=0.5, factor_tmin=10.
        )
        time_input_currents = wave.current_times[-7:]
        input_currents = wave.currents[-7:]

        n_sounding = 5
        dx = 20.
        hx = np.ones(n_sounding) * dx
        mesh = Mesh.TensorMesh([hx, hz], x0='00')
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
        rx_locations = np.c_[x, y, z]
        src_locations = np.c_[x, y, z]
        topo = np.c_[x, y, z-30.].astype(float)

        n_sounding = rx_locations.shape[0]

        rx_type_global = np.array(
            ["dBzdt"], dtype=str
        ).repeat(n_sounding, axis=0)
        field_type_global = np.array(
            ['secondary'], dtype=str
        ).repeat(n_sounding, axis=0)
        wave_type_global = np.array(
            ['general'], dtype=str
        ).repeat(n_sounding, axis=0)

        time_global = [time for i in range(n_sounding)]

        src_type_global = np.array(
            ["CircularLoop"], dtype=str
        ).repeat(n_sounding, axis=0)
        a_global = np.array(
            [13.], dtype=float
        ).repeat(n_sounding, axis=0)
        input_currents_global = [
            input_currents for i in range(n_sounding)
        ]
        time_input_currents_global = [
            time_input_currents for i in range(n_sounding)
        ]

        mapping = Maps.ExpMap(mesh)

        survey = GlobalEM1DSurveyTD(
            rx_locations=rx_locations,
            src_locations=src_locations,
            topo=topo,
            time=time_global,
            src_type=src_type_global,
            rx_type=rx_type_global,
            field_type=field_type_global,
            wave_type=wave_type_global,
            a=a_global,
            input_currents=input_currents_global,
            time_input_currents=time_input_currents_global
        )

        problem = GlobalEM1DProblemTD(
            mesh, sigmaMap=mapping, hz=hz, parallel=parallel, n_cpu=2
        )
        problem.pair(survey)

        survey.makeSyntheticData(mSynth)

        # Now set up the problem to do some minimization
        dmis = DataMisfit.l2_DataMisfit(survey)
        reg = Regularization.Tikhonov(mesh)
        opt = Optimization.InexactGaussNewton(
            maxIterLS=20, maxIter=10, tolF=1e-6,
            tolX=1e-6, tolG=1e-6, maxIterCG=6
        )
        invProb = InvProblem.BaseInvProblem(dmis, reg, opt, beta=0.)
        inv = Inversion.BaseInversion(invProb)
        self.inv = inv
        self.reg = reg
        self.p = problem
        self.mesh = mesh
        self.m0 = mSynth
        self.survey = survey
        self.dmis = dmis

    def test_misfit(self):
        passed = Tests.checkDerivative(
            lambda m: (
                self.survey.dpred(m),
                lambda mx: self.p.Jvec(self.m0, mx)
            ),
            self.m0,
            plotIt=False,
            num=3
        )
        self.assertTrue(passed)

    def test_adjoint(self):
        # Adjoint Test
        v = np.random.rand(self.mesh.nC)
        w = np.random.rand(self.survey.dobs.shape[0])
        wtJv = w.dot(self.p.Jvec(self.m0, v))
        vtJtw = v.dot(self.p.Jtvec(self.m0, w))
        passed = np.abs(wtJv - vtJtw) < 1e-10
        print('Adjoint Test', np.abs(wtJv - vtJtw), passed)
        self.assertTrue(passed)

    def test_dataObj(self):
        passed = Tests.checkDerivative(
            lambda m: [self.dmis(m), self.dmis.deriv(m)],
            self.m0,
            plotIt=False,
            num=3
        )
        self.assertTrue(passed)

class GlobalEM1DTD_Height(unittest.TestCase):

    def setUp(self, parallel=False):
        time = np.logspace(-6, -3, 21)
        time_input_currents = wave.current_times[-7:]
        input_currents = wave.currents[-7:]
        hz = get_vertical_discretization_time(
            time, facter_tmax=0.5, factor_tmin=10.
        )

        hz = np.r_[1.]
        n_sounding = 10
        dx = 20.
        hx = np.ones(n_sounding) * dx
        e = np.ones(n_sounding)
        mSynth = np.r_[e*np.log(1./100.), e*20]

        x = np.arange(n_sounding)
        y = np.zeros_like(x)
        z = np.ones_like(x) * 30.
        rx_locations = np.c_[x, y, z]
        src_locations = np.c_[x, y, z]
        topo = np.c_[x, y, z-30.].astype(float)

        rx_type_global = np.array(
            ["dBzdt"], dtype=str
        ).repeat(n_sounding, axis=0)
        field_type_global = np.array(
            ['secondary'], dtype=str
        ).repeat(n_sounding, axis=0)
        wave_type_global = np.array(
            ['general'], dtype=str
        ).repeat(n_sounding, axis=0)

        time_global = [time for i in range(n_sounding)]

        src_type_global = np.array(
            ["CircularLoop"], dtype=str
        ).repeat(n_sounding, axis=0)
        a_global = np.array(
            [13.], dtype=float
        ).repeat(n_sounding, axis=0)
        input_currents_global = [
            input_currents for i in range(n_sounding)
        ]
        time_input_currents_global = [
            time_input_currents for i in range(n_sounding)
        ]

        wires = Maps.Wires(('sigma', n_sounding),('h', n_sounding))
        expmap = Maps.ExpMap(nP=n_sounding)
        sigmaMap = expmap * wires.sigma

        survey = GlobalEM1DSurveyTD(
            rx_locations=rx_locations,
            src_locations=src_locations,
            topo=topo,
            time=time_global,
            src_type=src_type_global,
            rx_type=rx_type_global,
            field_type=field_type_global,
            wave_type=wave_type_global,
            a=a_global,
            input_currents=input_currents_global,
            time_input_currents=time_input_currents_global,
            half_switch=True
        )

        problem = GlobalEM1DProblemTD(
            [], sigmaMap=sigmaMap, hMap=wires.h, hz=hz, parallel=parallel, n_cpu=2
        )
        problem.pair(survey)

        survey.makeSyntheticData(mSynth)

        # Now set up the problem to do some minimization
        mesh = Mesh.TensorMesh([int(n_sounding * 2)])
        dmis = DataMisfit.l2_DataMisfit(survey)
        reg = Regularization.Tikhonov(mesh)
        opt = Optimization.InexactGaussNewton(
            maxIterLS=20, maxIter=10, tolF=1e-6,
            tolX=1e-6, tolG=1e-6, maxIterCG=6
        )
        invProb = InvProblem.BaseInvProblem(dmis, reg, opt, beta=0.)
        inv = Inversion.BaseInversion(invProb)
        self.inv = inv
        self.reg = reg
        self.p = problem
        self.mesh = mesh
        self.m0 = mSynth
        self.survey = survey
        self.dmis = dmis

    def test_misfit(self):
        passed = Tests.checkDerivative(
            lambda m: (
                self.survey.dpred(m),
                lambda mx: self.p.Jvec(self.m0, mx)
            ),
            self.m0,
            plotIt=False,
            num=3
        )
        self.assertTrue(passed)

    def test_adjoint(self):
        # Adjoint Test
        v = np.random.rand(self.mesh.nC)
        w = np.random.rand(self.survey.dobs.shape[0])
        wtJv = w.dot(self.p.Jvec(self.m0, v))
        vtJtw = v.dot(self.p.Jtvec(self.m0, w))
        passed = np.abs(wtJv - vtJtw) < 1e-10
        print('Adjoint Test', np.abs(wtJv - vtJtw), passed)
        self.assertTrue(passed)

    def test_dataObj(self):
        passed = Tests.checkDerivative(
            lambda m: [self.dmis(m), self.dmis.deriv(m)],
            self.m0,
            plotIt=False,
            num=3
        )
        self.assertTrue(passed)

if __name__ == '__main__':
    unittest.main()
