from __future__ import print_function
import unittest
import numpy as np
from simpegem1d import (
    GlobalEM1DProblemFD, GlobalEM1DSurveyFD,
    get_vertical_discretization_frequency
)
from SimPEG import (
    Regularization, Inversion, InvProblem,
    DataMisfit, Utils, Mesh, Maps, Optimization,
    Tests
)

np.random.seed(41)


class GlobalEM1DFD(unittest.TestCase):

    def setUp(self, parallel=True):
        frequency = np.array([900, 7200, 56000], dtype=float)
        hz = get_vertical_discretization_frequency(
            frequency, sigma_background=1./10.
        )

        n_sounding = 10
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
        mapping = Maps.ExpMap(mesh)
        survey = GlobalEM1DSurveyFD(
            rx_locations=rx_locations,
            src_locations=src_locations,
            frequency=frequency,
            offset=np.ones_like(frequency) * 8.,
            src_type="VMD",
            rx_type="Hz",
            field_type='secondary',
            topo=topo
        )

        problem = GlobalEM1DProblemFD(
            [], sigmaMap=mapping, hz=hz,
            parallel=parallel, n_cpu=5
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
        # u = np.random.rand(self.mesh.nC * self.survey.nSrc)
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
