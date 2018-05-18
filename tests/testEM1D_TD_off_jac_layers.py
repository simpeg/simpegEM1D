import unittest
from SimPEG import Mesh, Utils, Maps, Tests
import matplotlib.pyplot as plt
from simpegEM1D import EM1D, EM1DSurveyTD
import numpy as np


class EM1D_TD_Jac_layers_ProblemTests(unittest.TestCase):

    def setUp(self):

        nearthick = np.logspace(-1, 1, 5)
        deepthick = np.logspace(1, 2, 10)
        hx = np.r_[nearthick, deepthick]
        mesh1D = Mesh.TensorMesh([hx], [0.])
        depth = -mesh1D.gridN[:-1]
        LocSigZ = -mesh1D.gridCC

        TDsurvey = EM1DSurveyTD(
            rx_location=np.array([0., 0., 100.+50.]),
            src_location=np.array([0., 0., 100.+50.]),
            topo=np.r_[0., 0., 100.],
            depth=depth,
            field_type='secondary',
            rx_type='dBzdt',
            wave_type='stepoff',
            time=np.logspace(-5, -2, 64),
            src_type='CircularLoop',
            I=1e0,
            a=1e1
        )

        sig_half = 1e-2
        chi_half = 0.

        expmap = Maps.ExpMap(mesh1D)
        tau = 1e-3 * np.ones(TDsurvey.n_layer)
        eta = 2e-1 * np.ones(TDsurvey.n_layer)
        c = 1. * np.ones(TDsurvey.n_layer)
        m_1D = np.log(np.ones(TDsurvey.n_layer)*sig_half)
        chi = np.zeros(TDsurvey.n_layer)

        prob = EM1D(
            mesh1D, sigmaMap=expmap, chi=chi
        )
        prob.pair(TDsurvey)

        self.survey = TDsurvey
        self.prob = prob
        self.mesh1D = mesh1D
        self.showIt = True
        self.m_1D = m_1D
        self.sig_half = sig_half

    def test_EM1DTDJvec_Layers(self):
        sig = np.ones(self.prob.survey.n_layer)*self.sig_half
        m_1D = np.log(sig)

        def fwdfun(m):
            resp = self.prob.survey.dpred(m)
            return resp

        def jacfun(m, dm):
            Jvec = self.prob.Jvec(m, dm)
            return Jvec

        dm = m_1D*0.5
        derChk = lambda m: [fwdfun(m), lambda mx: jacfun(m, mx)]
        passed = Tests.checkDerivative(
            derChk, m_1D, num=4, dx=dm, plotIt=False, eps=1e-15
        )

        if passed:
            print ("EM1DTD-layers Jvec works")

    def test_EM1DTDJtvec_Layers(self):

        sig_blk = 0.1
        sig = np.ones(self.prob.survey.n_layer)*self.sig_half
        sig[3] = sig_blk

        m_true = np.log(sig)
        dobs = self.prob.survey.dpred(m_true)
        m_ini = np.log(np.ones(self.prob.survey.n_layer)*self.sig_half)
        resp_ini = self.prob.survey.dpred(m_ini)
        dr = resp_ini-dobs

        def misfit(m, dobs):
            dpred = self.survey.dpred(m)
            misfit = 0.5*np.linalg.norm(dpred-dobs)**2
            dmisfit = self.prob.Jtvec(m, dr)
            return misfit, dmisfit

        derChk = lambda m: misfit(m, dobs)
        passed = Tests.checkDerivative(
            derChk, m_ini, num=4, plotIt=False, eps=1e-26
        )
        self.assertTrue(passed)
        if passed:
            print ("EM1DTD-layers Jtvec works")


if __name__ == '__main__':
    unittest.main()
