import unittest
from SimPEG import *
import numpy as np
import matplotlib.pyplot as plt
from simpegEM1D import EM1D, EM1DSurveyTD
from simpegEM1D.Waveform import TriangleFun, TriangleFunDeriv


class EM1D_TD_general_Jac_layers_ProblemTests(unittest.TestCase):

    def setUp(self):

        nearthick = np.logspace(-1, 1, 5)
        deepthick = np.logspace(1, 2, 10)
        hx = np.r_[nearthick, deepthick]
        mesh1D = Mesh.TensorMesh([hx], [0.])
        depth = -mesh1D.gridN[:-1]
        LocSigZ = -mesh1D.gridCC

        # Triangular waveform
        time_input_currents = np.r_[0., 5.5*1e-4, 1.1*1e-3]
        input_currents = np.r_[0., 1., 0.]

        TDsurvey = EM1DSurveyTD(
            rx_location=np.array([0., 0., 100.+1e-5]),
            src_location=np.array([0., 0., 100.+1e-5]),
            topo=np.r_[0., 0., 100.],
            depth=depth,
            field_type='secondary',
            rx_type='Bz',
            wave_type='general',
            time_input_currents=time_input_currents,
            input_currents=input_currents,
            n_pulse=2,
            base_frequency=25.,
            time=np.logspace(-5, -2, 31),
            src_type='CircularLoop',
            I=1e0,
            a=2e1
        )

        sig_half = 1e-4
        chi_half = 0.

        expmap = Maps.ExpMap(mesh1D)
        m_1D = np.log(np.ones(TDsurvey.n_layer)*sig_half)
        chi = np.zeros(TDsurvey.n_layer)

        prob = EM1D(
            mesh1D, sigmaMap=expmap, chi=chi
        )
        prob.pair(TDsurvey)

        self.survey = TDsurvey
        self.prob = prob
        self.mesh1D = mesh1D
        self.showIt = False
        self.chi = chi
        self.m_1D = m_1D
        self.sig_half = sig_half
        self.expmap = expmap

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
