import unittest
from SimPEG import *
import matplotlib.pyplot as plt
from simpegEM1D import EM1D, EM1DAnalytics, DigFilter, EM1DSurveyFD
import numpy as np


class EM1D_FD_Jac_layers_ProblemTests(unittest.TestCase):

    def setUp(self):

        nearthick = np.logspace(-1, 1, 2)
        deepthick = np.logspace(1, 2, 5)
        hx = np.r_[nearthick, deepthick]
        mesh1D = Mesh.TensorMesh([hx], [0.])
        depth = -mesh1D.gridN[:-1]
        n_layer = depth.size

        FDsurvey = EM1DSurveyFD(
            rx_location=np.array([0., 0., 100.+50.]),
            src_location=np.array([0., 0., 100.+50.]),
            field_type='secondary',
            topo=np.r_[0., 0., 100.],
            depth=depth,
            frequency=np.logspace(2, 4, 10),
            src_type='CircularLoop'
        )

        sig_half = 1e-1
        chi_half = 0.

        expmap = Maps.ExpMap(mesh1D)
        tau = 1e-3
        eta = 2e-1
        c = 1.

        m_1D = np.log(np.ones(n_layer)*sig_half)
        FDsurvey.rxType = 'Hz'

        prob = EM1D(mesh1D, sigmaMap=expmap)
        prob.pair(FDsurvey)
        prob.chi = np.zeros(FDsurvey.n_layer)

        self.survey = FDsurvey
        self.prob = prob
        self.mesh1D = mesh1D
        self.showIt = False

    def test_EM1DFDJvec_Layers(self):

        I = 1e0
        a = 1e1
        self.prob.survey.I = I
        self.prob.survey.a = a

        sig_half = 0.01
        sig_blk = 0.1
        sig = np.ones(self.prob.survey.n_layer)*sig_half
        sig[3] = sig_blk
        m_1D = np.log(sig)

        Hz = self.prob.forward(m_1D, output_type='response')
        dHzdsig = self.prob.forward(
            m_1D, output_type='sensitivity_sigma'
        )

        def fwdfun(m):
            resp = self.prob.survey.dpred(m)
            return resp
            # return Hz

        def jacfun(m, dm):
            Jvec = self.prob.Jvec(m, dm)
            return Jvec

        dm = m_1D*0.5
        derChk = lambda m: [fwdfun(m), lambda mx: jacfun(m, mx)]
        passed = Tests.checkDerivative(
            derChk, m_1D, num=4, dx=dm, plotIt=False, eps=1e-15
        )

        if self.showIt is True:

            ilay = 3
            temp_r = Utils.mkvc((dHzdsig[:, ilay].copy()).real)
            temp_i = Utils.mkvc((dHzdsig[:, ilay].copy()).imag)
            frequency = Utils.mkvc(self.prob.survey.frequency)

            plt.loglog(frequency[temp_r > 0], temp_r[temp_r > 0], 'b.-')
            plt.loglog(frequency[temp_r < 0], -temp_r[temp_r < 0], 'b.--')
            plt.loglog(frequency[temp_i > 0], temp_i[temp_i > 0], 'r.-')
            plt.loglog(frequency[temp_i < 0], -temp_i[temp_i < 0], 'r.--')
            plt.show()

        if passed:
            print ("EM1DFD-layers Jvec works")

    def test_EM1DFDJtvec_Layers(self):

        I = 1e0
        a = 1e1
        self.prob.survey.I = I
        self.prob.survey.a = a

        sig_half = 0.01
        sig_blk = 0.1
        sig = np.ones(self.prob.survey.n_layer)*sig_half
        sig[3] = sig_blk
        m_true = np.log(sig)

        dobs = self.prob.survey.dpred(m_true)

        m_ini = np.log(
            np.ones(self.prob.survey.n_layer)*sig_half
        )
        resp_ini = self.prob.survey.dpred(m_ini)
        dr = resp_ini-dobs

        def misfit(m, dobs):
            dpred = self.survey.dpred(m)
            misfit = 0.5*np.linalg.norm(dpred-dobs)**2
            dmisfit = self.prob.Jtvec(m, dr)
            return misfit, dmisfit

        derChk = lambda m: misfit(m, dobs)
        passed = Tests.checkDerivative(
            derChk, m_ini, num=4, plotIt=False, eps=1e-27
        )
        self.assertTrue(passed)
        if passed:
            print ("EM1DFD-layers Jtvec works")


if __name__ == '__main__':
    unittest.main()
