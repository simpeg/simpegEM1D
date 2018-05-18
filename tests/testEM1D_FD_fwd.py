import unittest
from SimPEG import *
import matplotlib.pyplot as plt
from simpegEM1D import EM1D, EM1DAnalytics, EM1DSurveyFD
import numpy as np
from scipy.constants import mu_0


class EM1D_FD_FwdProblemTests(unittest.TestCase):

    def setUp(self):

        nearthick = np.logspace(-1, 1, 5)
        deepthick = np.logspace(1, 2, 10)
        hx = np.r_[nearthick, deepthick]
        mesh1D = Mesh.TensorMesh([hx], [0.])
        depth = -mesh1D.gridN[:-1]
        nlay = depth.size
        topo = np.r_[0., 0., 100.]

        FDsurvey = EM1DSurveyFD(
            rx_location=np.array([0., 0., 100.+1e-5]),
            src_location=np.array([0., 0., 100.+1e-5]),
            field_type='secondary',
            depth=depth,
            topo=topo,
            frequency=np.logspace(1, 8, 61),
            offset=10. * np.ones(61)
        )

        sig_half = 1e-2
        chi_half = 0.

        expmap = Maps.ExpMap(mesh1D)
        tau = 1e-3
        eta = 2e-1
        c = 1.

        m_1D = np.log(np.ones(nlay)*sig_half)
        FDsurvey.rx_type = 'Hz'

        prob = EM1D(
            mesh1D, sigmaMap=expmap
        )
        prob.pair(FDsurvey)
        prob.chi = np.zeros(FDsurvey.n_layer)

        self.survey = FDsurvey
        self.prob = prob
        self.mesh1D = mesh1D
        self.showIt = False
        self.tau = tau
        self.eta = eta
        self.c = c

    def test_EM1DFDfwd_VMD_RealCond(self):
        self.prob.survey.src_type = 'VMD'
        self.prob.survey.offset = np.ones(self.prob.survey.n_frequency) * 10.
        sig_half = 0.01
        m_1D = np.log(np.ones(self.prob.survey.n_layer)*sig_half)
        Hz = self.prob.forward(m_1D)
        Hzanal = EM1DAnalytics.Hzanal(
            sig_half, self.prob.survey.frequency,
            self.prob.survey.offset, 'secondary'
        )

        if self.showIt is True:

            plt.loglog(self.prob.survey.frequency, abs(Hz.real), 'b')
            plt.loglog(self.prob.survey.frequency, abs(Hzanal.real), 'b*')
            plt.loglog(self.prob.survey.frequency, abs(Hz.imag), 'r')
            plt.loglog(self.prob.survey.frequency, abs(Hzanal.imag), 'r*')
            plt.show()

        err = np.linalg.norm(Hz-Hzanal)/np.linalg.norm(Hzanal)
        self.assertTrue(err < 1e-5)
        print ("EM1DFD-VMD for real conductivity works")

    def test_EM1DFDfwd_VMD_ComplexCond(self):

        if self.prob.ispaired:
            self.prob.unpair()
        if self.survey.ispaired:
            self.survey.unpair()

        self.prob = EM1D(
            self.mesh1D,
            sigmaMap=Maps.IdentityMap(self.mesh1D),
            chi=np.zeros(self.survey.n_layer),
            eta=self.eta,
            tau=self.tau,
            c=self.c
        )
        self.prob.pair(self.survey)
        self.prob.survey.src_type = 'VMD'
        sig_half = 0.01
        m_1D = np.ones(self.prob.survey.n_layer)*sig_half
        Hz = self.prob.forward(m_1D)
        sigCole = EM1DAnalytics.ColeCole(
            self.survey.frequency, sig_half,
            self.eta, self.tau, self.c
            )
        Hzanal = EM1DAnalytics.Hzanal(
            sigCole, self.prob.survey.frequency,
            self.prob.survey.offset, 'secondary'
        )

        if self.showIt is True:

            plt.loglog(self.prob.survey.frequency, abs(Hz.real), 'b')
            plt.loglog(self.prob.survey.frequency, abs(Hzanal.real), 'b*')
            plt.loglog(self.prob.survey.frequency, abs(Hz.imag), 'r')
            plt.loglog(self.prob.survey.frequency, abs(Hzanal.imag), 'r*')
            plt.show()

        err = np.linalg.norm(Hz-Hzanal)/np.linalg.norm(Hzanal)
        self.assertTrue(err < 1e-5)
        print ("EM1DFD-VMD for complex conductivity works")

    def test_EM1DFDfwd_CircularLoop_RealCond(self):
        self.prob.survey.src_type = 'CircularLoop'
        I = 1e0
        a = 1e1
        self.prob.survey.I = I
        self.prob.survey.a = a
        sig_half = 0.01
        m_1D = np.log(np.ones(self.prob.survey.n_layer)*sig_half)
        Hz = self.prob.forward(m_1D)
        Hzanal = EM1DAnalytics.HzanalCirc(
            sig_half, self.prob.survey.frequency,
            I, a, 'secondary'
        )

        if self.showIt is True:

            plt.loglog(self.prob.survey.frequency, abs(Hz.real), 'b')
            plt.loglog(self.prob.survey.frequency, abs(Hzanal.real), 'b*')
            plt.loglog(self.prob.survey.frequency, abs(Hz.imag), 'r')
            plt.loglog(self.prob.survey.frequency, abs(Hzanal.imag), 'r*')
            plt.show()

        err = np.linalg.norm(Hz-Hzanal)/np.linalg.norm(Hzanal)
        self.assertTrue(err < 1e-5)
        print ("EM1DFD-CircularLoop for real conductivity works")

    def test_EM1DFDfwd_CircularLoop_ComplexCond(self):

        if self.prob.ispaired:
            self.prob.unpair()
        if self.survey.ispaired:
            self.survey.unpair()

        self.prob = EM1D(
            self.mesh1D,
            sigmaMap=Maps.IdentityMap(self.mesh1D),
            chi=np.zeros(self.survey.n_layer),
            eta=self.eta,
            tau=self.tau,
            c=self.c
        )

        self.prob.pair(self.survey)
        self.prob.survey.src_type = 'CircularLoop'
        I = 1e0
        a = 1e1
        self.prob.survey.I = I
        self.prob.survey.a = a

        sig_half = 0.01
        m_1D = np.ones(self.prob.survey.n_layer)*sig_half
        Hz = self.prob.forward(m_1D)
        sigCole = EM1DAnalytics.ColeCole(
            self.survey.frequency, sig_half, self.eta, self.tau, self.c
        )
        Hzanal = EM1DAnalytics.HzanalCirc(
            sigCole, self.prob.survey.frequency, I, a, 'secondary'
        )

        if self.showIt is True:

            plt.loglog(self.prob.survey.frequency, abs(Hz.real), 'b')
            plt.loglog(self.prob.survey.frequency, abs(Hzanal.real), 'b*')
            plt.loglog(self.prob.survey.frequency, abs(Hz.imag), 'r')
            plt.loglog(self.prob.survey.frequency, abs(Hzanal.imag), 'r*')
            plt.show()

        err = np.linalg.norm(Hz-Hzanal)/np.linalg.norm(Hzanal)
        self.assertTrue(err < 1e-5)
        print ("EM1DFD-CircularLoop for complex conductivity works")

    # def test_EM1DFDfwd_VMD_EM1D_sigchi(self):

    #     self.survey.rx_location = np.array([0., 0., 110.+1e-5])
    #     self.survey.src_location = np.array([0., 0., 110.+1e-5])
    #     self.survey.field_type = 'secondary'

    #     hx = np.r_[np.ones(3)*10]
    #     mesh1D = Mesh.TensorMesh([hx], [0.])
    #     depth = -mesh1D.gridN[:-1]
    #     nlay = depth.size
    #     topo = np.r_[0., 0., 100.]

    #     self.survey.depth = depth
    #     self.survey.topo = topo

    #     self.survey.frequency = np.logspace(-3, 5, 61)
    #     self.prob.unpair()
    #     mapping = Maps.ExpMap(mesh1D)
    #     # 1. Verification for variable conductivity
    #     chi = np.array([0., 0., 0.])
    #     sig = np.array([0.01, 0.1, 0.01])

    #     self.prob = EM1D(mesh1D, sigmaMap=mapping, chi=chi, jacSwitch=False)
    #     self.prob.pair(self.survey)
    #     self.prob.survey.src_type = 'VMD'
    #     self.prob.survey.offset = 10. * np.ones(self.survey.n_frequency)

    #     m_1D = np.log(sig)
    #     Hz = self.prob.forward(m_1D)
    #     from scipy import io
    #     mat = io.loadmat('em1dfm/VMD_3lay.mat')
    #     freq = mat['data'][:, 0]
    #     Hzanal = mat['data'][:, 1] + 1j*mat['data'][:, 2]

    #     if self.showIt is True:

    #         plt.loglog(self.prob.survey.frequency, abs(Hz.real), 'b')
    #         plt.loglog(self.prob.survey.frequency, abs(Hzanal.real), 'b*')
    #         plt.loglog(self.prob.survey.frequency, abs(Hz.imag), 'r')
    #         plt.loglog(self.prob.survey.frequency, abs(Hzanal.imag), 'r*')
    #         plt.show()

    #     err = np.linalg.norm(Hz-Hzanal)/np.linalg.norm(Hzanal)
    #     self.assertTrue(err < 0.08)

    #     chi = np.array([0., 1., 0.], dtype=float)
    #     sig = np.array([0.01, 0.01, 0.01], dtype=float)
    #     self.prob.chi = chi

    #     m_1D = np.log(sig)
    #     Hz = self.prob.forward(m_1D)

    #     # 2. Verification for variable susceptibility
    #     mat = io.loadmat('em1dfm/VMD_3lay_chi.mat')
    #     freq = mat['data'][:, 0]
    #     Hzanal = mat['data'][:, 1] + 1j*mat['data'][:, 2]

    #     if self.showIt is True:

    #         plt.loglog(self.prob.survey.frequency, abs(Hz.real), 'b')
    #         plt.loglog(self.prob.survey.frequency, abs(Hzanal.real), 'b*')
    #         plt.loglog(self.prob.survey.frequency, abs(Hz.imag), 'r')
    #         plt.loglog(self.prob.survey.frequency, abs(Hzanal.imag), 'r*')
    #         plt.show()

    #     err = np.linalg.norm(Hz-Hzanal)/np.linalg.norm(Hzanal)
    #     self.assertTrue(err < 0.08)

    #     print ("EM1DFD comprison of UBC code works")


if __name__ == '__main__':
    unittest.main()
