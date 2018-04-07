import unittest
import numpy as np
from SimPEG import Maps, Utils, Mesh
import matplotlib.pyplot as plt
from simpegem1d import EM1D, EM1DAnalytics, EM1DSurveyTD
from scipy import io


class EM1D_TD_FwdProblemTests(unittest.TestCase):

    def setUp(self):

        nearthick = np.logspace(-1, 1, 5)
        deepthick = np.logspace(1, 2, 10)
        hx = np.r_[nearthick, deepthick]
        mesh1D = Mesh.TensorMesh([hx], [0.])
        depth = -mesh1D.gridN[:-1]
        LocSigZ = -mesh1D.gridCC

        TDsurvey = EM1DSurveyTD(
            rx_location=np.array([0., 0., 100.+1e-5]),
            src_location=np.array([0., 0., 100.+1e-5]),
            topo=np.r_[0., 0., 100.],
            depth=depth,
            field_type='secondary',
            rx_type='Bz',
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
        prob = EM1D(mesh1D, sigmaMap=expmap)
        prob.pair(TDsurvey)
        prob.chi = np.zeros(TDsurvey.n_layer)

        self.survey = TDsurvey
        self.prob = prob
        self.mesh1D = mesh1D
        self.showIt = True
        self.tau = tau
        self.eta = eta
        self.c = c
        self.m_1D = m_1D
        self.sig_half = sig_half

    def test_EM1DTDfwd_CirLoop_RealCond(self):
        Hz = self.prob.fields(self.m_1D)
        print (Hz)
        BzTD = self.prob.survey.projectFields(u=Hz)
        Bzanal = EM1DAnalytics.BzAnalCircT(
            self.survey.a, self.survey.time, self.sig_half
        )

        # if self.showIt is True:

        #     plt.loglog(self.survey.time, (BzTD), 'b')
        #     plt.loglog(self.survey.time, (Bzanal), 'b*')
        #     plt.show()

        # err = np.linalg.norm(BzTD-Bzanal)/np.linalg.norm(Bzanal)
        # print ('Bz error = ', err)
        # self.assertTrue(err < 1e-2)

        self.survey.rxType = 'dBzdt'
        dBzdtTD = self.prob.survey.projectFields(u=Hz)
        dBzdtanal = EM1DAnalytics.dBzdtAnalCircT(a, self.survey.time, sig_half)

        if self.showIt is True:

            plt.loglog(self.survey.time, abs(dBzdtTD), 'k-')
            plt.loglog(self.survey.time, abs(dBzdtanal), 'b*')
            plt.show()

        err = np.linalg.norm(dBzdtTD-dBzdtanal)/np.linalg.norm(dBzdtanal)
        print ('dBzdt error = ', err)
        self.assertTrue(err < 1e-2)

        print ("EM1DTD-CirculurLoop for real conductivity works")

    # def test_EM1DTDfwd_CirLoop_ComplexCond(self):

    #     if self.prob.ispaired:
    #         self.prob.unpair()
    #     if self.survey.ispaired:
    #         self.survey.unpair()

    #     self.prob = EM1D.EM1D(self.mesh1D, self.modelComplex, **self.options)
    #     self.prob.chi = np.zeros(self.survey.n_layer)
    #     self.prob.pair(self.survey)

    #     self.prob.CondType = 'Complex'
    #     self.prob.survey.srcType = 'CircularLoop'
    #     self.prob.survey.offset = 10.
    #     sig_half = 0.01
    #     I = 1e0
    #     a = 1e1
    #     self.prob.survey.I = I
    #     self.prob.survey.a = a



    #     m_1D = np.log(np.ones(self.prob.survey.n_layer)*sig_half)
    #     Hz = self.prob.fields(m_1D)
    #     BzTD = self.prob.survey.projectFields(u=Hz)

    #     sigCole = EM1DAnalytics.ColeCole(self.survey.frequency, sig_half, self.eta, self.tau, self.c)
    #     Bzanal = EM1DAnalytics.BzAnalCircTCole(a, self.survey.time, sigCole)

    #     if self.showIt == True:

    #         plt.loglog(self.survey.time, abs(BzTD), 'b')
    #         plt.loglog(self.survey.time, abs(Bzanal), 'b*')
    #         plt.show()

    #     err = np.linalg.norm(BzTD-Bzanal)/np.linalg.norm(Bzanal)
    #     print 'Bz error = ', err
    #     self.assertTrue(err < 1e-2)

    #     self.survey.rxType = 'dBzdt'
    #     dBzdtTD = self.prob.survey.projectFields(u=Hz)
    #     dBzdtanal = EM1DAnalytics.dBzdtAnalCircTCole(a, self.survey.time, sigCole)

    #     if self.showIt == True:

    #         plt.loglog(self.survey.time, abs(dBzdtTD), 'b')
    #         plt.loglog(self.survey.time, abs(dBzdtanal), 'b*')
    #         plt.show()

    #     err = np.linalg.norm(dBzdtTD-dBzdtanal)/np.linalg.norm(dBzdtanal)
    #     print 'dBzdt error = ', err
    #     self.assertTrue(err < 1e-2)
    #     print "EM1DTD-CirculurLoop for Complex conductivity works"

if __name__ == '__main__':
    unittest.main()
