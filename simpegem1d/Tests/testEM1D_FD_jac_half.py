import unittest
from SimPEG import *
import matplotlib.pyplot as plt
from simpegem1d import EM1D, EM1DAnal, BaseEM1D, DigFilter
import numpy as np

class EM1D_FD_Jac_half_ProblemTests(unittest.TestCase):

    def setUp(self):

        FDsurvey = BaseEM1D.EM1DSurveyFD()
        FDsurvey.rxLoc = np.array([0., 0., 100.+1e-5])
        FDsurvey.srcLoc = np.array([0., 0., 100.+1e-5])
        FDsurvey.fieldtype = 'secondary'

        hx = np.r_[100.]

        mesh1D = Mesh.TensorMesh([hx], [0.])
        depth = -mesh1D.gridN[:-1]
        LocSigZ = -mesh1D.gridCC
        nlay = depth.size
        topo = np.r_[0., 0., 100.]
        FDsurvey.depth = depth
        FDsurvey.topo = topo
        FDsurvey.LocSigZ = LocSigZ

        FDsurvey.frequency = np.logspace(2, 4, 10)
        FDsurvey.Nfreq = FDsurvey.frequency.size
        FDsurvey.HalfSwitch = True
        FDsurvey.Setup1Dsystem()
        sig_half = 1e-1
        chi_half = 0.

        expmap = BaseEM1D.BaseEM1DMap(mesh1D)
        tau = 1e-3
        eta = 2e-1
        c = 1.
        options = {'Frequency': FDsurvey.frequency, 'tau': np.ones(nlay)*tau, 'eta':np.ones(nlay)*eta, 'c':np.ones(nlay)*c}
        colemap = BaseEM1D.BaseColeColeMap(mesh1D, **options)

        modelReal = expmap
        modelComplex = colemap * expmap
        m_1D = np.log(np.ones(nlay)*sig_half)

        FDsurvey.rxType = 'Hz'
        FDsurvey.switchRI = 'all'

        WT0, WT1, YBASE = DigFilter.LoadWeights()
        options = {'WT0': WT0, 'WT1': WT1, 'YBASE': YBASE}

        prob = EM1D.EM1D(mesh1D, modelReal, **options)
        prob.pair(FDsurvey)
        prob.chi = np.zeros(FDsurvey.nlay)


        self.survey = FDsurvey
        self.options = options
        self.modelReal = modelReal
        self.prob = prob
        self.mesh1D = mesh1D
        self.showIt = False


    def test_EM1DFDJvec_Half(self):
        self.prob.CondType = 'Real'
        self.prob.survey.srcType = 'CircularLoop'

        I = 1e0
        a = 1e1
        self.prob.survey.I = I
        self.prob.survey.a = a

        sig_half = np.r_[0.01]
        m_1D = np.log(np.ones(self.prob.survey.nlay)*sig_half)
        self.prob.jacSwitch = True

        Hz, dHzdsig = self.prob.fields(m_1D)
        dHzdsig = Utils.mkvc(dHzdsig)
        dHzdsiganal = EM1DAnal.dHzdsiganalCirc(sig_half, self.prob.survey.frequency, I, a, 'secondary')

        def fwdfun(m):
            self.prob.jacSwitch = False
            Hz = self.prob.fields(m)
            resp = self.prob.survey.projectFields(u=Hz)
            return resp

            # return Hz

        def jacfun(m, dm):
            self.prob.jacSwitch = True
            u = self.prob.fields(m)
            drespdmv = self.prob.Jvec(m, dm, u = u)
            return drespdmv


        if self.showIt == True:
            plt.loglog(self.prob.survey.frequency, abs(dHzdsig.imag), 'r')
            plt.loglog(self.prob.survey.frequency, abs(dHzdsig.real), 'b')
            plt.loglog(self.prob.survey.frequency, abs(dHzdsiganal.imag), 'r*')
            plt.loglog(self.prob.survey.frequency, abs(dHzdsiganal.real), 'b*')
            plt.show()

        dm = m_1D*0.1
        derChk = lambda m: [fwdfun(m), lambda mx: jacfun(m, mx)]
        passed = Tests.checkDerivative(derChk, m_1D, num=4, dx = dm, plotIt=False, eps = 1e-15)
        if passed:
            print "EM1DFD-half Jvec works"


    def test_EM1DFDJtvec_Half(self):
        self.prob.CondType = 'Real'
        self.prob.survey.srcType = 'CircularLoop'

        I = 1e0
        a = 1e1
        self.prob.survey.I = I
        self.prob.survey.a = a

        sig_half = np.r_[0.01]
        self.prob.jacSwitch = False
        m_true = np.log(np.ones(self.prob.survey.nlay)*sig_half)
        Hz_true = self.prob.fields(m_true)
        dobs = self.prob.survey.projectFields(u=Hz_true)

        m_ini  = np.log(np.ones(self.prob.survey.nlay)*sig_half*10)
        Hz_ini = self.prob.fields(m_ini)
        resp_ini = self.prob.survey.projectFields(u=Hz_ini)
        dr = resp_ini-dobs


        def misfit(m, dobs):
            self.prob.jacSwitch = True
            Hz = self.prob.fields(m)
            dpred = self.survey.dpred(m, u = Hz)
            misfit = 0.5*np.linalg.norm(dpred-dobs)**2
            dmisfit = self.prob.Jtvec(m, dr, u = Hz)
            return misfit, dmisfit

        derChk = lambda m: misfit(m, dobs)
        passed = Tests.checkDerivative(derChk, m_ini, num=4, plotIt=False, eps = 1e-23)
        self.assertTrue(passed)
        if passed:
            print "EM1DFD-half Jtvec works"

if __name__ == '__main__':
    unittest.main()
_
