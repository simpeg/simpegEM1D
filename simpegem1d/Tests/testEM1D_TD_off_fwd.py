import unittest
from SimPEG import *
import matplotlib.pyplot as plt
from simpegem1d import EM1D, EM1DAnal, BaseEM1D
from scipy import io

class EM1D_TD_FwdProblemTests(unittest.TestCase):

    def setUp(self):
        
        TDsurvey = BaseEM1D.EM1DSurveyTD()
        TDsurvey.rxLoc = np.array([0., 0., 100.+1e-5])
        TDsurvey.txLoc = np.array([0., 0., 100.+1e-5])
        TDsurvey.fieldtype = 'secondary'
        TDsurvey.rxType = 'Bz'
        TDsurvey.waveType = 'stepoff'

        nearthick = np.logspace(-1, 1, 5)
        deepthick = np.logspace(1, 2, 10)
        hx = np.r_[nearthick, deepthick]
        mesh1D = Mesh.TensorMesh([hx], [0.])
        depth = -mesh1D.gridN
        LocSigZ = -mesh1D.gridCC
        nlay = depth.size
        topo = np.r_[0., 0., 100.]
        TDsurvey.depth = depth
        TDsurvey.topo = topo
        TDsurvey.LocSigZ = LocSigZ
        TDsurvey.Setup1Dsystem()
        TDsurvey.time = np.logspace(-5, -2, 64)
        TDsurvey.setFrequency(TDsurvey.time)
        sig_half = 1e-2
        chi_half = 0.

        Logmodel = BaseEM1D.BaseEM1DModel(mesh1D)
        tau = 1e-3
        eta = 2e-1
        c = 1.
        options = {'Frequency': TDsurvey.frequency, 'tau': np.ones(nlay)*tau, 'eta':np.ones(nlay)*eta, 'c':np.ones(nlay)*c}
        Colemodel = BaseEM1D.BaseColeColeModel(mesh1D, **options)

        modelReal = Model.ComboModel(mesh1D, [Logmodel])
        modelComplex = Model.ComboModel(mesh1D, [Colemodel, Logmodel])                
        m_1D = np.log(np.ones(nlay)*sig_half)

        TDsurvey.rxType = 'Bz'        

        WT0 = np.load('../WT0.npy')
        WT1 = np.load('../WT1.npy')
        YBASE = np.load('../YBASE.npy')
        options = {'WT0': WT0, 'WT1': WT1, 'YBASE': YBASE}

        prob = EM1D.EM1D(modelReal, **options)
        prob.pair(TDsurvey)
        prob.chi = np.zeros(TDsurvey.nlay)

        self.survey = TDsurvey
        self.options = options
        self.modelComplex = modelComplex
        self.prob = prob
        self.mesh1D = mesh1D
        self.showIt = False
        self.tau = tau
        self.eta = eta
        self.c = c




    def test_EM1DTDfwd_CirLoop_RealCond(self):
        self.prob.CondType = 'Real'        
        self.prob.survey.txType = 'CircularLoop'
        self.prob.survey.offset = 10.
        sig_half = 0.01
        I = 1e0
        a = 1e1
        self.prob.survey.I = I
        self.prob.survey.a = a

        m_1D = np.log(np.ones(self.prob.survey.nlay)*sig_half)
        Hz = self.prob.fields(m_1D)
        BzTD = self.prob.survey.projectFields(u=Hz)
        Bzanal = EM1DAnal.BzAnalCircT(a, self.survey.time, sig_half)

        if self.showIt == True:

            plt.loglog(self.survey.time, (BzTD), 'b')
            plt.loglog(self.survey.time, (Bzanal), 'b*')
            plt.show()

        err = np.linalg.norm(BzTD-Bzanal)/np.linalg.norm(Bzanal)
        print 'Bz error = ', err
        self.assertTrue(err < 1e-2)

        self.survey.rxType = 'dBzdt'
        dBzdtTD = self.prob.survey.projectFields(u=Hz)
        dBzdtanal = EM1DAnal.dBzdtAnalCircT(a, self.survey.time, sig_half)

        if self.showIt == True:

            plt.loglog(self.survey.time, abs(dBzdtTD), 'b')
            plt.loglog(self.survey.time, abs(dBzdtanal), 'b*')
            plt.show()

        err = np.linalg.norm(dBzdtTD-dBzdtanal)/np.linalg.norm(dBzdtanal)
        print 'dBzdt error = ', err
        self.assertTrue(err < 1e-2)

        print "EM1DTD-CirculurLoop for real conductivity works"

    def test_EM1DTDfwd_CirLoop_ComplexCond(self):

        if self.prob.ispaired:
            self.prob.unpair()
        if self.survey.ispaired:
            self.survey.unpair()

        self.prob = EM1D.EM1D(self.modelComplex, **self.options)
        self.prob.chi = np.zeros(self.survey.nlay)
        self.prob.pair(self.survey)

        self.prob.CondType = 'Complex'        
        self.prob.survey.txType = 'CircularLoop'
        self.prob.survey.offset = 10.
        sig_half = 0.01
        I = 1e0
        a = 1e1
        self.prob.survey.I = I
        self.prob.survey.a = a

        

        m_1D = np.log(np.ones(self.prob.survey.nlay)*sig_half)
        Hz = self.prob.fields(m_1D)
        BzTD = self.prob.survey.projectFields(u=Hz)

        sigCole = EM1DAnal.ColeCole(self.survey.frequency, sig_half, self.eta, self.tau, self.c)
        Bzanal = EM1DAnal.BzAnalCircTCole(a, self.survey.time, sigCole)

        if self.showIt == True:

            plt.loglog(self.survey.time, abs(BzTD), 'b')
            plt.loglog(self.survey.time, abs(Bzanal), 'b*')
            plt.show()

        err = np.linalg.norm(BzTD-Bzanal)/np.linalg.norm(Bzanal)
        print 'Bz error = ', err        
        self.assertTrue(err < 1e-2)

        self.survey.rxType = 'dBzdt'
        dBzdtTD = self.prob.survey.projectFields(u=Hz)
        dBzdtanal = EM1DAnal.dBzdtAnalCircTCole(a, self.survey.time, sigCole)

        if self.showIt == True:

            plt.loglog(self.survey.time, abs(dBzdtTD), 'b')
            plt.loglog(self.survey.time, abs(dBzdtanal), 'b*')
            plt.show()

        err = np.linalg.norm(dBzdtTD-dBzdtanal)/np.linalg.norm(dBzdtanal)
        print 'dBzdt error = ', err
        self.assertTrue(err < 1e-2)        
        print "EM1DTD-CirculurLoop for Complex conductivity works"

if __name__ == '__main__':
    unittest.main()
