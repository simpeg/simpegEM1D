import unittest
from SimPEG import *
import matplotlib.pyplot as plt
from simpegem1d import EM1D, EM1DAnal, BaseEM1D


class EM1D_FD_JacProblemTests(unittest.TestCase):

    def setUp(self):
        
        FDsurvey = BaseEM1D.EM1DSurveyFD()
        FDsurvey.rxLoc = np.array([0., 0., 100.+1e-5])
        FDsurvey.txLoc = np.array([0., 0., 100.+1e-5])
        FDsurvey.fieldtype = 'secondary'

        nearthick = np.logspace(-1, 1, 5)
        deepthick = np.logspace(1, 2, 10)
        hx = np.r_[nearthick, deepthick]
        mesh1D = Mesh.TensorMesh([hx], [0.])
        depth = -mesh1D.gridN
        LocSigZ = -mesh1D.gridCC
        nlay = depth.size
        topo = np.r_[0., 0., 100.]
        FDsurvey.depth = depth
        FDsurvey.topo = topo
        FDsurvey.LocSigZ = LocSigZ

        FDsurvey.frequency = np.logspace(1, 8, 61)
        FDsurvey.Nfreq = FDsurvey.frequency.size
        FDsurvey.Setup1Dsystem()
        sig_half = 1e-2
        chi_half = 0.

        Logmodel = BaseEM1D.BaseEM1DModel(mesh1D)
        tau = 1e-3
        eta = 2e-1
        c = 1.
        options = {'Frequency': FDsurvey.frequency, 'tau': np.ones(nlay)*tau, 'eta':np.ones(nlay)*eta, 'c':np.ones(nlay)*c}
        Colemodel = BaseEM1D.BaseColeColeModel(mesh1D, **options)

        modelReal = Model.ComboModel(mesh1D, [Logmodel])
        modelComplex = Model.ComboModel(mesh1D, [Colemodel, Logmodel])                
        m_1D = np.log(np.ones(nlay)*sig_half)

        FDsurvey.rxType = 'Hz'        

        WT0 = np.load('../WT0.npy')
        WT1 = np.load('../WT1.npy')
        YBASE = np.load('../YBASE.npy')
        options = {'WT0': WT0, 'WT1': WT1, 'YBASE': YBASE}

        prob = EM1D.EM1D(modelReal, **options)
        prob.pair(FDsurvey)
        prob.chi = np.zeros(FDsurvey.nlay)

        self.survey = FDsurvey
        self.options = options
        self.modelComplex = modelComplex
        self.prob = prob
        self.mesh1D = mesh1D
        self.showIt = False
        self.tau = tau
        self.eta = eta
        self.c = c


    def test_EM1DFDfwd_VMD_RealCond(self):
        self.prob.CondType = 'Real'        
        self.prob.survey.txType = 'VMD'
        self.prob.survey.offset = 10.
        sig_half = 0.01
        m_1D = np.log(np.ones(self.prob.survey.nlay)*sig_half)
        Hz, dHzdsig = self.prob.fields(m_1D)
        Hzanal = EM1DAnal.Hzanal(sig_half, self.prob.survey.frequency, self.prob.survey.offset, 'secondary')

        if self.showIt == True:

            plt.loglog(self.prob.survey.frequency, abs(Hz.real), 'b')
            plt.loglog(self.prob.survey.frequency, abs(Hzanal.real), 'b*')
            plt.loglog(self.prob.survey.frequency, abs(Hz.imag), 'r')
            plt.loglog(self.prob.survey.frequency, abs(Hzanal.imag), 'r*')
            plt.show()
        plt.loglog(self.prob.survey.frequency, abs(dHzdsig[1,:].imag), 'r')
        plt.loglog(self.prob.survey.frequency, abs(dHzdsig[1,:].real), 'b')
        plt.show()
        # print dHzdsig[2,:].imag
        err = np.linalg.norm(Hz-Hzanal)/np.linalg.norm(Hzanal)
        self.assertTrue(err < 1e-5)
        print "EM1DFD-VMD for real conductivity works"
        # self.prob.Jvec(m_1D, 0., 0.)




 


if __name__ == '__main__':
    unittest.main()
_