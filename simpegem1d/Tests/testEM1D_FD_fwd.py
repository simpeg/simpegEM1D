import unittest
from SimPEG import *
import matplotlib.pyplot as plt
from simpegem1d import EM1D, EM1DAnal, BaseEM1D, DigFilter


class EM1D_FD_FwdProblemTests(unittest.TestCase):

    def setUp(self):

        FDsurvey = BaseEM1D.EM1DSurveyFD()
        FDsurvey.rxLoc = np.array([0., 0., 100.+1e-5])
        FDsurvey.srcLoc = np.array([0., 0., 100.+1e-5])
        FDsurvey.fieldtype = 'secondary'

        nearthick = np.logspace(-1, 1, 5)
        deepthick = np.logspace(1, 2, 10)
        hx = np.r_[nearthick, deepthick]
        mesh1D = Mesh.TensorMesh([hx], [0.])
        depth = -mesh1D.gridN[:-1]
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

        expmap = BaseEM1D.BaseEM1DMap(mesh1D)
        tau = 1e-3
        eta = 2e-1
        c = 1.
        options = {'Frequency': FDsurvey.frequency, 'tau': np.ones(nlay)*tau, 'eta':np.ones(nlay)*eta, 'c':np.ones(nlay)*c}
        colemap = BaseEM1D.BaseColeColeMap(mesh1D, **options)

        modelReal = expmap
        modelComplex = colemap*expmap
        m_1D = np.log(np.ones(nlay)*sig_half)
        FDsurvey.rxType = 'Hz'

        WT0, WT1, YBASE = DigFilter.LoadWeights()
        options = {'WT0': WT0, 'WT1': WT1, 'YBASE': YBASE}

        prob = EM1D.EM1D(mesh1D, mapping = modelReal, **options)
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
        self.prob.survey.srcType = 'VMD'
        self.prob.survey.offset = 10.
        self.prob.survey.SetOffset()
        sig_half = 0.01
        m_1D = np.log(np.ones(self.prob.survey.nlay)*sig_half)
        Hz = self.prob.fields(m_1D)
        Hzanal = EM1DAnal.Hzanal(sig_half, self.prob.survey.frequency, self.prob.survey.offset, 'secondary')

        if self.showIt == True:

            plt.loglog(self.prob.survey.frequency, abs(Hz.real), 'b')
            plt.loglog(self.prob.survey.frequency, abs(Hzanal.real), 'b*')
            plt.loglog(self.prob.survey.frequency, abs(Hz.imag), 'r')
            plt.loglog(self.prob.survey.frequency, abs(Hzanal.imag), 'r*')
            plt.show()

        err = np.linalg.norm(Hz-Hzanal)/np.linalg.norm(Hzanal)
        self.assertTrue(err < 1e-5)
        print "EM1DFD-VMD for real conductivity works"


    def test_EM1DFDfwd_VMD_ComplexCond(self):

        if self.prob.ispaired:
            self.prob.unpair()
        if self.survey.ispaired:
            self.survey.unpair()

        self.prob = EM1D.EM1D(self.mesh1D, mapping = self.modelComplex, **self.options)
        self.prob.chi = np.zeros(self.survey.nlay)
        self.prob.pair(self.survey)

        self.prob.CondType = 'Complex'
        self.prob.survey.srcType = 'VMD'
        self.prob.survey.offset = 10.
        self.prob.survey.SetOffset()
        sig_half = 0.01
        m_1D = np.log(np.ones(self.prob.survey.nlay)*sig_half)
        Hz = self.prob.fields(m_1D)
        sigCole = EM1DAnal.ColeCole(self.survey.frequency, sig_half, self.eta, self.tau, self.c)
        Hzanal = EM1DAnal.Hzanal(sigCole, self.prob.survey.frequency, self.prob.survey.offset, 'secondary')

        if self.showIt == True:

            plt.loglog(self.prob.survey.frequency, abs(Hz.real), 'b')
            plt.loglog(self.prob.survey.frequency, abs(Hzanal.real), 'b*')
            plt.loglog(self.prob.survey.frequency, abs(Hz.imag), 'r')
            plt.loglog(self.prob.survey.frequency, abs(Hzanal.imag), 'r*')
            plt.show()

        err = np.linalg.norm(Hz-Hzanal)/np.linalg.norm(Hzanal)
        self.assertTrue(err < 1e-5)
        print "EM1DFD-VMD for complex conductivity works"

    def test_EM1DFDfwd_CircularLoop_RealCond(self):
        self.prob.CondType = 'Real'
        self.prob.survey.srcType = 'CircularLoop'
        I = 1e0
        a = 1e1
        self.prob.survey.I = I
        self.prob.survey.a = a
        sig_half = 0.01
        m_1D = np.log(np.ones(self.prob.survey.nlay)*sig_half)
        Hz = self.prob.fields(m_1D)
        Hzanal = EM1DAnal.HzanalCirc(sig_half, self.prob.survey.frequency, I, a, 'secondary')

        if self.showIt == True:

            plt.loglog(self.prob.survey.frequency, abs(Hz.real), 'b')
            plt.loglog(self.prob.survey.frequency, abs(Hzanal.real), 'b*')
            plt.loglog(self.prob.survey.frequency, abs(Hz.imag), 'r')
            plt.loglog(self.prob.survey.frequency, abs(Hzanal.imag), 'r*')
            plt.show()

        err = np.linalg.norm(Hz-Hzanal)/np.linalg.norm(Hzanal)
        self.assertTrue(err < 1e-5)
        print "EM1DFD-CircularLoop for real conductivity works"

    def test_EM1DFDfwd_CircularLoop_ComplexCond(self):

        if self.prob.ispaired:
            self.prob.unpair()
        if self.survey.ispaired:
            self.survey.unpair()

        self.prob = EM1D.EM1D(self.mesh1D, self.modelComplex, **self.options)
        self.prob.chi = np.zeros(self.survey.nlay)
        self.prob.pair(self.survey)

        self.prob.CondType = 'Complex'
        self.prob.survey.srcType = 'CircularLoop'
        I = 1e0
        a = 1e1
        self.prob.survey.I = I
        self.prob.survey.a = a

        sig_half = 0.01
        m_1D = np.log(np.ones(self.prob.survey.nlay)*sig_half)
        Hz = self.prob.fields(m_1D)
        sigCole = EM1DAnal.ColeCole(self.survey.frequency, sig_half, self.eta, self.tau, self.c)
        Hzanal = EM1DAnal.HzanalCirc(sigCole, self.prob.survey.frequency, I, a, 'secondary')

        if self.showIt == True:

            plt.loglog(self.prob.survey.frequency, abs(Hz.real), 'b')
            plt.loglog(self.prob.survey.frequency, abs(Hzanal.real), 'b*')
            plt.loglog(self.prob.survey.frequency, abs(Hz.imag), 'r')
            plt.loglog(self.prob.survey.frequency, abs(Hzanal.imag), 'r*')
            plt.show()

        err = np.linalg.norm(Hz-Hzanal)/np.linalg.norm(Hzanal)
        self.assertTrue(err < 1e-5)
        print "EM1DFD-CircularLoop for complex conductivity works"

    def test_EM1DFDfwd_VMD_EM1D_sigchi(self):


        self.survey.rxLoc = np.array([0., 0., 110.+1e-5])
        self.survey.srcLoc = np.array([0., 0., 110.+1e-5])
        self.survey.fieldtype = 'secondary'

        hx = np.r_[np.ones(3)*10]
        mesh1D = Mesh.TensorMesh([hx], [0.])
        depth = -mesh1D.gridN[:-1]
        LocSigZ = -mesh1D.gridCC
        nlay = depth.size
        topo = np.r_[0., 0., 100.]

        self.survey.depth = depth
        self.survey.topo = topo
        self.survey.LocSigZ = LocSigZ

        self.survey.frequency = np.logspace(-3, 5, 61)
        self.survey.Nfreq = self.survey.frequency.size
        self.survey.Setup1Dsystem()
        self.prob.unpair()
        mapping = BaseEM1D.BaseEM1DMap(mesh1D)
        self.prob = EM1D.EM1D(mesh1D, mapping = mapping, **self.options)
        self.prob.pair(self.survey)
        self.prob.CondType = 'Real'
        self.prob.survey.srcType = 'VMD'
        self.prob.survey.offset = 10.
        self.prob.survey.SetOffset()

        # 1. Verification for variable conductivity
        chi = np.array([0., 0., 0.], dtype=complex)
        sig = np.array([0.01, 0.1, 0.01], dtype=complex)
        self.prob.chi = chi

        m_1D = np.log(sig)
        Hz = self.prob.fields(m_1D)
        from scipy import io
        mat = io.loadmat('em1dfm/VMD_3lay.mat')
        freq = mat['data'][:,0]
        Hzanal = mat['data'][:,1] + 1j*mat['data'][:,2]

        if self.showIt == True:

            plt.loglog(self.prob.survey.frequency, abs(Hz.real), 'b')
            plt.loglog(self.prob.survey.frequency, abs(Hzanal.real), 'b*')
            plt.loglog(self.prob.survey.frequency, abs(Hz.imag), 'r')
            plt.loglog(self.prob.survey.frequency, abs(Hzanal.imag), 'r*')
            plt.show()

        err = np.linalg.norm(Hz-Hzanal)/np.linalg.norm(Hzanal)
        self.assertTrue(err < 0.08)

        chi = np.array([0., 1., 0.], dtype=complex)
        sig = np.array([0.01, 0.01, 0.01], dtype=complex)
        self.prob.chi = chi

        m_1D = np.log(sig)
        Hz = self.prob.fields(m_1D)

        # 2. Verification for variable susceptibility
        mat = io.loadmat('em1dfm/VMD_3lay_chi.mat')
        freq = mat['data'][:,0]
        Hzanal = mat['data'][:,1] + 1j*mat['data'][:,2]

        if self.showIt == True:

            plt.loglog(self.prob.survey.frequency, abs(Hz.real), 'b')
            plt.loglog(self.prob.survey.frequency, abs(Hzanal.real), 'b*')
            plt.loglog(self.prob.survey.frequency, abs(Hz.imag), 'r')
            plt.loglog(self.prob.survey.frequency, abs(Hzanal.imag), 'r*')
            plt.show()

        err = np.linalg.norm(Hz-Hzanal)/np.linalg.norm(Hzanal)
        self.assertTrue(err < 0.08)

        print "EM1DFD comprison of UBC code works"


if __name__ == '__main__':
    unittest.main()
