from SimPEG import *
import BaseEM1D
# from future import division
from scipy.constants import mu_0
from Kernels import HzKernel_layer, HzkernelCirc_layer
from DigFilter import EvalDigitalFilt
from RTEfun import rTEfun

class EM1D(Problem.BaseProblem):
    """
        Pseudo analytic solutions for frequency and time domain EM problems assuming
        Layered earth (1D).
    """
    surveyPair = BaseEM1D.BaseEM1DSurvey
    modelPair = BaseEM1D.BaseEM1DModel
    # modelPair = Model.BaseModel
    CondType = 'Real'
    WT1 = None
    WT0 = None
    YBASE = None
    chi = None

    def __init__(self, model, **kwargs):

        Problem.BaseProblem.__init__(self, model, **kwargs)
        self.WT0 = kwargs['WT0']
        self.WT1 = kwargs['WT1']
        self.YBASE = kwargs['YBASE']

    @Utils.requires('survey')
    def fields(self, m):
        """
            Return Bz or dBzdt

            .. math ::

        """

        f = self.survey.frequency
        nfreq = self.survey.Nfreq
        flag = self.survey.fieldtype
        r = self.survey.offset
        sig = self.model.transform(m)
        #TODO: In corporate suseptibility in to the model !!
        chi = self.chi
        nlay = self.survey.nlay
        depth = self.survey.depth
        h = self.survey.h
        z = self.survey.z
        HzFHT = np.zeros(nfreq, dtype = complex)
        if self.CondType == 'Real':
            if self.survey.txType == 'VMD':
                r = self.survey.offset
                for ifreq in range(nfreq):
                    kernel = lambda x: HzKernel_layer(x, f[ifreq], nlay, sig, chi, depth, h, z, flag)
                    HzFHT[ifreq] = EvalDigitalFilt(self.YBASE, self.WT0, kernel, r)

            elif self.survey.txType == 'CircularLoop':
                I = self.survey.I
                a = self.survey.a
                for ifreq in range(nfreq):
                    kernel = lambda x: HzkernelCirc_layer(x, f[ifreq], nlay, sig, chi, depth, h, z, I, a, flag)
                    HzFHT[ifreq] = EvalDigitalFilt(self.YBASE, self.WT1, kernel, a)

        elif self.CondType == 'Complex':
            sig_temp = np.zeros(self.survey.nlay, dtype = complex)
            if self.survey.txType == 'VMD':
                r = self.survey.offset
                for ifreq in range(nfreq):
                    sig_temp = Utils.mkvc(sig[ifreq, :])
                    kernel = lambda x: HzKernel_layer(x, f[ifreq], nlay, sig_temp, chi, depth, h, z, flag)
                    HzFHT[ifreq] = EvalDigitalFilt(self.YBASE, self.WT0, kernel, r)

            elif self.survey.txType == 'CircularLoop':
                I = self.survey.I
                a = self.survey.a
                for ifreq in range(nfreq):
                    sig_temp = Utils.mkvc(sig[ifreq, :])
                    kernel = lambda x: HzkernelCirc_layer(x, f[ifreq], nlay, sig_temp, chi, depth, h, z, I, a, flag)
                    HzFHT[ifreq] = EvalDigitalFilt(self.YBASE, self.WT1, kernel, a)
        else :

            raise Exception("Not implemented!!")

        return  HzFHT


    @Utils.timeIt
    def Jvec(self, m, v, u=None):
        """
            Computing Jacobian multiplied by vector

        """
        pass
    @Utils.timeIt
    def Jtvec(self, m, v, u=None):
        """
            Computing Jacobian^T multiplied by vector.

        """
        pass


if __name__ == '__main__':
    # hx = np.ones(10)
    # M = Mesh.TensorMesh([hx])

    # model = Model.LogModel(M)
    # prob = EM1D(M)

    test1 = np.load('WT1.npy')
    test2 = np.load('WT0.npy')
