from SimPEG import *
import BaseEM1D
from scipy.constants import mu_0
from Kernels import HzKernel_layer, HzkernelCirc_layer
from RTEfun import rTEfun

class EM1D(Problem.BaseProblem):
    """
        Pseudo analytic solutions for frequency and time domain EM problems assuming
        Layered earth (1D).
    """

    surveyPair = BaseEM1D.BaseEM1DSurvey
    modelPair = BaseEM1D.BaseEM1DModel
    

    @Utils.requires('survey')
    def __init__(self, model, **kwargs):
        Problem.BaseProblem.__init__(self, model, **kwargs)
        self.WT1 = np.load('WT0.npy') 
        self.WT0 = np.load('WT1.npy') 
        self.YBASE = np.load('YBASE.npy') 


    def fields(self, m):
        """
            Return Bz or dBzdt

            .. math ::

        """
        
        f = self.survey.frequency
        nfreq = self.survey.Nfreq
        flag = self.survey.fieldtype
        r = self.offset
        
        field = []

        for isd in range(self.survey.Nsd):
            HzFHT = np.zeros(nfreq, dtype = complex)

            if self.survey.txType == 'VMD':
                for ifreq in range(nfreq):
                    kernel = lambda x: HzKernel_layer(x, f[ifreq], nlay, m[isd], chi[isd], depth, h[isd], z[isd], flag)
                    HzFHT[ifreq] = EvalDigitalFilt(YBASE, WT0, kernel, r)

            elif self.survey.txType == 'CircularLoop':
                for ifreq in range(nfreq):            
                    kernel = lambda x: HzkernelCirc_layer(x, f[ifreq], nlay, m[isd], chi[isd], depth, h[isd], z[isd], I, a, flag)
                    HzFHT[ifreq] = EvalDigitalFilt(YBASE, WT1, kernel, a)
            field.append(HzFHT)                
        
        return  field


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
    import matplotlib.pyplot as plt
