from SimPEG import Model, Survey, Utils, np, sp
from scipy.constants import mu_0


class BaseEM1DSurvey(Survey.BaseSurvey):
    """Base EM1D Survey"""

    rxLoc = None #: receiver locations
    rxType = None #: receiver type

    def __init__(self, **kwargs):
        Survey.BaseSurvey.__init__(self, **kwargs)


    @property
    def Qcc(self):
        if getattr(self, '_Qcc', None) is None:
            self._Qcc = self.prob.mesh.getInterpolationMat(self.rxLoc,'CC')
        return self._Qcc

    def projectFields(self, u):
        """
            This function projects the fields onto the data space.

        """
        pass
        # return

    @Utils.count
    def projectFieldsDeriv(self, B):
        """
            This function projects the fields onto the data space.

            .. math::

                \\frac{\partial d_\\text{pred}}{\partial \mathbf{B}} = \mathbf{P}
        """
        pass
        # return

class EM1DSurveyTD(object):
    """docstring for EM1DSurveyTD"""
    def __init__(self, **kwargs):
        Survey.BaseData.__init__(self, **kwargs)

    def projectFields(self, B):

        pass
        # return



class BaseEM1DModel(Model.BaseModel):
    """BaseEM1DModel"""

    def __init__(self, mesh, **kwargs):
        Model.BaseModel.__init__(self, mesh)

    def transform(self, m):
        pass
        # return

    def transformDeriv(self, m):
        pass
        # return

