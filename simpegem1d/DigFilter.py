import numpy as np
from scipy.constants import mu_0

def EvalDigitalFilt(base, weight, fun, r):
    """
        Evaluating Digital filtering based on given base and weight

    """
    return  np.dot(weight, fun(base/r))/r

