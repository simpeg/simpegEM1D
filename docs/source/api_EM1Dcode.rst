.. _api_EM1Dcode:


AirEM1D code
============

Here, we used SimPEG's frame work so that we have following modules:


* Problem: EM1D
    a. Computing reflection coefficients
    b. Evaluating Hankel transform with Digital filtering

* Survey: BaseEM1Dsurvey
    a. EM1DTDsurvey
    b. EM1DTDsurvey


EM1D problem
************

.. autoclass:: simpegem1d.EM1D.EM1D
    :show-inheritance:
    :members:
    :undoc-members:
    :inherited-members:

Computing reflection coefficients
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. automodule:: simpegem1d.RTEfun
    :show-inheritance:
    :members:
    :undoc-members:
    :inherited-members:


Digital filtering
^^^^^^^^^^^^^^^^^

.. automodule:: simpegem1d.DigFilter
    :show-inheritance:
    :members:
    :undoc-members:
    :inherited-members:

Tx Waveform
^^^^^^^^^^^^^^^^^^^^

.. automodule:: simpegem1d.Waveform
    :show-inheritance:
    :members:
    :undoc-members:
    :inherited-members:

EM1D survey
***********

.. autoclass:: simpegem1d.BaseEM1D.BaseEM1DSurvey
    :members:
    :undoc-members:

Frequency domain survey
^^^^^^^^^^^^^^^^^^^^^^^
.. autoclass:: simpegem1d.BaseEM1D.EM1DSurveyFD
    :members:
    :undoc-members:    

Time domain survey
^^^^^^^^^^^^^^^^^^
.. autoclass:: simpegem1d.BaseEM1D.EM1DSurveyTD
    :members:
    :undoc-members:

EM1D analaytic solutions
************************

Some terminologies
^^^^^^^^^^^^^^^^^^
    
    Here we provide analytic solutions, which are related to airborn EM problems. We have two types of sources:

    * Vertical magnetic dipole (VMD)
    * Circular loop with horizontal coplanar (HCP) geometry

    Dervations of these solutions are extracted from [#ref1]_.
        
        * \\(\a\\): Tx-loop radius
        * \\(\I\\): Current intensity
        * \\(\\sigma\\): conductivity [S/m]
        * \\(\r\\): Tx-Rx offset
        * \\(\m\\): magnetic dipole moment
        * \\(\k\\): propagation constant            

    .. math::

        k = \omega^2\epsilon\mu - \imath\omega\mu\sigma

        \theta = \sqrt{\frac{\sigma\mu}{4t}}

EM1DAnal
^^^^^^^^
.. automodule:: simpegem1d.EM1DAnal
    :show-inheritance:
    :members:
    :undoc-members:
    :inherited-members:

References
**********

    .. [#ref1] Ward and Hohmann, Electromagnetic Theory for Geophysical Applications, p131-311: `Link <http://library.seg.org/doi/abs/10.1190/1.9781560802631.ch4>`_
    .. [#ref2] ??

