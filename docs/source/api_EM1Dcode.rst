.. _api_EM1Dcode:


simpegEM1D code
===============

Here, we used SimPEG's frame work so that we have following modules:


* Problem: EM1D
* Survey: BaseEM1Dsurvey
* Mapping: BaseEM1Dmap

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

Source Waveform
^^^^^^^^^^^^^^^^^^^^

.. automodule:: simpegem1d.Waveform
    :show-inheritance:
    :members:
    :undoc-members:
    :inherited-members:

EM1D survey
***********

.. autoclass:: simpegem1d.BaseEM1D.BaseEM1DSurvey
    :show-inheritance:
    :members:
    :inherited-members:

Frequency domain survey
^^^^^^^^^^^^^^^^^^^^^^^
.. autoclass:: simpegem1d.BaseEM1D.EM1DSurveyFD
    :show-inheritance:
    :members:

Time domain survey
^^^^^^^^^^^^^^^^^^
.. autoclass:: simpegem1d.BaseEM1D.EM1DSurveyTD
    :show-inheritance:
    :members:


EM1D analaytic solutions
************************

.. automodule:: simpegem1d.EM1DAnal
    :show-inheritance:
    :members:
    :undoc-members:
    :inherited-members:


