.. _api_EM1D:

.. math::

    \renewcommand{\div}{\nabla\cdot\,}
    \newcommand{\grad}{\vec \nabla}
    \newcommand{\curl}{{\vec \nabla}\times\,}
    \newcommand {\J}{{\vec J}}
    \renewcommand{\H}{{\vec H}}
    \renewcommand{\B}{{\vec B}}
    \renewcommand{\dBdt}{{\frac{d\vec B}{dt}}}
    \newcommand {\E}{{\vec E}}
    \renewcommand{\Re}{\mathsf{Re}}
    \renewcommand{\Im}{\mathsf{Im}}
    \newcommand{\bon}{b^{on}(t)}
    \newcommand{\bp}{b^{p}}
    \newcommand{\dbondt}{\frac{db^{on}(t)}{dt}}
    \newcommand{\dfdt}{\frac{df(t)}{dt}}
    \newcommand{\dfdtdsiginf}{\frac{\partial\frac{df(t)}{dt}}{\partial\siginf}}
    \newcommand{\dfdsiginf}{\frac{\partial f(t)}{\partial\siginf}}
    \newcommand{\dbgdsiginf}{\frac{\partial b^g(t)}{\partial\siginf}}
    \newcommand{\digint}{\frac{2}{\pi}\int_0^{\infty}}

AirEM1D
*******

Motivation
==========

Airborne Electromagnetic (EM) methods in geophysical applications has been successfully applied for several decades to map interesting geological structure of the earth in large scale. A natural way to categorize this airborne EM methods might be frequency domain and time domain EM systems. Famous frequency domain systems are `DIGEHM and RESOLVE <http://www.cgg.com/default.aspx?cid=7739&lang=1>`_  of CGG; time domain systems are `VTEM <http://www.geotech.ca/vtem>`_ of Geotech and `AeroTEM <http://www.aeroquestairborne.com/AeroTEM>`_ of Aeroquest. Each instrument has its own advantage and disadvantage depends on purposes of geophysical survey so that indentifying those are crucial for successful geophysical application.

One of the most used interpretation tools of these airborne EM data is 1D inversion, which assumes the earth structure as layers. Since we can derive solutions for this case pseudo-analytically, this can be evaluated relatively fast compared to solving differential equations in 2D or 3D. Therefore, this is really useful tool that we can use for first order survey design and interpretation in reaility. Furthermore, this is really nice education tool for students who are studying geophysics, since they can play with EM responses by manipulating conductivity or susceptibiltiy of the layered earth. While they are playing with this tool, if they want to recognize EM responses more seriously, then they can see how we derived these responses.

However, although it has been more than ten years since these tools were developed, as far as I know, there are no avaiable open source, modular, well-documented 1D EM forward modeling and inversion program that we can use for airborn EM applications. Therefore, here, we try to make this program applicable for both

* Practical applications for most airborne EM system (real data inversion)
* Education tools (easy implentation and well-documented)

In order to satisfy those components, first we derive solutions of frequency and time domain EM problems, and develop some modules that we can compute forward EM responses. We use SimPEG's frame work, to make this algorithm modular. Next, we apply inversion frame in SimPEG to our forward problem.


Forward problem
===============

Freqeuncy domain EM
^^^^^^^^^^^^^^^^^^^

Maxwell's equations in frequency domain can be written as

.. math:: \curl \E = -\imath\omega\B
   :label: maxeq1

.. math:: \curl \H  - \J = \J_s
   :label: maxeq2

where \\(\\E\\)

Euler's identity, equation :eq:`euler`, was elected one of the most
beautiful mathematical formulas

Time domain EM
^^^^^^^^^^^^^^

Inverse problem
===============

