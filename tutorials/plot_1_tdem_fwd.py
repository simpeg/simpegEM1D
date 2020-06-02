"""
Forward Simulation of 1D Time-Domain Data
==============================================





"""

#####################################################
# Import Modules
# --------------
#

import numpy as np
from matplotlib import pyplot as plt

from SimPEG import maps
import simpegEM1D as em1d
from simpegEM1D.analytics import ColeCole


#####################################################################
# Create Survey
# -------------
#
#

source_location = np.array([0., 0., 0.])  
source_orientation = "z"  # "x", "y" or "z"
source_current = 1.
source_radius = 10.

receiver_location = np.array([0., 0., 0.])
receiver_orientation = "z"  # "x", "y" or "z"
field_type = "secondary"  # "secondary", "total" or "ppm"

times = np.logspace(-6, -1, 51)

# Receiver list
receiver_list = []
receiver_list.append(
    em1d.receivers.TimeDomainPointReceiver(
        receiver_location, times, orientation=receiver_orientation,
        component="b"
    )
)
receiver_list.append(
    em1d.receivers.TimeDomainPointReceiver(
        receiver_location, times, orientation=receiver_orientation,
        component="dbdt"
    )
)

# Sources
source_list = [
    em1d.sources.TimeDomainHorizontalLoopSource(
        receiver_list=receiver_list, location=source_location,
        I=source_current, a=source_radius
    )
]


# Survey
survey = em1d.survey.EM1DSurveyTD(source_list)


###############################################
# Defining a 1D Layered Earth Model
# ---------------------------------
#
# Here, we define the layer thicknesses and electrical conductivities for our
# 1D simulation. If we have N layers, we define N electrical conductivity
# values and N-1 layer thicknesses. The lowest layer is assumed to extend to
# infinity.
#

# Layer thicknesses
thicknesses = np.array([40., 40.])
n_layer = len(thicknesses) + 1

# half-space physical properties
sigma = 1e-2
eta = 0.5
tau = 0.01
c = 0.5
chi = 0.

# physical property models
sigma_model = sigma * np.ones(n_layer)
eta_model = eta * np.ones(n_layer)
tau_model =  tau * np.ones(n_layer)
c_model = c * np.ones(n_layer)
chi_model = chi * np.ones(n_layer)

# Define a mapping for conductivities
model_mapping = maps.IdentityMap(nP=n_layer)

# Compute and plot complex conductivity at all frequencies
frequencies = np.logspace(-3, 6, 91)
sigma_complex = ColeCole(frequencies, sigma, eta, tau, c)

fig = plt.figure()
ax = fig.add_subplot(111)
ax.semilogx(frequencies, sigma*np.ones(len(frequencies)), "b", lw=3)
ax.semilogx(frequencies, np.real(sigma_complex), "r", lw=3)
ax.semilogx(frequencies, np.imag(sigma_complex), "r--", lw=3)
ax.set_xlim(np.min(frequencies), np.max(frequencies))
ax.set_ylim(0., 1.1*sigma)
ax.set_xlabel("Frequency (Hz)")
ax.set_ylabel("Conductivity")
ax.legend(["$\sigma_{DC}$", "$Re[\sigma (\omega)]$", "$Im[\sigma (\omega)]$"])

#######################################################################
# Define the Forward Simulation and Predict MT Data
# -------------------------------------------------
#
# Here we predict MT data. If the keyword argument *rhoMap* is
# defined, the simulation will expect a resistivity model. If the keyword
# argument *sigmaMap* is defined, the simulation will expect a conductivity model.
#

# Simulate response for static conductivity
simulation = em1d.simulation.EM1DTMSimulation(
    survey=survey, thicknesses=thicknesses, sigmaMap=model_mapping,
    chi=chi_model
)

dpred = simulation.dpred(sigma_model)

# Simulate response for complex conductivity
simulation_colecole = em1d.simulation.EM1DTMSimulation(
    survey=survey, thicknesses=thicknesses, sigmaMap=model_mapping,
    eta=eta, tau=tau, c=c, chi=chi_model
)

dpred_colecole = simulation_colecole.dpred(sigma_model)


#######################################################################
# Analytic Solution
# -------------------------------------------------
#
#
b = dpred[0:len(times)]
b_colecole = dpred_colecole[0:len(times)]
dbdt = dpred[len(times):]
dbdt_colecole = dpred_colecole[len(times):]

k1 = b_colecole > 0.
k2 = dbdt_colecole > 0.

fig = plt.figure(figsize = (6, 5))
ax = fig.add_axes([0.1, 0.1, 0.8, 0.85])
ax.loglog(times, np.abs(dpred[0:len(times)]), 'b', lw=3)
ax.loglog(times[k1], np.abs(b_colecole[k1]), 'r', lw=3)
ax.loglog(times[~k1], np.abs(b_colecole[~k1]), 'r--', lw=3)
ax.legend(["Non-chargeable", "Cole-Cole"])
ax.set_xlabel("Times (s)")
ax.set_ylabel("B (T)")
ax.set_title("Response for a chargeable and non-chargeable half-space")


fig = plt.figure(figsize = (6, 5))
ax = fig.add_axes([0.1, 0.1, 0.8, 0.85])
ax.loglog(times, np.abs(dpred[len(times):]), 'b--', lw=3)
ax.loglog(times[~k2], np.abs(dbdt_colecole[~k2]), 'r--', lw=3)
ax.loglog(times[k2], np.abs(dbdt_colecole[k2]), 'r', lw=3)
ax.legend(["Non-chargeable", "Cole-Cole"])
ax.set_xlabel("Times (s)")
ax.set_ylabel("dB/dt (T/s)")
ax.set_title("Response for a chargeable and non-chargeable half-space")


























