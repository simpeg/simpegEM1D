"""
Forward Simulation over a Chargeable Earth
==========================================





"""

#####################################################
# Import Modules
# --------------
#

import numpy as np
from matplotlib import pyplot as plt

from SimPEG import maps
import simpegEM1D as em1d


#####################################################################
# Create Survey
# -------------
#
#

source_location = np.array([0., 0., 10.])  
source_orientation = "z"  # "x", "y" or "z"
source_current = 1.
source_radius = np.sqrt(1/np.pi)

phi = (np.pi/4)*np.r_[1, 3, 5, 7, 1]
node_locations = np.c_[np.cos(phi), np.sin(phi), np.zeros(len(phi))]

receiver_location = np.array([0.1, 0., 10.])
receiver_orientation = "z"  # "x", "y" or "z"
field_type = "secondary"  # "secondary", "total" or "ppm"

frequencies = np.logspace(-1, 8, 51)

# Receiver list
receiver_list = []
receiver_list.append(
    em1d.receivers.HarmonicPointReceiver(
        receiver_location, frequencies, orientation=receiver_orientation,
        field_type=field_type, component="real"
    )
)

receiver_list.append(
    em1d.receivers.HarmonicPointReceiver(
        receiver_location, frequencies, orientation=receiver_orientation,
        field_type=field_type, component="imag"
    )
)

# Sources
source_list = []

source_list.append(
    em1d.sources.HarmonicMagneticDipoleSource(
        receiver_list=receiver_list, location=source_location,
        orientation=source_orientation, I=source_current
    )
)
    
source_list.append(
    em1d.sources.HarmonicHorizontalLoopSource(
        receiver_list=receiver_list, location=source_location,
        a=source_radius, I=source_current
    )
)

#source_list.append(
#    em1d.sources.HarmonicLineSource(
#        receiver_list=receiver_list, location=node_locations,
#        I=source_current
#    )
#)

# Survey
survey = em1d.survey.EM1DSurveyFD(source_list)


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
thicknesses = np.array([20., 20.])
n_layer = len(thicknesses) + 1

# half-space physical properties
sigma = 1e-2
eta = 0.5
tau = 0.001
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

#######################################################################
# Define the Forward Simulation and Predic Data
# ----------------------------------------------
#


# Simulate response for static conductivity
simulation = em1d.simulation.EM1DFMSimulation(
    survey=survey, thicknesses=thicknesses, sigmaMap=model_mapping,
    chi=chi_model
)

dpred = simulation.dpred(sigma_model)

#######################################################################
# Plotting Results
# -------------------------------------------------
#
#

dpred = np.reshape(dpred, (4, len(frequencies))).T


fig, ax = plt.subplots(1,1, figsize = (7, 7))
ax.loglog(frequencies, np.abs(dpred[:,0]), 'b-o', lw=2)
ax.loglog(frequencies, np.abs(dpred[:,1]), 'b--o', lw=2)
ax.loglog(frequencies, np.abs(dpred[:,2]), 'r-', lw=2)
ax.loglog(frequencies, np.abs(dpred[:,3]), 'r--', lw=2)
ax.set_xlabel("Frequency (Hz)")
ax.set_ylabel("|H| (A/m)")
ax.set_title("Magnetic Field as a Function of Frequency")
ax.legend((
    'Real (dipole)', 'Imaginary (dipole)',
    'Real (loop)', 'Imaginary (loop)',
    'Real (line)', 'Imaginary (line)'
))
























