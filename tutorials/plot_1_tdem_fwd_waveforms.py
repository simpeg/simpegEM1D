"""
Forward Simulation of with Different Waveforms
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
from simpegEM1D.Waveform import RectFun
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

times = np.r_[3.179659e-06, 7.651805e-06, 1.312977e-05, 2.006194e-05, 2.900623e-05, 4.040942e-05, 5.483263e-05, 7.272122e-05, 9.508194e-05, 1.229211e-04, 1.581381e-04, 2.033051e-04, 2.599911e-04, 3.314300e-04, 4.218804e-04, 5.351367e-04, 6.781304e-04, 8.584694e-04]
#times = np.logspace(-6, -1, 51)

# Receiver list
receiver_list = [
    em1d.receivers.TimeDomainPointReceiver(
        receiver_location, times, orientation=receiver_orientation,
        component="dbdt"
    )
]

# Sources
source_list = []

# Step off
source_list.append(
    em1d.sources.TimeDomainHorizontalLoopSource(
        receiver_list=receiver_list, location=source_location,
        I=source_current, a=source_radius
    )
)



# Triangle Waveform
time_input_currents = np.r_[-np.logspace(-3, -5, 21), 0.]
input_currents = RectFun(time_input_currents, -0.0008, 0.)
source_list.append(
    em1d.sources.TimeDomainHorizontalLoopSource(
        receiver_list=receiver_list, location=source_location,
        I=source_current,
        a=source_radius,
        wave_type="general",
        time_input_currents=time_input_currents,
        input_currents=input_currents,
        n_pulse = 1,
        base_frequency = 25.,
        use_lowpass_filter=False,
        high_cut_frequency=210*1e3
    )
)



# Custom waveform
time_input_currents = np.r_[-1.0000E-03, -8.0000E-04,-7.6473E-04,-6.2818E-04,-4.3497E-04,-9.2197E-05,-2.0929E-06,6.6270E-08,6.9564E-07,2.1480E-06,3.7941E-06,5.6822E-06,7.1829E-06,8.5385E-06,1.0136E-05,1.1976E-05,1.3138E-05]
input_currents = np.r_[0.0000E+00, 0.0000E+00, 6.3431E-02, 2.4971E-01, 4.7453E-01, 8.9044E-01, 1.0000E+00,1.0000E+00,9.7325E-01,7.9865E-01,5.3172E-01,2.7653E-01,1.5062E-01,7.5073E-02,3.1423E-02,7.9197E-03,0.0000E+00]
source_list.append(
    em1d.sources.TimeDomainHorizontalLoopSource(
        receiver_list=receiver_list, location=source_location,
        I=source_current,
        a=source_radius,
        wave_type="general",
        time_input_currents=time_input_currents,
        input_currents=input_currents,
        n_pulse = 1,
        base_frequency = 25.,
        use_lowpass_filter=False,
        high_cut_frequency=210*1e3
    )
)


# Survey
survey = em1d.survey.EM1DSurveyTD(source_list)

###############################################
# Plot the Waveforms
# ------------------
#
#

fig = plt.figure(figsize=(6, 4))
ax = fig.add_axes([0.1, 0.1, 0.85, 0.8])
ax.plot(time_input_currents, input_currents, 'k')





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
hz = em1d.EM1DSimulation.get_vertical_discretization_time(times, facter_tmax=0.5, factor_tmin=10.)
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


fig = plt.figure(figsize = (6, 5))
d = np.reshape(dpred, (len(source_list), len(times))).T
ax = fig.add_axes([0.1, 0.1, 0.8, 0.85])
ax.loglog(times, np.abs(d))
ax.legend(["Step-off", "Custom"])
ax.set_xlabel("Times (s)")
ax.set_ylabel("dB/dt (T/s)")
ax.set_title("Response for a chargeable and non-chargeable half-space")


fig = plt.figure(figsize = (6, 5))
d = np.reshape(dpred_colecole, (len(source_list), len(times))).T
ax = fig.add_axes([0.1, 0.1, 0.8, 0.85])
ax.loglog(times, np.abs(d))
ax.legend(["Step-off", "Custom"])
ax.set_xlabel("Times (s)")
ax.set_ylabel("dB/dt (T/s)")
ax.set_title("Response for a chargeable and non-chargeable half-space")


























