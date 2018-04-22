import unittest
from SimPEG import Mesh, Maps, Utils
import matplotlib.pyplot as plt
from simpegEM1D import EM1D, EM1DSurveyTD, EM1DAnalytics
from simpegEM1D.Waveforms import piecewise_ramp
import numpy as np
from scipy import io
from scipy.interpolate import interp1d


class EM1D_TD_FwdProblemTests(unittest.TestCase):

    def setUp(self):

        nearthick = np.logspace(-1, 1, 5)
        deepthick = np.logspace(1, 2, 10)
        hx = np.r_[nearthick, deepthick]
        mesh1D = Mesh.TensorMesh([hx], [0.])
        depth = -mesh1D.gridN[:-1]
        LocSigZ = -mesh1D.gridCC

        # Triangular waveform
        time_input_currents = np.r_[0., 5.5*1e-4, 1.1*1e-3]
        input_currents = np.r_[0., 1., 0.]

        TDsurvey = EM1DSurveyTD(
            rx_location=np.array([0., 0., 100.+1e-5]),
            src_location=np.array([0., 0., 100.+1e-5]),
            topo=np.r_[0., 0., 100.],
            depth=depth,
            field_type='secondary',
            rx_type='Bz',
            wave_type='general',
            time_input_currents=time_input_currents,
            input_currents=input_currents,
            n_pulse=2,
            base_frequency=25.,
            time=np.logspace(-5, -2, 31),
            src_type='CircularLoop',
            I=1e0,
            a=2e1
        )

        sig_half = 1e-4
        chi_half = 0.

        expmap = Maps.ExpMap(mesh1D)
        m_1D = np.log(np.ones(TDsurvey.n_layer)*sig_half)
        chi = np.zeros(TDsurvey.n_layer)

        prob = EM1D(
            mesh1D, sigmaMap=expmap, chi=chi
        )
        prob.pair(TDsurvey)

        self.survey = TDsurvey
        self.prob = prob
        self.mesh1D = mesh1D
        self.showIt = False
        self.chi = chi
        self.m_1D = m_1D
        self.sig_half = sig_half
        self.expmap = expmap

    def test_em1dtd_circular_loop_single_pulse(self):
        f = self.prob.forward(self.m_1D)
        BzTD = self.prob.survey.projectFields(f)

        def step_func_Bzdt(time):
            return EM1DAnalytics.BzAnalCircT(
                self.survey.a, time, self.sig_half
            )

        BzTD_analytic = piecewise_ramp(
            step_func_Bzdt, self.survey.time,
            self.survey.time_input_currents, self.survey.input_currents
        )

        if self.showIt:
            plt.subplot(121)
            plt.loglog(self.survey.time, BzTD, 'b*')
            plt.loglog(self.survey.time, BzTD_analytic, 'b')
            plt.subplot(122)
            plt.loglog(
                self.survey.time, abs((BzTD-BzTD_analytic)/BzTD_analytic), 'r:'
            )
            plt.show()

        err = np.linalg.norm(BzTD-BzTD_analytic)/np.linalg.norm(BzTD_analytic)
        print ('Bz error = ', err)
        self.assertTrue(err < 5e-2)

        self.survey.rx_type = 'dBzdt'
        dBzdtTD = self.prob.survey.projectFields(f)

        def step_func_dBzdt(time):
            return EM1DAnalytics.dBzdtAnalCircT(
                self.survey.a, time, self.sig_half
            )

        dBzdtTD_analytic = piecewise_ramp(
            step_func_dBzdt, self.survey.time,
            self.survey.time_input_currents, self.survey.input_currents
        )

        if self.showIt:
            plt.subplot(121)
            plt.loglog(self.survey.time, abs(dBzdtTD), 'b*')
            plt.loglog(
                self.survey.time,
                abs(dBzdtTD_analytic), 'b'
            )
            plt.subplot(122)
            plt.loglog(
                self.survey.time,
                abs((dBzdtTD-dBzdtTD_analytic)/dBzdtTD_analytic), 'r:'
            )
            plt.show()

        err = (
            np.linalg.norm(dBzdtTD-dBzdtTD_analytic)/
            np.linalg.norm(dBzdtTD_analytic)
        )

        print ('dBzdt error = ', err)
        self.assertTrue(err < 5e-2)

        print ("EM1DTD-CirculurLoop-general for real conductivity works")


if __name__ == '__main__':
    unittest.main()
