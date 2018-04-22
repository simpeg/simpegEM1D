import unittest
from SimPEG import Mesh, Maps, Utils
import matplotlib.pyplot as plt
from simpegEM1D import (
    EM1D, EM1DSurveyTD, EM1DAnalytics, piecewise_pulse, set_mesh_1d,
    skytem_HM_2015, skytem_LM_2015, get_vertical_discretization_time
)
import numpy as np
from scipy import io
from scipy.interpolate import interp1d


class EM1D_TD_FwdProblemTests(unittest.TestCase):

    def setUp(self):

        wave_HM = skytem_HM_2015()
        wave_LM = skytem_LM_2015()
        time_HM = wave_HM.time_gate_center[0::2]
        time_LM = wave_LM.time_gate_center[0::2]

        hz = get_vertical_discretization_time(
            np.unique(np.r_[time_HM, time_LM]), facter_tmax=0.5, factor_tmin=10.
        )
        mesh1D = set_mesh_1d(hz)
        depth = -mesh1D.gridN[:-1]
        LocSigZ = -mesh1D.gridCC

        time_input_currents_HM = wave_HM.current_times[-7:]
        input_currents_HM = wave_HM.currents[-7:]
        time_input_currents_LM = wave_LM.current_times[-13:]
        input_currents_LM = wave_LM.currents[-13:]

        TDsurvey = EM1DSurveyTD(
            rx_location=np.array([0., 0., 100.]),
            src_location=np.array([0., 0., 100.]),
            topo=np.r_[0., 0., 100.],
            depth=depth,
            rx_type='dBzdt',
            wave_type='general',
            src_type='CircularLoop',
            a=13.,
            I=1.,
            time=time_HM,
            time_input_currents=time_input_currents_HM,
            input_currents=input_currents_HM,
            n_pulse=2,
            base_frequency=25.,
            use_lowpass_filter=False,
            high_cut_frequency=7e4,
            moment_type='dual',
            time_dual_moment=time_HM,
            time_input_currents_dual_moment=time_input_currents_LM,
            input_currents_dual_moment=input_currents_LM,
            base_frequency_dual_moment=210,
        )

        sig_half=1e-2
        chi_half=0.

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

        dBzdtTD = self.survey.dpred(self.m_1D)
        dBzdtTD_HM = dBzdtTD[:self.survey.n_time]
        dBzdtTD_LM = dBzdtTD[self.survey.n_time:]

        def step_func_dBzdt(time):
            return EM1DAnalytics.dBzdtAnalCircT(
                self.survey.a, time, self.sig_half
            )

        dBzdtTD_analytic_HM = piecewise_pulse(
            step_func_dBzdt, self.survey.time,
            self.survey.time_input_currents,
            self.survey.input_currents,
            self.survey.period
        )

        dBzdtTD_analytic_LM = piecewise_pulse(
            step_func_dBzdt, self.survey.time,
            self.survey.time_input_currents_dual_moment,
            self.survey.input_currents_dual_moment,
            self.survey.period_dual_moment
        )
        if self.showIt:
            plt.loglog(self.survey.time, -dBzdtTD_HM)
            plt.loglog(self.survey.time, -dBzdtTD_LM)
            plt.loglog(self.survey.time, -dBzdtTD_analytic_HM, 'x')
            plt.loglog(self.survey.time, -dBzdtTD_analytic_LM, 'x')
            plt.show()

        err = (
            np.linalg.norm(dBzdtTD_HM-dBzdtTD_analytic_HM)/
            np.linalg.norm(dBzdtTD_analytic_HM)
        )

        print ('dBzdt error (HM) = ', err)

        self.assertTrue(err < 5e-2)
        err = (
            np.linalg.norm(dBzdtTD_LM-dBzdtTD_analytic_LM)/
            np.linalg.norm(dBzdtTD_analytic_LM)
        )

        print ('dBzdt error (LM) = ', err)
        self.assertTrue(err < 5e-2)

        print ("EM1DTD-CirculurLoop-general for real conductivity works")


if __name__ == '__main__':
    unittest.main()
