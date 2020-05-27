import numpy as np
from .RTEfun_vec import rTEfunfwd, rTEfunjac

try:
    from simpegEM1D.m_rTE_Fortran import rte_fortran
except ImportError as e:
    rte_fortran = None


def hz_kernel_vertical_magnetic_dipole(
    simulation, lamda, f, n_layer, sig, chi, depth, h, z,
    flag, I, output_type='response'
):

    """
        Kernel for vertical magnetic component (Hz) due to
        vertical magnetic diopole (VMD) source in (kx,ky) domain

    """
    u0 = lamda
    coefficient_wavenumber = 1/(4*np.pi)*lamda**3/u0

    n_frequency = simulation.survey.n_frequency
    n_layer = simulation.survey.n_layer
    n_filter = simulation.n_filter

    if output_type == 'sensitivity_sigma':
        drTE = np.zeros(
            [n_layer, n_frequency, n_filter],
            dtype=np.complex128, order='F'
        )
        if rte_fortran is None:
            drTE = rTEfunjac(
                n_layer, f, lamda, sig, chi, depth, simulation.survey.half_switch
            )
        else:
            rte_fortran.rte_sensitivity(
                f, lamda, sig, chi, depth, simulation.survey.half_switch, drTE,
                n_layer, n_frequency, n_filter
                )

        kernel = drTE * np.exp(-u0*(z+h)) * coefficient_wavenumber
    else:
        rTE = np.empty(
            [n_frequency, n_filter], dtype=np.complex128, order='F'
        )
        if rte_fortran is None:
                rTE = rTEfunfwd(
                    n_layer, f, lamda, sig, chi, depth,
                    simulation.survey.half_switch
                )
        else:
            rte_fortran.rte_forward(
                f, lamda, sig, chi, depth, simulation.survey.half_switch,
                rTE, n_layer, n_frequency, n_filter
            )

        kernel = rTE * np.exp(-u0*(z+h)) * coefficient_wavenumber
        if output_type == 'sensitivity_height':
            kernel *= -2*u0

    return kernel * I

    # Note
    # Here only computes secondary field.
    # I am not sure why it does not work if we add primary term.
    # This term can be analytically evaluated, where h = 0.
    #     kernel = (
    #         1./(4*np.pi) *
    #         (np.exp(u0*(z-h))+rTE * np.exp(-u0*(z+h)))*lamda**3/u0
    #     )

# TODO: make this to take a vector rather than a single frequency
def hz_kernel_circular_loop(
    simulation, lamda, f, n_layer, sig, chi, depth, h, z, I, a,
    flag,  output_type='response'
):

    """

    Kernel for vertical magnetic component (Hz) at the center
    due to circular loop source in (kx,ky) domain

    .. math::

        H_z = \\frac{Ia}{2} \int_0^{\infty} [e^{-u_0|z+h|} +
        \\r_{TE}e^{u_0|z-h|}]
        \\frac{\lambda^2}{u_0} J_1(\lambda a)] d \lambda

    """

    n_frequency = simulation.survey.n_frequency
    n_layer = simulation.survey.n_layer
    n_filter = simulation.n_filter

    w = 2*np.pi*f
    u0 = lamda
    radius = np.empty([n_frequency, n_filter], order='F')
    radius[:, :] = np.tile(a.reshape([-1, 1]), (1, n_filter))

    coefficient_wavenumber = I*radius*0.5*lamda**2/u0

    if output_type == 'sensitivity_sigma':
        drTE = np.empty(
            [n_layer, n_frequency, n_filter],
            dtype=np.complex128, order='F'
        )
        if rte_fortran is None:
                drTE[:, :] = rTEfunjac(
                    n_layer, f, lamda, sig, chi, depth,
                    simulation.survey.half_switch
                )
        else:
            rte_fortran.rte_sensitivity(
                f, lamda, sig, chi, depth, simulation.survey.half_switch,
                drTE, n_layer, n_frequency, n_filter
            )

        kernel = drTE * np.exp(-u0*(z+h)) * coefficient_wavenumber
    else:
        rTE = np.empty(
            [n_frequency, n_filter], dtype=np.complex128, order='F'
        )
        if rte_fortran is None:
            rTE[:, :] = rTEfunfwd(
                n_layer, f, lamda, sig, chi, depth, simulation.survey.half_switch
            )
        else:
            rte_fortran.rte_forward(
                f, lamda, sig, chi, depth, simulation.survey.half_switch,
                rTE, n_layer, n_frequency, n_filter
            )

        if flag == 'secondary':
            kernel = rTE * np.exp(-u0*(z+h)) * coefficient_wavenumber
        else:
            kernel = rTE * (
                np.exp(-u0*(z+h)) + np.exp(u0*(z-h))
            ) * coefficient_wavenumber

        if output_type == 'sensitivity_height':
            kernel *= -2*u0

    return kernel

def hz_kernel_horizontal_electric_dipole(
    simulation, lamda, f, n_layer, sig, chi, depth, h, z,
    flag, output_type='response'
):

    """
        Kernel for vertical magnetic field (Hz) due to
        horizontal electric diopole (HED) source in (kx,ky) domain

    """
    n_frequency = simulation.survey.n_frequency
    n_layer = simulation.survey.n_layer
    n_filter = simulation.n_filter

    u0 = lamda
    coefficient_wavenumber = 1/(4*np.pi)*lamda**2/u0

    if output_type == 'sensitivity_sigma':
        drTE = np.zeros(
            [n_layer, n_frequency, n_filter], dtype=np.complex128,
            order='F'
        )
        if rte_fortran is None:
            drTE = rTEfunjac(
                n_layer, f, lamda, sig, chi, depth, simulation.survey.half_switch
            )
        else:
            rte_fortran.rte_sensitivity(
                f, lamda, sig, chi, depth, simulation.survey.half_switch,
                drTE, n_layer, n_frequency, n_filter
            )

        kernel = drTE * np.exp(-u0*(z+h)) * coefficient_wavenumber
    else:
        rTE = np.empty(
            [n_frequency, n_filter], dtype=np.complex128, order='F'
        )
        if rte_fortran is None:
            rTE = rTEfunfwd(
                    n_layer, f, lamda, sig, chi, depth,
                    simulation.survey.half_switch
            )
        else:
            rte_fortran.rte_forward(
                f, lamda, sig, chi, depth, simulation.survey.half_switch,
                rTE, n_layer, n_frequency, n_filter
            )

        kernel = rTE * np.exp(-u0*(z+h)) * coefficient_wavenumber
        if output_type == 'sensitivity_height':
            kernel *= -2*u0

    return kernel













# import numpy as np
# from scipy.constants import mu_0
# from .DigFilter import EvalDigitalFilt
# from RTEfun import rTEfun

# def HzKernel_layer(lamda, f, nlay, sig, chi, depth, h, z, flag):

#     """

#         Kernel for vertical magnetic component (Hz) due to vertical magnetic
#         diopole (VMD) source in (kx,ky) domain

#     """
#     u0 = lamda
#     rTE, M00, M01, M10, M11 = rTEfun(nlay, f, lamda, sig, chi, depth)

#     if flag=='secondary':
#         # Note
#         # Here only computes secondary field.
#         # I am not sure why it does not work if we add primary term.
#         # This term can be analytically evaluated, where h = 0.

#         kernel = 1/(4*np.pi)*(rTE*np.exp(-u0*(z+h)))*lamda**3/u0

#     else:
#         kernel = 1/(4*np.pi)*(np.exp(u0*(z-h))+ rTE*np.exp(-u0*(z+h)))*lamda**3/u0

#     return  kernel

# def HzkernelCirc_layer(lamda, f, nlay, sig, chi, depth, h, z, I, a, flag):

#     """

#         Kernel for vertical magnetic component (Hz) at the center
#         due to circular loop source in (kx,ky) domain

#         .. math::

#             H_z = \\frac{Ia}{2} \int_0^{\infty} [e^{-u_0|z+h|} + r_{TE}e^{u_0|z-h|}] \\frac{\lambda^2}{u_0} J_1(\lambda a)] d \lambda

#     """

#     w = 2*np.pi*f
#     rTE = np.zeros(lamda.size, dtype=complex)
#     u0 = lamda
#     rTE, M00, M01, M10, M11 = rTEfun(nlay, f, lamda, sig, chi, depth)

#     if flag == 'secondary':
#         kernel = I*a*0.5*(rTE*np.exp(-u0*(z+h)))*lamda**2/u0
#     else:
#         kernel = I*a*0.5*(np.exp(u0*(z-h))+rTE*np.exp(-u0*(z+h)))*lamda**2/u0

#     return  kernel

#TODO: Get rid of below two functions and put in in main class
# def HzFreq_layer(nlay, sig, chi, depth, f, z, h, r, flag, YBASE, WT0):
#     """

#     """
#     nfreq = np.size(f)
#     HzFHT = np.zeros(nfreq, dtype = complex)
#     for ifreq in range(nfreq):

#         kernel = lambda x: HzKernel_layer(x, f[ifreq], nlay, sig, chi, depth, h, z, flag)
#         HzFHT[ifreq] = EvalDigitalFilt(YBASE, WT0, kernel, r)

#     return HzFHT

# def HzCircFreq_layer(nlay, sig, chi, depth, f, z, h, I, a, flag, YBASE, WT1):

#     """

#     """
#     nfreq = np.size(f)
#     HzFHT = np.zeros(nfreq, dtype = complex)
#     for ifreq in range(nfreq):

#         kernel = lambda x: HzkernelCirc_layer(x, f[ifreq], nlay, sig, chi, depth, h, z, I, a, flag)
#         HzFHT[ifreq] = EvalDigitalFilt(YBASE, WT1, kernel, a)

#     return HzFHT
