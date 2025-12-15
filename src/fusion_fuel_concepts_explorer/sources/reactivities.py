import numpy as np
from numpy import float64
from numpy.typing import NDArray

def sigmav_DT_Hively(ion_temp_profile: NDArray[float64]) -> NDArray[float64]:
    r"""Deuterium-Tritium reaction.

    Calculate :math:`\langle \sigma v \rangle` for a given characteristic ion energy.
    Formulation from table 1, column S5 in :cite:`hively_convenient_1977`.
    Curvefit was performed for the range of [1,80]keV.

    Args:
        ion_temp_profile: ion temperature profile [keV]

    Returns:
        :math:`\langle \sigma v \rangle` in cm^3/s.
    """
    A = [-21.377692, -25.204054, -7.1013427 * 1e-2, 1.9375451 * 1e-4, 4.9246592 * 1e-6, -3.9836572 * 1e-8]
    r = 0.2935
    sigmav = np.exp(
        A[0] / ion_temp_profile**r
        + A[1]
        + A[2] * ion_temp_profile
        + A[3] * ion_temp_profile**2.0
        + A[4] * ion_temp_profile**3.0
        + A[5] * ion_temp_profile**4.0
    )
    return sigmav  # type: ignore[no-any-return] # [cm^3/s]

def sigmav_DT_BoschHale(ion_temp_profile: NDArray[float64]) -> NDArray[float64]:
    r"""Deuterium-Tritium reaction.

    Calculate :math:`\langle \sigma v \rangle` product for a given characteristic ion energy using Bosch Hale method.

    :func:`sigmav_DT_BoschHale` is more accurate than :func:`sigmav_DT` for ion_temp_profile > ~48.45 keV (estimate based on
    linear interp between errors found at available datapoints).
    Maximum error = 1.4% within range 50-1000 keV from available NRL data.

    Formulation from :cite:`bosch_improved_1992`

    Args:
        ion_temp_profile: ion temperature profile [keV]

    Returns:
        :math:`\langle \sigma v \rangle` in cm^3/s.

    """
    # Bosch Hale coefficients for DT reaction
    C = [0.0, 1.173e-9, 1.514e-2, 7.519e-2, 4.606e-3, 1.35e-2, -1.068e-4, 1.366e-5]
    B_G = 34.3827
    mr_c2 = 1124656

    theta = ion_temp_profile / (
        1
        - (ion_temp_profile * (C[2] + ion_temp_profile * (C[4] + ion_temp_profile * C[6])))
        / (1 + ion_temp_profile * (C[3] + ion_temp_profile * (C[5] + ion_temp_profile * C[7])))
    )
    eta = (B_G**2 / (4 * theta)) ** (1 / 3)
    sigmav = C[1] * theta * np.sqrt(eta / (mr_c2 * ion_temp_profile**3)) * np.exp(-3 * eta)
    return sigmav  # type: ignore[no-any-return] # [cm^3/s]


def sigmav_DD_Hively(ion_temp_profile: NDArray[float64]) -> tuple[NDArray[float64], NDArray[float64], NDArray[float64]]:
    r"""Deuterium-Deuterium reaction.

    Calculate :math:`\langle \sigma v \rangle` for a given characteristic ion energy.
    Formulation from column S5, in table 3 and 4 in :cite:`hively_convenient_1977`.
    Curvefit was performed for the range of [1,80]keV.

    Args:
        ion_temp_profile: ion temperature profile [keV]

    Returns:
        :math:`\langle \sigma v \rangle` tuple (total, D(d,p)T, D(d,n)3He) in cm^3/s.
    """
    a_1 = [
        -15.511891,
        -35.318711,
        -1.2904737 * 1e-2,
        2.6797766 * 1e-4,
        -2.9198685 * 1e-6,
        1.2748415 * 1e-8,
    ]  # For D(d,p)T
    r_1 = 0.3735
    a_2 = [
        -15.993842,
        -35.017640,
        -1.3689787 * 1e-2,
        2.7089621 * 1e-4,
        -2.9441547 * 1e-6,
        1.2841202 * 1e-8,
    ]  # For D(d,n)3He
    r_2 = 0.3725
    # Ti in units of keV, sigmav in units of cm^3/s
    sigmav_1: NDArray[float64] = np.exp(
        a_1[0] / ion_temp_profile**r_1
        + a_1[1]
        + a_1[2] * ion_temp_profile
        + a_1[3] * ion_temp_profile**2.0
        + a_1[4] * ion_temp_profile**3.0
        + a_1[5] * ion_temp_profile**4.0
    )
    sigmav_2: NDArray[float64] = np.exp(
        a_2[0] / ion_temp_profile**r_2
        + a_2[1]
        + a_2[2] * ion_temp_profile
        + a_2[3] * ion_temp_profile**2.0
        + a_2[4] * ion_temp_profile**3.0
        + a_2[5] * ion_temp_profile**4.0
    )
    sigmav_tot: NDArray[float64] = sigmav_1 + sigmav_2
    return sigmav_tot, sigmav_1, sigmav_2  # [cm^3/s]


def sigmav_DD_BoschHale(ion_temp_profile: NDArray[float64]) -> tuple[NDArray[float64], NDArray[float64], NDArray[float64]]:
    r"""Deuterium-Deuterium reaction.

    Calculate :math:`\langle \sigma v \rangle` product for a given characteristic ion energy using Bosch Hale method.

    Function tested on available data at [1, 2, 5, 10, 20, 50, 100] keV.
    Maximum error = 3.8% within range 5-50 keV and increases significantly outside of [5, 50] keV.

    Uses DD cross section formulation from :cite:`bosch_improved_1992`.

    Other form in :cite:`langenbrunner_analytic_2017`.

    Args:
        ion_temp_profile: ion temperature profile [keV]

    Returns:
        :math:`\langle \sigma v \rangle` tuple (total, D(d,p)T, D(d,n)3He) in cm^3/s.
    """
    # For D(d,n)3He
    cBH_1 = [((31.3970**2) / 4.0) ** (1.0 / 3.0), 5.65718e-12, 3.41e-03, 1.99e-03, 0, 1.05e-05, 0, 0]  # 3.72e-16,

    mc2_1 = 937814.0

    # For D(d,p)T
    cBH_2 = [((31.3970**2) / 4.0) ** (1.0 / 3.0), 5.43360e-12, 5.86e-03, 7.68e-03, 0, -2.96e-06, 0, 0]  # 3.57e-16,

    mc2_2 = 937814.0

    thetaBH_1 = ion_temp_profile / (
        1
        - (
            (cBH_1[2] * ion_temp_profile + cBH_1[4] * ion_temp_profile**2 + cBH_1[6] * ion_temp_profile**3)
            / (1 + cBH_1[3] * ion_temp_profile + cBH_1[5] * ion_temp_profile**2 + cBH_1[7] * ion_temp_profile**3)
        )
    )

    thetaBH_2 = ion_temp_profile / (
        1
        - (
            (cBH_2[2] * ion_temp_profile + cBH_2[4] * ion_temp_profile**2 + cBH_2[6] * ion_temp_profile**3)
            / (1 + cBH_2[3] * ion_temp_profile + cBH_2[5] * ion_temp_profile**2 + cBH_2[7] * ion_temp_profile**3)
        )
    )

    etaBH_1: float = cBH_1[0] / (thetaBH_1 ** (1.0 / 3.0))
    etaBH_2: float = cBH_2[0] / (thetaBH_2 ** (1.0 / 3.0))

    sigmav_1: NDArray[float64] = cBH_1[1] * thetaBH_1 * np.sqrt(etaBH_1 / (mc2_1 * (ion_temp_profile**3.0))) * np.exp(-3.0 * etaBH_1)
    sigmav_2: NDArray[float64] = cBH_2[1] * thetaBH_2 * np.sqrt(etaBH_2 / (mc2_2 * (ion_temp_profile**3.0))) * np.exp(-3.0 * etaBH_2)
    sigmav_tot: NDArray[float64] = sigmav_1 + sigmav_2

    return sigmav_tot, sigmav_1, sigmav_2  # [cm^3/s]


def sigmav_DHe3_BoschHale(ion_temp_profile: NDArray[float64]) -> NDArray[float64]:
    r"""Deuterium-Helium-3 reaction.

    Calculate :math:`\langle \sigma v \rangle` for a given characteristic ion energy.

    Function tested on available data at [1, 2, 5, 10, 20, 50, 100] keV.
    Maximum error = 8.4% within range 2-100 keV and should not be used outside range [2, 100] keV.

    Uses DD cross section formulation :cite:`bosch_improved_1992`.

    Args:
        ion_temp_profile: ion temperature profile [keV]

    Returns:
        :math:`\langle \sigma v \rangle` in cm^3/s.
    """
    # For He3(d,p)4He
    cBH_1 = [
        ((68.7508**2) / 4.0) ** (1.0 / 3.0),
        5.51036e-10,  # 3.72e-16,
        6.41918e-03,
        -2.02896e-03,
        -1.91080e-05,
        1.35776e-04,
        0,
        0,
    ]

    mc2_1 = 1124572.0

    thetaBH_1 = ion_temp_profile / (
        1
        - (
            (cBH_1[2] * ion_temp_profile + cBH_1[4] * ion_temp_profile**2 + cBH_1[6] * ion_temp_profile**3)
            / (1 + cBH_1[3] * ion_temp_profile + cBH_1[5] * ion_temp_profile**2 + cBH_1[7] * ion_temp_profile**3.0)
        )
    )

    etaBH_1: float = cBH_1[0] / (thetaBH_1 ** (1.0 / 3.0))

    sigmav: NDArray[float64] = cBH_1[1] * thetaBH_1 * np.sqrt(etaBH_1 / (mc2_1 * (ion_temp_profile**3.0))) * np.exp(-3.0 * etaBH_1)

    return sigmav  # [cm^3/s]


# ---------------------------------------------------------------------------
# Maxwellian-averaged TT and T+He3 reactivities from NRL Plasma Formulary table
# (temperature in keV, <σv> in cm^3/s). Values are interpolated in log-log
# space; extrapolation beyond the table uses the edge slopes.
# Table columns correspond to "T - T" and "T - He3" entries in:
#   NRL Plasma Formulary, Maxwellian-averaged reactivities.
# ---------------------------------------------------------------------------

_T_REF_KEV = np.array([1.0, 2.0, 5.0, 10.0, 20.0, 50.0, 100.0, 200.0, 500.0, 1000.0], dtype=float)
_SIG_TT = np.array([3.3e-22, 7.1e-21, 1.4e-19, 7.2e-19, 2.5e-18, 8.7e-18, 1.9e-17, 4.2e-17, 8.4e-17, 8.0e-17], dtype=float)
_SIG_THe3 = np.array([1.0e-28, 1.0e-25, 2.1e-22, 1.2e-20, 2.6e-19, 5.3e-18, 2.7e-17, 9.2e-17, 2.9e-16, 5.2e-16], dtype=float)


def _log_interp(x: NDArray[float64], xp: NDArray[float64], fp: NDArray[float64]) -> NDArray[float64]:
    """Log-log linear interpolation with edge extrapolation."""
    x_log = np.log(x)
    xp_log = np.log(xp)
    fp_log = np.log(fp)
    interp = np.interp(x_log, xp_log, fp_log)
    return np.exp(interp)


def sigmav_TT_NRL(ion_temp_profile: NDArray[float64]) -> NDArray[float64]:
    r"""Tritium–Tritium reaction.

    Maxwellian-averaged reactivity <σv> (cm^3/s) from the NRL Plasma Formulary
    table (TT column). Implemented via log-log interpolation between tabulated
    points at 1–1000 keV; extrapolation outside this range uses the edge slopes.
    """
    t_arr = np.asarray(ion_temp_profile, dtype=float)
    return _log_interp(t_arr, _T_REF_KEV, _SIG_TT)  # [cm^3/s]


def sigmav_THe3_NRL(ion_temp_profile: NDArray[float64]) -> NDArray[float64]:
    r"""Tritium–Helium-3 reaction.

    Maxwellian-averaged reactivity <σv> (cm^3/s) from the NRL Plasma Formulary
    table (T–He3 column). Implemented via log-log interpolation between
    tabulated points at 1–1000 keV; extrapolation outside this range uses the
    edge slopes.
    """
    t_arr = np.asarray(ion_temp_profile, dtype=float)
    return _log_interp(t_arr, _T_REF_KEV, _SIG_THe3)  # [cm^3/s]
