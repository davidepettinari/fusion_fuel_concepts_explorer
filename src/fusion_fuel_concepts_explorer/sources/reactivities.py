from __future__ import annotations

import numpy as np
from numpy import float64
from numpy.typing import NDArray

ArrayLike = float | NDArray[np.float64]

_AVOGADRO_NUMBER = 6.02214076e23
_KEV_TO_T9 = 1.160451812e-2

def sigmav_DT_BoschHale(ion_temp_profile: ArrayLike) -> ArrayLike:
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
    return sigmav*1e-6  # type: ignore[no-any-return]


def sigmav_DD_BoschHale(ion_temp_profile: ArrayLike) -> tuple[ArrayLike, ArrayLike, ArrayLike]:
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
    cBH_1 = [((31.3970**2) / 4.0) ** (1.0 / 3.0), 5.65718e-12, 3.41e-03, 1.99e-03, 0, 1.05e-05, 0, 0]
    mc2_1 = 937814.0
    cBH_2 = [((31.3970**2) / 4.0) ** (1.0 / 3.0), 5.43360e-12, 5.86e-03, 7.68e-03, 0, -2.96e-06, 0, 0]
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

    etaBH_1 = cBH_1[0] / (thetaBH_1 ** (1.0 / 3.0))
    etaBH_2 = cBH_2[0] / (thetaBH_2 ** (1.0 / 3.0))

    sigmav_1 = cBH_1[1] * thetaBH_1 * np.sqrt(etaBH_1 / (mc2_1 * (ion_temp_profile**3.0))) * np.exp(-3.0 * etaBH_1)
    sigmav_2 = cBH_2[1] * thetaBH_2 * np.sqrt(etaBH_2 / (mc2_2 * (ion_temp_profile**3.0))) * np.exp(-3.0 * etaBH_2)
    sigmav_tot = sigmav_1 + sigmav_2

    return sigmav_tot*1e-6, sigmav_1*1e-6, sigmav_2*1e-6


def sigmav_DDn_BoschHale(ion_temp_profile: ArrayLike) -> ArrayLike:
    """Return the D(d,n)He3 Bosch-Hale branch only."""
    return sigmav_DD_BoschHale(ion_temp_profile)[2]


def sigmav_DDp_BoschHale(ion_temp_profile: ArrayLike) -> ArrayLike:
    """Return the D(d,p)T Bosch-Hale branch only."""
    return sigmav_DD_BoschHale(ion_temp_profile)[1]


def sigmav_DHe3_BoschHale(ion_temp_profile: ArrayLike) -> ArrayLike:
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
    cBH_1 = [
        ((68.7508**2) / 4.0) ** (1.0 / 3.0),
        5.51036e-10,
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

    etaBH_1 = cBH_1[0] / (thetaBH_1 ** (1.0 / 3.0))
    sigmav = cBH_1[1] * thetaBH_1 * np.sqrt(etaBH_1 / (mc2_1 * (ion_temp_profile**3.0))) * np.exp(-3.0 * etaBH_1)
    return sigmav*1e-6


def sigmav_TT_CF88(T_i: ArrayLike) -> ArrayLike:
    r"""Tritium-Tritium reaction using the CF88 parametrization."""
    T9 = np.asarray(T_i, dtype=float) * _KEV_TO_T9
    sigmav = (
        1.67e9
        / (T9 ** (2 / 3))
        * np.exp(-4.872 / (T9 ** (1 / 3)))
        * (
            1
            + 0.086 * (T9 ** (1 / 3))
            - 0.455 * (T9 ** (2 / 3))
            - 0.272 * T9
            + 0.148 * (T9 ** (4 / 3))
            + 0.225 * (T9 ** (5 / 3))
        )
    )
    return sigmav / _AVOGADRO_NUMBER * 1e-6  # type: ignore[no-any-return]


def sigmav_He3He3_CF88(T_i: ArrayLike) -> ArrayLike:
    r"""Helium-3/Helium-3 reaction using the CF88 parametrization."""
    T9 = np.asarray(T_i, dtype=float) * _KEV_TO_T9
    sigmav = (
        6.04e10
        / (T9 ** (2 / 3))
        * np.exp(-12.276 / (T9 ** (1 / 3)))
        * (
            1
            + 0.034 * (T9 ** (1 / 3))
            - 0.522 * (T9 ** (2 / 3))
            - 0.124 * T9
            + 0.353 * (T9 ** (4 / 3))
            + 0.213 * (T9 ** (5 / 3))
        )
    )
    return sigmav / _AVOGADRO_NUMBER * 1e-6  # type: ignore[no-any-return]


def sigmav_THe3_D_CF88(T_i: ArrayLike) -> ArrayLike:
    r"""T + 3He -> 4He + D branch using the CF88 parametrization."""
    T9 = np.asarray(T_i, dtype=float) * _KEV_TO_T9
    T9A = T9 / (1 + 0.128 * T9)
    sigmav = 5.46e9 * (T9A ** (5 / 6)) / (T9 ** (3 / 2)) * np.exp(-7.733 / (T9A ** (1 / 3)))
    return sigmav / _AVOGADRO_NUMBER * 1e-6  # type: ignore[no-any-return]


def sigmav_THe3_np_CF88(T_i: ArrayLike) -> ArrayLike:
    r"""T + 3He -> 4He + n + p branch using the CF88 parametrization."""
    T9 = np.asarray(T_i, dtype=float) * _KEV_TO_T9
    T9A = T9 / (1 + 0.115 * T9)
    sigmav = 7.71e9 * (T9A ** (5 / 6)) / (T9 ** (3 / 2)) * np.exp(-7.733 / (T9A ** (1 / 3)))
    return sigmav / _AVOGADRO_NUMBER * 1e-6  # type: ignore[no-any-return]


# ---------------------------------------------------------------------------
# Maxwellian-averaged TT and T+He3 reactivities from NRL Plasma Formulary table
# (temperature in keV, <σv> in cm^3/s). Values are interpolated in log-log
# space; extrapolation beyond the table uses the edge slopes.
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