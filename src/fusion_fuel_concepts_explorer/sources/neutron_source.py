from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Mapping, Sequence, Union, Literal

import numpy as np
from numpy.typing import NDArray

from .reactivities import (
    sigmav_DD_BoschHale,
    sigmav_DD_Hively,
    sigmav_DT_BoschHale,
    sigmav_DT_Hively,
    sigmav_DHe3_BoschHale,
    sigmav_THe3_NRL,
    sigmav_TT_NRL,
)


Number = Union[float, int]
ArrayLike = Union[Number, Sequence[Number], NDArray[np.float64]]


@dataclass
class NeutronEmissionResult:
    """Container for neutron emission probabilities and supporting rates."""

    probabilities: dict[str, float]
    reaction_rates_s: dict[str, float]
    total_neutron_rate_s: float


def _as_profile(values: ArrayLike) -> NDArray[np.float64]:
    """Convert scalars or sequences to a 1D numpy array."""
    arr = np.asarray(values, dtype=float)
    if arr.ndim == 0:
        arr = np.array([float(arr)], dtype=float)
    return arr


def _broadcast_base(
    temperature_keV: ArrayLike,
    total_density_cm3: ArrayLike | None,
    species_profiles: list[ArrayLike],
) -> tuple[NDArray[np.float64], NDArray[np.float64] | None, list[NDArray[np.float64]]]:
    """Broadcast all profiles to the same shape."""
    base_arrays: list[NDArray[np.float64]] = [_as_profile(temperature_keV)]

    density_array: NDArray[np.float64] | None = None
    if total_density_cm3 is not None:
        density_array = _as_profile(total_density_cm3)
        base_arrays.append(density_array)

    species_arrays = [_as_profile(val) for val in species_profiles]
    base_arrays.extend(species_arrays)

    broadcasted = np.broadcast_arrays(*base_arrays)

    idx = 0
    temperature_b = broadcasted[idx]
    idx += 1
    density_b: NDArray[np.float64] | None = None
    if density_array is not None:
        density_b = broadcasted[idx]
        idx += 1

    species_b: list[NDArray[np.float64]] = []
    for _ in species_arrays:
        species_b.append(broadcasted[idx])
        idx += 1

    return temperature_b, density_b, species_b


def _select_sigmav_dt(temperature_keV: NDArray[np.float64], model: Literal["bosch_hale", "hively", "auto"]) -> NDArray[np.float64]:
    """Select DT reactivity with optional temperature-based switch."""
    if model == "bosch_hale":
        return sigmav_DT_BoschHale(temperature_keV)
    if model == "hively":
        return sigmav_DT_Hively(temperature_keV)
    if model == "auto":
        sig_dt_bh = sigmav_DT_BoschHale(temperature_keV)
        sig_dt_hv = sigmav_DT_Hively(temperature_keV)
        return np.where(temperature_keV < 50.0, sig_dt_hv, sig_dt_bh)
    raise ValueError(f"Unsupported reactivity_model '{model}'. Use 'bosch_hale', 'hively', or 'auto'.")


def _select_sigmav_dd(
    temperature_keV: NDArray[np.float64], model: Literal["bosch_hale", "hively", "auto"]
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    """Select DD reactivity tuple (total, D(d,p)T, D(d,n)3He)."""
    if model in ("bosch_hale", "auto"):
        return sigmav_DD_BoschHale(temperature_keV)
    if model == "hively":
        return sigmav_DD_Hively(temperature_keV)
    raise ValueError(f"Unsupported reactivity_model '{model}'. Use 'bosch_hale', 'hively', or 'auto'.")


def calculate_neutron_emission_probabilities(
    fuel_composition: Mapping[str, ArrayLike],
    ion_temperature_keV: ArrayLike,
    total_ion_density_cm3: ArrayLike | None,
    volume_profile_cm3: ArrayLike | None = None,
    *,
    absolute_densities: bool = False,
    normalize_fractions: bool = True,
    reactivity_model: Literal["bosch_hale", "hively", "auto"] = "bosch_hale",
    include_secondary_equilibrium: bool = True,
    tau_particle_s: float | None = None,
    reactivity_tt: Callable[[NDArray[np.float64]], NDArray[np.float64]] | None = None,
    reactivity_the3: Callable[[NDArray[np.float64]], NDArray[np.float64]] | None = None,
) -> NeutronEmissionResult:
    """Calculate neutron emission probabilities for each neutron energy.

    Supports both 0D averages and 1D profiles. If a profile is provided for any
    input (temperature, composition, density, volume), all other inputs are
    broadcast to match. For profiles, use ``volume_profile_cm3`` to weight
    shell-wise reaction rates; if a scalar volume is passed, it is evenly
    distributed across the profile points.

    Args:
        fuel_composition: Mapping of species symbol to either a fractional
            abundance or an absolute number density profile [cm^-3]. Fractions
            are normalized if ``normalize_fractions`` is True.
        ion_temperature_keV: Ion temperature (scalar or profile) in keV.
        total_ion_density_cm3: Total ion density [cm^-3] (scalar or profile).
            Required when ``absolute_densities`` is False. Ignored when True.
        volume_profile_cm3: Total plasma volume [cm^3] or a profile of shell
            volumes matching the other profiles. If omitted, unit volumes are
            assumed (suitable for relative probabilities).
        absolute_densities: If True, values in ``fuel_composition`` are treated
            as absolute number densities [cm^-3]; otherwise they are fractions.
        normalize_fractions: When using fractional composition, normalize the
            fractions so they sum to 1.0 at each point.
        reactivity_model: Either ``"bosch_hale"``, ``"hively"``, or ``"auto"`` to select the
            <sigma v> fit used. ``"auto"`` uses Hively below 50 keV for DT and Bosch-Hale otherwise.
        include_secondary_equilibrium: If True, include equilibrium T and He3 bred from DD
            branches feeding DT and DHe3. Set False to ignore secondary breeding.
        tau_particle_s: Optional particle confinement time [s] used to balance bred T/He3 against
            burn-up and losses. If None, losses are ignored (infinite confinement).
        reactivity_tt: Optional callable returning <σv>_TT [cm^3/s] for a temperature array.
            If omitted, uses NRL table fit (TT column). Includes T+T neutron production (3.77 MeV).
        reactivity_the3: Optional callable returning <σv>_{THe3} [cm^3/s] for a temperature array.
            If omitted, uses NRL table fit (T–He3 column). Includes T+He3 neutron production (4.03 MeV).

    Returns:
        NeutronEmissionResult with:
            - probabilities: neutron emission probability per energy (keys:
              ``"2.45 MeV (DDn)"``, ``"14.1 MeV (DTn)"``, ``"3.77 MeV (TTn)"``,
              ``"4.03 MeV (THe3 n)"``).
            - reaction_rates_s: total neutron production rate [1/s] per channel.
            - total_neutron_rate_s: total neutron production rate [1/s].

    Notes:
        Neutron-producing channels considered:
            - D(d,n)^3He at ~2.45 MeV.
            - D(t,n)^4He at ~14.1 MeV.
            - T(t,2n)^4He at ~3.77 MeV (using NRL table fit by default).
            - T(He3,n)^alpha+p at ~4.03 MeV (using NRL table fit by default).
    """
    if not absolute_densities and total_ion_density_cm3 is None:
        raise ValueError("total_ion_density_cm3 is required when using fractional composition.")

    species_keys = list(fuel_composition.keys())
    temperature_b, density_b, species_profiles = _broadcast_base(
        ion_temperature_keV, total_ion_density_cm3, [fuel_composition[k] for k in species_keys]
    )

    # Build species densities.
    if absolute_densities:
        species_densities = {k: species_profiles[idx] for idx, k in enumerate(species_keys)}
    else:
        fractions_raw = {k: species_profiles[idx] for idx, k in enumerate(species_keys)}
        fraction_sum = np.zeros_like(temperature_b, dtype=float)
        for frac in fractions_raw.values():
            fraction_sum = fraction_sum + frac
        if normalize_fractions:
            fraction_sum = np.where(fraction_sum == 0.0, 1.0, fraction_sum)
        species_densities = {
            k: density_b * frac / fraction_sum for k, frac in fractions_raw.items()  # type: ignore[arg-type]
        }

    # Volume handling.
    if volume_profile_cm3 is None:
        volume_profile = np.ones_like(temperature_b)
    else:
        volume_profile = _as_profile(volume_profile_cm3)
        # If a scalar total volume is provided with a profile, split evenly.
        if volume_profile.size == 1 and temperature_b.size > 1:
            volume_profile = np.full_like(temperature_b, float(volume_profile.item()) / temperature_b.size)
        else:
            volume_profile = np.broadcast_to(volume_profile, temperature_b.shape)

    # Choose reactivity fits.
    sigmav_dt = _select_sigmav_dt(temperature_b, reactivity_model)
    sigmav_dd_tot, sigmav_ddp, sigmav_ddn = _select_sigmav_dd(temperature_b, reactivity_model)
    sigmav_dhe3 = sigmav_DHe3_BoschHale(temperature_b)  # only BH available here
    sigmav_tt_fn = reactivity_tt
    sigmav_the3_fn = reactivity_the3

    n_d = species_densities.get("D", np.zeros_like(temperature_b))
    n_t_base = species_densities.get("T", np.zeros_like(temperature_b))
    n_he3_base = species_densities.get("He3", np.zeros_like(temperature_b))

    # Primary DD branches (per cm^3 s^-1).
    ddp_rate = 0.5 * n_d * n_d * sigmav_ddp  # D(d,p)T
    ddn_rate = 0.5 * n_d * n_d * sigmav_ddn  # D(d,n)3He

    if include_secondary_equilibrium:
        loss_term = 0.0 if tau_particle_s is None else 1.0 / float(tau_particle_s)
        # Avoid division by zero when density or reactivity is zero.
        denom_t = n_d * sigmav_dt + loss_term
        denom_t = np.where(denom_t == 0.0, np.inf, denom_t)
        n_t_bred = ddp_rate / denom_t

        denom_he3 = n_d * sigmav_dhe3 + loss_term
        denom_he3 = np.where(denom_he3 == 0.0, np.inf, denom_he3)
        n_he3_bred = ddn_rate / denom_he3

        n_t = n_t_base + n_t_bred
        n_he3 = n_he3_base + n_he3_bred
    else:
        n_t = n_t_base
        n_he3 = n_he3_base

    # Reaction rates per cm^3.
    dt_rate = n_d * n_t * sigmav_dt
    dhe3_rate = n_d * n_he3 * sigmav_dhe3
    tt_rate = n_t * n_t * 0.5 * (sigmav_tt_fn or sigmav_TT_NRL)(temperature_b)  # factor 0.5 for identical reactants
    the3_rate = n_t * n_he3 * (sigmav_the3_fn or sigmav_THe3_NRL)(temperature_b)

    dt_neutrons = float(np.sum(dt_rate * volume_profile))
    dd_neutrons = float(np.sum(ddn_rate * volume_profile))
    tt_neutrons = float(np.sum(2 * tt_rate * volume_profile)) # each TT reaction produces 2 neutrons
    the3_neutrons = float(np.sum(the3_rate * volume_profile))

    total_neutrons = dt_neutrons + dd_neutrons + tt_neutrons + the3_neutrons
    if total_neutrons == 0.0:
        probabilities = {
            "2.45 MeV (DDn)": 0.0,
            "14.1 MeV (DTn)": 0.0,
            "3.77 MeV (TTn)": 0.0,
            "4.03 MeV (THe3 n)": 0.0,
        }
    else:
        probabilities = {
            "2.45 MeV (DDn)": dd_neutrons / total_neutrons,
            "14.1 MeV (DTn)": dt_neutrons / total_neutrons,
            "3.77 MeV (TTn)": tt_neutrons / total_neutrons,
            "4.03 MeV (THe3 n)": the3_neutrons / total_neutrons,
        }

    reaction_rates = {
        "DDp_rate_s": float(np.sum(ddp_rate * volume_profile)),
        "DDn_2.45MeV": dd_neutrons,
        "DTn_14.1MeV": dt_neutrons,
        "DHe3_rate_s": float(np.sum(dhe3_rate * volume_profile)),
        "TT_rate_s": tt_neutrons,
        "THe3_rate_s": the3_neutrons,
    }

    return NeutronEmissionResult(probabilities=probabilities, reaction_rates_s=reaction_rates, total_neutron_rate_s=total_neutrons)
