"""Steady-state plasma composition and neutron-source solver."""

from __future__ import annotations

import math
from pprint import pformat
from typing import Callable

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import root

from .reactivities import (
    sigmav_DDn_BoschHale,
    sigmav_DDp_BoschHale,
    sigmav_DHe3_BoschHale,
    sigmav_DT_BoschHale,
    sigmav_He3He3_CF88,
    sigmav_THe3_D_CF88,
    sigmav_THe3_np_CF88,
    sigmav_TT_CF88,
)

ProfileInput = float | Callable[[float], float] | NDArray[np.float64] | list[float] | tuple[float, ...]

_TRACKED_SPECIES = {
    "D": {"fraction_field": "f_D", "tau_field": "tau_p_D"},
    "T": {"fraction_field": "f_T", "tau_field": "tau_p_T"},
    "He3": {"fraction_field": "f_He3", "tau_field": "tau_p_He3"},
    "He4": {"fraction_field": "f_He4", "tau_field": "tau_p_He4"},
}
_INERT_SPECIES = {
    "imp": {"fraction_field": "f_imp"},
}
_ALL_SOURCE_SPECIES = {**_TRACKED_SPECIES, **_INERT_SPECIES}
# Central reaction registry: reactivity model plus metadata used for neutron and power bookkeeping.
_REACTIONS = {
    "DT": {"reactivity": sigmav_DT_BoschHale, "q_mev": 17.5893007, "neutron_energy_ev": 14.1e6, "neutrons": 1.0},
    "DDn": {"reactivity": sigmav_DDn_BoschHale, "q_mev": 3.2689087, "neutron_energy_ev": 2.45e6, "neutrons": 1.0},
    "DDp": {"reactivity": sigmav_DDp_BoschHale, "q_mev": 4.0326634, "neutron_energy_ev": None, "neutrons": 0.0},
    "DHe3": {"reactivity": sigmav_DHe3_BoschHale, "q_mev": 18.3530554, "neutron_energy_ev": None, "neutrons": 0.0},
    "TT": {"reactivity": sigmav_TT_CF88, "q_mev": 11.3320710, "neutron_energy_ev": 3.77e6, "neutrons": 2.0},
    "He3He3": {"reactivity": sigmav_He3He3_CF88, "q_mev": 12.8595804, "neutron_energy_ev": None, "neutrons": 0.0},
    "THe3_D": {"reactivity": sigmav_THe3_D_CF88, "q_mev": 14.3203920, "neutron_energy_ev": None, "neutrons": 0.0},
    "THe3_np": {"reactivity": sigmav_THe3_np_CF88, "q_mev": 12.0958257, "neutron_energy_ev": 4.03e6, "neutrons": 1.0},
}
# Default injected species mix associated with each named source configuration.
_SOURCE_MIXES = {
    "DT": {"D": 0.5, "T": 0.5, "He3": 0.0, "He4": 0.0},
    "DD": {"D": 1.0, "T": 0.0, "He3": 0.0, "He4": 0.0},
    "DHe3": {"D": 0.5, "T": 0.0, "He3": 0.5, "He4": 0.0},
}
_J_PER_MEV = 1.602176634e-13
_DEFAULT_RHO_GRID = np.linspace(0.0, 1.0, 101, dtype=float)
_FRACTION_SUM_TOL = 1.0e-9


def plasma_balance_ode(
    n_D: float,
    n_T: float,
    n_He3: float,
    n_He4: float,
    sigmav_DT: float,
    sigmav_DDn: float,
    sigmav_DDp: float,
    sigmav_DHe3: float,
    sigmav_TT: float,
    sigmav_He3He3: float,
    sigmav_THe3_D: float,
    sigmav_THe3_np: float,
    tau_p_D: float | None,
    tau_p_T: float | None,
    tau_p_He3: float | None,
    tau_p_He4: float | None,
    *,
    injection_fractions: np.ndarray | tuple[float, float, float, float] | None = None,
) -> tuple[float, float, float, float]:
    """Return the tracked-species ODE with implicit fueling to hold total density fixed.

    Args:
        n_D: Deuterium density.
        n_T: Tritium density.
        n_He3: Helium-3 density.
        n_He4: Helium-4 density.
        sigmav_DT: DT reactivity.
        sigmav_DDn: DDn reactivity.
        sigmav_DDp: DDp reactivity.
        sigmav_DHe3: DHe3 reactivity.
        sigmav_TT: TT reactivity.
        sigmav_He3He3: He3He3 reactivity.
        sigmav_THe3_D: THe3-to-D branch reactivity.
        sigmav_THe3_np: THe3-to-np branch reactivity.
        tau_p_D: Deuterium particle confinement time.
        tau_p_T: Tritium particle confinement time.
        tau_p_He3: Helium-3 particle confinement time.
        tau_p_He4: Helium-4 particle confinement time.
        injection_fractions: Optional fuel split ``(f_D, f_T, f_He3, f_He4)``. When
            ``None``, use the local state fractions.

    Returns:
        Time derivatives ``(dn_D/dt, dn_T/dt, dn_He3/dt, dn_He4/dt)`` including
        the implicit fueling source.
    """
    # Combine fusion production, fusion burn, and particle losses for D.
    dn_D_dt = (
        - n_D * n_T * sigmav_DT
        - n_D**2 * (sigmav_DDn + sigmav_DDp)
        - n_D * n_He3 * sigmav_DHe3
        + n_T * n_He3 * sigmav_THe3_D
        - (0.0 if tau_p_D is None else 1.0 / float(tau_p_D)) * n_D
    )

    # Combine fusion production, fusion burn, and particle losses for T.
    dn_T_dt = (
        + 0.5 * n_D**2 * sigmav_DDp
        - n_D * n_T * sigmav_DT
        - n_T**2 * sigmav_TT
        - n_T * n_He3 * (sigmav_THe3_D + sigmav_THe3_np)
        - (0.0 if tau_p_T is None else 1.0 / float(tau_p_T)) * n_T
    )

    # Combine fusion production, fusion burn, and particle losses for He3.
    dn_He3_dt = (
        + 0.5 * n_D**2 * sigmav_DDn
        - n_D * n_He3 * sigmav_DHe3
        - n_He3**2 * sigmav_He3He3
        - n_T * n_He3 * (sigmav_THe3_D + sigmav_THe3_np)
        - (0.0 if tau_p_He3 is None else 1.0 / float(tau_p_He3)) * n_He3
    )

    # Combine fusion production and particle losses for He4 ash.
    dn_He4_dt = (
        + n_D * n_T * sigmav_DT
        + n_D * n_He3 * sigmav_DHe3
        + 0.5 * n_T**2 * sigmav_TT
        + 0.5 * n_He3**2 * sigmav_He3He3
        + n_T * n_He3 * (sigmav_THe3_D + sigmav_THe3_np)
        - (0.0 if tau_p_He4 is None else 1.0 / float(tau_p_He4)) * n_He4
    )

    # Determine the fueling split that keeps the total density fixed.
    total_density = n_D + n_T + n_He3 + n_He4
    if total_density <= 0.0:
        # Skip the implicit source if the total density is zero.
        return dn_D_dt, dn_T_dt, dn_He3_dt, dn_He4_dt

    if injection_fractions is None:
        # Default to feeding the same split as the local state.
        feed = np.asarray(
            [n_D, n_T, n_He3, n_He4],
            dtype=float,
        ) / total_density
    else:
        # Validate and normalize the provided injection split.
        feed = np.asarray(injection_fractions, dtype=float)
        if feed.shape != (4,) or not np.isfinite(feed).all() or np.any(feed < 0.0):
            raise ValueError("injection_fractions must be a length-4 non-negative vector")
        if float(np.sum(feed)) <= 0.0:
            raise ValueError("injection_fractions must sum to a positive value")
        feed = feed / float(np.sum(feed))

    # Compute the total balance and distribute the source to hold total density.
    net_balance = dn_D_dt + dn_T_dt + dn_He3_dt + dn_He4_dt
    source = -net_balance * feed

    return (
        dn_D_dt + float(source[0]),
        dn_T_dt + float(source[1]),
        dn_He3_dt + float(source[2]),
        dn_He4_dt + float(source[3]),
    )


def steady_state_plasma_composition(
    n_D: np.ndarray,
    n_T: np.ndarray,
    n_He3: np.ndarray,
    n_He4: np.ndarray,
    sigmav_DT: np.ndarray,
    sigmav_DDn: np.ndarray,
    sigmav_DDp: np.ndarray,
    sigmav_DHe3: np.ndarray,
    sigmav_TT: np.ndarray,
    sigmav_He3He3: np.ndarray,
    sigmav_THe3_D: np.ndarray,
    sigmav_THe3_np: np.ndarray,
    tau_p_D: float | None,
    tau_p_T: float | None,
    tau_p_He3: float | None,
    tau_p_He4: float | None,
    *,
    tol: float = 1e-10,
    max_iter: int = 500,
    method: str = "hybr",
) -> dict[str, np.ndarray]:
    """Solve the pointwise steady-state D/T/He3/He4 composition.

    Uses a root-finding approach to solve the local composition ODE at each
    point. The implicit fueling source keeps the total density fixed, and its
    species split mirrors the seeded input densities at each point.

    #NOTE: impurities are not included since they do not participate in fusion reactions.
    Since the function considers densities and not fractions, impurity density is not needed.

    Args:
        n_D: Seeded deuterium density profile.
        n_T: Seeded tritium density profile.
        n_He3: Seeded helium-3 density profile.
        n_He4: Seeded helium-4 density profile.
        sigmav_DT: DT reactivity profile.
        sigmav_DDn: DDn reactivity profile.
        sigmav_DDp: DDp reactivity profile.
        sigmav_DHe3: DHe3 reactivity profile.
        sigmav_TT: TT reactivity profile.
        sigmav_He3He3: He3He3 reactivity profile.
        sigmav_THe3_D: THe3 to D branch reactivity profile.
        sigmav_THe3_np: THe3 to np branch reactivity profile.
        tau_p_D: Deuterium particle confinement time.
        tau_p_T: Tritium particle confinement time.
        tau_p_He3: Helium-3 particle confinement time.
        tau_p_He4: Helium-4 particle confinement time.
        tol: Root-solver tolerance.
        max_iter: Maximum root-solver evaluations per radial point.
        method: SciPy root-solver method.

    Returns:
        Solved tracked density profiles.
    """
    # Validate the profile inputs before the pointwise solve.
    profiles = (
        n_D,
        n_T,
        n_He3,
        n_He4,
        sigmav_DT,
        sigmav_DDn,
        sigmav_DDp,
        sigmav_DHe3,
        sigmav_TT,
        sigmav_He3He3,
        sigmav_THe3_D,
        sigmav_THe3_np,
    )
    # Ensure every input profile is a 1D numpy array.
    if not all(isinstance(v, np.ndarray) and v.ndim == 1 for v in profiles):
        raise TypeError(
            "Density and reactivity inputs must already be 1D numpy arrays before relation evaluation."
        )
    # Ensure every input profile has the same length.
    if len({v.size for v in profiles}) != 1:
        raise ValueError("Density and reactivity profiles must all have the same length")

    # Validate that confinement times are positive when supplied.
    for name, tau in (
        ("tau_p_D", tau_p_D),
        ("tau_p_T", tau_p_T),
        ("tau_p_He3", tau_p_He3),
        ("tau_p_He4", tau_p_He4),
    ):
        if tau is not None and ((tau := float(tau)) <= 0.0 or not math.isfinite(tau)):
            raise ValueError(f"{name} must be positive or None")

    n_points = n_D.size

    # Allocate the solved density profiles.
    out_D = np.zeros_like(n_D, dtype=float)
    out_T = np.zeros_like(n_T, dtype=float)
    out_He3 = np.zeros_like(n_He3, dtype=float)
    out_He4 = np.zeros_like(n_He4, dtype=float)

    # Solve the steady-state composition independently at each radial point.
    for i in range(n_points):
        # Pack the seed densities for this point and validate them.
        input_densities = np.asarray([n_D[i], n_T[i], n_He3[i], n_He4[i]], dtype=float)
        if not np.isfinite(input_densities).all() or np.any(input_densities < 0.0):
            raise ValueError("Seeded densities must be finite and non-negative")
        total_density = float(np.sum(input_densities))
        # Skip points with zero total density.
        if total_density <= 0.0:
            continue
        # Normalize the seed densities to get the implicit injection split.
        initial_density_fractions = input_densities / total_density

        def residual(fractions: np.ndarray) -> np.ndarray:
            """Compute the local steady-state residual with an implicit source.

            Args:
                fractions: Candidate density fractions at this radial point.

            Returns:
                Residual vector for D, T, He3, and the total-density constraint.
            """
            # Work in fractions to keep the solver scale near unity.
            fractions = np.asarray(fractions, dtype=float)
            # Convert fractions back to physical densities.
            state = total_density * fractions

            # Evaluate the balance with the seeded fuel split.
            dn_D_dt, dn_T_dt, dn_He3_dt, dn_He4_dt = plasma_balance_ode(
                n_D=float(state[0]),
                n_T=float(state[1]),
                n_He3=float(state[2]),
                n_He4=float(state[3]),
                sigmav_DT=float(sigmav_DT[i]),
                sigmav_DDn=float(sigmav_DDn[i]),
                sigmav_DDp=float(sigmav_DDp[i]),
                sigmav_DHe3=float(sigmav_DHe3[i]),
                sigmav_TT=float(sigmav_TT[i]),
                sigmav_He3He3=float(sigmav_He3He3[i]),
                sigmav_THe3_D=float(sigmav_THe3_D[i]),
                sigmav_THe3_np=float(sigmav_THe3_np[i]),
                tau_p_D=tau_p_D,
                tau_p_T=tau_p_T,
                tau_p_He3=tau_p_He3,
                tau_p_He4=tau_p_He4,
                injection_fractions=initial_density_fractions,
            )

            # Enforce D, T, and He3 balances together with the total-density target.
            return np.asarray(
                [
                    dn_D_dt / total_density,
                    dn_T_dt / total_density,
                    dn_He3_dt / total_density,
                    float(np.sum(fractions)) - 1.0,
                ],
                dtype=float,
            )

        #NOTE: the solver uses fractions instead of densities to keep the solution scale near unity, which generally improves convergence behavior.
        # The densities are retrieved after the solve.
        result = root(
            residual,
            initial_density_fractions,
            method=method,
            tol=tol,
            options={"maxfev": max_iter, "diag": [1.0, 1.0, 1.0, 1.0]},
        )
        solved_fraction = np.asarray(result.x, dtype=float)

        # Reject invalid converged states before the final residual check.
        if (not np.isfinite(solved_fraction).all()) or (np.any(solved_fraction < 0.0)):
            raise RuntimeError("Steady-state composition solve produced an invalid state")

        # Check that the nonlinear solve satisfied every balance tightly.
        residual_vector = residual(solved_fraction)
        residual_norm = float(np.linalg.norm(residual_vector))
        if (
            not math.isfinite(residual_norm)
            or residual_norm > 1e-10
            or abs(float(residual_vector[3])) > 1e-12
        ):
            raise RuntimeError(
                "Steady-state composition solve did not converge tightly enough "
                f"(status={result.success}, message={result.message!r}, "
                f"D={residual_vector[0]:.3e}, T={residual_vector[1]:.3e}, "
                f"He3={residual_vector[2]:.3e}, total={residual_vector[3]:.3e})"
            )

        # Scale the solved fractions back to densities and store the result.
        solved_state = total_density * solved_fraction
        out_D[i], out_T[i], out_He3[i], out_He4[i] = solved_state

    return {"n_D": out_D, "n_T": out_T, "n_He3": out_He3, "n_He4": out_He4}


class SteadyStateSource:
    """Steady-state composition and neutron-source solver.

    Construction eagerly executes the full solve pipeline. Inputs and derived
    values are stored directly on the instance.
    """

    def __init__(
        self,
        source_type: str,
        volume_m3: float,
        n_i: ProfileInput,
        T_i: ProfileInput,
        *,
        tau_p: float | None = None,
        tau_p_D: float | None = None,
        tau_p_T: float | None = None,
        tau_p_He3: float | None = None,
        tau_p_He4: float | None = None,
        f_D: float | None = None,
        f_T: float | None = None,
        f_He3: float | None = None,
        f_He4: float | None = None,
        f_imp: float | None = None,
        solver_tol: float = 1e-10,
        solver_maxfev: int = 500,
    ) -> None:
        self.source_type = source_type
        self.volume_m3 = float(volume_m3)
        self.n_i = n_i
        self.T_i = T_i
        self.tau_p = None if tau_p is None else float(tau_p)
        self.tau_p_D = None if tau_p_D is None else float(tau_p_D)
        self.tau_p_T = None if tau_p_T is None else float(tau_p_T)
        self.tau_p_He3 = None if tau_p_He3 is None else float(tau_p_He3)
        self.tau_p_He4 = None if tau_p_He4 is None else float(tau_p_He4)
        self.solver_tol = float(solver_tol)
        self.solver_maxfev = int(float(solver_maxfev))

        if self.source_type not in _SOURCE_MIXES:
            raise ValueError("source_type must be one of 'DT', 'DD', or 'DHe3'")

        for name, value in (
            ("tau_p", self.tau_p),
            ("tau_p_D", self.tau_p_D),
            ("tau_p_T", self.tau_p_T),
            ("tau_p_He3", self.tau_p_He3),
            ("tau_p_He4", self.tau_p_He4),
        ):
            if value is not None and (value <= 0.0 or math.isnan(value)):
                raise ValueError(f"{name} must be positive or None")

        self.density_fraction_input_by_species = {
            "D": None if f_D is None else float(f_D),
            "T": None if f_T is None else float(f_T),
            "He3": None if f_He3 is None else float(f_He3),
            "He4": None if f_He4 is None else float(f_He4),
            "imp": None if f_imp is None else float(f_imp),
        }
        explicit_fractions = any(value is not None for value in self.density_fraction_input_by_species.values())
        self.resolved_density_fraction_by_species = (
            {
                species: 0.0 if (value := self.density_fraction_input_by_species[species]) is None else value
                for species in _ALL_SOURCE_SPECIES
            }
            if explicit_fractions
            else {**_SOURCE_MIXES[self.source_type], "imp": 0.0}
        )
        if any(value < 0.0 for value in self.resolved_density_fraction_by_species.values()):
            raise ValueError("Explicit fractions must be >= 0")
        total = sum(self.resolved_density_fraction_by_species.values())
        if not math.isclose(total, 1.0, rel_tol=0.0, abs_tol=_FRACTION_SUM_TOL):
            raise ValueError("Explicit density fractions must sum to 1.0")

        profile_inputs = {"n_i": self.n_i, "T_i": self.T_i}
        profile_sizes: list[int] = []
        for name, value in profile_inputs.items():
            if callable(value) or np.isscalar(value):
                continue
            value = np.asarray(value, dtype=float)
            if value.ndim != 1 or value.size == 0:
                raise ValueError(f"{name} array input must be a non-empty 1D profile")
            profile_inputs[name] = value
            profile_sizes.append(int(value.size))
        if profile_sizes and len(set(profile_sizes)) != 1:
            raise ValueError("n_i and T_i array inputs must have the same length")
        self.rho_grid_profile = (
            np.linspace(0.0, 1.0, profile_sizes[0], dtype=float)
            if profile_sizes
            else _DEFAULT_RHO_GRID.copy()
        )
        for name, value in profile_inputs.items():
            if callable(value):
                values = np.asarray([value(float(rho)) for rho in self.rho_grid_profile], dtype=float)
            elif np.isscalar(value):
                values = np.full_like(self.rho_grid_profile, float(value), dtype=float)
            else:
                values = value
                if values.size != self.rho_grid_profile.size:
                    raise ValueError(f"{name} array input must have the same length as rho_grid_profile")

            if np.any(~np.isfinite(values)):
                raise ValueError(f"{name} must contain only finite values")
            if np.any(values < 0.0):
                raise ValueError(f"{name} must contain values >= 0")
            if name == "n_i":
                self.total_density_m3_profile = values
            else:
                self.temperature_keV_profile = values

        positive_temperature_mask = self.temperature_keV_profile > 0.0
        temperature_keV_safe_profile = np.where(positive_temperature_mask, self.temperature_keV_profile, 1.0)
        self.reactivity_m3_s_profile = {
            reaction: np.where(
                positive_temperature_mask,
                np.asarray(meta["reactivity"](temperature_keV_safe_profile), dtype=float),
                0.0,
            )
            for reaction, meta in _REACTIONS.items()
        }
        base_tau = math.inf if self.tau_p is None else self.tau_p

        self.tau_p_map: dict[str, float] = {}
        for species in _TRACKED_SPECIES:
            tau_value = getattr(self, _TRACKED_SPECIES[species]["tau_field"])
            self.tau_p_map[species] = base_tau if tau_value is None else tau_value
        self.species_density_m3_profile = {
            species: np.asarray(
                self.total_density_m3_profile * self.resolved_density_fraction_by_species[species],
                dtype=float,
            )
            for species in _ALL_SOURCE_SPECIES
        }
        if not all(math.isinf(value) for value in self.tau_p_map.values()):
            solved_density_profiles = steady_state_plasma_composition(
                n_D=self.species_density_m3_profile["D"],
                n_T=self.species_density_m3_profile["T"],
                n_He3=self.species_density_m3_profile["He3"],
                n_He4=self.species_density_m3_profile["He4"],
                sigmav_DT=np.asarray(self.reactivity_m3_s_profile["DT"], dtype=float),
                sigmav_DDn=np.asarray(self.reactivity_m3_s_profile["DDn"], dtype=float),
                sigmav_DDp=np.asarray(self.reactivity_m3_s_profile["DDp"], dtype=float),
                sigmav_DHe3=np.asarray(self.reactivity_m3_s_profile["DHe3"], dtype=float),
                sigmav_TT=np.asarray(self.reactivity_m3_s_profile["TT"], dtype=float),
                sigmav_He3He3=np.asarray(self.reactivity_m3_s_profile["He3He3"], dtype=float),
                sigmav_THe3_D=np.asarray(self.reactivity_m3_s_profile["THe3_D"], dtype=float),
                sigmav_THe3_np=np.asarray(self.reactivity_m3_s_profile["THe3_np"], dtype=float),
                tau_p_D=None if math.isinf(self.tau_p_map["D"]) else self.tau_p_map["D"],
                tau_p_T=None if math.isinf(self.tau_p_map["T"]) else self.tau_p_map["T"],
                tau_p_He3=None if math.isinf(self.tau_p_map["He3"]) else self.tau_p_map["He3"],
                tau_p_He4=None if math.isinf(self.tau_p_map["He4"]) else self.tau_p_map["He4"],
                tol=self.solver_tol,
                max_iter=self.solver_maxfev,
            )
            self.species_density_m3_profile["D"] = solved_density_profiles["n_D"]
            self.species_density_m3_profile["T"] = solved_density_profiles["n_T"]
            self.species_density_m3_profile["He3"] = solved_density_profiles["n_He3"]
            self.species_density_m3_profile["He4"] = solved_density_profiles["n_He4"]

        positive_density_mask = self.total_density_m3_profile > 0.0
        self.species_fraction_profile = {
            species: np.divide(
                self.species_density_m3_profile[species],
                self.total_density_m3_profile,
                out=np.full_like(
                    self.total_density_m3_profile,
                    self.resolved_density_fraction_by_species[species],
                    dtype=float,
                ),
                where=positive_density_mask,
            )
            for species in _ALL_SOURCE_SPECIES
        }
        n_D = self.species_density_m3_profile["D"]
        n_T = self.species_density_m3_profile["T"]
        n_He3 = self.species_density_m3_profile["He3"]
        self.reaction_rate_m3_s_profile = {
            "DT": n_D * n_T * self.reactivity_m3_s_profile["DT"],
            "DDn": 0.5 * n_D * n_D * self.reactivity_m3_s_profile["DDn"],
            "DDp": 0.5 * n_D * n_D * self.reactivity_m3_s_profile["DDp"],
            "DHe3": n_D * n_He3 * self.reactivity_m3_s_profile["DHe3"],
            "TT": 0.5 * n_T * n_T * self.reactivity_m3_s_profile["TT"],
            "He3He3": 0.5 * n_He3 * n_He3 * self.reactivity_m3_s_profile["He3He3"],
            "THe3_D": n_T * n_He3 * self.reactivity_m3_s_profile["THe3_D"],
            "THe3_np": n_T * n_He3 * self.reactivity_m3_s_profile["THe3_np"],
        }
        self.reaction_rates_s = {
            reaction: float(
                self.volume_m3 * (
                    profile[0]
                    if self.rho_grid_profile.size == 1
                    else np.trapezoid(2.0 * self.rho_grid_profile * profile, self.rho_grid_profile)
                )
            )
            for reaction, profile in self.reaction_rate_m3_s_profile.items()
        }

        neutron_channels = [reaction for reaction, meta in _REACTIONS.items() if meta["neutrons"]]
        self.neutron_production_rates_s = np.asarray(
            [
                self.reaction_rates_s[reaction] * float(_REACTIONS[reaction]["neutrons"])
                for reaction in neutron_channels
            ],
            dtype=float,
        )
        self.total_neutron_rate_s = float(np.sum(self.neutron_production_rates_s))
        self.neutron_emission_probabilities = (
            self.neutron_production_rates_s / self.total_neutron_rate_s
            if self.total_neutron_rate_s > 0.0
            else np.zeros(len(neutron_channels), dtype=float)
        )
        self.neutron_energies_eV = np.asarray(
            [_REACTIONS[reaction]["neutron_energy_ev"] for reaction in neutron_channels],
            dtype=float,
        )
        self.fusion_power_by_channel_W = {
            reaction: self.reaction_rates_s[reaction] * float(meta["q_mev"]) * _J_PER_MEV
            for reaction, meta in _REACTIONS.items()
        }

        self.f_D = float(
            self.species_fraction_profile["D"][0]
            if self.rho_grid_profile.size == 1
            else np.trapezoid(2.0 * self.rho_grid_profile * self.species_fraction_profile["D"], self.rho_grid_profile)
        )
        self.f_T = float(
            self.species_fraction_profile["T"][0]
            if self.rho_grid_profile.size == 1
            else np.trapezoid(2.0 * self.rho_grid_profile * self.species_fraction_profile["T"], self.rho_grid_profile)
        )
        self.f_He3 = float(
            self.species_fraction_profile["He3"][0]
            if self.rho_grid_profile.size == 1
            else np.trapezoid(2.0 * self.rho_grid_profile * self.species_fraction_profile["He3"], self.rho_grid_profile)
        )
        self.f_He4 = float(
            self.species_fraction_profile["He4"][0]
            if self.rho_grid_profile.size == 1
            else np.trapezoid(2.0 * self.rho_grid_profile * self.species_fraction_profile["He4"], self.rho_grid_profile)
        )
        self.f_imp = float(
            self.species_fraction_profile["imp"][0]
            if self.rho_grid_profile.size == 1
            else np.trapezoid(2.0 * self.rho_grid_profile * self.species_fraction_profile["imp"], self.rho_grid_profile)
        )
        self.total_fusion_power_W = float(sum(self.fusion_power_by_channel_W.values()))
        self.average_density_m3 = float(
            self.total_density_m3_profile[0]
            if self.rho_grid_profile.size == 1
            else np.trapezoid(2.0 * self.rho_grid_profile * self.total_density_m3_profile, self.rho_grid_profile)
        )
        self.average_temperature_keV = float(
            self.temperature_keV_profile[0]
            if self.rho_grid_profile.size == 1
            else np.trapezoid(2.0 * self.rho_grid_profile * self.temperature_keV_profile, self.rho_grid_profile)
        )

    def __repr__(self) -> str:
        lines = [f"{type(self).__name__}("]
        display_fields = {
            "source_type": self.source_type,
            "volume_m3": self.volume_m3,
            "n_i": self.n_i,
            "T_i": self.T_i,
            "tau_p": self.tau_p,
            "tau_p_D": self.tau_p_D,
            "tau_p_T": self.tau_p_T,
            "tau_p_He3": self.tau_p_He3,
            "tau_p_He4": self.tau_p_He4,
            "density_fraction_input_by_species": self.density_fraction_input_by_species,
            "resolved_density_fraction_by_species": self.resolved_density_fraction_by_species,
            "f_imp": self.f_imp,
            "solver_tol": self.solver_tol,
            "solver_maxfev": self.solver_maxfev,
        }
        lines += [
            f"  {name}={pformat(value, sort_dicts=False, width=100)},"
            for name, value in display_fields.items()
        ]
        lines += [
            f"  f_D={self.f_D!r},",
            f"  f_T={self.f_T!r},",
            f"  f_He3={self.f_He3!r},",
            f"  f_He4={self.f_He4!r},",
            f"  f_imp={self.f_imp!r},",
            f"  average_density_m3={self.average_density_m3!r},",
            f"  average_temperature_keV={self.average_temperature_keV!r},",
            ")",
        ]
        return "\n".join(lines)

    def plot(self) -> tuple[object, object]:
        try:
            import matplotlib.pyplot as plt
        except ImportError as exc:
            raise ImportError("plot() requires matplotlib to be installed") from exc

        fig, ax_density = plt.subplots(figsize=(10, 6))
        ax_temperature = ax_density.twinx()

        ax_density.plot(
            self.rho_grid_profile,
            self.total_density_m3_profile,
            color="tab:blue",
            linewidth=2,
            label="Density [m^-3]",
        )
        ax_temperature.plot(
            self.rho_grid_profile,
            self.temperature_keV_profile,
            color="tab:red",
            linewidth=2,
            linestyle="--",
            label="Temperature [keV]",
        )

        ax_density.set_xlabel("rho")
        ax_density.set_ylabel("Density [m^-3]", color="tab:blue")
        ax_temperature.set_ylabel("Temperature [keV]", color="tab:red")
        ax_density.tick_params(axis="y", labelcolor="tab:blue")
        ax_temperature.tick_params(axis="y", labelcolor="tab:red")
        ax_density.set_title(f"{type(self).__name__}: density and temperature")
        ax_density.grid(True, alpha=0.3)

        handles_density, labels_density = ax_density.get_legend_handles_labels()
        handles_temperature, labels_temperature = ax_temperature.get_legend_handles_labels()
        ax_density.legend(handles_density + handles_temperature, labels_density + labels_temperature, loc="best")
        fig.tight_layout()
        return fig, ax_density
