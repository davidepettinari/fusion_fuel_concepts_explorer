[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.24106293.svg)](https://doi.org/10.5281/zenodo.24106293)

# Fusion Fuel Concepts Explorer (FFCE)

Fusion Fuel Concepts Explorer is a Python project for comparing the radiation
environment produced by different fusion fuels in a tokamak geometry. The model
connects plasma composition and neutron source calculations to Monte Carlo
simulations with OpenMC. Its main focus is neutron flux, energy deposition, and
radiation damage in magnet components as the fuel, coolant, and shielding
materials change.

The repository contains a reusable package, simulation cases, and analysis
notebooks. The available transport cases compare **deuterium–tritium (D–T)** and
**deuterium–helium-3 (D–³He)**; the source module also supports D–D.

## How the model works

The workflow consists of three steps:

1. `SteadyStateSource` uses density, temperature, initial composition, and particle
   confinement times to calculate composition, reaction rates, fusion power, and
   neutron production.
2. The `openmc_model.py` scripts define materials, geometry, source distributions,
   and tallies, then run neutron and photon transport with OpenMC.
3. The `postprocessing.ipynb` notebooks and those in `models/comparison/` convert
   the results into physical quantities, profiles, spectra, plots, and comparison
   tables.

### Plasma composition and neutron source

The main class is
[`SteadyStateSource`](src/fusion_fuel_concepts_explorer/sources/source_equilibrium_solver.py),
exported by `fusion_fuel_concepts_explorer.sources`. Calculations run when the
object is created. The main inputs are:

| Parameter | Meaning |
| --- | --- |
| `source_type` | Default mixture: `"DT"`, `"DD"`, or `"DHe3"`. D–T and D–³He start with equimolar fractions. |
| `volume_m3` | Plasma volume in m³, used to integrate local rates. |
| `n_i`, `T_i` | Total ion density in m⁻³ and ion temperature in keV: scalars, array profiles, or functions of `rho` between 0 and 1. |
| `tau_p` | Common particle confinement time in seconds. It can be overridden for each species using `tau_p_D`, `tau_p_T`, `tau_p_He3`, and `tau_p_He4`. |
| `f_D`, `f_T`, `f_He3`, `f_He4`, `f_imp` | Custom initial fractions. If any fraction is specified, omitted fractions are set to zero and the sum must equal one. |

The model tracks D, T, ³He, and ⁴He, with an optional inert impurity fraction.
It includes eight channels: D–T, both D–D branches, D–³He, T–T, ³He–³He, and
both T–³He branches. Reactivities are implemented in
[`reactivities.py`](src/fusion_fuel_concepts_explorer/sources/reactivities.py):
Bosch–Hale parameterizations for D–T, D–D, and D–³He, and CF88 for the remaining
channels.

When at least one confinement time is finite, the solver seeks a steady-state
composition independently at each radial point. It balances fusion consumption
and production, particle losses, and implicit fueling that keeps the total
density of the tracked species constant. The fueling composition is fixed by
the initial fractions. If all confinement times are omitted or infinite, the
calculation instead retains the specified composition and evaluates its rates
without solving for equilibrium.

Local rates are proportional to `n_a * n_b * <sigma v>`, with a factor of `1/2`
for identical reactants. Profile integration uses the radial weight `2*rho` and
the specified volume. Outputs include:

- `species_density_m3_profile`, `species_fraction_profile`, and
  `resolved_fractions`: composition profiles and average fractions;
- `reaction_rates_s`: reactions per second for each channel;
- `fusion_power_by_channel_W` and `total_fusion_power_W`: fusion power by channel
  and in total, calculated from the reaction Q-values;
- `neutron_energies_eV`, `neutron_emission_probabilities`, and
  `total_neutron_rate_s`: discrete energy distribution and neutron emission rate.

The neutron-producing channels are D–T, the neutron-emitting D–D branch, T–T,
and the T–³He → ⁴He + n + p branch. The representative energies assigned by the
code are **14.1, 2.45, 3.77, and 4.03 MeV**, respectively. Probabilities are
normalized to total neutron production and account for the two neutrons emitted
per T–T reaction. These discrete energies are a simplified representation of
the spectra.

A plasma fueled with D–³He can therefore also produce neutrons: D–D reactions
generate neutrons and tritium, which participates in further secondary reactions.

### Geometry and OpenMC transport

Component outlines are defined in the R–Z plane, in centimeters, in
[`geometry/component_nodes.py`](src/fusion_fuel_concepts_explorer/geometry/component_nodes.py).
The scripts build cells through the `tokamak_radiation_environment` dependency,
using the `core_group`, `tfcoil_group`, and `pfcoil_group` functions. The available
cases model a **20°** toroidal sector, from −10° to +10°, with reflective side
boundaries and an outer enclosure with a vacuum boundary condition.

The geometry includes the plasma, scrape-off layer (SOL), tungsten first wall,
Inconel 718 vacuum vessel structures, cooling channel, an intermediate layer
that may contain beryllium, blanket, and SS304 steel shield. It also includes the
toroidal field (TF) coil, six poloidal field (PF) coils, and six central solenoid
sections. Each magnet consists of a homogenized winding pack containing YBCO tape,
surrounded by fiberglass insulation and an SS316L casing.

The SOL region is void. In the D–T cases, the plasma cell is also void; in the
D–³He cases, it is filled with a D/³He material at a fixed density. This transport
material is not updated with the composition calculated by the solver.

In the current scripts, the first wall is 0.5 cm thick; the vessel layers are
2, 2, 1, and 2 cm thick, followed by a 60 cm blanket and a 2 cm shield.
Materials and thicknesses are defined directly in each script.

The OpenMC source is a fixed, isotropic neutron source distributed along an arc
at `R = 330 cm` and `z = 0`, within the toroidal sector. Energy probabilities come
from `SteadyStateSource`: the radial plasma profiles determine the integrated
rates and spectrum, but **are not sampled as the spatial distribution of the
transport source**. Secondary photon transport is enabled; variance reduction
uses the CADIS weight windows stored in the fuel directories.

Tallies record neutron flux, energy deposition, gas production, tritium production
through `(n,Xt)`, damage energy (`damage-energy`), neutron spectra using TRIPOLI-315
groups, and radial profiles. Some quantities are tallied on meshes, others in the
TF magnet, insulation, and casing cells. Damage energy is an input to subsequent
DPA estimates, rather than an already normalized DPA value.

### Available configurations

| Directory under `models/` | Fuel | Material configuration |
| --- | --- | --- |
| `d_t/flibe/` | D–T | FLiBe in the channel and blanket, with lithium enriched to 90% ⁶Li; Be multiplier layer. |
| `d_t/inconel_water_config/` | D–T | Homogeneous mixture of 10% Inconel and 90% water by volume in the channel and blanket region; Be layer. |
| `d_he3/flibe/` | D–³He | FLiBe imported from the `tokamak_radiation_environment` material catalog, with a Be layer. |
| `d_he3/be_water_inconel/` | D–³He | Water–Inconel mixture with a Be layer. |
| `d_he3/water_inconel/` | D–³He | Water–Inconel mixture also used in the intermediate layer, without a Be multiplier. |
| `d_he3/borated_water_inconel/` | D–³He | Inconel–borated water mixture with 1000 ppm boron in the water, without a Be multiplier. |

The name `blanket` also identifies a geometric region in the water-based cases,
which do not contain FLiBe for tritium breeding. The presence of tallies named
`TBR` does not by itself imply a self-sufficient tritium fuel cycle.

## Repository structure

```text
fusion_fuel_concepts_explorer/
├── README.md
├── pyproject.toml                 # Package metadata and dependencies
├── LICENSE                        # MIT license
├── src/fusion_fuel_concepts_explorer/
│   ├── __init__.py
│   ├── sources/
│   │   ├── source_equilibrium_solver.py  # Species balances, composition, and source
│   │   ├── reactivities.py              # Fusion reactivities
│   │   ├── README_sources.md
│   │   └── *.ipynb                      # Source examples and studies
│   ├── geometry/
│   │   └── component_nodes.py           # Component outlines in R–Z
│   └── materials/
│       └── cross_sections/              # Intended location for local data
├── models/
│   ├── d_t/
│   │   ├── flibe/
│   │   ├── inconel_water_config/
│   │   └── weight_windows.cadis.wwinp
│   ├── d_he3/
│   │   ├── flibe/
│   │   ├── be_water_inconel/
│   │   ├── water_inconel/
│   │   ├── borated_water_inconel/
│   │   └── weight_windows.cadis.wwinp
│   ├── geometry/openmc_model.py   # Stochastic volume calculation
│   └── comparison/                # Comparisons, plots, and LaTeX tables
└── figures/                       # Model and result images
```

Transport cases contain an `openmc_model.py` script; several cases also include
an exported `model.xml` and a `postprocessing.ipynb` notebook. Material
compositions are defined in the scripts or imported from
`tokamak_radiation_environment.materials`: the FFCE package's `materials/`
directory currently has no independent material catalog.

The script in `models/geometry/` performs stochastic volume calculations. Its
current configuration selects the three TF cells and the outer cell, rather
than running a transport simulation of the complete reactor. The notebooks in
`models/comparison/` compare D–T with D–³He and the different D–³He variants;
`tf_coil_tables.ipynb` generates tables from results saved in the postprocessing
notebooks.

## Installation and usage

The prerequisites are OpenMC with its executable and Python interface, the
`tokamak_radiation_environment` dependency, and cross-section libraries compatible
with the materials and neutron and photon transport. Other declared dependencies
are NumPy, SciPy, pandas, and PyTables; the notebooks also require Jupyter and
Matplotlib.

The current source code requires **Python ≥ 3.10** and **NumPy ≥ 2.0** for its type
aliases and use of `np.trapezoid`. These requirements are stricter than the
metadata currently declared in `pyproject.toml`.

From the repository root, in an environment with these prerequisites:

```bash
python3 -m pip install -e .
```

Cross-section libraries are not distributed with the repository. Set
`OPENMC_CROSS_SECTIONS` to point to your `cross_sections.xml` file. The
`src/fusion_fuel_concepts_explorer/materials/cross_sections/` directory can hold
local data, which are excluded from version control.

Example of a source-only calculation with uniform profiles:

```python
from fusion_fuel_concepts_explorer.sources import SteadyStateSource

source = SteadyStateSource(
    source_type="DHe3",
    volume_m3=150.0,
    n_i=1.3e20,       # m^-3
    T_i=70.0,        # keV
    tau_p=2.5,       # s
)

print(source.resolved_fractions)
print(source.total_neutron_rate_s)       # neutrons/s
print(source.total_fusion_power_W)       # W
```

To run the D–T case with FLiBe, for example:

```bash
cd models/d_t/flibe
python3 openmc_model.py
```

Run each script from its own directory: the weight window path is relative
(`../weight_windows.cadis.wwinp`). The scripts immediately launch OpenMC and write
results to the working directory. Batch, particle, and thread counts are set in
the code and differ between some cases.

## Results and modeling assumptions

OpenMC generates `statepoint.*.h5` files containing tallies and data needed for
postprocessing. The notebooks read these files and use neutron emission rates,
volumes, and unit conversions to calculate fluxes, powers, power densities, and
damage indicators. Fixed-source tallies are normalized per source particle:
absolute values require normalization consistent with `total_neutron_rate_s`,
the reflective sector, and component volumes.

Generated HDF5 and VTK files are excluded from version control. Reusing the
notebooks may require local results or result archives provided with some cases.
Check statepoint paths before execution: several notebooks look for files in
`results/`, while the scripts write to the current directory. Some comparison
notebooks, including `tf_coil_tables.ipynb`, retain references to earlier directory
layouts and require path updates. Saved XML files and outputs should also be
checked against the current scripts.

The fusion power returned by the source includes the energy of charged products
and neutrons; it is not the same as the power deposited in components.

The source model assumes prescribed density and temperature profiles, local
species balances, and simplified radial weighting. It does not solve the plasma
energy balance, magnetic equilibrium, or radial particle transport. The OpenMC
model uses a sector geometry and homogenized materials; results depend on source
settings, cross sections, Monte Carlo statistics, and postprocessing assumptions.

## License

The project is distributed under the [MIT license](LICENSE).
