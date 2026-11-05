import openmc
import numpy as np
from pathlib import Path
import tokamak_radiation_environment as tre
from fusion_fuel_concepts_explorer.sources import SteadyStateSource
from fusion_fuel_concepts_explorer.geometry import component_nodes as cn

# define materials
dt_plasma = openmc.Material(name='dt_plasma')
dt_plasma.add_nuclide('H2', 1.0)
dt_plasma.add_nuclide('He3', 1.0)
dt_plasma.set_density('g/cm3', 1e-5)

tungsten = openmc.Material(name='tungsten')
tungsten.add_element('W', 1.0)
tungsten.set_density('g/cm3', 19.3)

inconel = openmc.Material(name='Inconel 718')
inconel.add_element('Al', 0.52e-2, 'wo')
inconel.add_element('C', 0.021e-2, 'wo')
inconel.add_element('Co', 0.11e-2, 'wo')
inconel.add_element('Cr', 19.06e-2, 'wo')
inconel.add_element('Cu', 0.02e-2, 'wo')
inconel.add_element('Fe', 18.15e-2, 'wo')
inconel.add_element('Mo', 3.04e-2, 'wo')
inconel.add_element('Ti', 0.93e-2, 'wo')
inconel.add_element('Nb', 5.08e-2, 'wo')
inconel.add_element('Ni', 53.0e-2, 'wo')
inconel.set_density('g/cm3', 8.19)

# beryllium = tre.materials.beryllium
# eurofer = tre.materials.eurofer97
ss304 = tre.materials.ss304
ss316L = tre.materials.ss316L
b4c = tre.materials.b4c
wc = tre.materials.wc

borated_water = openmc.model.borated_water(
    boron_ppm=20000.0,
    temperature=343.15,
    pressure=0.1013,
    temp_unit="K",
    press_unit="MPa",
    name="Borated water",
)

water_for_mix = openmc.Material(name="Borated water without SAB")

for nuclide in borated_water.nuclides:
    water_for_mix.add_nuclide(
        nuclide.name,
        nuclide.percent,
        nuclide.percent_type,
    )

water_for_mix.set_density(
    "g/cm3",
    borated_water.get_mass_density(),
)

new_bz_structures = openmc.Material.mix_materials(
    materials=[inconel, water_for_mix],
    fracs=[0.10, 0.90],
    percent_type="vo",
    name="10% Inconel + 90% Borated water",
    temperature=293.0,
)

new_bz_structures.add_s_alpha_beta("c_H_in_H2O")

windingpack = tre.materials.windingpack
fiberglass = tre.materials.fiberglass

materials = openmc.Materials(
    [dt_plasma, tungsten, new_bz_structures, wc, inconel, ss304, windingpack, fiberglass, ss316L])

# geometry

# boundary surfaces
angle = (-10, 10)

# components

# core
plasma, sol, first_wall, vessel_inner_structure, vessel_cooling_channel, vessel_neutron_multiplier, vessel_outer_structure, blanket, shield = tre.components.core_group(
    plasma_outer_nodes=cn.plasma_out, plasma_material=dt_plasma,
    firstwall_inner_nodes=cn.fw_in, firstwall_thickness=.5, firstwall_material=tungsten,
    vv_stri_thickness=2., vv_stri_material=inconel,
    vv_channel_thickness=2., vv_channel_material=new_bz_structures, #
    vv_multiplier_thickness=1., vv_multiplier_material=new_bz_structures,
    vv_stro_thickness=2., vv_stro_material=inconel,
    blanket_thickness=60., blanket_material=new_bz_structures, #
    shield_thickness=2., shield_material=ss304,
    angle=angle)

# tf coil
tf_coil_magnet, tf_coil_insulation, tf_coil_case = tre.components.tfcoil_group(
    magnet_inner_nodes=cn.tf_in, magnet_thickness=9., magnet_material=windingpack,
    insulation_thickness=14., insulation_material=fiberglass,
    case_thickness=14., case_material=ss316L,
    angle=angle)

# pf coils
# u1
pf_u1_magnet, pf_u1_insulation, pf_u1_case = tre.components.pfcoil_group(
    magnet_nodes=cn.pf_u1, magnet_material=windingpack,
    insulation_thickness=10., insulation_material=fiberglass,
    case_thickness=10., case_material=ss316L,
    angle=angle)

# u2
pf_u2_magnet, pf_u2_insulation, pf_u2_case = tre.components.pfcoil_group(
    magnet_nodes=cn.pf_u2, magnet_material=windingpack,
    insulation_thickness=10., insulation_material=fiberglass,
    case_thickness=10., case_material=ss316L,
    angle=angle)

# u3
pf_u3_magnet, pf_u3_insulation, pf_u3_case = tre.components.pfcoil_group(
    magnet_nodes=cn.pf_u3, magnet_material=windingpack,
    insulation_thickness=10., insulation_material=fiberglass,
    case_thickness=10., case_material=ss316L,
    angle=angle)

# l1
pf_l1_magnet, pf_l1_insulation, pf_l1_case = tre.components.pfcoil_group(
    magnet_nodes=cn.pf_l1, magnet_material=windingpack,
    insulation_thickness=10., insulation_material=fiberglass,
    case_thickness=10., case_material=ss316L,
    angle=angle)

# l2
pf_l2_magnet, pf_l2_insulation, pf_l2_case = tre.components.pfcoil_group(
    magnet_nodes=cn.pf_l2, magnet_material=windingpack,
    insulation_thickness=10., insulation_material=fiberglass,
    case_thickness=10., case_material=ss316L,
    angle=angle)

# l3
pf_l3_magnet, pf_l3_insulation, pf_l3_case = tre.components.pfcoil_group(
    magnet_nodes=cn.pf_l3, magnet_material=windingpack,
    insulation_thickness=10., insulation_material=fiberglass,
    case_thickness=10., case_material=ss316L,
    angle=angle)

# central solenoid
# u1
cs_u1_magnet, cs_u1_insulation, cs_u1_case = tre.components.pfcoil_group(
    magnet_nodes=cn.cs_u1, magnet_material=windingpack,
    insulation_thickness=5., insulation_material=fiberglass,
    case_thickness=5., case_material=ss316L,
    angle=angle)

# u2
cs_u2_magnet, cs_u2_insulation, cs_u2_case = tre.components.pfcoil_group(
    magnet_nodes=cn.cs_u2, magnet_material=windingpack,
    insulation_thickness=5., insulation_material=fiberglass,
    case_thickness=5., case_material=ss316L,
    angle=angle)

# u3
cs_u3_magnet, cs_u3_insulation, cs_u3_case = tre.components.pfcoil_group(
    magnet_nodes=cn.cs_u3, magnet_material=windingpack,
    insulation_thickness=5., insulation_material=fiberglass,
    case_thickness=5., case_material=ss316L,
    angle=angle)

# l1
cs_l1_magnet, cs_l1_insulation, cs_l1_case = tre.components.pfcoil_group(
    magnet_nodes=cn.cs_l1, magnet_material=windingpack,
    insulation_thickness=5., insulation_material=fiberglass,
    case_thickness=5., case_material=ss316L,
    angle=angle)

# l2
cs_l2_magnet, cs_l2_insulation, cs_l2_case = tre.components.pfcoil_group(
    magnet_nodes=cn.cs_l2, magnet_material=windingpack,
    insulation_thickness=5., insulation_material=fiberglass,
    case_thickness=5., case_material=ss316L,
    angle=angle)

# l3
cs_l3_magnet, cs_l3_insulation, cs_l3_case = tre.components.pfcoil_group(
    magnet_nodes=cn.cs_l3, magnet_material=windingpack,
    insulation_thickness=5., insulation_material=fiberglass,
    case_thickness=5., case_material=ss316L,
    angle=angle)


reactor_components = [plasma, sol, first_wall, vessel_inner_structure, vessel_cooling_channel, vessel_neutron_multiplier, vessel_outer_structure, blanket, shield,
                      cs_u1_magnet, cs_u1_insulation, cs_u1_case, cs_u2_magnet, cs_u2_insulation, cs_u2_case, cs_u3_magnet, cs_u3_insulation, cs_u3_case,
                      cs_l1_magnet, cs_l1_insulation, cs_l1_case, cs_l2_magnet, cs_l2_insulation, cs_l2_case, cs_l3_magnet, cs_l3_insulation, cs_l3_case,
                      pf_u1_magnet, pf_u1_insulation, pf_u1_case, pf_u2_magnet, pf_u2_insulation, pf_u2_case, pf_u3_magnet, pf_u3_insulation, pf_u3_case,
                      pf_l1_magnet, pf_l1_insulation, pf_l1_case, pf_l2_magnet, pf_l2_insulation, pf_l2_case, pf_l3_magnet, pf_l3_insulation, pf_l3_case,
                      tf_coil_magnet, tf_coil_insulation, tf_coil_case]


# building enclosure
enclosure_surf = openmc.Sphere(r=5000, boundary_type='vacuum')
enclosure_region = -enclosure_surf
for rc in reactor_components:
    enclosure_region = enclosure_region & ~(rc.region)
enclosure_cell = openmc.Cell(region=enclosure_region, fill=None)

root = [plasma.cell, sol.cell, first_wall.cell, vessel_inner_structure.cell, vessel_cooling_channel.cell,
        vessel_neutron_multiplier.cell, vessel_outer_structure.cell, blanket.cell, shield.cell,
        cs_u1_magnet.cell, cs_u1_insulation.cell, cs_u1_case.cell, cs_u2_magnet.cell, cs_u2_insulation.cell, cs_u2_case.cell,
        cs_u3_magnet.cell, cs_u3_insulation.cell, cs_u3_case.cell, cs_l1_magnet.cell, cs_l1_insulation.cell, cs_l1_case.cell,
        cs_l2_magnet.cell, cs_l2_insulation.cell, cs_l2_case.cell, cs_l3_magnet.cell, cs_l3_insulation.cell, cs_l3_case.cell,
        pf_u1_magnet.cell, pf_u1_insulation.cell, pf_u1_case.cell, pf_u2_magnet.cell, pf_u2_insulation.cell, pf_u2_case.cell,
        pf_u3_magnet.cell, pf_u3_insulation.cell, pf_u3_case.cell, pf_l1_magnet.cell, pf_l1_insulation.cell, pf_l1_case.cell,
        pf_l2_magnet.cell, pf_l2_insulation.cell, pf_l2_case.cell, pf_l3_magnet.cell, pf_l3_insulation.cell, pf_l3_case.cell,
        tf_coil_magnet.cell, tf_coil_insulation.cell, tf_coil_case.cell,
        enclosure_cell]

geometry = openmc.Geometry(root=root)

geometry.merge_surfaces = True

# settings

# weight windows from attila4mc
ww = openmc.wwinp_to_wws("../weight_windows.cadis.wwinp")

# source definition
source = openmc.IndependentSource()
source.particle = 'neutron'
radius = openmc.stats.Discrete([330], [1])
z_values = openmc.stats.Discrete([0], [1])
angle = openmc.stats.Uniform(a=np.radians(-10), b=np.radians(10))
source.space = openmc.stats.CylindricalIndependent(
    r=radius, phi=angle, z=z_values, origin=(0., 0., 0.))
source.angle = openmc.stats.Isotropic()
# source.energy = openmc.stats.muir(e0=14.08e6, m_rat=5, kt=20000)

# Source Data
avg_density_Dhe3 = 1.3e20
avg_temperature_Dhe3 = 70.0
density_peaking_DHe3 = 2.5
temperature_peaking_DHe3 = 2
density_profile_DHe3 = lambda rho: avg_density_Dhe3 * density_peaking_DHe3 * ((1.0 - rho**2.0) ** (density_peaking_DHe3 - 1.0))
temperature_profile_DHe3 = lambda rho: avg_temperature_Dhe3 * temperature_peaking_DHe3 * ((1.0 - rho**2.0) ** (temperature_peaking_DHe3 - 1.0))

res_avg = SteadyStateSource(
    source_type="DHe3",
    volume_m3=150.0,
    n_i=density_profile_DHe3,
    T_i=temperature_profile_DHe3,
    tau_p=2.5,
)
order = np.argsort(res_avg.neutron_energies_eV)
energies = res_avg.neutron_energies_eV[order]
weights = res_avg.neutron_emission_probabilities[order]
source.energy = openmc.stats.Discrete(energies, weights)

# settings' settings
settings = openmc.Settings(run_mode='fixed source')
settings.photon_transport = True
# settings.electron_treatment = 'ttb'
settings.weight_windows = ww
settings.source = source
settings.batches = 10
settings.particles = int(1e6)  #1e6
# settings.statepoint = {'batches': [
#      30, 50, 100]}
settings.output = {'tallies': False}

# filters
particle_filter = openmc.ParticleFilter(
    ['neutron', 'photon', 'electron', 'positron'])

neutron_filter = openmc.ParticleFilter(['neutron'])

# energyfilter
tripoli315 = openmc.mgxs.GROUP_STRUCTURES['TRIPOLI-315']
energy_filter = openmc.EnergyFilter(tripoli315)

# mesh filter
# regular mesh
mesh = openmc.RegularMesh()
mesh.dimension = [128, 48, 200]
mesh.lower_left = [40, -120, -500]
mesh.upper_right = [680, 120, 500]
globalmesh_filter = openmc.MeshFilter(mesh)

# Mesh for inner radial plot
inner_radial_mesh = openmc.RegularMesh()
nxr_in = 695*2
nyr_in = 1
nzr_in = 1
inner_radial_mesh.lower_left  = (139.5, -1.0, -10.0)
inner_radial_mesh.upper_right = (209.0, 0.0, 10.0)
inner_radial_mesh.dimension=[nxr_in, nyr_in, nzr_in]
inner_radial_mesh_filter = openmc.MeshFilter(inner_radial_mesh)

# Mesh for outer radial plot
outer_radial_mesh = openmc.RegularMesh()
nxr_out = 695*2
outer_radial_mesh.lower_left  = (454.0, -1.0, -10.0)
outer_radial_mesh.upper_right = (523.5, 0.0, 10.0)
outer_radial_mesh.dimension=[nxr_out, nyr_in, nzr_in]
outer_radial_mesh_filter = openmc.MeshFilter(outer_radial_mesh)

# cell filter
channel_filter = openmc.CellFilter(root[4])
breeder_filter = openmc.CellFilter(root[7])
tf_coil_magnet_filter = openmc.CellFilter(root[45])
tf_coil_insulation_filter = openmc.CellFilter(root[46])
tf_coil_case_filter = openmc.CellFilter(root[47])

# energyfilter
tripoli315 = openmc.mgxs.GROUP_STRUCTURES['TRIPOLI-315']
energy_filter = openmc.EnergyFilter(tripoli315)

# tallies
tallies = openmc.Tallies()
# mesh tally - nflux
tally1 = openmc.Tally(tally_id=1, name="nflux_mesh")
tally1.filters = [neutron_filter, globalmesh_filter]
tally1.scores = ["flux"]
tallies.append(tally1)

# mesh tally - heating
tally2 = openmc.Tally(tally_id=2, name="heating_mesh")
tally2.filters = [particle_filter, globalmesh_filter]
tally2.scores = ["heating"]
tallies.append(tally2)

# mesh tally - gas production
tally3 = openmc.Tally(tally_id=3, name="gas_pruduction_mesh")
tally3.filters = [globalmesh_filter]
tally3.scores = ["H1-production", "H2-production", "H3-production",
                 "He3-production", "He4-production"]
tallies.append(tally3)

# tbr
tbr_total_tally = openmc.Tally(name="TBR_total")
tbr_total_tally.scores = ["(n,Xt)"]
tallies.append(tbr_total_tally)

tbr_channel_tally = openmc.Tally(name="TBR_channel")
tbr_channel_tally.scores = ["(n,Xt)"]
tbr_channel_tally.filters = [channel_filter]
tallies.append(tbr_channel_tally)

tbr_breeder_tally = openmc.Tally(name="TBR_breeder")
tbr_breeder_tally.scores = ["(n,Xt)"]
tbr_breeder_tally.filters = [breeder_filter]
tallies.append(tbr_breeder_tally)

# qvol
# Total power deposition
heating_full_tally = openmc.Tally(name="heating_full")
heating_full_tally.scores = ["heating"]
tallies.append(heating_full_tally)

q_vol_tf_coil_magnet_tally = openmc.Tally(name="q_vol_tf_coil_magnet")
q_vol_tf_coil_magnet_tally.scores = ["heating"]
q_vol_tf_coil_magnet_tally.filters = [tf_coil_magnet_filter]
tallies.append(q_vol_tf_coil_magnet_tally)

q_vol_tf_coil_insulation_tally = openmc.Tally(name="q_vol_tf_coil_insulation")
q_vol_tf_coil_insulation_tally.scores = ["heating"]
q_vol_tf_coil_insulation_tally.filters = [tf_coil_insulation_filter]
tallies.append(q_vol_tf_coil_insulation_tally)

q_vol_tf_coil_case_tally = openmc.Tally(name="q_vol_tf_coil_case")
q_vol_tf_coil_case_tally.scores = ["heating"]
q_vol_tf_coil_case_tally.filters = [tf_coil_case_filter]
tallies.append(q_vol_tf_coil_case_tally)

# DPA
damage_tf_coil_magnet_tally = openmc.Tally(name="dpa_tf_coil_magnet")
damage_tf_coil_magnet_tally.scores = ["damage-energy"]
damage_tf_coil_magnet_tally.filters = [tf_coil_magnet_filter]
tallies.append(damage_tf_coil_magnet_tally)

damage_tf_coil_insulation_tally = openmc.Tally(name="dpa_tf_coil_insulation")
damage_tf_coil_insulation_tally.scores = ["damage-energy"]
damage_tf_coil_insulation_tally.filters = [tf_coil_insulation_filter]
tallies.append(damage_tf_coil_insulation_tally)

damage_tf_coil_case_tally = openmc.Tally(name="dpa_tf_coil_case")
damage_tf_coil_case_tally.scores = ["damage-energy"]
damage_tf_coil_case_tally.filters = [tf_coil_case_filter]
tallies.append(damage_tf_coil_case_tally)

dpa_ybco = openmc.Tally(name='dpa_ybco')
dpa_ybco.scores = ['damage-energy']
dpa_ybco.filters = [tf_coil_magnet_filter]
dpa_ybco.nuclides = [
    openmc.Nuclide("Y89"),
    openmc.Nuclide("Ba130"),
    openmc.Nuclide("Ba132"),
    openmc.Nuclide("Ba134"),
    openmc.Nuclide("Ba135"),
    openmc.Nuclide("Ba136"),
    openmc.Nuclide("Ba137"),
    openmc.Nuclide("Ba138"),
    openmc.Nuclide("Cu63"),
    openmc.Nuclide("Cu65"),
    openmc.Nuclide("O16"),
    openmc.Nuclide("O17"),
    openmc.Nuclide("O18"),
]
tallies.append(dpa_ybco)

# NEUTRON SPECTRA
spectra_tf_coil_magnet_tally = openmc.Tally(name='neutron_spectra_tf_coil_magnet')
spectra_tf_coil_magnet_tally.filters = [tf_coil_magnet_filter, neutron_filter, energy_filter]
spectra_tf_coil_magnet_tally.scores = ['flux']
tallies.append(spectra_tf_coil_magnet_tally)

spectra_tf_coil_insulation_tally = openmc.Tally(name='neutron_spectra_tf_coil_insulation')
spectra_tf_coil_insulation_tally.filters = [tf_coil_insulation_filter, neutron_filter, energy_filter]
spectra_tf_coil_insulation_tally.scores = ['flux']
tallies.append(spectra_tf_coil_insulation_tally)

spectra_tf_coil_case_tally = openmc.Tally(name='neutron_spectra_tf_coil_case')
spectra_tf_coil_case_tally.filters = [tf_coil_case_filter, neutron_filter, energy_filter]
spectra_tf_coil_case_tally.scores = ['flux']
tallies.append(spectra_tf_coil_case_tally)

# RADIAL PROFILES
# Neutron flux
i_r_n_flux_tally = openmc.Tally(name="inner_radial_neutron_flux")
i_r_n_flux_tally.scores = ["flux"]
i_r_n_flux_tally.filters = [inner_radial_mesh_filter, neutron_filter]
tallies.append(i_r_n_flux_tally)

o_r_n_flux_tally = openmc.Tally(name="outer_radial_neutron_flux")
o_r_n_flux_tally.scores = ["flux"]
o_r_n_flux_tally.filters = [outer_radial_mesh_filter, neutron_filter]
tallies.append(o_r_n_flux_tally)

# q vol on mesh
i_r_q_vol_tally = openmc.Tally(name="i_r_q_vol")
i_r_q_vol_tally.scores = ["heating-local"]
i_r_q_vol_tally.filters = [inner_radial_mesh_filter]
tallies.append(i_r_q_vol_tally)

o_r_q_vol_tally = openmc.Tally(name="o_r_q_vol")
o_r_q_vol_tally.scores = ["heating-local"]
o_r_q_vol_tally.filters = [outer_radial_mesh_filter]
tallies.append(o_r_q_vol_tally)

model = openmc.Model(materials=materials, geometry=geometry,
                     settings=settings, tallies=tallies)

model.run(threads=12, geometry_debug=False)
