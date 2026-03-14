import openmc
import numpy as np
from pathlib import Path
import tokamak_radiation_environment as tre
from fusion_fuel_concepts_explorer.sources import neutron_source
from fusion_fuel_concepts_explorer.geometry import component_nodes as cn

# define materials
dt_plasma = openmc.Material(name='dt_plasma')
dt_plasma.add_nuclide('H2', 1.0)
dt_plasma.add_nuclide('H3', 1.0)
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

flibe = openmc.Material(name='FLiBe')  #breeding blanket (Li-6 enrichment has been kept constant to 90 %), FLiBe
flibe.add_element('F', 4.)
flibe.add_element('Be', 1.)
flibe.add_nuclide('Li6', 1.8)
flibe.add_nuclide('Li7', 0.2)
flibe.set_density('g/cm3', 1.94)

wc = openmc.Material(name='wc')
wc.add_element('W', 1.0)
wc.add_element('C', 1.0)
wc.set_density('g/cm3', 15.63)

ss304 = openmc.Material(name='ss304')
ss304.add_element('Cr', 0.19, 'wo')
ss304.add_element('Mn', 0.01, 'wo')
ss304.add_element('Fe', 0.71, 'wo')
ss304.add_element('Ni', 0.09, 'wo')
ss304.set_density('g/cm3', 7.80)

ss316L = openmc.Material(name='ss316L')
ss316L.add_element('Cr', 0.166, 'wo')
ss316L.add_element('Mo', 0.02, 'wo')
ss316L.add_element('Mn', 0.002, 'wo')
ss316L.add_element('Fe', 0.712, 'wo')
ss316L.add_element('Ni', 0.1, 'wo')
ss316L.set_density('g/cm3', 7.80)

beryllium = openmc.Material(name='beryllium')
beryllium.add_element('Be', 1.0)
beryllium.set_density('g/cm3', 1.85)

fiberglass = openmc.Material(name='Fiberglass')
fiberglass.add_element('B', 0.022803, percent_type='wo')
fiberglass.add_element('O', 0.471950, percent_type='wo')
fiberglass.add_element('F', 0.004895, percent_type='wo')
fiberglass.add_element('Na', 0.007262, percent_type='wo')
fiberglass.add_element('Mg', 0.014759, percent_type='wo')
fiberglass.add_element('Al', 0.072536, percent_type='wo')
fiberglass.add_element('Si', 0.247102, percent_type='wo')
fiberglass.add_element('K', 0.008127, percent_type='wo')
fiberglass.add_element('Ca', 0.143428, percent_type='wo')
fiberglass.add_element('Ti', 0.004400, percent_type='wo')
fiberglass.add_element('Fe', 0.002739, percent_type='wo')
fiberglass.set_density('g/cm3', 2.57)

copper = openmc.Material(name='copper')
copper.add_element('Cu', 1.0)
copper.set_density('g/cm3', 8.96)

silver = openmc.Material(name='silver')
silver.add_element('Ag', 1.0)
silver.set_density('g/cm3', 10.49)

hastelloy_c276 = openmc.Material(name='hastelloy_c276')
hastelloy_c276.add_element('Ni', 0.5456, percent_type='wo')
hastelloy_c276.add_element('Co', 0.025, percent_type='wo')
hastelloy_c276.add_element('Cr', 0.16, percent_type='wo')
hastelloy_c276.add_element('Mo', 0.16, percent_type='wo')
hastelloy_c276.add_element('Fe', 0.05, percent_type='wo')
hastelloy_c276.add_element('W', 0.04, percent_type='wo')
hastelloy_c276.add_element('Mn', 0.01, percent_type='wo')
hastelloy_c276.add_element('V', 0.0035, percent_type='wo')
hastelloy_c276.add_element('Si', 0.0008, percent_type='wo')
hastelloy_c276.add_element('C', 0.0001, percent_type='wo')
hastelloy_c276.add_element('Cu', 0.005, percent_type='wo')
hastelloy_c276.set_density('g/cm3', 8.89)

ybco = openmc.Material(name='ybco')
ybco.add_elements_from_formula('YBa2Cu3O7')
ybco.set_density('g/cm3', 6.3)

nitronic50 = openmc.Material(name='nitronic50')
nitronic50.add_element('Cr', 0.22, percent_type='wo')
nitronic50.add_element('Ni', 0.125, percent_type='wo')
nitronic50.add_element('Mo', 0.0225, percent_type='wo')
nitronic50.add_element('Nb', 0.002, percent_type='wo')
nitronic50.add_element('Mn', 0.05, percent_type='wo')
nitronic50.add_element('Si', 0.004, percent_type='wo')
nitronic50.add_element('C', 0.0003, percent_type='wo')
nitronic50.add_element('S', 0.0001, percent_type='wo')
nitronic50.add_element('P', 0.0004, percent_type='wo')
nitronic50.add_element('V', 0.0002, percent_type='wo')
nitronic50.add_element('N', 0.003, percent_type='wo')
nitronic50.add_element('Fe', 0.5725, percent_type='wo')
nitronic50.set_density('g/cm3', 7.88)

pbsn = openmc.Material(name='pbsn')
pbsn.add_element('Pb', 0.37, percent_type='wo')
pbsn.add_element('Sn', 0.63, percent_type='wo')
pbsn.set_density('g/cm3', 8.8)

# mixes
# https://doi.org/10.1038/s41598-021-81559-z. Assuming 50um hastelloy.
tape = openmc.Material.mix_materials([copper, silver, hastelloy_c276, ybco], [
                                     10/65.35, 3/65.35, 50/65.35, 2.35/65.35], percent_type='vo')
windingpack = openmc.Material.mix_materials([copper, nitronic50, pbsn, tape], [
                                            0.15, 0.5, 0.05, 0.2], percent_type='vo', name="Windingpack")

materials = openmc.Materials(
    [dt_plasma, tungsten, beryllium, flibe, wc, inconel, ss304, windingpack, fiberglass, ss316L])

# geometry

# boundary surfaces
angle = (-10, 10)

# components

# core
plasma, sol, first_wall, vessel_inner_structure, vessel_cooling_channel, vessel_neutron_multiplier, vessel_outer_structure, blanket, shield = tre.components.core_group(
    plasma_outer_nodes=cn.plasma_out, plasma_material=None,
    firstwall_inner_nodes=cn.fw_in, firstwall_thickness=0.5, firstwall_material=tungsten,
    vv_stri_thickness=2., vv_stri_material=inconel,
    vv_channel_thickness=2., vv_channel_material=flibe,
    vv_multiplier_thickness=1., vv_multiplier_material=beryllium,
    vv_stro_thickness=2., vv_stro_material=inconel,
    blanket_thickness=60., blanket_material=flibe,
    shield_thickness=2., shield_material=ss304,
    angle=angle)

reactor_components = [
    plasma,
    first_wall,
    vessel_inner_structure,
    vessel_cooling_channel,
    vessel_neutron_multiplier,
    vessel_outer_structure,
    sol,
    blanket,
    shield,
]

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

root_tf = [tf_coil_magnet.cell, tf_coil_insulation.cell, tf_coil_case.cell,
        enclosure_cell]
geometry = openmc.Geometry(root=root_tf)

geometry.merge_surfaces = True

# settings
# --- choose a bounding box that contains the whole reactor ---
# adapt these numbers if needed
lower_left = (-3000.0, -3000.0, -3000.0)
upper_right = ( 3000.0,  3000.0,  3000.0)

# --- stochastic volume calculation ---
vol_calc = openmc.VolumeCalculation(
    domains=root_tf,
    samples=1000000,
    lower_left=lower_left,
    upper_right=upper_right,
)

# optional: keep iterating until relative error is below threshold
vol_calc.set_trigger(0.01, 'rel_err')   # 1%

settings = openmc.Settings()
settings.run_mode = 'volume'
settings.volume_calculations = [vol_calc]

model = openmc.Model(geometry=geometry, materials=materials, settings=settings)

# run the stochastic volume calculation
model.run()

# load results
vol_calc.load_results('volume_1.h5')

# print results
for cell, (vol, std) in vol_calc.volumes.items():
    print(f"Cell {cell.id:3d} | {cell.name:25s} | V = {vol:.6e} ± {std:.6e} cm^3")