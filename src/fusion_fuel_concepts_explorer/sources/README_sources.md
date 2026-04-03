# Steady-state neutron source evaluation

Use `SteadyStateSource` to solve the steady-state D/T/He3/He4 composition and obtain neutron source inputs for OpenMC.

The relevant outputs are:

- `neutron_emission_probabilities` [DT, DDn, TT, THe3n]
- `neutron_energies_eV` [DT, DDn, TT, THe3n]
- `total_neutron_rate_s` 
- `reaction_rates_s`
- `fusion_power_by_channel_W`

The neutron probability for an energy bin is based on the neutron production rate of the corresponding channel:

$p_n = \frac{\text{rr}_n}{\sum \text{rr}}$

where `rr_n` is the reaction rate of the neutron-producing channel and `∑rr` is the total reaction rate.  
This represents neutrons per second in that channel, normalized by the total neutron rate.
