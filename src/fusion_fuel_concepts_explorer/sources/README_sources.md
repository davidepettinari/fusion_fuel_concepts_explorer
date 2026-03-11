# Evaluation of a 0D, isotropic neutron source from a generic fuel mixture

For an OpenMC simulation using an isotropic ring source, it is sufficient to specify the emission probability for each neutron type (DD, DT).

The probability that a neutron is emitted from a given reaction in a mixed fuel is:

$p_n = \frac{\text{rr}_n}{\sum \text{rr}}$

where `rr_n` is the reaction rate of the neutron-producing channel and `∑rr` is the total reaction rate.  
This represents *neutrons per reaction*.

To obtain the probability of emitting a neutron with energy **X**, compute:

$p_X = \frac{p_{n,X}}{\sum p_n}$

That is, divide the neutrons/reaction for the neutron-producing channel that yields energy **X** by the total neutrons/reaction from all neutron-producing reactions.

BE CAREFUL! The function does not automatically return energies and probabilities formatted for use in openmc.
Information must be extracted from the dictionary and converted from MeV to eV!