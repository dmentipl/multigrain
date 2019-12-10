"""Set up TW Hya multigrain calculation."""

import pathlib

import numpy as np
import phantomsetup

# Constants
igas = phantomsetup.defaults.PARTICLE_TYPE['igas']
idust = phantomsetup.defaults.PARTICLE_TYPE['idust']
earthm = phantomsetup.constants.earthm
solarm = phantomsetup.constants.solarm
year = phantomsetup.constants.year

# Instantiate Setup object
setup = phantomsetup.Setup()

# Prefix
setup.prefix = 'twhya'

# Units
length_unit = phantomsetup.units.unit_string_to_cgs('au')
mass_unit = phantomsetup.units.unit_string_to_cgs('solarm')
gravitational_constant = 1.0
setup.set_units(
    length=length_unit, mass=mass_unit, gravitational_constant_is_unity=True
)
time_unit = setup.units['time']

# Resolution
number_of_particles_gas = 10_000_000
number_of_particles_dust = 250_000

# Dust
number_of_dust_species = 2
dust_to_gas_ratio = (0.025, 0.025)
grain_size_cgs = [0.01, 0.1]
grain_size = tuple([s / length_unit for s in grain_size_cgs])
grain_density_cgs = 3.0
grain_density = grain_density_cgs / (mass_unit / length_unit ** 3)
setup.set_dust(
    dust_method='largegrains',
    drag_method='Epstein/Stokes',
    grain_size=grain_size,
    grain_density=grain_density,
)

# Star
stellar_mass = 0.8
stellar_accretion_radius = 10.0
stellar_position = (0.0, 0.0, 0.0)
stellar_velocity = (0.0, 0.0, 0.0)
setup.add_sink(
    mass=stellar_mass,
    accretion_radius=stellar_accretion_radius,
    position=stellar_position,
    velocity=stellar_velocity,
)

# Equation of state
ieos = 3
q_index = 0.125
aspect_ratio = 0.034
reference_radius = 10.0
polyk = phantomsetup.eos.polyk_for_locally_isothermal_disc(
    q_index, reference_radius, aspect_ratio, stellar_mass, gravitational_constant
)
setup.set_equation_of_state(ieos=ieos, polyk=polyk, qfacdisc=q_index)

# Dissipation
alpha_artificial = 0.1
setup.set_dissipation(disc_viscosity=True, alpha=alpha_artificial)

# Gas disc
radius_min = 10.0
radius_max = 200.0
disc_mass = 7.5e-4
p_index = 0.5
density_distribution = phantomsetup.disc.power_law_with_zero_inner_boundary
disc = phantomsetup.Disc(
    particle_type=igas,
    number_of_particles=number_of_particles_gas,
    disc_mass=disc_mass,
    density_distribution=density_distribution,
    radius_range=(radius_min, radius_max),
    q_index=q_index,
    aspect_ratio=aspect_ratio,
    reference_radius=reference_radius,
    stellar_mass=stellar_mass,
    gravitational_constant=gravitational_constant,
    extra_args=(radius_min, reference_radius, p_index),
)
setup.add_container(disc)

# Dust discs
radius_min = 10.0
radius_max = 80.0
p_index = 0.0
density_distribution = phantomsetup.disc.power_law
for idx in range(number_of_dust_species):
    disc_mass = dust_to_gas_ratio[idx] * 7.5e-4
    disc = phantomsetup.Disc(
        particle_type=idust + idx,
        number_of_particles=number_of_particles_dust,
        disc_mass=disc_mass,
        density_distribution=density_distribution,
        radius_range=(radius_min, radius_max),
        q_index=q_index,
        aspect_ratio=aspect_ratio,
        reference_radius=reference_radius,
        stellar_mass=stellar_mass,
        gravitational_constant=gravitational_constant,
        pressureless=True,
        extra_args=(reference_radius, p_index),
    )
    setup.add_container(disc)

# Planets
planet_accretion_radius_fraction_hill_radius = 0.5
orbital_radii = (24.0, 41.0, 94.0)
planet_masses_earthm = (4.0, 4.0, 95.3)
planet_masses = tuple([mp * earthm / solarm for mp in planet_masses_earthm])

for planet_mass, orbital_radius in zip(planet_masses, orbital_radii):
    planet_position = (orbital_radius, 0.0, 0.0)
    planet_velocity = np.sqrt(gravitational_constant * stellar_mass / orbital_radius)
    planet_velocity = (0.0, planet_velocity, 0.0)
    planet_hill_radius = phantomsetup.orbits.hill_sphere_radius(
        orbital_radius, planet_mass, stellar_mass
    )
    planet_accretion_radius = (
        planet_accretion_radius_fraction_hill_radius * planet_hill_radius
    )
    setup.add_sink(
        mass=planet_mass,
        accretion_radius=planet_accretion_radius,
        position=planet_position,
        velocity=planet_velocity,
    )

# Output time from orbital period of second planet
planet_index = 1
n_orbits = 100
ndumps = 250
nfulldump = 5
period_in_years = np.sqrt(orbital_radii[planet_index] ** 3 / stellar_mass)
tmax = (year / time_unit) * period_in_years * n_orbits
setup.set_output(tmax=tmax, ndumps=ndumps, nfulldump=nfulldump)

# Write to file
working_dir = pathlib.Path('~/runs/multigrain/twhya').expanduser()
setup.write_dump_file(directory=working_dir)
setup.write_in_file(directory=working_dir)

# Compile Phantom
result = setup.compile_phantom(phantom_dir='~/repos/phantom', working_dir=working_dir)
