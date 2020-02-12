"""Setup and run dustybox calculations.

Time evolution of the differential velocity.
"""

import copy
import pathlib

import click
import multigrain
import numpy as np
import phantombuild
import pint

units = pint.UnitRegistry(system='cgs')


# ------------------------------------------------------------------------------------ #
# MAKE CHANGES AS REQUIRED

# Required Phantom version.
PHANTOM_VERSION = 'e0d6986df99d980267f712d2218376dd50701117'

# Path to HDF5 library.
HDF5ROOT = '/usr/local/opt/hdf5'


# Choose parameters for each run.
def set_parameters():
    """Generate a dictionary of parameter dictionaries.

    The dictionary is as follows:
        {
            'name_of_run1': parameters1,
            'name_of_run2': parameters2,
            ...
        }

    The 'parameters' dictionary has keys with the name of the run, which
    will be the name of its directory, and the values are the parameters
    dictionaries for that run.

    Each dictionary for each run needs the following keys:

        'prefix'
        'length_unit'
        'mass_unit'
        'time_unit'
        'sound_speed'
        'box_width'
        'lattice'
        'number_of_particles_in_x_gas'
        'number_of_particles_in_x_dust'
        'density_gas'
        'dust_to_gas_ratio'
        'drag_method'
        'grain_size'
        'grain_density'
        'velocity_delta'
        'maximum_time'
        'number_of_dumps'

    All float or ndarray variables can have units.

    The length of 'dust_to_gas_ratio', 'grain_size', and
    'velocity_delta' should be the same, i.e. the number of dust
    species.
    """
    # Dictionary of parameters common to all runs.
    _parameters = {
        'prefix': 'dustybox',
        'length_unit': 1.0 * units['cm'],
        'mass_unit': 1.0 * units['g'],
        'time_unit': 1.0 * units['s'],
        'sound_speed': 1.0 * units['cm/s'],
        'box_width': 1.0 * units['cm'],
        'lattice': 'close packed',
        'number_of_particles_in_x_gas': 32,
        'number_of_particles_in_x_dust': 16,
        'density_gas': 1.0e-13 * units['g / cm^3'],
        'drag_method': 'Epstein/Stokes',
        'grain_density': 0.5e-14 * units['g / cm^3'],
        'maximum_time': 0.1 * units['s'],
        'number_of_dumps': 100,
    }

    # Each value in tuple multiplicatively generates a new simulation.
    grain_size = ([1.0], [0.562, 1.78], [0.1, 0.316, 1.0, 3.16, 10.0])
    total_dust_to_gas_ratio = (0.01, 0.5)

    # Iterate over dust-to-gas ratio and grain sizes.
    parameters = dict()
    for f in total_dust_to_gas_ratio:
        for size in grain_size:
            N = len(size)
            label = f'f={f:.2f}_N={N}'
            parameters[label] = copy.copy(_parameters)
            dust_to_gas_ratio = tuple(f / N * np.ones(N))
            parameters[label]['dust_to_gas_ratio'] = dust_to_gas_ratio
            parameters[label]['grain_size'] = size * units['cm']
            velocity_delta = tuple(np.ones(N)) * units['cm / s']
            parameters[label]['velocity_delta'] = velocity_delta

    return parameters


# ------------------------------------------------------------------------------------ #
# DO NOT CHANGE BELOW


def remove_units(parameters):
    """Remove units to eventually pass on to phantom-setup."""
    for params in parameters.values():
        length_unit = params.pop('length_unit')
        mass_unit = params.pop('mass_unit')
        time_unit = params.pop('time_unit')
        for key, value in params.items():
            if isinstance(value, units.Quantity):
                d = value.dimensionality
                new_units = (
                    length_unit ** d['[length]']
                    * mass_unit ** d['[mass]']
                    * time_unit ** d['[time]']
                )
                params[key] = value.to(new_units).magnitude
        if isinstance(length_unit, units.Quantity):
            params['length_unit'] = length_unit.to_base_units().magnitude
        if isinstance(mass_unit, units.Quantity):
            params['mass_unit'] = mass_unit.to_base_units().magnitude
        if isinstance(time_unit, units.Quantity):
            params['time_unit'] = time_unit.to_base_units().magnitude

    return parameters


@click.command()
@click.option(
    '--run_directory', required=True, help='the directory for the calculations'
)
@click.option(
    '--hdf5_directory',
    help='the path to the HDF5 library',
    default=HDF5ROOT,
    show_default=True,
)
def cli(run_directory, hdf5_directory):
    """CLI interface."""
    parameters_dict = set_parameters()
    parameters_dict = remove_units(parameters_dict)
    run_directory = pathlib.Path(run_directory).expanduser()
    hdf5_directory = pathlib.Path(hdf5_directory).expanduser()
    phantom_dir = run_directory.parent / '.phantom'
    phantombuild.get_phantom(phantom_dir=phantom_dir)
    phantombuild.checkout_phantom_version(
        phantom_dir=phantom_dir, required_phantom_git_commit_hash=PHANTOM_VERSION
    )
    multigrain.setup_multiple_calculations(
        simulation_to_setup='dustybox',
        run_root_directory=run_directory,
        parameters_dict=parameters_dict,
        phantom_dir=phantom_dir,
        hdf5root=hdf5_directory,
    )
    multigrain.run_multiple_calculations(run_root_directory=run_directory)


if __name__ == "__main__":
    cli()  # pylint: disable=no-value-for-parameter
