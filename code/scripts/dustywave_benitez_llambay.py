"""Setup and run dustywave calculations.

Exactly as Benitez-Llambay et al. (2019).
"""

import copy
import pathlib

import click
import multigrain
import phantombuild
import pint
from multigrain import HDF5ROOT, PHANTOM_VERSION

units = pint.UnitRegistry(system='cgs')

# ------------------------------------------------------------------------------------ #
# MAKE CHANGES BELOW AS REQUIRED


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
        'number_of_particles_in_x_gas'
        'number_of_particles_in_x_dust'
        'omega'
        'density_gas'
        'delta_density_gas'
        'delta_v_gas'
        'density_dust'
        'delta_density_dust'
        'delta_v_dust'
        'K_drag'
        'maximum_time'
        'number_of_dumps'

    All float or ndarray variables can have units.
    """
    # Dictionary of parameters common to all runs.
    _parameters = {
        'prefix': 'dustywave',
        'length_unit': 1.0,
        'mass_unit': 1.0,
        'time_unit': 1.0,
        'sound_speed': 1.0,
        'box_width': 1.0,
        'number_of_particles_in_x_gas': 128,
        'number_of_particles_in_x_dust': 128,
        'density_gas': 1.0,
        'wave_amplitude': 1.0e-4,
        'maximum_time': 2.0,
        'number_of_dumps': 100,
    }
    parameters = dict()

    # Two species
    d = copy.copy(_parameters)
    d['delta_density_gas'] = 1.0
    d['delta_v_gas'] = -0.701960 - 0.304924j
    d['omega'] = 1.915896 - 4.410541j
    d['density_dust'] = (2.24,)
    d['delta_density_dust'] = (0.165251 - 1.247801j,)
    d['delta_v_dust'] = (-0.221645 + 0.368534j,)
    tstop = (0.4,)
    d['K_drag'] = tuple([rho_d / ts for rho_d, ts in zip(d['density_dust'], tstop)])
    parameters['two species'] = d

    # Five species
    d = copy.copy(_parameters)
    d['delta_density_gas'] = 1.0
    d['delta_v_gas'] = -0.874365 - 0.145215j
    d['omega'] = 0.912414 - 5.493800j
    d['density_dust'] = (0.1, 0.233333, 0.366667, 0.5)
    d['delta_density_dust'] = (
        0.080588 - 0.048719j,
        0.091607 - 0.134955j,
        0.030927 - 0.136799j,
        0.001451 - 0.090989j,
    )
    d['delta_v_dust'] = (
        -0.775380 + 0.308952j,
        -0.427268 + 0.448704j,
        -0.127928 + 0.313967j,
        -0.028963 + 0.158693j,
    )
    tstop = (0.1, 0.215443, 0.464159, 1.0)
    d['K_drag'] = tuple([rho_d / ts for rho_d, ts in zip(d['density_dust'], tstop)])
    parameters['five species'] = d

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
