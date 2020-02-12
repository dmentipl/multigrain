"""Setup and run dusty shock calculations.

Exactly as Benitez-Llambay et al. (2019).
"""

import copy
import pathlib

import multigrain
import phantombuild
import pint
from multigrain import HDF5ROOT, PHANTOM_VERSION

units = pint.UnitRegistry(system='cgs')

# ------------------------------------------------------------------------------------ #
# MAKE CHANGES BELOW AS REQUIRED

RUN_DIRECTORY = '~/runs/multigrain/dustyshock'


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
        'number_of_particles_in_x_R'
        'density_L'
        'density_R'
        'dust_to_gas_ratio'
        'K_const'
        'velocity_L'
        'velocity_R'
        'maximum_time'
        'number_of_dumps'

    All float or ndarray variables can have units.

    The length of 'dust_to_gas_ratio', 'grain_size', and
    'velocity_delta' should be the same, i.e. the number of dust
    species.
    """
    # Mach number
    mach = 2.0

    # Dictionary of parameters common to all runs.
    _parameters = {
        'prefix': 'dustyshock',
        'length_unit': 1.0,
        'mass_unit': 1.0,
        'time_unit': 1.0,
        'sound_speed': 1.0,
        'box_width': 200,
        'lattice': 'close packed',
        'number_of_particles_in_x_R': 128,
        'density_L': 1.0,
        'velocity_L': mach,
        'maximum_time': 10.0,
        'number_of_dumps': 100,
    }

    # Each value in tuple multiplicatively generates a new simulation.
    K_drag = ([1.0], [1.0, 3.0, 5.0])
    density_R = (8.0, 16.0)
    velocity_R = (mach * 0.125, mach * 0.0625)

    # Iterate over dust-to-gas ratio and grain sizes.
    parameters = dict()
    for K, rho_R, v_R in zip(K_drag, density_R, velocity_R):
        N = len(K)
        label = f'N={N}'
        parameters[label] = copy.copy(_parameters)
        parameters[label]['K_drag'] = K
        parameters[label]['density_R'] = rho_R
        parameters[label]['velocity_R'] = v_R

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


def main():
    parameters_dict = set_parameters()
    parameters_dict = remove_units(parameters_dict)
    run_directory = pathlib.Path(RUN_DIRECTORY).expanduser()
    hdf5_directory = pathlib.Path(HDF5ROOT).expanduser()
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
    main()
