"""Setup and run dustybox calculations.

Check accuracy by changing C_force.
"""

import copy

from multigrain import run_script
from multigrain.config import UNITS

# ------------------------------------------------------------------------------------ #
# MAKE CHANGES BELOW AS REQUIRED

SIMULATION = 'dustybox'
RUN_DIRECTORY = '~/runs/multigrain/dustybox/accuracy'


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
        'C_force'

    All float or ndarray variables can have units.

    The length of 'dust_to_gas_ratio', 'grain_size', and
    'velocity_delta' should be the same, i.e. the number of dust
    species.
    """
    # Dictionary of parameters common to all runs.
    _parameters = {
        'prefix': 'dustybox',
        'length_unit': 1.0 * UNITS['cm'],
        'mass_unit': 1.0 * UNITS['g'],
        'time_unit': 1.0 * UNITS['s'],
        'sound_speed': 1.0 * UNITS['cm/s'],
        'box_width': 1.0 * UNITS['cm'],
        'lattice': 'close packed',
        'number_of_particles_in_x_gas': 32,
        'number_of_particles_in_x_dust': 16,
        'density_gas': 1.0e-13 * UNITS['g / cm^3'],
        'drag_method': 'Epstein/Stokes',
        'grain_density': 0.5e-14 * UNITS['g / cm^3'],
        'grain_size': [0.1, 0.316, 1.0, 3.16, 10.0] * UNITS['cm'],
        'velocity_delta': [1.0, 1.0, 1.0, 1.0, 1.0] * UNITS['cm / s'],
        'dust_to_gas_ratio': [0.1, 0.1, 0.1, 0.1, 0.1],
        'maximum_time': 1.0 * UNITS['s'],
        'number_of_dumps': 10,
    }

    # Each value in tuple multiplicatively generates a new simulation.
    C_forces = [0.1, 0.25, 0.5, 1.0, 2.0]

    # Iterate over C_forces
    parameters = dict()
    for C_force in C_forces:
        label = f'C_force={C_force:.2f}'
        parameters[label] = copy.copy(_parameters)
        parameters[label]['C_force'] = C_force

    return parameters


# ------------------------------------------------------------------------------------ #
# DO NOT CHANGE BELOW

if __name__ == "__main__":
    parameters = set_parameters()
    run_script(SIMULATION, parameters, RUN_DIRECTORY)
