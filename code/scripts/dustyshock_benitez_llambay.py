"""Setup and run dusty shock calculations.

Exactly as Benitez-Llambay et al. (2019).
"""

import copy

from multigrain import run_script

# ------------------------------------------------------------------------------------ #
# MAKE CHANGES BELOW AS REQUIRED

SIMULATION = 'dustyshock'
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

if __name__ == "__main__":
    parameters = set_parameters()
    run_script(SIMULATION, parameters, RUN_DIRECTORY)
