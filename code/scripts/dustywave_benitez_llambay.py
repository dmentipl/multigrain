"""Setup and run dustywave calculations.

Exactly as Benitez-Llambay et al. (2019).
"""

import copy

from multigrain import run_script

SIMULATION = None
RUN_DIRECTORY = None
PATCH_FILE = None

# ------------------------------------------------------------------------------------ #
# MAKE CHANGES BELOW AS REQUIRED

SIMULATION = 'dustywave'
RUN_DIRECTORY = '~/runs/multigrain/dustywave'


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

if __name__ == "__main__":
    parameters = set_parameters()
    run_script(
        simulation_to_setup=SIMULATION,
        parameters_dict=parameters,
        run_directory=RUN_DIRECTORY,
        phantom_patch_file=PATCH_FILE,
    )
