"""Set up and run dusty shock calculations.

Exactly as Benitez-Llambay et al. (2019).

Need to set the following variables:
    SIMULATION
        The simulation type; here it is 'dustybox'.
    PARAMETERS
        The parameters dictionary of dictionaries for each run.
    RUN_DIRECTORY
        The path to the directory to store the runs.
    PATCH_FILE
        An optional Phantom patch file.

The PARAMETERS variable is a dictionary of parameter dictionaries.

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

The length of 'dust_to_gas_ratio', 'grain_size', and 'velocity_delta'
should be the same, i.e. the number of dust species.
"""

import copy
import pathlib
import sys

path = pathlib.Path(__file__).parent / '..' / 'modules'
sys.path.insert(0, str(path))

from multigrain import run_script

SIMULATION = None
PARAMETERS = None
RUN_DIRECTORY = None
PATCH_FILE = None

# ------------------------------------------------------------------------------------ #
# MAKE CHANGES BELOW AS REQUIRED

SIMULATION = 'dustyshock'
RUN_DIRECTORY = '~/runs/multigrain/dustyshock'

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
PARAMETERS = dict()
for K, rho_R, v_R in zip(K_drag, density_R, velocity_R):
    N = len(K)
    label = f'N={N}'
    PARAMETERS[label] = copy.copy(_parameters)
    PARAMETERS[label]['K_drag'] = K
    PARAMETERS[label]['density_R'] = rho_R
    PARAMETERS[label]['velocity_R'] = v_R

# ------------------------------------------------------------------------------------ #
# DO NOT CHANGE BELOW

run_script(
    simulation_to_setup=SIMULATION,
    parameters_dict=PARAMETERS,
    run_directory=RUN_DIRECTORY,
    phantom_patch_file=PATCH_FILE,
)
