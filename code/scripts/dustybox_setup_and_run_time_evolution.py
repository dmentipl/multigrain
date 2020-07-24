"""Set up and run dustybox calculations.

Time evolution of the differential velocity.

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

The length of 'dust_to_gas_ratio', 'grain_size', and 'velocity_delta'
should be the same, i.e. the number of dust species.
"""

import copy
import pathlib
import sys

import numpy as np

path = pathlib.Path(__file__).parent / '..' / 'modules'
sys.path.insert(0, str(path))

from multigrain import run_script
from multigrain.config import UNITS

# Variables to set
SIMULATION = None
PARAMETERS = None
RUN_DIRECTORY = None
PATCH_FILE = None

# ------------------------------------------------------------------------------------ #
# MAKE CHANGES BELOW AS REQUIRED

SIMULATION = 'dustybox'
RUN_DIRECTORY = '~/runs/multigrain/dustybox/time_evolution'

# Dictionary of parameters common to all runs.
_parameters = {
    'prefix': 'dustybox',
    'length_unit': 1.0 * UNITS['cm'],
    'mass_unit': 1.0 * UNITS['g'],
    'time_unit': 1.0 * UNITS['s'],
    'sound_speed': 1.0 * UNITS['cm/s'],
    'box_width': 1.0 * UNITS['cm'],
    'lattice': 'close packed',
    'number_of_particles_in_x_gas': 8,
    'number_of_particles_in_x_dust': 8,
    'density_gas': 1.0e-13 * UNITS['g / cm^3'],
    'drag_method': 'Epstein/Stokes',
    'grain_density': 0.5e-14 * UNITS['g / cm^3'],
    'maximum_time': 0.1 * UNITS['s'],
    'number_of_dumps': 20,
    'C_force': 0.25,
}

# Each value in tuple multiplicatively generates a new simulation.
grain_size = ([1.0], [0.562, 1.78], [0.1, 0.316, 1.0, 3.16, 10.0])
total_dust_to_gas_ratio = (0.01, 0.5)

# Iterate over dust-to-gas ratio and grain sizes.
PARAMETERS = dict()
for f in total_dust_to_gas_ratio:
    for size in grain_size:
        N = len(size)
        label = f'f_{f:.2f}-N_{N}'
        PARAMETERS[label] = copy.copy(_parameters)
        dust_to_gas_ratio = tuple(f / N * np.ones(N))
        PARAMETERS[label]['dust_to_gas_ratio'] = dust_to_gas_ratio
        PARAMETERS[label]['grain_size'] = size * UNITS['cm']
        velocity_delta = tuple(np.ones(N)) * UNITS['cm / s']
        PARAMETERS[label]['velocity_delta'] = velocity_delta

# ------------------------------------------------------------------------------------ #
# DO NOT CHANGE BELOW

run_script(
    simulation_to_setup=SIMULATION,
    parameters_dict=PARAMETERS,
    run_directory=RUN_DIRECTORY,
    phantom_patch_file=PATCH_FILE,
)
