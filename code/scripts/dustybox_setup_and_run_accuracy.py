"""Set up and run dustybox calculations.

Check accuracy by changing C_force.

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

Optional parameters:

    'C_force'

All float or ndarray variables can have units.

The length of 'dust_to_gas_ratio', 'grain_size', and 'velocity_delta'
should be the same, i.e. the number of dust species.
"""

import copy
import sys
import pathlib

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
RUN_DIRECTORY = '~/runs/multigrain/dustybox/accuracy'
PATCH_FILE = (
    pathlib.Path(__file__).resolve().parent.parent
    / 'patches'
    / 'phantom-666da9e8-dustybox.patch'
)

# Dictionary of parameters common to all runs.
_parameters = {
    'prefix': 'dustybox',
    'length_unit': 1.0 * UNITS['au'],
    'mass_unit': 1.99e33 * UNITS['g'],
    'time_unit': 1.0 * UNITS['year'],
    'sound_speed': 0.2e5 * UNITS['cm/s'],
    'box_width': 1.0 * UNITS['au'],
    'lattice': 'close packed',
    'number_of_particles_in_x_gas': 32,
    'number_of_particles_in_x_dust': 32,
    'density_gas': 1.0e-13 * UNITS['g / cm^3'],
    'drag_method': 'Epstein/Stokes',
    'grain_density': 1.0 * UNITS['g / cm^3'],
    'maximum_time': 3.0e7 * UNITS['s'],
    'number_of_dumps': 3,
}

# Parameter in Phantom patch: "comparing" dt_force and dt_drag
DTFORCE_TO_DTDRAG = 1.0

# Grain sizes
grain_size = [0.01, 0.03]
n_dust = len(grain_size)
_parameters['grain_size'] = grain_size * UNITS['cm']
_parameters['velocity_delta'] = [1.0 for _ in range(n_dust)] * UNITS['cm / s']

# Generate one simulation per element of the Cartesian product of lists below
largest = 0.20
num = 5
dtdrag_fac = [largest / 2 ** n for n in range(num)]
dust_to_gas = [0.01, 0.5]
hfacts = [1.0, 2.5]

# Iterate over dtdrag_fac and dust_to_gas to generate simulations
PARAMETERS = dict()
for hfact in hfacts:
    for eps in dust_to_gas:
        for dtdrag in dtdrag_fac:
            C_force = dtdrag / DTFORCE_TO_DTDRAG
            label = f'hfact_{hfact:.1f}-eps_{eps:.2f}-C_force_{C_force:.4f}'
            PARAMETERS[label] = copy.copy(_parameters)
            PARAMETERS[label]['C_force'] = C_force
            PARAMETERS[label]['dust_to_gas_ratio'] = [
                eps / n_dust for _ in range(n_dust)
            ]

# ------------------------------------------------------------------------------------ #
# DO NOT CHANGE BELOW

run_script(
    simulation_to_setup=SIMULATION,
    parameters_dict=PARAMETERS,
    run_directory=RUN_DIRECTORY,
    phantom_patch_file=PATCH_FILE,
)
