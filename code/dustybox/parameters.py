"""
Generate a dictionary of Parameter objects for the dustybox tests

The dictionary is as follows:
    {
        'name_of_run1': parameters1,
        'name_of_run2': parameters2,
        ...
    }

The 'parameters' dictionary has keys with the name of the run, which
will be the name of its directory, and the values are the Parameters
object for that run.
"""

from setup import set_parameters

# Each value in dust_to_gas_ratio generates a dustybox setup.
_dust_to_gas_ratio = (0.01, 0.1, 1.0, 10.0)

# Dictionary of parameters common to all runs.
_parameters_asdict = {
    'prefix': 'dustybox',
    'length_unit': 'cm',
    'mass_unit': 'g',
    'time_unit': 's',
    'sound_speed': (1.0, 'cm / s'),
    'box_boundary': ([-0.5, 0.5, -0.5, 0.5, -0.5, 0.5], 'cm'),
    'number_of_particles_gas': 50_000,
    'number_of_particles_dust': 10_000,
    'density_gas': (1.0e-13, 'g / cm^3'),
    'dust_method': 'largegrains',
    'drag_method': 'Epstein/Stokes',
    'grain_size': ([0.1, 0.316, 1.0, 3.16, 10.0], 'cm'),
    'grain_density': (0.5e-14, 'g / cm^3'),
    'velocity_delta': (1.0, 'cm / s'),
    'maximum_time': (0.1, 's'),
    'number_of_dumps': 100,
}

# Initialize the 'parameters' dictionary.
parameters = dict()

# Iterate over dust-to-gas ratio and generate a Parameters object for each value.
for val in _dust_to_gas_ratio:
    f = val / len(_dust_to_gas_ratio)
    _parameters_asdict['dust_to_gas_ratio'] = (f, f, f, f, f)
    _parameters_asdataclass = set_parameters({**_parameters_asdict})
    parameters[f'Epstein-f={f}'] = _parameters_asdataclass