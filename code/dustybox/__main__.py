import argparse
import pathlib
from typing import Dict, List
from pathlib import Path

import phantombuild

from dustybox_parameters import set_parameters
from dustybox_setup import setup_all
from dustybox_run import run_all

PHANTOM_DIR = pathlib.Path('~/repos/phantom').expanduser()
REQUIRED_PHANTOM_GIT_COMMIT_HASH = '6666c55feea1887b2fd8bb87fbe3c2878ba54ed7'


def main():

    run_root_dir = get_command_line()

    phantombuild.get_phantom(phantom_dir=PHANTOM_DIR)
    phantombuild.checkout_phantom_version(
        phantom_dir=PHANTOM_DIR,
        required_phantom_git_commit_hash=REQUIRED_PHANTOM_GIT_COMMIT_HASH,
    )

    parameters_all = parameters_for_Epstein_tests()
    setup_all(run_root_dir, parameters_all)
    run_all(run_root_dir)


def parameters_for_Epstein_tests():

    dust_to_gas_ratio = (0.01, 0.1, 1.0, 10.0)
    parameters_all = dict()
    parameters_dict = {
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

    for f in dust_to_gas_ratio:
        parameters_dict['dust_to_gas_ratio'] = (f/5, f/5, f/5, f/5, f/5)
        parameters = set_parameters({**parameters_dict})
        parameters_all[f'Epstein-f={f}'] = parameters

    return parameters_all


def get_command_line():
    """
    Get command line options.

    Returns
    -------
    run_root_dir : Path
        The path to the root directory for the calculations.
    """
    parser = argparse.ArgumentParser(description='Set up dustybox calculations')
    parser.add_argument(
        '-r',
        '--run_root_dir',
        help='the root directory for the calculations',
        required=True,
    )
    args = parser.parse_args()
    run_root_dir = pathlib.Path(args.run_root_dir).resolve()
    return run_root_dir

if __name__ == '__main__':
    main()
