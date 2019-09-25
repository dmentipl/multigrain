"""
Setup dustybox calculations.

This script does the following.

- Clone Phantom, and check out a particular version.
- Set up several Phantom calculations.

The parameters for the problem are set in the Parameters data class. The
script sets up one calculation per value in K_DRAG. The only difference
between each calculation is the K_drag value.

Phantom is compiled with HDF5. So the library must be available.

The following global variables are set.

- PHANTOM_DIR
- REQUIRED_PHANTOM_GIT_COMMIT_HASH
- HDF5ROOT
"""

from __future__ import annotations

import argparse
import pathlib
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Union

import numpy as np
import tomlkit
import phantombuild
import phantomsetup

PHANTOM_DIR = pathlib.Path('~/repos/phantom').expanduser()
REQUIRED_PHANTOM_GIT_COMMIT_HASH = '6666c55feea1887b2fd8bb87fbe3c2878ba54ed7'
if sys.platform == 'darwin':
    HDF5ROOT = pathlib.Path('/usr/local/opt/hdf5')
elif sys.platform == 'linux':
    HDF5ROOT = pathlib.Path('/usr/lib/x86_64-linux-gnu/hdf5/serial')


@dataclass
class Parameters:
    """
    Dusty box parameters.

    If drag_method is set to 'Epstein/Stokes' you must set grain_size
    and grain_density. If drag_method is set to 'K_drag' you must set
    K_drag. Note that the length of dust_to_gas_ratio must match
    grain_size and grain_density.
    """

    prefix: str = 'dustybox'
    length_unit: str = 'au'
    mass_unit: str = 'solarm'
    time_unit: str = 'year'
    ieos: int = 1
    sound_speed: float = 1.0
    box_boundary: tuple = (-0.5, 0.5, -0.5, 0.5, -0.5, 0.5)
    number_of_particles_gas: int = 50_000
    number_of_particles_dust: int = 10_000
    density_gas: float = 1.0
    dust_to_gas_ratio: tuple = (0.01, 0.01, 0.01, 0.01, 0.01)
    drag_method: str = 'Epstein/Stokes'
    K_drag: float = 1.0
    grain_size: tuple = (0.01, 0.03, 0.1, 0.3, 1.0)
    grain_density: float = 3.0
    velocity_delta: float = 1.0
    maximum_time: float = 0.1
    number_of_dumps: int = 20

    def write_to_file(self, filename: Union[str, Path]) -> Parameters:
        """
        Write the parameters to TOML file.

        Parameters
        ----------
        filename : str or Path
            The name of the file to write. Should have extension
            '.toml'.
        """
        document = tomlkit.document()
        for key, value in self.__dict__.items():
            if isinstance(value, tuple):
                value = list(value)
            document.add(key, value)
        with open(filename, 'w') as fp:
            fp.write(tomlkit.dumps(document))
        return self

    def read_from_file(self, filename: Union[str, Path]) -> Parameters:
        """
        Read parameters from TOML file.

        Parameters
        ----------
        filename : str or Path
            The name of the file to read. Should have extension '.toml'.

        Returns
        -------
        Parameters
            The Parameters object generated from the parameters file.
        """

        if not pathlib.Path(filename).exists():
            raise ValueError('parameter file does not exist')

        with open(filename, 'r') as fp:
            t = tomlkit.loads(fp.read())

        d = dict()
        for key, val in t.items():
            if isinstance(val, list):
                val = tuple(val)
            d[key] = val

        return Parameters(**d)


def main():

    run_root_dir, parameter_files = get_command_line()
    parameters_dict = get_parameters(parameter_files)

    phantombuild.get_phantom(phantom_dir=PHANTOM_DIR)
    phantombuild.checkout_phantom_version(
        phantom_dir=PHANTOM_DIR,
        required_phantom_git_commit_hash=REQUIRED_PHANTOM_GIT_COMMIT_HASH,
    )

    setups = setup_all(run_root_dir, parameters_dict)
    return setups


def get_command_line():
    """
    Get command line options.

    Returns
    -------
    run_root_dir : Path
        The path to the root directory for the calculations.
    parameter_files : List[Path]
        The path to the parameter files.
    """
    parser = argparse.ArgumentParser(description='Set up dustybox calculations')
    parser.add_argument(
        '-r',
        '--run_root_dir',
        help='the root directory for the calculations',
        required=True,
    )
    parser.add_argument(
        '-p', '--parameters', nargs='+', help='a list of parameter files', required=True
    )
    args = parser.parse_args()
    run_root_dir = pathlib.Path(args.run_root_dir).resolve()
    parameter_files = [
        pathlib.Path(parameter_file).resolve() for parameter_file in args.parameters
    ]
    for parameter_file in parameter_files:
        if not parameter_file.exists():
            raise ValueError('parameter_file does not exist')
    return run_root_dir, parameter_files


def get_parameters(parameter_files: List[Path]) -> Dict[str, Parameters]:
    """
    Get Parameter objects from parameter files.

    Parameters
    ----------
    parameter_files
        The list of parameter files.

    Returns
    -------
    Dict[str, Parameters]   
        A dictionary of Parameters. The key is the "run label" which
        will be the sub-directory of the root directory. The value is
        the Parameters data object for the run.
    """
    parameters_dict = {}
    for parameter_file in parameter_files:
        parameters_dict[parameter_file.stem] = Parameters().read_from_file(
            parameter_file
        )
    return parameters_dict


def setup_all(run_root_directory: pathlib.Path, parameters_dict: Dict[str, Parameters]):
    """
    Setup multiple calculations.

    Parameters
    ----------
    run_root_directory
        The path to the root directory for this series of runs.
    parameters_dict
        A dictionary of Parameters. The key is the "run label" which
        will be the sub-directory of the root directory. The value is
        the Parameters data object for the run.

    Returns
    -------
    List[phantomsetup.Setup]
        A list of Setup objects.
    """

    print('\n' + 72 * '-')
    print('>>> Setting up calculations <<<')
    print(72 * '-' + '\n')

    if not run_root_directory.exists():
        run_root_directory.mkdir(parents=True)

    setups = list()
    for run_label, parameters in parameters_dict.items():
        print(f'Setting up {run_label}...')
        run_directory = run_root_directory / run_label
        run_directory.mkdir()
        setups.append(
            setup_dustybox(parameters=parameters, run_directory=run_directory)
        )

    return setups


def setup_dustybox(
    parameters: Parameters, run_directory: Union[str, Path]
) -> phantomsetup.Setup:
    """
    Setup a Phantom dustybox calculation.

    Parameters
    ----------
    parameters
        The parameters for this calculation.
    run_directory
        The path to the directory containing the run.

    Returns
    -------
    phantomsetup.Setup
    """

    # Constants
    igas = phantomsetup.defaults.PARTICLE_TYPE['igas']
    idust = phantomsetup.defaults.PARTICLE_TYPE['idust']

    # Setup
    setup = phantomsetup.Setup()
    setup.prefix = parameters.prefix

    setup.set_compile_option('IND_TIMESTEPS', False)
    setup.set_output(
        tmax=parameters.maximum_time, ndumps=parameters.number_of_dumps, nfulldump=1
    )

    length_unit = phantomsetup.units.unit_string_to_cgs(parameters.length_unit)
    mass_unit = phantomsetup.units.unit_string_to_cgs(parameters.mass_unit)
    time_unit = phantomsetup.units.unit_string_to_cgs(parameters.time_unit)
    setup.set_units(length=length_unit, mass=mass_unit, time=time_unit)

    setup.set_equation_of_state(ieos=parameters.ieos, polyk=parameters.sound_speed ** 2)

    number_of_dust_species = len(parameters.dust_to_gas_ratio)
    density_dust = [
        eps * parameters.density_gas for eps in parameters.dust_to_gas_ratio
    ]
    if parameters.drag_method == 'Epstein/Stokes':
        setup.set_dust(
            dust_method='largegrains',
            drag_method='Epstein/Stokes',
            grain_size=parameters.grain_size,
            grain_density=parameters.grain_density,
        )
    elif parameters.drag_method == 'K_drag':
        setup.set_dust(
            dust_method='largegrains',
            drag_method='K_drag',
            drag_constant=parameters.K_drag,
            number_of_dust_species=number_of_dust_species,
        )
    else:
        raise ValueError('Cannot set up dust')

    setup.set_boundary(parameters.box_boundary, periodic=True)

    def velocity_gas(xyz: np.ndarray) -> np.ndarray:
        """Gas has zero initial velocity."""
        vxyz = np.zeros_like(xyz)
        return vxyz

    box = phantomsetup.Box(*parameters.box_boundary)
    box.add_particles(
        particle_type=igas,
        number_of_particles=parameters.number_of_particles_gas,
        density=parameters.density_gas,
        velocity_distribution=velocity_gas,
    )
    setup.add_box(box)

    def velocity_dust(xyz: np.ndarray) -> np.ndarray:
        """Dust has uniform initial velocity."""
        vxyz = np.zeros_like(xyz)
        vxyz[:, 0] = parameters.velocity_delta
        return vxyz

    for idx in range(number_of_dust_species):
        box = phantomsetup.Box(*parameters.box_boundary)
        box.add_particles(
            particle_type=idust + idx,
            number_of_particles=parameters.number_of_particles_dust,
            density=density_dust[idx],
            velocity_distribution=velocity_dust,
        )
        setup.add_box(box)

    alpha = np.zeros(setup.total_number_of_particles, dtype=np.single)
    setup.add_array_to_particles('alpha', alpha)

    # Write to file
    setup.write_dump_file(directory=run_directory)
    setup.write_in_file(directory=run_directory)

    # Compile Phantom
    setup.compile_phantom(
        phantom_dir=PHANTOM_DIR, hdf5root=HDF5ROOT, working_dir=run_directory
    )

    # Return setup
    return setup


if __name__ == '__main__':
    main()
