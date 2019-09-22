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

- RUN_ROOT_DIR
- PHANTOM_DIR
- REQUIRED_PHANTOM_GIT_COMMIT_HASH
- HDF5ROOT
- K_DRAG
"""

import pathlib
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Union

import numpy as np

import phantombuild
import phantomsetup

RUN_ROOT_DIR = pathlib.Path('~/runs/multigrain/dustybox').expanduser()
PHANTOM_DIR = pathlib.Path('~/repos/phantom').expanduser()

REQUIRED_PHANTOM_GIT_COMMIT_HASH = '6666c55feea1887b2fd8bb87fbe3c2878ba54ed7'

if sys.platform == 'darwin':
    HDF5ROOT = pathlib.Path('/usr/local/opt/hdf5')
elif sys.platform == 'linux':
    HDF5ROOT = pathlib.Path('/usr/lib/x86_64-linux-gnu/hdf5/serial')

K_DRAG = (0.1, 1.0, 10.0)


@dataclass
class Parameters:

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
    dust_to_gas_ratio: tuple = (0.01, 0.02, 0.03, 0.04, 0.05)
    drag_method: str = 'K_const'
    K_drag: float = 1.0
    velocity_delta: float = 1.0
    maximum_time: float = 0.1
    number_of_dumps: int = 20


def main():

    phantombuild.get_phantom(phantom_dir=PHANTOM_DIR)
    phantombuild.checkout_phantom_version(
        phantom_dir=PHANTOM_DIR,
        required_phantom_git_commit_hash=REQUIRED_PHANTOM_GIT_COMMIT_HASH,
    )

    setups = setup_all(RUN_ROOT_DIR)
    return setups


def setup_all(run_root_directory: pathlib.Path):

    print('\n' + 72 * '-')
    print('>>> Setting up calculations <<<')
    print(72 * '-' + '\n')

    if not run_root_directory.exists():
        run_root_directory.mkdir(parents=True)

    setups = list()
    for K_drag in K_DRAG:
        run_label = f'K={K_drag}'
        print(f'Setting up {run_label}...')

        maximum_time = 0.1 / K_drag
        parameters = Parameters(K_drag=K_drag, maximum_time=maximum_time)

        run_directory = run_root_directory / run_label
        run_directory.mkdir()

        setups.append(
            setup_dustybox(parameters=parameters, run_directory=run_directory)
        )

    return setups


def setup_dustybox(
    parameters: Parameters, run_directory: Union[str, Path]
) -> phantomsetup.Setup:

    # Constants
    igas = phantomsetup.defaults.PARTICLE_TYPE['igas']
    idust = phantomsetup.defaults.PARTICLE_TYPE['idust']

    # Setup
    setup = phantomsetup.Setup()
    setup.prefix = parameters.prefix

    setup.set_compile_option('IND_TIMESTEPS', False)
    setup.set_output(tmax=parameters.maximum_time, ndumps=parameters.number_of_dumps)

    length_unit = phantomsetup.units.unit_string_to_cgs(parameters.length_unit)
    mass_unit = phantomsetup.units.unit_string_to_cgs(parameters.mass_unit)
    time_unit = phantomsetup.units.unit_string_to_cgs(parameters.time_unit)
    setup.set_units(length=length_unit, mass=mass_unit, time=time_unit)

    setup.set_equation_of_state(ieos=parameters.ieos, polyk=parameters.sound_speed ** 2)

    number_of_dust_species = len(parameters.dust_to_gas_ratio)
    density_dust = [
        eps * parameters.density_gas for eps in parameters.dust_to_gas_ratio
    ]
    setup.set_dust(
        dust_method='largegrains',
        drag_method=parameters.drag_method,
        drag_constant=parameters.K_drag,
        number_of_dust_species=number_of_dust_species,
    )

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
    setups = main()
