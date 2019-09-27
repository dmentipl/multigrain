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
- HDF5ROOT
"""

import pathlib
import sys
from pathlib import Path
from typing import Dict, List, Union

import numpy as np
import phantomsetup
from numpy import ndarray

from dustybox_parameters import Parameters

PHANTOM_DIR = pathlib.Path('~/repos/phantom').expanduser()
if sys.platform == 'darwin':
    HDF5ROOT = pathlib.Path('/usr/local/opt/hdf5')
elif sys.platform == 'linux':
    HDF5ROOT = pathlib.Path('/usr/lib/x86_64-linux-gnu/hdf5/serial')


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

    if isinstance(parameters.length_unit, str):
        length_unit = phantomsetup.units.unit_string_to_cgs(parameters.length_unit)
    else:
        length_unit = parameters.length_unit
    if isinstance(parameters.mass_unit, str):
        mass_unit = phantomsetup.units.unit_string_to_cgs(parameters.mass_unit)
    else:
        mass_unit = parameters.mass_unit
    if isinstance(parameters.time_unit, str):
        time_unit = phantomsetup.units.unit_string_to_cgs(parameters.time_unit)
    else:
        time_unit = parameters.time_unit
    setup.set_units(length=length_unit, mass=mass_unit, time=time_unit)

    setup.set_equation_of_state(ieos=1, polyk=parameters.sound_speed ** 2)

    number_of_dust_species = len(parameters.dust_to_gas_ratio)
    density_dust = [
        eps * parameters.density_gas for eps in parameters.dust_to_gas_ratio
    ]
    if parameters.drag_method == 'Epstein/Stokes':
        setup.set_dust(
            dust_method=parameters.dust_method,
            drag_method=parameters.drag_method,
            grain_size=parameters.grain_size,
            grain_density=parameters.grain_density,
        )
    elif parameters.drag_method == 'K_const':
        setup.set_dust(
            dust_method=parameters.dust_method,
            drag_method=parameters.drag_method,
            drag_constant=parameters.K_drag,
            number_of_dust_species=number_of_dust_species,
        )
    else:
        raise ValueError('Cannot set up dust')

    setup.set_boundary(parameters.box_boundary, periodic=True)

    def velocity_gas(xyz: ndarray) -> ndarray:
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

    def velocity_dust(xyz: ndarray) -> ndarray:
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