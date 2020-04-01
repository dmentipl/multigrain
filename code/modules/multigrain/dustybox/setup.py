"""Set up a dustybox calculation."""

from pathlib import Path
from typing import Any, Dict, Tuple

import numpy as np
import phantomsetup
from numpy import ndarray

from ..config import EXTRA_COMPILER_ARGUMENTS


def setup_calculation(
    params: Dict[str, Any], run_directory: Path, phantom_dir: Path, hdf5root: Path
) -> phantomsetup.Setup:
    """Set up a Phantom dustybox calculation.

    Parameters
    ----------
    params
        The parameters for this calculation.
    run_directory
        The path to the directory containing the run.
    phantom_dir
        The path to the Phantom repository.
    hdf5root
        The path to the root directory containing the HDF5 library.

    Returns
    -------
    phantomsetup.Setup

    Note
    ----
    The params dictionary run needs the following keys:

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

    The length of 'dust_to_gas_ratio', 'grain_size', and
    'velocity_delta' should be the same, i.e. the number of dust
    species.
    """
    # Constants
    igas = phantomsetup.defaults.PARTICLE_TYPE['igas']
    idust = phantomsetup.defaults.PARTICLE_TYPE['idust']

    # Setup
    setup = phantomsetup.Setup()
    setup.prefix = params['prefix']

    # Units
    setup.set_units(
        length=params['length_unit'], mass=params['mass_unit'], time=params['time_unit']
    )

    setup.set_compile_option('IND_TIMESTEPS', False)
    setup.set_output(
        tmax=params['maximum_time'], ndumps=params['number_of_dumps'], nfulldump=1
    )

    setup.set_equation_of_state(ieos=1, polyk=params['sound_speed'] ** 2)

    number_of_dust_species = len(params['dust_to_gas_ratio'])
    density_dust = [eps * params['density_gas'] for eps in params['dust_to_gas_ratio']]
    if params['drag_method'] == 'Epstein/Stokes':
        setup.set_dust(
            dust_method='largegrains',
            drag_method=params['drag_method'],
            grain_size=params['grain_size'],
            grain_density=params['grain_density'],
        )
    elif params['drag_method'] == 'K_const':
        setup.set_dust(
            dust_method='largegrains',
            drag_method=params['drag_method'],
            drag_constant=params['K_drag'],
            number_of_dust_species=number_of_dust_species,
        )
    else:
        raise ValueError('Cannot set up dust')

    # Boxes
    boxes = list()

    lattice = params['lattice']
    box_width = params['box_width']

    xmin = -box_width / 2
    xmax = box_width / 2
    ymin = -box_width / 2
    ymax = box_width / 2
    zmin = -box_width / 2
    zmax = box_width / 2

    box_boundary = (xmin, xmax, ymin, ymax, zmin, zmax)

    setup.set_boundary(box_boundary, periodic=True)

    def velocity_gas(
        x: ndarray, y: ndarray, z: ndarray
    ) -> Tuple[ndarray, ndarray, ndarray]:
        """Gas has zero initial velocity."""
        vx, vy, vz = np.zeros(x.shape), np.zeros(y.shape), np.zeros(z.shape)
        return vx, vy, vz

    # Gas
    box = phantomsetup.Box(
        box_boundary=box_boundary,
        particle_type=igas,
        number_of_particles_in_x=params['number_of_particles_in_x_gas'],
        density=params['density_gas'],
        velocity_distribution=velocity_gas,
        lattice=lattice,
    )
    boxes.append(box)

    for idx in range(number_of_dust_species):

        def velocity_dust(
            x: ndarray, y: ndarray, z: ndarray
        ) -> Tuple[ndarray, ndarray, ndarray]:
            """Dust has uniform initial velocity."""
            vx, vy, vz = np.zeros(x.shape), np.zeros(y.shape), np.zeros(z.shape)
            vx = params['velocity_delta'][idx]
            return vx, vy, vz

        box = phantomsetup.Box(
            box_boundary=box_boundary,
            particle_type=idust + idx,
            number_of_particles_in_x=params['number_of_particles_in_x_dust'],
            density=density_dust[idx],
            velocity_distribution=velocity_dust,
            lattice=lattice,
        )
        boxes.append(box)

    # Add extra quantities
    for box in boxes:
        alpha = np.zeros(box.number_of_particles, dtype=np.single)
        box.set_array('alpha', alpha)

    # Add boxes to setup
    for box in boxes:
        setup.add_container(box)

    # Set dissipation
    setup.set_dissipation(alpha=0.0, alphamax=0.0)

    # Set C_force
    C_force = params.get('C_force')
    if C_force is not None:
        setup.set_run_option('C_force', C_force)

    # Set hfact
    hfact = params.get('hfact')
    if hfact is not None:
        setup.set_run_option('hfact', hfact)

    # Write to file
    setup.write_dump_file(directory=run_directory)
    setup.write_in_file(directory=run_directory)

    # Compile Phantom
    setup.compile_phantom(
        phantom_dir=phantom_dir,
        hdf5root=hdf5root,
        working_dir=run_directory,
        extra_compiler_arguments=EXTRA_COMPILER_ARGUMENTS,
    )

    # Return setup
    return setup
