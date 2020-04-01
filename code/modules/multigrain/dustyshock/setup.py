"""Setup and run dustyshock calculations."""

from pathlib import Path
from typing import Any, Dict, Tuple

import numpy as np
import phantomsetup
import pint
from numpy import ndarray

from ..config import EXTRA_COMPILER_ARGUMENTS

units = pint.UnitRegistry(system='cgs')


def setup_calculation(
    params: Dict[str, Any], run_directory: Path, phantom_dir: Path, hdf5root: Path
) -> phantomsetup.Setup:
    """Set up a Phantom dustyshock calculation.

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
    """
    # Constants
    igas = phantomsetup.defaults.PARTICLE_TYPE['igas']
    idust = phantomsetup.defaults.PARTICLE_TYPE['idust']
    # iboundary = phantomsetup.defaults.PARTICLE_TYPE['iboundary']

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

    # Equation of state
    setup.set_equation_of_state(ieos=1, polyk=params['sound_speed'] ** 2)

    # Dust method
    number_of_dust_species = len(params['K_drag'])
    setup.set_dust(
        dust_method='largegrains',
        drag_method='K_const',
        drag_constant=params['K_drag'],
        number_of_dust_species=number_of_dust_species,
    )

    # Domain
    box_width = params['box_width']

    n_particles_in_yz = 8
    dx_L = 0.5 * box_width / params['number_of_particles_in_x_R']
    y_width = n_particles_in_yz * dx_L
    z_width = n_particles_in_yz * dx_L

    xmin = -box_width / 2
    xmax = box_width / 2
    ymin = -y_width / 2 * np.sqrt(3) / 2
    ymax = y_width / 2 * np.sqrt(3) / 2
    zmin = -z_width / 2 * np.sqrt(6) / 3
    zmax = z_width / 2 * np.sqrt(6) / 3

    domain_boundary = (xmin - 1000 * dx_L, xmax + 1000 * dx_L, ymin, ymax, zmin, zmax)

    setup.set_boundary(domain_boundary, periodic=True)

    # Box: left of shock
    box_boundary_L = (xmin, 0, ymin, ymax, zmin, zmax)
    n_L = params['number_of_particles_in_x_R'] * (
        params['density_L'] / params['density_R']
    ) ** (1 / 3)
    rho_L = params['density_L']
    v_L = params['velocity_L']

    def velocity_L(
        x: ndarray, y: ndarray, z: ndarray
    ) -> Tuple[ndarray, ndarray, ndarray]:
        """Initial left velocity."""
        vx, vy, vz = v_L * np.ones(x.shape), np.zeros(y.shape), np.zeros(z.shape)
        return vx, vy, vz

    # Box: right of shock
    box_boundary_R = (0, xmax, ymin, ymax, zmin, zmax)
    n_R = params['number_of_particles_in_x_R']
    rho_R = params['density_R']
    v_R = params['velocity_R']

    def velocity_R(
        x: ndarray, y: ndarray, z: ndarray
    ) -> Tuple[ndarray, ndarray, ndarray]:
        """Initial right velocity."""
        vx, vy, vz = v_R * np.ones(x.shape), np.zeros(y.shape), np.zeros(z.shape)
        return vx, vy, vz

    # Boxes
    boxes = list()

    # Gas box: left of shock
    box = phantomsetup.Box(
        box_boundary=box_boundary_L,
        particle_type=igas,
        number_of_particles_in_x=n_L,
        density=rho_L,
        velocity_distribution=velocity_L,
        lattice='close packed',
    )
    boxes.append(box)

    # Gas box: right of shock
    box = phantomsetup.Box(
        box_boundary=box_boundary_R,
        particle_type=igas,
        number_of_particles_in_x=n_R,
        density=rho_R,
        velocity_distribution=velocity_R,
        lattice='close packed',
    )
    boxes.append(box)

    for idx in range(number_of_dust_species):

        # Dust box: left of shock
        box = phantomsetup.Box(
            box_boundary=box_boundary_L,
            particle_type=idust + idx,
            number_of_particles_in_x=n_L,
            density=rho_L,
            velocity_distribution=velocity_L,
            lattice='close packed',
        )
        boxes.append(box)

        # Dust box: right of shock
        box = phantomsetup.Box(
            box_boundary=box_boundary_R,
            particle_type=idust + idx,
            number_of_particles_in_x=n_R,
            density=rho_R,
            velocity_distribution=velocity_R,
            lattice='close packed',
        )
        boxes.append(box)

    # # Set boundary particles
    # for box in boxes:
    #     x = box.arrays['position']
    #     boundary_particles = np.argwhere((x < xmin + dx_L) | (x > xmax - dx_L))
    #     particle_type = box.arrays['particle_type']
    #     particle_type[boundary_particles] = iboundary
    #     box.set_array('particle_type', particle_type)

    # Add extra quantities
    for box in boxes:
        alpha = np.zeros(box.number_of_particles, dtype=np.single)
        box.set_array('alpha', alpha)

    # Add boxes to setup
    for box in boxes:
        setup.add_container(box)

    # Set dissipation
    setup.set_dissipation(alpha=1.0, alphamax=1.0)

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
