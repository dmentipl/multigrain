"""Setup and run dustywave calculations."""

from pathlib import Path
from typing import Any, Dict, Tuple

import numba
import numpy as np
import phantomsetup
import pint
from numba import float64
from numpy import ndarray

from ..config import EXTRA_COMPILER_ARGUMENTS

units = pint.UnitRegistry(system='cgs')


def setup_calculation(
    params: Dict[str, Any], run_directory: Path, phantom_dir: Path, hdf5root: Path
) -> phantomsetup.Setup:
    """Set up a Phantom dustywave calculation.

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

    # Dust grains
    number_of_dust_species = len(params['density_dust'])
    density_dust = params['density_dust']
    setup.set_dust(
        dust_method='largegrains',
        drag_method='K_const',
        drag_constant=params['K_drag'],
        number_of_dust_species=number_of_dust_species,
    )

    # Boxes
    boxes = list()

    box_width = params['box_width']

    n_particles_in_yz = 8
    dx = box_width / params['number_of_particles_in_x_gas']
    y_width = n_particles_in_yz * dx
    z_width = n_particles_in_yz * dx

    xmin = -box_width / 2
    xmax = box_width / 2
    ymin = -y_width / 2 * np.sqrt(3) / 2
    ymax = y_width / 2 * np.sqrt(3) / 2
    zmin = -z_width / 2 * np.sqrt(6) / 3
    zmax = z_width / 2 * np.sqrt(6) / 3

    box_boundary = (xmin, xmax, ymin, ymax, zmin, zmax)

    setup.set_boundary(box_boundary, periodic=True)

    # Density perturbation
    rho = params['density_gas']
    drho = params['delta_density_gas']
    kwave = 2 * np.pi / box_width
    ampl = params['wave_amplitude']

    @numba.vectorize([float64(float64)])
    def density_function(x):
        x = rho + ampl * (
            drho.real * np.cos(kwave * (x + box_width / 2))
            - drho.imag * np.sin(kwave * (x + box_width / 2))
        )
        return x

    # Velocity perturbation
    dv = params['delta_v_gas']

    def velocity_perturbation(
        x: ndarray, y: ndarray, z: ndarray
    ) -> Tuple[ndarray, ndarray, ndarray]:
        """Initialize velocity perturbation."""
        vx, vy, vz = np.zeros(x.shape), np.zeros(y.shape), np.zeros(z.shape)
        vx = ampl * (
            dv.real * np.cos(kwave * (x + box_width / 2))
            - dv.imag * np.sin(kwave * (x + box_width / 2))
        )
        return vx, vy, vz

    # Gas
    box = phantomsetup.Box(
        box_boundary=box_boundary,
        particle_type=igas,
        number_of_particles_in_x=params['number_of_particles_in_x_gas'],
        density=params['density_gas'],
        velocity_distribution=velocity_perturbation,
        lattice='close packed',
    )
    position = phantomsetup.geometry.stretch_map(
        density_function, box.arrays['position'], box_boundary[0], box_boundary[1]
    )
    box.arrays['position'] = position
    boxes.append(box)

    # Dust
    for idx in range(number_of_dust_species):

        # Density perturbation
        rho = params['density_dust'][idx]
        drho = params['delta_density_dust'][idx]
        kwave = 2 * np.pi / box_width
        ampl = params['wave_amplitude']

        @numba.vectorize([float64(float64)])
        def density_function(x):
            x = rho + ampl * (
                drho.real * np.cos(kwave * (x + box_width / 2))
                - drho.imag * np.sin(kwave * (x + box_width / 2))
            )
            return x

        # Velocity perturbation
        dv = params['delta_v_dust'][idx]

        def velocity_perturbation(
            x: ndarray, y: ndarray, z: ndarray
        ) -> Tuple[ndarray, ndarray, ndarray]:
            """Initialize velocity perturbation."""
            vx, vy, vz = np.zeros(x.shape), np.zeros(y.shape), np.zeros(z.shape)
            vx = ampl * (
                dv.real * np.cos(kwave * (x + box_width / 2))
                - dv.imag * np.sin(kwave * (x + box_width / 2))
            )
            return vx, vy, vz

        box = phantomsetup.Box(
            box_boundary=box_boundary,
            particle_type=idust + idx,
            number_of_particles_in_x=params['number_of_particles_in_x_dust'],
            density=density_dust[idx],
            velocity_distribution=velocity_perturbation,
            lattice='close packed',
        )
        position = phantomsetup.geometry.stretch_map(
            density_function, box.arrays['position'], box_boundary[0], box_boundary[1]
        )
        box.arrays['position'] = position
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
