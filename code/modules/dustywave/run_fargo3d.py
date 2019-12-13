"""Setup and run dustywave calculations."""

import copy
import pathlib
import subprocess
from pathlib import Path
from typing import Any, Dict, List, Tuple

import click
import numba
import numpy as np
import phantombuild
import phantomsetup
import pint
from numba import float64
from numpy import ndarray

units = pint.UnitRegistry(system='cgs')

# Required Phantom version.
PHANTOM_VERSION = '19d7c66baff909133e4d1122bc3fb943d1a71ce4'

# Phantom patch.
PHANTOM_PATCH = pathlib.Path(__file__).resolve().parent.parent.parent / 'phantom.patch'

# Path to HDF5 library.
HDF5ROOT = '/usr/local/opt/hdf5'


def set_parameters():
    """Generate a dictionary of parameter dictionaries.

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
        'number_of_particles_in_x_gas'
        'number_of_particles_in_x_dust'
        'omega'
        'density_gas'
        'delta_density_gas'
        'delta_v_gas'
        'density_dust'
        'delta_density_dust'
        'delta_v_dust'
        'K_drag'
        'maximum_time'
        'number_of_dumps'

    All float or ndarray variables can have units.
    """
    # Dictionary of parameters common to all runs.
    _parameters = {
        'prefix': 'dustywave',
        'length_unit': 1.0,
        'mass_unit': 1.0,
        'time_unit': 1.0,
        'sound_speed': 1.0,
        'box_width': 1.0,
        'number_of_particles_in_x_gas': 32,
        'number_of_particles_in_x_dust': 32,
        'density_gas': 1.0,
        'wave_amplitude': 1.0e-4,
        'maximum_time': 2.0,
        'number_of_dumps': 100,
    }
    parameters = dict()

    # Two species
    d = copy.copy(_parameters)
    d['delta_density_gas'] = 1.0
    d['delta_v_gas'] = -0.701960 - 0.304924j
    d['omega'] = 1.915896 - 4.410541j
    d['density_dust'] = (2.24,)
    d['delta_density_dust'] = (0.165251 - 1.247801j,)
    d['delta_v_dust'] = (-0.221645 + 0.368534j,)
    tstop = (0.4,)
    d['K_drag'] = tuple([1 / ts for ts in tstop])
    parameters['two species'] = d

    # Five species
    d = copy.copy(_parameters)
    d['delta_density_gas'] = 1.0
    d['delta_v_gas'] = -0.874365 - 0.145215j
    d['omega'] = 0.912414 - 5.493800j
    d['density_dust'] = (0.1, 0.233333, 0.366667, 0.5)
    d['delta_density_dust'] = (
        0.080588 - 0.048719j,
        0.091607 - 0.134955j,
        0.030927 - 0.136799j,
        0.001451 - 0.090989j,
    )
    d['delta_v_dust'] = (
        -0.775380 + 0.308952j,
        -0.427268 + 0.448704j,
        -0.127928 + 0.313967j,
        -0.028963 + 0.158693j,
    )
    tstop = (0.1, 0.215443, 0.464159, 1.0)
    d['K_drag'] = tuple([1 / ts for ts in tstop])
    parameters['five species'] = d

    return parameters


def setup_all_calculations(
    run_root_directory: Path,
    parameters_dict: Dict[str, dict],
    phantom_dir: Path,
    hdf5root: Path,
) -> List[phantomsetup.Setup]:
    """Set up multiple calculations.

    Parameters
    ----------
    run_root_directory
        The path to the root directory for this series of runs.
    parameters_dict
        A dictionary of dictionaries. The key is the "run label" which
        will be the sub-directory of the root directory. The value is
        the parameters dictionary for the run.
    phantom_dir
        The path to the Phantom repository.
    hdf5root
        The path to the root directory containing the HDF5 library.

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
    for run_label, params in parameters_dict.items():
        print(f'Setting up {run_label}...')
        run_directory = run_root_directory / run_label
        run_directory.mkdir()
        setups.append(
            setup_one_calculation(
                params=params,
                run_directory=run_directory,
                phantom_dir=phantom_dir,
                hdf5root=hdf5root,
            )
        )

    return setups


def setup_one_calculation(
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
    params = copy.copy(params)

    # Constants
    igas = phantomsetup.defaults.PARTICLE_TYPE['igas']
    idust = phantomsetup.defaults.PARTICLE_TYPE['idust']

    # Setup
    setup = phantomsetup.Setup()
    setup.prefix = params.pop('prefix')

    # Units
    length_unit = params.pop('length_unit')
    mass_unit = params.pop('mass_unit')
    time_unit = params.pop('time_unit')
    for key, value in params.items():
        if isinstance(value, units.Quantity):
            d = value.dimensionality
            new_units = (
                length_unit ** d['[length]']
                * mass_unit ** d['[mass]']
                * time_unit ** d['[time]']
            )
            params[key] = value.to(new_units).magnitude
    if isinstance(length_unit, units.Quantity):
        length_unit = length_unit.to_base_units().magnitude
    if isinstance(mass_unit, units.Quantity):
        mass_unit = mass_unit.to_base_units().magnitude
    if isinstance(time_unit, units.Quantity):
        time_unit = time_unit.to_base_units().magnitude
    setup.set_units(length=length_unit, mass=mass_unit, time=time_unit)

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

    n_particles_in_yz = 6
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
    drho = params['delta_density_gas']
    kwave = 2 * np.pi / box_width
    ampl = params['wave_amplitude']

    @numba.vectorize([float64(float64)])
    def density_function(x):
        x = 1.0 + ampl * (
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
        drho = params['delta_density_dust'][idx]
        kwave = 2 * np.pi / box_width
        ampl = params['wave_amplitude']

        @numba.vectorize([float64(float64)])
        def density_function(x):
            x = 1.0 + ampl * (
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
    extra_compiler_arguments = ['FC=gfortran-9']
    setup.compile_phantom(
        phantom_dir=phantom_dir,
        hdf5root=hdf5root,
        working_dir=run_directory,
        extra_compiler_arguments=extra_compiler_arguments,
    )

    # Return setup
    return setup


def run_all_calculations(run_root_directory: Path) -> List[subprocess.CompletedProcess]:
    """Run dustywave calculations.

    Parameters
    ----------
    run_root_directory
        Root directory containing the run directories.

    Returns
    -------
    List[subprocess.CompletedProcess]
        A list with the outputs from each completed process.
    """
    print('\n' + 72 * '-')
    print('>>> Running calculations <<<')
    print(72 * '-' + '\n')

    results = list()
    for directory in sorted(run_root_directory.iterdir()):
        if not directory.is_dir():
            continue
        print(f'Running {directory.name}...')
        in_files = list(directory.glob('*.in'))
        if len(in_files) > 1:
            raise ValueError('Too many .in files in directory')
        in_file = in_files[0].name
        log_file = f'{in_files[0].stem}01.log'
        with open(directory / log_file, 'w') as fp:
            result = subprocess.run(
                [directory / 'phantom', in_file], cwd=directory, stdout=fp, stderr=fp
            )
        results.append(result)

    return results


@click.command()
@click.option(
    '--run_directory', required=True, help='the directory for the calculations'
)
@click.option(
    '--hdf5_directory',
    help='the path to the HDF5 libary',
    default=HDF5ROOT,
    show_default=True,
)
def cli(run_directory, hdf5_directory):
    """CLI interface."""
    parameters_dict = set_parameters()
    run_directory = pathlib.Path(run_directory).expanduser()
    hdf5_directory = pathlib.Path(hdf5_directory).expanduser()
    phantom_dir = run_directory.parent / '.phantom'
    phantombuild.get_phantom(phantom_dir=phantom_dir)
    phantombuild.checkout_phantom_version(
        phantom_dir=phantom_dir, required_phantom_git_commit_hash=PHANTOM_VERSION
    )
    phantombuild.patch_phantom(
        phantom_dir=phantom_dir, phantom_patch=PHANTOM_PATCH,
    )
    setup_all_calculations(
        run_root_directory=run_directory,
        parameters_dict=parameters_dict,
        phantom_dir=phantom_dir,
        hdf5root=hdf5_directory,
    )
    run_all_calculations(run_root_directory=run_directory)


if __name__ == "__main__":
    cli()
