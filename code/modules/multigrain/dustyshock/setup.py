"""Setup and run dustyshock calculations."""

import copy
import pathlib
import subprocess
from pathlib import Path
from typing import Any, Dict, List, Tuple

import click
import numpy as np
import phantombuild
import phantomsetup
import pint
from numpy import ndarray

units = pint.UnitRegistry(system='cgs')

# Required Phantom version.
PHANTOM_VERSION = '0ee3d8da8d8756cd29e6969b341a16d24ee4752b'

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
        'lattice'
        'number_of_particles_in_x_R'
        'density_L'
        'density_R'
        'dust_to_gas_ratio'
        'K_const'
        'velocity_L'
        'velocity_R'
        'maximum_time'
        'number_of_dumps'

    All float or ndarray variables can have units.

    The length of 'dust_to_gas_ratio', 'grain_size', and
    'velocity_delta' should be the same, i.e. the number of dust
    species.
    """
    # Mach number
    mach = 2.0

    # Dictionary of parameters common to all runs.
    _parameters = {
        'prefix': 'dustyshock',
        'length_unit': 1.0,
        'mass_unit': 1.0,
        'time_unit': 1.0,
        'sound_speed': 1.0,
        'box_width': 200,
        'lattice': 'close packed',
        'number_of_particles_in_x_R': 128,
        'density_L': 1.0,
        'velocity_L': mach,
        'maximum_time': 10.0,
        'number_of_dumps': 100,
    }

    # Each value in tuple multiplicatively generates a new simulation.
    K_drag = ([1.0], [1.0, 3.0, 5.0])
    density_R = (8.0, 16.0)
    velocity_R = (mach * 0.125, mach * 0.0625)

    # Iterate over dust-to-gas ratio and grain sizes.
    parameters = dict()
    for K, rho_R, v_R in zip(K_drag, density_R, velocity_R):
        N = len(K)
        label = f'N={N}'
        parameters[label] = copy.copy(_parameters)
        parameters[label]['K_drag'] = K
        parameters[label]['density_R'] = rho_R
        parameters[label]['velocity_R'] = v_R

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
    params = copy.copy(params)

    # Constants
    igas = phantomsetup.defaults.PARTICLE_TYPE['igas']
    idust = phantomsetup.defaults.PARTICLE_TYPE['idust']
    # iboundary = phantomsetup.defaults.PARTICLE_TYPE['iboundary']

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
    n_L = params['number_of_particles_in_x_R'] * (params['density_L']/params['density_R']) ** (1/3)
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
    """Run dustyshock calculations.

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
    setup_all_calculations(
        run_root_directory=run_directory,
        parameters_dict=parameters_dict,
        phantom_dir=phantom_dir,
        hdf5root=hdf5_directory,
    )
    run_all_calculations(run_root_directory=run_directory)


if __name__ == "__main__":
    cli()
