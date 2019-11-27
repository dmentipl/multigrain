"""Setup and run dustybox calculations."""

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
PHANTOM_VERSION = '6666c55feea1887b2fd8bb87fbe3c2878ba54ed7'

# Phantom patch.
PHANTOM_PATCH = pathlib.Path(__file__).resolve().parent.parent / 'phantom.patch'

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
        'number_of_particles_gas'
        'number_of_particles_dust'
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
    # Dictionary of parameters common to all runs.
    _parameters = {
        'prefix': 'dustybox',
        'length_unit': 1.0 * units['cm'],
        'mass_unit': 1.0 * units['g'],
        'time_unit': 1.0 * units['s'],
        'sound_speed': 1.0 * units['cm/s'],
        'box_width': 1.0 * units['cm'],
        'number_of_particles_gas': 50_000,
        'number_of_particles_dust': 10_000,
        'density_gas': 1.0e-13 * units['g / cm^3'],
        'drag_method': 'Epstein/Stokes',
        'grain_size': [0.1, 0.316, 1.0, 3.16, 10.0] * units['cm'],
        'grain_density': 0.5e-14 * units['g / cm^3'],
        'velocity_delta': [1.0, 1.0, 1.0, 1.0, 1.0] * units['cm / s'],
        'maximum_time': 0.1 * units['s'],
        'number_of_dumps': 100,
    }

    # Each value in dust_to_gas_ratio generates a dustybox setup.
    total_dust_to_gas_ratio = (0.01, 0.1, 1.0, 10.0)

    # Distribute dust mass between bins
    dust_mass_distribution = dict()
    size = _parameters['grain_size']
    # Equal mass in each dust bin
    dust_mass_distribution['equal'] = np.ones(len(size)) / len(size)
    # MRN-distributed mass in each dust bin
    dust_mass_distribution['MRN'] = np.sqrt(size) / np.sum(np.sqrt(size))

    # Iterate over dust-to-gas ratio and dust-mass-distributions.
    parameters = dict()
    for f in total_dust_to_gas_ratio:
        for dist in dust_mass_distribution.keys():
            parameters[f'Epstein-f={f}-{dist}'] = copy.copy(_parameters)
            dust_to_gas_ratio = f * dust_mass_distribution[dist]
            parameters[f'Epstein-f={f}-{dist}']['dust_to_gas_ratio'] = tuple(
                dust_to_gas_ratio
            )

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

    box_boundary = (
        -params['box_width'] / 2,
        params['box_width'] / 2,
        -params['box_width'] / 2,
        params['box_width'] / 2,
        -params['box_width'] / 2,
        params['box_width'] / 2,
    )

    setup.set_boundary(box_boundary, periodic=True)

    # Boxes
    lattice = 'cubic'
    boxes = list()

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
        number_of_particles=params['number_of_particles_gas'],
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
            number_of_particles=params['number_of_particles_dust'],
            density=density_dust[idx],
            velocity_distribution=velocity_dust,
            lattice=lattice,
        )

    # Add extra quantities
    for box in boxes:
        alpha = np.zeros(box.number_of_particles, dtype=np.single)
        box.set_array('alpha', alpha)

    # Add boxes to setup
    for box in boxes:
        setup.add_container(box)

    # Write to file
    setup.write_dump_file(directory=run_directory)
    setup.write_in_file(directory=run_directory)

    # Compile Phantom
    setup.compile_phantom(
        phantom_dir=phantom_dir, hdf5root=hdf5root, working_dir=run_directory
    )

    # Return setup
    return setup


def run_all_calculations(run_root_directory: Path) -> List[subprocess.CompletedProcess]:
    """Run dustybox calculations.

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
