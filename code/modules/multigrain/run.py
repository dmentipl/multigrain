"""Setup and run calculations."""

import subprocess
from pathlib import Path
from typing import Dict, List

import phantomsetup

from . import dustybox, dustywave, dustyshock


def setup_multiple_calculations(
    simulation_to_setup: str,
    run_root_directory: Path,
    parameters_dict: Dict[str, dict],
    phantom_dir: Path,
    hdf5root: Path,
) -> List[phantomsetup.Setup]:
    """Set up multiple calculations.

    Parameters
    ----------
    simulation_to_setup
        The simulation to setup: 'dustybox', 'dustywave', 'dustyshock'.
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

        if simulation_to_setup == 'dustybox':
            setup_calculation = dustybox.setup_calculation
        elif simulation_to_setup == 'dustywave':
            setup_calculation = dustywave.setup_calculation
        elif simulation_to_setup == 'dustyshock':
            setup_calculation = dustyshock.setup_calculation
        else:
            raise ValueError(
                f'Simulation_to_setup: {simulation_to_setup} not available'
            )

        setups.append(
            setup_calculation(
                params=params,
                run_directory=run_directory,
                phantom_dir=phantom_dir,
                hdf5root=hdf5root,
            )
        )

    return setups


def run_multiple_calculations(
    run_root_directory: Path,
) -> List[subprocess.CompletedProcess]:
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
