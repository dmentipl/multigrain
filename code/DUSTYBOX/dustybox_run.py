"""
Write Phantom setup and in files for DUSTYBOX.

Daniel Mentiplay, 2019.
"""

import pathlib
import shutil
import subprocess
import sys

import phantombuild as pb
import phantomconfig as pc


class SetupError(Exception):
    pass


class RunError(Exception):
    pass


# ------------------------------------------------------------------------------------ #

# Runs: there is one Phantom calculation per item in K_DRAG_LIST
K_DRAG_LIST = [0.1, 1.0, 10.0]

# Number of grains: there is one per item in EPS_FOR_K_IS_UNITY
# This variable specifies the dust-to-gas ratio for K=1.0.
EPS_FOR_K_IS_UNITY = [0.01, 0.02, 0.03, 0.04, 0.05]

# ------------------------------------------------------------------------------------ #

# Gas properties
CS = 1.0
NPARTX_GAS = 32
RHOZERO_GAS = 1.0
ILATTICE = 2

# Dust properties
NPARTX_DUST = 32

# Directories
RUN_ROOT_DIR = pathlib.Path('~/runs/multigrain/dustybox').expanduser()
PHANTOM_DIR = pathlib.Path('~/repos/phantom').expanduser()
CODE_DIR = pathlib.Path('~/repos/multigrain/code/DUSTYBOX').expanduser()

# Phantom version
REQUIRED_PHANTOM_GIT_COMMIT_HASH = '6666c55feea1887b2fd8bb87fbe3c2878ba54ed7'

# Phantom patch
PHANTOM_PATCH = CODE_DIR / 'dustybox.patch'

# ------------------------------------------------------------------------------------ #
# Set up calculations


def setup_calculations(run_root_directory: pathlib.Path, phantom_dir: pathlib.Path):

    print('\n' + 72*'-')
    print('>>> Setting up calculations <<<')
    print(72*'-' + '\n')

    for K_drag in K_DRAG_LIST:

        run_label = f'K={K_drag}'

        print(f'Setting up {run_label}...')

        run_directory = run_root_directory / run_label
        try:
            run_directory.mkdir()
        except FileExistsError:
            raise SetupError('Run directory already exists.')

        shutil.copy(phantom_dir / 'bin/phantom', run_directory)
        shutil.copy(phantom_dir / 'bin/phantomsetup', run_directory)
        shutil.copy(phantom_dir / 'bin/phantom_version', run_directory)

        setup_file = f'dustybox-{run_label}.setup'
        dust_to_gas_ratio = [eps * K_drag for eps in EPS_FOR_K_IS_UNITY]
        write_setup_file(K_drag, dust_to_gas_ratio, setup_file, run_directory)

        with open(run_directory / 'dustybox00.log', 'w') as fp:
            subprocess.run(
                [run_directory / 'phantomsetup', setup_file],
                cwd=run_directory,
                stdout=fp,
                stderr=fp,
            )

        in_file = f'dustybox-{run_label}.in'
        write_in_file(K_drag, in_file, run_directory)


def write_setup_file(K_drag: float, eps: list, filename: str, output_dir: pathlib.Path):

    params = {
        'cs': [CS, 'sound speed (sets polyk)', 'gas properties'],
        'npartx_gas': [
            NPARTX_GAS,
            'number of particles in x direction',
            'gas properties',
        ],
        'rhozero_gas': [RHOZERO_GAS, 'initial density', 'gas properties'],
        'ilattice_gas': [
            ILATTICE,
            'lattice type (1=cubic, 2=closepacked)',
            'gas properties',
        ],
    }

    rhozero_dust = [RHOZERO_GAS * _ for _ in eps]
    ndustlarge = len(rhozero_dust)

    params.update(
        {'ndustlarge': [ndustlarge, 'number of dust species', 'dust properties']}
    )

    for idust in range(ndustlarge):
        block = f'dust: {idust+1:2}'
        dust_dict = {
            f'npartx_dust{idust+1}': [
                NPARTX_DUST,
                'number of particles in x direction',
                block,
            ],
            f'rhozero_dust{idust+1}': [rhozero_dust[idust], 'initial density', block],
            f'ilattice_dust{idust+1}': [
                ILATTICE,
                'lattice type (1=cubic, 2=closepacked)',
                block,
            ],
        }
        params.update(dust_dict)

    cf = pc.read_dict(params)
    cf.header = ['input file for dustybox setup routine']

    if not output_dir.exists():
        raise FileExistsError(f'{output_dir} does not exist')

    cf.write_phantom(output_dir / filename)


def write_in_file(K_drag: float, filename: str, output_dir: pathlib.Path):

    if not output_dir.exists():
        raise FileExistsError(f'{output_dir} does not exist')

    run_label = f'K={K_drag}'

    template_dir = pathlib.Path(__file__).resolve().parent / 'templates'

    in_file = pc.read_config(template_dir / 'dustybox-Kdrag.in-template')

    in_file.change_value('K_code', K_drag)
    in_file.change_value('dumpfile', f'dustybox-{run_label}_00000.tmp')

    in_file.write_phantom(output_dir / filename)


# ------------------------------------------------------------------------------------ #
# Run calculations


def run_calculations(run_root_directory: pathlib.Path):

    print('\n' + 72*'-')
    print('>>> Running calculations <<<')
    print(72*'-' + '\n')

    for K_drag in K_DRAG_LIST:

        run_label = f'K={K_drag}'

        print(f'Running {run_label}...')

        run_directory = run_root_directory / run_label
        in_file = f'dustybox-{run_label}.in'

        with open(run_directory / 'dustybox01.log', 'w') as fp:
            subprocess.run(
                [run_directory / 'phantom', in_file],
                cwd=run_directory,
                stdout=fp,
                stderr=fp,
            )


# ------------------------------------------------------------------------------------ #

if __name__ == '__main__':

    pb.get_phantom(PHANTOM_DIR)

    pb.checkout_phantom_version(PHANTOM_DIR, REQUIRED_PHANTOM_GIT_COMMIT_HASH)

    pb.patch_phantom(PHANTOM_DIR, PHANTOM_PATCH)

    if sys.platform == 'darwin':
        hdf5_location = pathlib.Path('/usr/local/opt/hdf5')
    elif sys.platform == 'linux':
        hdf5_location = pathlib.Path('/usr/lib/x86_64-linux-gnu/hdf5/serial')

    extra_makefile_options = {'MAXP': '10000000'}

    pb.build_phantom(
        PHANTOM_DIR, 'dustybox', 'gfortran', hdf5_location, extra_makefile_options
    )

    setup_calculations(RUN_ROOT_DIR, PHANTOM_DIR)

    run_calculations(RUN_ROOT_DIR)
