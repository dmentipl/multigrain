"""
Write Phantom setup and in files for DUSTYBOX.

Daniel Mentiplay, 2019.
"""

import pathlib
import subprocess
import sys

import phantom_config as pc


class PatchError(Exception):
    pass


class CompileError(Exception):
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
REQUIRED_PHANTOM_GIT_SHA = '6666c55feea1887b2fd8bb87fbe3c2878ba54ed7'

# ------------------------------------------------------------------------------------ #
# Check Phantom version


def check_phantom_version():

    print('>>> Checking Phantom version <<<')

    phantom_git_sha = (
        subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=PHANTOM_DIR)
        .strip()
        .decode()
    )
    if phantom_git_sha != REQUIRED_PHANTOM_GIT_SHA:
        print(f'Checking out Phantom version: {REQUIRED_PHANTOM_GIT_SHA}')
        subprocess.check_output(['git', 'checkout', '6666c55f'], cwd=PHANTOM_DIR)
    else:
        print('Required version of Phantom already checked out')

    # Apply patch
    git_status = (
        subprocess.check_output(
            ['git', 'status', '--porcelain', '--', 'setup_dustybox.f90'],
            cwd=PHANTOM_DIR / 'src/setup',
        )
        .strip()
        .decode()
    )
    if git_status == '':
        print('Applying patch to Phantom')
        subprocess.check_output(
            ['git', 'apply', CODE_DIR / 'dustybox.patch'], cwd=PHANTOM_DIR
        )
    else:
        try:
            subprocess.check_output(
                [
                    'diff',
                    '-I',
                    '!  $Id: ',
                    CODE_DIR / 'setup_dustybox.f90',
                    PHANTOM_DIR / 'src/setup/setup_dustybox.f90',
                ]
            )
        except subprocess.CalledProcessError:
            raise PatchError('Cannot apply patch to Phantom')
        print('Already patched')


# ------------------------------------------------------------------------------------ #
# Build Phantom


def build_phantom():

    print('>>> Building Phantom <<<')

    if sys.platform == 'darwin':
        HDF5ROOT = '/usr/local/opt/hdf5'
    elif sys.platform == 'linux':
        HDF5ROOT = '/usr/lib/x86_64-linux-gnu/hdf5/serial'
    if not pathlib.Path(HDF5ROOT).exists():
        raise FileNotFoundError('Cannot determine HDF5 library location')

    with open(PHANTOM_DIR / 'build' / 'build-output.log', 'w') as fp:
        result = subprocess.run(
            [
                'make',
                'SETUP=dustybox',
                'SYSTEM=gfortran',
                'HDF5=yes',
                'HDF5ROOT=' + HDF5ROOT,
                'MAXP=10000000',
                'phantom',
                'setup',
            ],
            cwd=PHANTOM_DIR,
            stdout=fp,
            stderr=fp,
        )
    if result.returncode != 0:
        raise CompileError('Phantom failed compiling')
    print(f'Successfully built; see {PHANTOM_DIR / "build"} for make output')


# ------------------------------------------------------------------------------------ #
# Set up calculations


def setup_calculations(run_root_directory: pathlib.Path):

    print('>>> Setting up calculations <<<')

    for K_drag in K_DRAG_LIST:

        run_label = f'K={K_drag}'

        print(f'Setting up {run_label}...')

        run_directory = run_root_directory / run_label
        run_directory.mkdir()

        subprocess.run(['cp', PHANTOM_DIR / 'bin/phantom', run_directory])
        subprocess.run(['cp', PHANTOM_DIR / 'bin/phantomsetup', run_directory])
        subprocess.run(['cp', PHANTOM_DIR / 'bin/phantom_version', run_directory])

        setup_file = f'dustybox-{run_label}.setup'
        dust_to_gas_ratio = [eps / K_drag for eps in EPS_FOR_K_IS_UNITY]
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

    run_label = f'K={K_drag}'

    template_dir = pathlib.Path(__file__).resolve().parent / 'templates'

    config_dictionary = pc.read_config(
        template_dir / 'dustybox-Kdrag.in-template'
    ).to_ordered_dict()

    config_dictionary['K_code'][0] = K_drag
    config_dictionary['dumpfile'][0] = f'dustybox-{run_label}_00000.tmp'

    if not output_dir.exists():
        raise FileExistsError(f'{output_dir} does not exist')

    pc.read_dict(config_dictionary).write_phantom(output_dir / filename)


# ------------------------------------------------------------------------------------ #
# Run calculations


def run_calculations(run_root_directory: pathlib.Path):

    print('>>> Running calculations <<<')

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

    check_phantom_version()
    build_phantom()
    setup_calculations(RUN_ROOT_DIR)
    run_calculations(RUN_ROOT_DIR)
