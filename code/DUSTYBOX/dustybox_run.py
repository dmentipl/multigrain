"""
DUSTYBOX run: run Phantom simulations.

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
# DUSTYBOX parameters

# Drag type: options (1) Epstein/Stokes (not implemented yet), and (2) K drag
IDRAG = 2

# Gas properties
CS = 1.0
NPARTX_GAS = 32
RHOZERO_GAS = 1.0
ILATTICE = 2

# Dust properties
K = 1.0
NPARTX_DUST = 32
EPS_DUST = [0.01, 0.02, 0.03, 0.04, 0.05]

# Directories
RUN_ROOT_DIR = pathlib.Path('~/runs/multigrain/dustybox').expanduser()
CODE_DIR = pathlib.Path('~/repos/multigrain/code/DUSTYBOX').expanduser()
PHANTOM_DIR = pathlib.Path('~/repos/phantom').expanduser()

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

    with open(RUN_DIRECTORY / 'build-output.log', 'w') as fp:
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

    subprocess.run(['cp', PHANTOM_DIR / 'bin/phantom', RUN_DIRECTORY])
    subprocess.run(['cp', PHANTOM_DIR / 'bin/phantomsetup', RUN_DIRECTORY])
    subprocess.run(['cp', PHANTOM_DIR / 'bin/phantom_version', RUN_DIRECTORY])


# ------------------------------------------------------------------------------------ #
# Set up calculation


def setup_calculation():

    print('>>> Setting up calculation <<<')

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

    rhozero_dust = [RHOZERO_GAS * eps for eps in EPS_DUST]
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
    cf.write_phantom('dustybox.setup')

    subprocess.run(['mv', 'dustybox.setup', RUN_DIRECTORY])

    with open(RUN_DIRECTORY / 'dustybox00.log', 'w') as fp:
        subprocess.run(
            [RUN_DIRECTORY / 'phantomsetup', 'dustybox'],
            cwd=RUN_DIRECTORY,
            stdout=fp,
            stderr=fp,
        )

    if IDRAG == 1:
        in_file = 'dustybox-Epstein-Stokes.in'
    elif IDRAG == 2:
        in_file = 'dustybox-Kdrag.in'
    else:
        raise ValueError('Cannot determine drag type')

    config_dictionary = pc.read_config(in_file).to_ordered_dict()
    config_dictionary['K_code'][0] = K
    pc.read_dict(config_dictionary).write_phantom(RUN_DIRECTORY / 'dustybox.in')


# ------------------------------------------------------------------------------------ #
# Run calculation


def run_calculation():

    print('>>> Running calculation <<<')

    with open(RUN_DIRECTORY / 'dustybox01.log', 'w') as fp:
        subprocess.run(
            [RUN_DIRECTORY / 'phantom', 'dustybox.in'],
            cwd=RUN_DIRECTORY,
            stdout=fp,
            stderr=fp,
        )


# ------------------------------------------------------------------------------------ #
# Main program


if __name__ == '__main__':

    if IDRAG == 1:
        # TODO: implement Epstein drag in setup_dustybox.f90
        run_subdir = '????'
        raise NotImplementedError('Have not implemented Epstein/Stokes drag yet')
    elif IDRAG == 2:
        run_subdir = f'K={K}_' + '_'.join([f'eps={eps:.3g}' for eps in EPS_DUST])
    else:
        raise ValueError('Cannot determine drag type')

    RUN_DIRECTORY = RUN_ROOT_DIR / run_subdir
    RUN_DIRECTORY.mkdir()

    check_phantom_version()
    build_phantom()
    setup_calculation()
    run_calculation()
