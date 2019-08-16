"""
DUSTYBOX run: run Phantom simulations.

Daniel Mentiplay, 2019.
"""

import pathlib
import subprocess
import sys

import phantom_config as pc


class PatchError(Exception):
    """Error patching Phantom."""

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
RUN_DIR = pathlib.Path('~/runs/multigrain/dustybox').expanduser()
CODE_DIR = pathlib.Path('~/repos/multigrain/code/DUSTYBOX').expanduser()
PHANTOM_DIR = pathlib.Path('~/repos/phantom').expanduser()

# Phantom version
REQUIRED_PHANTOM_GIT_SHA = '6666c55feea1887b2fd8bb87fbe3c2878ba54ed7'

# ------------------------------------------------------------------------------------ #
# Check Phantom version

phantom_git_sha = (
    subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=PHANTOM_DIR)
    .strip()
    .decode()
)
if phantom_git_sha != REQUIRED_PHANTOM_GIT_SHA:
    subprocess.check_output(['git', 'checkout', '6666c55f'], cwd=PHANTOM_DIR)

# Apply patch
modified = (
    subprocess.check_output(
        ['git', 'status', '--porcelain', 'setup_dustybox.f90'], cwd=PHANTOM_DIR
    )
    .strip()
    .decode()
)
if modified != '':
    git_patch_output = subprocess.check_output(
        ['git', 'apply', CODE_DIR / 'dustybox.patch'], cwd=PHANTOM_DIR
    )
else:
    diff = (
        subprocess.check_output(
            [
                'diff',
                CODE_DIR / 'setup_dustybox.f90',
                PHANTOM_DIR / 'src/setup/setup_dustybox.f90',
            ]
        )
        .strip()
        .decode()
    )
    if diff != '':
        raise PatchError('Cannot apply patch')

# ------------------------------------------------------------------------------------ #
# Build Phantom

'_'.join([f'eps={eps:.3g}' for eps in EPS_DUST])
if IDRAG == 1:
    # TODO: implement Epstein drag in setup_dustybox.f90
    run_subdir = '????'
    raise NotImplementedError('Have not implemented Epstein/Stokes drag yet')
elif IDRAG == 2:
    run_subdir = f'K={K}_' + '_'.join([f'eps={eps:.3g}' for eps in EPS_DUST])
else:
    raise ValueError('Cannot determine drag type')

run_directory = RUN_DIR / run_subdir
run_directory.mkdir()

if sys.platform == 'darwin':
    HDF5ROOT = '/usr/local/opt/hdf5'
elif sys.platform == 'linux':
    HDF5ROOT = '/usr/lib/x86_64-linux-gnu/hdf5/serial'
if not pathlib.Path(HDF5ROOT).exists():
    raise FileNotFoundError('Cannot determine HDF5 library location')

with open(run_directory / 'Makefile', 'w') as fp:
    subprocess.run(
        [PHANTOM_DIR / 'scripts/writemake.sh', 'dustybox'], stdout=fp, stderr=fp
    )
with open(run_directory / 'build-output.log', 'w') as fp:
    subprocess.run(
        [
            'make',
            'SYSTEM=gfortran',
            'HDF5=yes',
            'HDF5ROOT=' + HDF5ROOT,
            'MAXP=10000000',
            'phantom',
            'setup',
        ],
        cwd=run_directory,
        stdout=fp,
        stderr=fp,
    )

# ------------------------------------------------------------------------------------ #
# Set up calculation

params = {
    'cs': [CS, 'sound speed (sets polyk)', 'gas properties'],
    'npartx_gas': [NPARTX_GAS, 'number of particles in x direction', 'gas properties'],
    'rhozero_gas': [RHOZERO_GAS, 'initial density', 'gas properties'],
    'ilattice_gas': [
        ILATTICE,
        'lattice type (1=cubic, 2=closepacked)',
        'gas properties',
    ],
}

rhozero_dust = [RHOZERO_GAS * eps for eps in EPS_DUST]
ndustlarge = len(rhozero_dust)

params.update({'ndustlarge': [ndustlarge, 'number of dust species', 'dust properties']})

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

subprocess.run(['mv', 'dustybox.setup', run_directory])

with open(run_directory / 'dustybox00.log', 'w') as fp:
    subprocess.run(
        [run_directory / 'phantomsetup', 'dustybox'],
        cwd=run_directory,
        stdout=fp,
        stderr=fp,
    )

# ------------------------------------------------------------------------------------ #
# Run calculation

if IDRAG == 1:
    in_file = 'dustybox-Epstein-Stokes.in'
elif IDRAG == 2:
    in_file = 'dustybox-Kdrag.in'
else:
    raise ValueError('Cannot determine drag type')

subprocess.run(['cp', CODE_DIR / in_file, run_directory / 'dustybox.in'])

with open(run_directory / 'dustybox01.log', 'w') as fp:
    subprocess.run(
        [run_directory / 'phantom', 'dustybox.in'],
        cwd=run_directory,
        stdout=fp,
        stderr=fp,
    )
