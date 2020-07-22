"""Setup and schedule dustyshock runs.

1. Generate Phantom .in and .setup files for dustyshock with
   phantomconfig and template files.
2. Setup and schedule runs with phantombuild from TOML template file.
"""

from pathlib import Path

import phantombuild as pb
import phantomconfig as pc

FILE_DIR = Path(__file__).resolve().parent.parent / 'initial-conditions' / 'dustyshock'
OUTPUT_DIR = Path('~/runs/multigrain/dustyshock/_initial_conditions').expanduser()
CONFIG_FILE = Path(__file__).resolve().parent / 'dustyshock.toml.j2'

PARAMETERS = {
    'nx': [32, 64, 128, 256],
    'smooth_fac': [2.0, 5.0],
    'hfact': [1.0, 1.2, 1.5, 1.8],
}


def generate_files():
    for N in [1, 3]:
        pc.parameter_sweep(
            filename='dustyshock.setup',
            template=pc.read_config(FILE_DIR / f'dustyshock-N_{N}.setup'),
            parameters=PARAMETERS,
            dummy_parameters=['hfact'],
            prefix=f'N_{N}-',
            output_dir=OUTPUT_DIR,
        )
        pc.parameter_sweep(
            filename='dustyshock.in',
            template=pc.read_config(FILE_DIR / f'dustyshock-N_{N}.in'),
            parameters=PARAMETERS,
            dummy_parameters=['nx', 'smooth_fac'],
            prefix=f'N_{N}-',
            output_dir=OUTPUT_DIR,
        )


def setup_and_schedule_runs():
    conf = pb.read_config(CONFIG_FILE)
    pb.build_phantom(**conf['phantom'])
    phantom_path = conf['phantom']['path']
    for run in conf.get('runs', []):
        run_path = run.pop('path')
        pb.setup_calculation(run_path=run_path, phantom_path=phantom_path, **run)


if __name__ == '__main__':
    generate_files()
    setup_and_schedule_runs()
