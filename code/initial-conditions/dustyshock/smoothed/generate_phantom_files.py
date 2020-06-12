"""Generate Phantom .in and .setup files for dustyshock."""

from pathlib import Path

from phantomconfig import parameter_sweep, read_config

FILE_DIR = Path(__file__).parent
OUTPUT_DIR = Path('~/runs/multigrain/dustyshock/_initial_conditions').expanduser()

PARAMETERS = {'nx': [32, 128], 'smooth_fac': [2.0, 5.0], 'hfact': [1.0, 1.2, 1.5]}


def _main():

    for N in [1, 3]:
        parameter_sweep(
            filename='dustyshock.setup',
            template=read_config(FILE_DIR / f'dustyshock-N_{N}.setup'),
            parameters=PARAMETERS,
            dummy_parameters=['hfact'],
            prefix=f'N-{N}_',
            output_dir=OUTPUT_DIR,
        )
        parameter_sweep(
            filename='dustyshock.in',
            template=read_config(FILE_DIR / f'dustyshock-N_{N}.in'),
            parameters=PARAMETERS,
            dummy_parameters=['nx', 'smooth_fac'],
            prefix=f'N-{N}_',
            output_dir=OUTPUT_DIR,
        )


if __name__ == '__main__':
    _main()
