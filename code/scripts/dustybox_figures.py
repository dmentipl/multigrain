"""Dusty box figures.

Make figures showing time evolution of differential velocity in the
dusty box test:

- for 1, 2, and 5 dust species (time_evolution),
- for dust-to-gas ratio of 0.01 and 0.5 (time_evolution_error).

Also, make figures showing the time evolution of the error of the above
tests.
"""

import sys
from pathlib import Path

import plonk

sys.path.insert(0, str(Path(__file__).resolve().parent / '../modules'))
from multigrain import dustybox

PATH = '~/runs/multigrain/dustybox/time_evolution'


def get_paths():
    print('Get path to data...')
    root_directory = Path(PATH).expanduser()
    _paths = sorted(list(root_directory.glob('f_*-N_*')))
    paths = {p.name: p for p in _paths}

    return paths


def calculate_velocity_differential(paths):
    print('Calculate velocity differential time evolution...')
    velocity_differential = dict()
    for name, path in paths.items():
        print(f'Running analysis for {name}...')
        sim = plonk.load_sim(prefix='dustybox', directory=path)
        velocity_differential[name] = dustybox.calculate_differential_velocity(sim)

    return velocity_differential


def time_evolution(velocity_differential):
    print('Plotting figure...')
    fig = dustybox.plot_differential_velocity_all(
        velocity_differential, figsize=(15, 8)
    )
    for ax in fig.axes:
        ax.grid()
    name = 'dustybox_differential_velocity_comparison.pdf'
    print(f'Saving figure to {name}')
    fig.savefig(name, bbox_inches='tight', pad_inches=0)


def time_evolution_error(velocity_differential):
    error = dict()
    for name, vd in velocity_differential.items():
        print(f'Calculating error for {name}...')
        error[name] = dustybox.calculate_error(vd)

    print('Plotting figure...')
    fig = dustybox.plot_error_all(error, figsize=(15, 8))
    for ax in fig.axes:
        ax.grid()
    name = 'dustybox_differential_velocity_error.pdf'
    print(f'Saving figure to {name}')
    fig.savefig(name, bbox_inches='tight', pad_inches=0)


if __name__ == "__main__":
    paths = get_paths()
    velocity_differential = calculate_velocity_differential(paths)
    time_evolution(velocity_differential)
    time_evolution_error(velocity_differential)
