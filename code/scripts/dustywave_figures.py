"""Dusty wave figures.

Make figures showing time evolution of the velocity and density in the
dusty wave test:

- for 1 and 4 dust species.
"""

import sys
from pathlib import Path

import plonk

sys.path.insert(0, str(Path(__file__).resolve().parent / '../modules'))
from multigrain import dustywave


PATH = '~/runs/multigrain/dustywave'

SOUND_SPEED = 1.0
AMPLITUDE = 1e-4
NUM_PARTICLES_X = 128


def _get_paths():
    # Path to data
    print('Get path to data...')
    root_directory = Path(PATH).expanduser()
    _paths = sorted(list(root_directory.glob('*')))
    paths = {p.name: p for p in _paths}

    return paths


def time_evolution():
    print('')
    print('Time evolution of velocity and density')
    print('--------------------------------------')
    print('')

    paths = _get_paths()

    # Calculate velocity and density time evolution
    print('Calculate velocity and density time evolution...')
    dataframes = dict()
    for name, path in paths.items():
        print(f'Running analysis for {name}...')
        sim = plonk.load_sim(prefix='dustywave', directory=path)
        dataframes[name] = dustywave.calculate_velocity_density(
            sim, AMPLITUDE, SOUND_SPEED, NUM_PARTICLES_X
        )

    # Plot results
    print('Plotting figure...')
    fig = dustywave.plot_velocity_density(dataframes, figsize=(10, 8))
    axs = fig.axes
    for ax in axs:
        ax.grid()
    axs[0].set_title('1 dust species')
    axs[1].set_title('4 dust species')
    axs[0].legend(loc='lower right')
    axs[1].legend(loc='lower right')
    name = 'dustywave_velocity_density.pdf'
    print(f'Saving figure to {name}')
    fig.savefig(name, bbox_inches='tight', pad_inches=0.05)


if __name__ == "__main__":
    time_evolution()
