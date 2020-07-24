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


def _get_paths():
    print('Get path to data...')
    root_directory = Path(PATH).expanduser()
    _paths = sorted(list(root_directory.glob('f_*-N_*')))
    paths = {p.name: p for p in _paths}

    return paths


def _calculate_velocity_differential(same_times=False):
    paths = _get_paths()

    data, exact1, exact2 = dict(), dict(), dict()

    print('Calculate velocity differential time evolution...')
    for name, path in paths.items():
        print(f'Running analysis for {name}...')
        sim = plonk.load_sim(prefix='dustybox', directory=path)
        data[name] = dustybox.calculate_differential_velocity(sim)
        times = None
        if same_times:
            times = data['time'].to_numpy()
        exact1[name] = dustybox.calculate_differential_velocity_exact(
            sim, times=times, backreaction=True
        )
        exact2[name] = dustybox.calculate_differential_velocity_exact(
            sim, times=times, backreaction=False
        )

    return data, exact1, exact2


def time_evolution():
    print('')
    print('Time evolution of velocity differential')
    print('---------------------------------------')
    print('')

    data, exact1, exact2 = _calculate_velocity_differential()

    print('Plotting figure...')
    fig = dustybox.plot_differential_velocity_all(data, exact1, exact2, figsize=(15, 8))
    axs = fig.axes
    for ax in axs:
        ax.grid()
    axs[0].set_title('1 dust species')
    axs[1].set_title('2 dust species')
    axs[2].set_title('5 dust species')
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    for ax in axs[:3]:
        ax.text(0.10, 0.9, r'$\varepsilon = 0.01$', ha='right', bbox=props)
    for ax in axs[3:]:
        ax.text(0.10, 0.9, r'$\varepsilon = 0.5$', ha='right', bbox=props)
    name = 'dustybox_differential_velocity_comparison.pdf'
    print(f'Saving figure to {name}')
    fig.savefig(name, bbox_inches='tight', pad_inches=0.05)


def time_evolution_error():
    print('')
    print('Time evolution of velocity differential error')
    print('---------------------------------------------')
    print('')

    paths = _get_paths()
    error = dict()

    for name, path in paths.items():
        print(f'Calculating error for {name}...')
        sim = plonk.load_sim(prefix='dustybox', directory=path)
        error[name] = dustybox.calculate_error(sim)

    print('Plotting figure...')
    fig = dustybox.plot_error_all(error, figsize=(15, 8))
    for ax in fig.axes:
        ax.grid()
    name = 'dustybox_differential_velocity_error.pdf'
    print(f'Saving figure to {name}')
    fig.savefig(name, bbox_inches='tight', pad_inches=0.05)


if __name__ == "__main__":
    time_evolution()
    # time_evolution_error()
