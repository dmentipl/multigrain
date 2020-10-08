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

import matplotlib.pyplot as plt
import numpy as np
import plonk

sys.path.insert(0, str(Path(__file__).resolve().parent / '../modules'))
from multigrain import dustybox

PATH = '~/runs/multigrain/dustybox'


def _get_paths(name, glob):
    print('Get path to data...')
    root_directory = (Path(PATH) / name).expanduser()
    _paths = sorted(list(root_directory.glob(glob)))
    paths = {p.name: p for p in _paths}

    return paths


def _calculate_velocity_differential_all(same_times=False):
    paths = _get_paths(name='time_evolution', glob='f_*-N_*')

    data, exact1, exact2 = dict(), dict(), dict()

    print('Calculate velocity differential time evolution...')
    for name, path in paths.items():
        print(f'Running analysis for {name}...')
        data[name], exact1[name], exact2[name] = _calculate_velocity_differential(
            path, same_times=False
        )

    return data, exact1, exact2


def _calculate_velocity_differential(path, same_times=False):
    sim = plonk.load_simulation(prefix='dustybox', directory=path)
    data = dustybox.calculate_differential_velocity(sim)
    times = None
    if same_times:
        times = data['time'].to_numpy()
    exact1 = dustybox.calculate_differential_velocity_exact(
        sim, times=times, backreaction=True
    )
    exact2 = dustybox.calculate_differential_velocity_exact(
        sim, times=times, backreaction=False
    )
    # Scale time by the shortest stopping time
    _, stopping_time = dustybox.get_dust_properties(sim.snaps[0])
    data['time'] = data['time'] / stopping_time[0]
    exact1['time'] = exact1['time'] / stopping_time[0]
    exact2['time'] = exact2['time'] / stopping_time[0]

    return data, exact1, exact2


def time_evolution():
    print('')
    print('Time evolution of velocity differential')
    print('---------------------------------------')
    print('')

    data, exact1, exact2 = _calculate_velocity_differential_all()

    print('Plotting figure...')
    fig = dustybox.plot_differential_velocity_all(data, exact1, exact2, figsize=(15, 8))
    axs = fig.axes
    axs[0].set_title('1 dust species')
    axs[1].set_title('2 dust species')
    axs[2].set_title('5 dust species')
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    for ax in axs[:3]:
        ax.text(
            0.9,
            0.9,
            r'$\varepsilon = 0.01$',
            ha='right',
            transform=ax.transAxes,
            bbox=props,
        )
    for ax in axs[3:]:
        ax.text(
            0.9,
            0.9,
            r'$\varepsilon = 0.5$',
            ha='right',
            transform=ax.transAxes,
            bbox=props,
        )
    name = 'dustybox_differential_velocity_comparison.pdf'
    print(f'Saving figure to {name}')
    fig.savefig(name, bbox_inches='tight', pad_inches=0.05)


def time_evolution_zoomed():
    print('')
    print('Time evolution of velocity differential zoomed')
    print('----------------------------------------------')
    print('')

    paths = _get_paths(name='time_evolution', glob='f_*-N_*')
    data, exact1, exact2 = _calculate_velocity_differential(paths['f_0.50-N_5'])

    print('Plotting figure...')
    fig, ax = plt.subplots()
    dustybox.plot_differential_velocity(data, exact1, exact2, ax)
    ax.set_xlabel('Time')
    ax.set_ylabel('Differential velocity')
    ax.set_ylim([-0.05, 0.1])
    name = 'dustybox_differential_velocity_comparison_zoomed.pdf'
    print(f'Saving figure to {name}')
    fig.savefig(name, bbox_inches='tight', pad_inches=0.05)


def time_evolution_error():
    print('')
    print('Time evolution of velocity differential error')
    print('---------------------------------------------')
    print('')

    paths = _get_paths(name='time_evolution', glob='f_*-N_*')
    error = dict()

    for name, path in paths.items():
        print(f'Calculating error for {name}...')
        sim = plonk.load_simulation(prefix='dustybox', directory=path)
        error[name] = dustybox.calculate_error(sim)

    print('Plotting figure...')
    fig = dustybox.plot_error_all(error, figsize=(15, 8))
    name = 'dustybox_differential_velocity_error.pdf'
    print(f'Saving figure to {name}')
    fig.savefig(name, bbox_inches='tight', pad_inches=0.05)


def accuracy():
    print('')
    print('Accuracy of method')
    print('------------------')
    print('')

    paths = _get_paths(name='accuracy', glob='hfact_*-eps_*-C_force_*')
    error = dict()

    for name, path in paths.items():
        print(f'Calculating error for {name}...')
        sim = plonk.load_simulation(prefix='dustybox', directory=path)
        error[name] = dustybox.calculate_error(sim)

    print('Plotting figure...')

    C_drag = {}
    error_norm = {}
    for key, val in error.items():
        name = key[:18]
        if C_drag.get(name) is None:
            C_drag[name] = list()
        C_drag[name].append(_get_val(key, 'C_force'))
        if error_norm.get(name) is None:
            error_norm[name] = list()
        error_norm[name].append(
            _error_norm_fn([val['error.1'], val['error.2']], method=2)
        )

    def line(x, m=2, c=0):
        return [m * _x + c for _x in x]

    m, c = 2, -0.1

    fig, ax = plt.subplots()

    def plot_error_norm(eta, marker, label, ax):
        _C_drag = {
            key: val for key, val in C_drag.items() if _get_val(key, 'hfact') == eta
        }
        _error_norm = {
            key: val for key, val in error_norm.items() if _get_val(key, 'hfact') == eta
        }

        for (eps, dt), err in zip(_C_drag.items(), _error_norm.values()):
            _eps = _get_val(eps, 'dust_to_gas')
            if label:
                _label = rf'$\eta = {eta}, \epsilon = {_eps}$'
            else:
                _label = None
            ax.plot(np.log10(dt), np.log10(err), marker, label=_label)

        return [np.log10(dt[0]), np.log10(dt[-1])]

    plot_error_norm(eta=1.0, marker='d', label=True, ax=ax)
    x = plot_error_norm(eta=2.5, marker='s', label=True, ax=ax)
    ax.plot(x, line(x, m=m, c=c), '--', color='gray')
    ax.set(xlabel=r'$\log_{10}$(dt)', ylabel=r'$\log_{10}$(error)')
    ax.legend()

    name = 'dustybox_accuracy.pdf'
    print(f'Saving figure to {name}')
    fig.savefig(name, bbox_inches='tight', pad_inches=0.05)


def _get_val(path, q):
    nums = [float(s.split('_')[-1]) for s in path.split('-')]
    if q == 'hfact':
        return nums[0]
    if q == 'dust_to_gas':
        return nums[1]
    if q == 'C_force':
        return nums[2]
    raise ValueError


def _error_norm_fn(errors, method=2):
    if method == 1:
        return np.sqrt(np.sum([np.mean(err) ** 2 for err in errors]))
    elif method == 2:
        return np.sqrt(np.sum([err ** 2 for err in errors]))
    elif method == 3:
        return np.sum([err for err in errors])
    elif method == 4:
        return np.sum([err.iloc[1] for err in errors])
    raise ValueError


if __name__ == "__main__":
    time_evolution()
    time_evolution_zoomed()
    # time_evolution_error()
    accuracy()
