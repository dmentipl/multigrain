"""Dusty shock figures.

Make figures showing the final time velocity and density for the dusty
shock test:

- for 1 and 3 dust species (final_velocity_density).

Also, make figures showing

- the smoothed initial conditions (initial_conditions),
- the variation with hfact (variation_hfact).
"""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import plonk

sys.path.insert(0, str(Path(__file__).resolve().parent / '../modules'))
from multigrain import dustyshock


def final_velocity_density():
    print('')
    print('Final time velocity and density')
    print('-------------------------------')
    print('')

    # Get data
    paths = _get_paths()

    # Set parameters for "final time" plot
    Ns = [1, 3]
    hfact = 1.8
    nx = 128
    smooth_fac = 2.0

    drag_coefficients = [[1.0], [1.0, 3.0, 5.0]]
    xrange = (-5, 15)

    # Get last snaps for each simulation
    print('Load data...')
    sim_names = [f'N_{N}-nx_{nx}-smooth_fac_{smooth_fac}-hfact_{hfact}' for N in Ns]
    snaps = [dustyshock.last_snap(paths[name]) for name in sim_names]

    # Find shock x-positions.
    print('Calculation x-position of shock...')
    x_shock = [
        dustyshock.find_x_shock(snap=snap, drag_coefficients=K, xrange=xrange)
        for snap, K in zip(snaps, drag_coefficients)
    ]

    # Plot the x-velocity and density for the gas and each dust species
    # at the final time.
    print('Plotting figure...')
    fig = dustyshock.plot_numerical_vs_exact(
        snaps=snaps,
        xrange=xrange,
        drag_coefficients=drag_coefficients,
        x_shock=x_shock,
        plot_type='profile',
        n_bins=40,
        fig_kwargs={'width': 10, 'height': 4},
    )
    for ax in fig.axes:
        ax.grid()
    name = 'dustyshock_velocity_density.pdf'
    print(f'Saving figure to {name}')
    fig.savefig(name, bbox_inches='tight', pad_inches=0)


def initial_conditions():
    print('')
    print('Initial conditions')
    print('------------------')
    print('')

    # Get data
    paths = _get_paths()

    # Set parameters for "final time" plot
    N = 1
    hfact = 1.8
    nx = 128
    smooth_fac = 2.0
    xrange = (-5, 15)

    # Get snap
    print('Load data...')
    sim_name = f'N_{N}-nx_{nx}-smooth_fac_{smooth_fac}-hfact_{hfact}'
    snap = dustyshock.first_snap(paths[sim_name])

    # Make plot
    print('Plotting figure...')
    fig, axs = plt.subplots(nrows=2, sharex=True, squeeze=False)
    axs[0, 0].set_ylabel('Velocity')
    axs[1, 0].set_ylabel('Density')
    axs[1, 0].set_xlabel('x')
    dustyshock.plot_velocity_density_as_profile(snaps=[snap], xrange=xrange, axs=axs)
    name = 'dustyshock_initial.pdf'
    print(f'Saving figure to {name}')
    fig.savefig(name, bbox_inches='tight', pad_inches=0)


def initial_conditions_particles():
    print('')
    print('Initial conditions particles')
    print('----------------------------')
    print('')

    DIR = Path('~/runs/multigrain/dustyshock').expanduser()
    RUN = 'N_1-nx_32-smooth_fac_2.0-hfact_1.0'
    FILE = 'dustyshock_00000.h5'
    PATH = DIR / RUN / FILE

    snap = plonk.load_snap(PATH)
    subsnap = snap[np.abs(snap['z']) < 2]['gas']

    ax = plonk.visualize.particle_plot(snap=subsnap, color='k', ms=2.0)
    ax.set_aspect('equal')
    ax.set_xlim(-20, 20)
    ax.set(xlabel='x', ylabel='y')

    fig = ax.figure
    fig.savefig('dustyshock_initial_particles.pdf', bbox_inches='tight', pad_inches=0)


def variation_hfact():
    print('')
    print('Variation with hfact')
    print('--------------------')
    print('')

    # Get data
    paths = _get_paths()

    # Set parameters
    Ns = [1, 3]
    hfacts = [1.2, 1.5, 1.8]
    nx = 128
    smooth_fac = 2.0
    drag_coefficients = [
        [[1.0], [1.0], [1.0]],
        [[1.0, 3.0, 5.0], [1.0, 3.0, 5.0], [1.0, 3.0, 5.0]],
    ]
    xrange = (-5, 15)

    # Figure
    fig, axs = plt.subplots(
        ncols=len(hfacts),
        nrows=len(Ns),
        sharex=True,
        sharey='row',
        squeeze=False,
        figsize=(15, 8),
    )
    fig.subplots_adjust(hspace=0.1, wspace=0.1)

    for idx, ax in enumerate(axs[1]):
        ax.set_xlabel('x')
    axs.T[0][0].set_ylabel('Density')
    axs.T[0][1].set_ylabel('Density')

    for idx, N in enumerate(Ns):
        _variation_hfact(
            N, axs[idx], paths, hfacts, nx, smooth_fac, drag_coefficients[idx], xrange
        )

    name = 'dustyshock_hfact.pdf'
    print(f'Saving figure to {name}')
    fig.savefig(name, bbox_inches='tight', pad_inches=0)


def _variation_hfact(N, axs, paths, hfacts, nx, smooth_fac, drag_coefficients, xrange):

    # Get last snaps for each simulation
    print('Load data...')
    sim_names = [
        f'N_{N}-nx_{nx}-smooth_fac_{smooth_fac}-hfact_{hfact}' for hfact in hfacts
    ]
    snaps = [dustyshock.last_snap(paths[name]) for name in sim_names]

    # Find shock x-positions.
    print('Calculation x-position of shock...')
    x_shock = [
        dustyshock.find_x_shock(snap=snap, drag_coefficients=K, xrange=xrange)
        for snap, K in zip(snaps, drag_coefficients)
    ]

    # Plot the x-velocity and density for the gas and each dust species
    # at the final time.
    print('Plotting figure...')
    dustyshock.plot_numerical_vs_exact_density(
        snaps=snaps,
        xrange=xrange,
        drag_coefficients=drag_coefficients,
        x_shock=x_shock,
        axs=axs,
        plot_type='profile',
        n_bins=40,
    )
    for ax in axs:
        ax.grid()


def _get_paths():
    print('Get path to data...')
    _paths = (
        Path('~/runs/multigrain/dustyshock')
        .expanduser()
        .glob('N_*-nx_*-smooth_fac_*-hfact_*')
    )
    return {p.name: p for p in _paths}


if __name__ == '__main__':

    initial_conditions()
    final_velocity_density()
    variation_hfact()
