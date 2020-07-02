"""Dusty shock figures.

Make figures showing the final time velocity and density for the dusty
shock test:

- for 1 and 3 dust species.

Also, make figures showing

- the smoothed initial conditions,
- the variation with hfact.
"""

import sys
from pathlib import Path

sys.path.insert(0, '../modules')
from multigrain import dustyshock


# Get data
print('Get path to data...')
_paths = (
    Path('~/runs/multigrain/dustyshock')
    .expanduser()
    .glob('N_*-nx_*-smooth_fac_*-hfact_*')
)
paths = {p.name: p for p in _paths}

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

# Plot the x-velocity and density for the gas and each dust species at the final time.
print('Plotting figure...')
fig = dustyshock.plot_numerical_vs_exact(
    snaps=snaps,
    xrange=xrange,
    drag_coefficients=drag_coefficients,
    x_shock=x_shock,
    plot_type='profile',
    n_bins=40,
    fig_kwargs={'width': 15, 'height': 4},
)
for ax in fig.axes:
    ax.grid()
name = 'dustyshock_velocity_density.pdf'
print(f'Saving figure to {name}')
fig.savefig(name)
