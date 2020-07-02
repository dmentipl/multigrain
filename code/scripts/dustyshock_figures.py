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
from multigrain.dustyshock import last_snap, plot_numerical_vs_exact, find_x_shock


# Get data
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
sim_names = [f'N_{N}-nx_{nx}-smooth_fac_{smooth_fac}-hfact_{hfact}' for N in Ns]
snaps = [last_snap(paths[name]) for name in sim_names]

# Find shock x-positions.
x_shock = [
    find_x_shock(snap=snap, drag_coefficients=drag_coefficients, xrange=xrange)
    for snap in snaps
]

# Plot the x-velocity and density for the gas and each dust species at the final time.
plot_numerical_vs_exact(
    snaps=snaps,
    xrange=xrange,
    drag_coefficients=drag_coefficients,
    x_shock=x_shock,
    plot_type='profile',
)