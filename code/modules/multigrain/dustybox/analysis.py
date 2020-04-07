"""Dusty box analysis."""

from pathlib import Path
from typing import Any, Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plonk
from numpy import ndarray
from pandas import DataFrame
from plonk import Simulation, Snap

from bokeh.io import show
from bokeh.layouts import gridplot
from bokeh.palettes import Spectral11
from bokeh.plotting import figure

from . import exact as exact_solution


def load_data(root_directory: Path) -> List[Simulation]:
    """Load simulation data.

    Parameters
    ----------
    root_directory
        The root directory containing the simulation directories.

    Returns
    -------
    A list of Simulation objects.
    """
    paths = sorted(list(root_directory.glob('*')))
    sims = [plonk.load_sim(prefix='dustybox', directory=p) for p in paths]
    return sims


def get_dust_properties(snap: Snap) -> Tuple[ndarray, ndarray]:
    """Get dust properties.

    Calculate the dust-to-gas ratio and stopping times.

    Parameters
    ----------
    sim
        The Simulation object.

    Returns
    -------
    dust_fraction
        The dust fraction on each species.
    stopping_time
        The stopping time on each species.
    """
    subsnaps = [snap['gas']] + snap['dust']
    density = np.array([subsnap['density'].mean() for subsnap in subsnaps])
    c_s = np.sqrt(snap.properties['polytropic_constant'])
    y = snap.properties['adiabatic_index']
    s = snap.properties['grain_size'].to(snap.units['length']).magnitude
    rho_m = snap.properties['grain_density'].to(snap.units['density']).magnitude
    rho_g = density[0]
    rho_d = density[1:]
    drag_coeff = rho_g * rho_d * c_s / (np.sqrt(np.pi * y / 8) * s * rho_m)
    dust_fraction = density[1:] / np.sum(density)
    stopping_time = density.sum() / drag_coeff

    return dust_fraction, stopping_time


def generate_results(sim: Simulation) -> DataFrame:
    """Generate results.

    The results for each simulation is a DataFrame of differential
    velocities for each dust species. The columns are for the
    differential velocity at each time:
        - from the simulation data
        - from the analytic solution with back reaction
        - from the analytic solution without back reaction

    Parameters
    ----------
    sim
        The simulation object.

    Returns
    -------
    The velocity differential DataFrame.
    """
    n_dust = sim.snaps[0].num_dust_species

    # Snapshot times
    time = sim.properties['time'].magnitude

    # Velocity differential: simulation data
    data = np.zeros((len(time), n_dust))
    for idx, snap in enumerate(sim.snaps):
        subsnaps = [snap['gas']] + snap['dust']
        vx = np.array([subsnap['velocity_x'].mean() for subsnap in subsnaps])
        data[idx, :] = vx[1:] - vx[0]

    # Velocity differential: analytical solutions
    dust_fraction, stopping_time = get_dust_properties(sim.snaps[0])
    delta_vx_init = data[0, :]
    exact1 = np.zeros((len(time), n_dust))
    exact2 = np.zeros((len(time), n_dust))
    for idxi, t in enumerate(time):
        exact1[idxi, :] = exact_solution.delta_vx(
            t, stopping_time, dust_fraction, delta_vx_init
        )
        for idxj in range(n_dust):
            exact2[idxi, idxj] = exact_solution.delta_vx(
                t, stopping_time[idxj], dust_fraction[idxj], delta_vx_init[idxj]
            )

    # Generate DataFrame
    arrays = np.hstack((time[:, np.newaxis], data, exact1, exact2))
    columns = (
        ['time']
        + [f'data.{idx}' for idx in range(1, n_dust + 1)]
        + [f'exact1.{idx}' for idx in range(1, n_dust + 1)]
        + [f'exact2.{idx}' for idx in range(1, n_dust + 1)]
    )
    dataframe = pd.DataFrame(arrays, columns=columns)

    return dataframe


def plot_results(df: DataFrame) -> Any:
    """Plot results.

    Plot the data as circle markers, the analytical solution with back
    reaction as solid lines, and the analytical solution without back
    reaction as dashed lines.

    Parameters
    ----------
    df
        A DataFrame with the differential velocity.

    Returns
    -------
    bokeh.plotting.figure.Figure
    """
    n_dust = int((len(df.columns) - 1) / 3)
    palette = Spectral11[:n_dust]

    x = [df['time'] for col in df.columns if col.startswith('data')]
    y_data = [df[col] for col in df.columns if col.startswith('data')]
    y_exact1 = [df[col] for col in df.columns if col.startswith('exact1')]
    y_exact2 = [df[col] for col in df.columns if col.startswith('exact2')]

    fig = figure()
    fig.multi_line(x, y_exact1, line_dash='solid', line_color=palette, line_width=3)
    fig.multi_line(x, y_exact2, line_dash=[10, 10], line_color=palette, line_width=3)
    for xx, yy, color in zip(x, y_data, palette):
        fig.scatter(xx, yy, line_color=color, fill_color=None, size=8)

    return fig


def plot_all_results(dataframes: Dict[str, DataFrame], ncols: int) -> Any:
    """Plot all results.

    Plot the data as circle markers, the analytical solution with back
    reaction as solid lines, and the analytical solution without back
    reaction as dashed lines.

    Parameters
    ----------
    dataframes
        A dictionary of DataFrames, one per simulation.
    ncols
        The number of columns.

    Returns
    -------
    bokeh.models.layouts.Column
    """
    figs = list()
    for df in dataframes.values():
        figs.append(plot_results(df=df))
    p = gridplot(figs, ncols=ncols, sizing_mode='stretch_width', plot_height=300)
    show(p)
    return p


def plot_all_results_pdf(dataframes: Dict[str, DataFrame], fig: Any, axes: Any) -> None:
    """Plot all results and save pdf.

    Plot the data as circle markers, the analytical solution with back
    reaction as solid lines, and the analytical solution without back
    reaction as dashed lines.

    Parameters
    ----------
    dataframes
        A dictionary of DataFrames, one per simulation.
    fig
        The matplotlib figure.
    axes
        The array matplotlib axes.
    """
    n_dust = int((len(list(dataframes.values())[0].columns) - 1) / 3)
    prop_cycle = plt.rcParams['axes.prop_cycle']
    color = prop_cycle.by_key()['color'][:n_dust]
    x_range = (-0.01, 0.11)
    y_range = (-0.1, 1.1)
    for df, ax in zip(dataframes.values(), axes.ravel()):
        data_cols = ['time'] + [f'data.{idx}' for idx in range(1, n_dust + 1)]
        exact1_cols = ['time'] + [f'exact1.{idx}' for idx in range(1, n_dust + 1)]
        exact2_cols = ['time'] + [f'exact2.{idx}' for idx in range(1, n_dust + 1)]
        df[data_cols].plot(
            'time',
            color=color,
            linestyle='None',
            marker='.',
            fillstyle='none',
            ax=ax,
            legend=False,
        )
        df[exact1_cols].plot('time', color=color, linestyle='-', ax=ax, legend=False)
        df[exact2_cols].plot('time', color=color, linestyle='--', ax=ax, legend=False)
        ax.set_xlabel('Time')
        ax.set_ylabel(r'$\Delta v_x$')
        ax.set_xlim(*x_range)
        ax.set_ylim(*y_range)
    fig.savefig('dustybox.pdf')
    return None
