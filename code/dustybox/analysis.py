"""Dusty box analysis."""

import importlib
import pathlib
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
import plonk
from numpy import ndarray
from pandas import DataFrame
from plonk import Simulation

from bokeh.io import output_notebook, show
from bokeh.layouts import gridplot, row
from bokeh.palettes import Spectral11
from bokeh.plotting import figure

path = pathlib.Path(__file__).parent / 'exact.py'
loader = importlib.machinery.SourceFileLoader('exact_solution', str(path))
exact_solution = loader.load_module()


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


def get_dust_properties(sim: Simulation) -> Tuple[ndarray, ndarray]:
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
    snap = sim.snaps[0]
    density = (
        snap.to_dataframe(('dust_id', 'density'))
        .groupby('dust_id')
        .mean()['density']
        .to_numpy()
    )
    sound_speed = np.sqrt(snap.properties['polyk'])
    gamma = snap.properties['gamma']
    grain_size = snap.properties['grain size']
    grain_dens = snap.properties['grain density']
    drag_coeff = (
        density[0]
        * density[1:]
        * sound_speed
        / (np.sqrt(np.pi * gamma / 8) * grain_size * grain_dens)
    )
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
    dust_ids = sorted(np.unique(sim.snaps[0]['dust_id']))
    n_dust = len(dust_ids) - 1

    dust_fraction, stopping_time = get_dust_properties(sim)

    # Snapshot times
    _time = list()
    for snap in sim.snaps:
        _time.append(snap.properties['time'])
    time = np.array(_time)

    # Velocity differential: simulation data
    data = np.zeros((len(time), n_dust))
    for idx, snap in enumerate(sim.snaps):
        df = snap.to_dataframe(('dust_id', 'vx')).groupby('dust_id').mean()
        data[idx, :] = (df.iloc[1:] - df.iloc[0])['vx']

    # Velocity differential: analytical solutions
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

    data_cols = [f'data.{idx}' for idx in range(1, n_dust + 1)]
    exact1_cols = [f'exact1.{idx}' for idx in range(1, n_dust + 1)]
    exact2_cols = [f'exact2.{idx}' for idx in range(1, n_dust + 1)]

    x = [df['time'] for col in data_cols]
    y_data = [df[col] for col in data_cols]
    y_exact1 = [df[col] for col in exact1_cols]
    y_exact2 = [df[col] for col in exact2_cols]

    tools = "hover, box_zoom, undo, crosshair"
    fig = figure(tools=tools)

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
