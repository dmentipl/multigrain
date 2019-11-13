"""Dusty box analysis."""

import importlib
import pathlib
from pathlib import Path
from typing import Any, Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plonk
from numpy import ndarray
from pandas import DataFrame
from plonk import Simulation

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


def _get_dust_properties(sim: Simulation) -> Tuple[ndarray, ndarray]:
    """Get dust properties.

    Calculate the dust-to-gas ratio and stopping times.

    Parameters
    ----------
    sim
        The Simulation object.

    Returns
    -------
    dust_to_gas
        The dust-to-gas ratio on each species.
    stopping_time
        The stopping time on each species.
    """
    snap = sim.snaps[0]
    dust_ids = sorted(np.unique(snap['dust_id']))
    species = [snap[snap['dust_id'] == idx] for idx in dust_ids]
    density = np.array([specie['density'].mean() for specie in species])
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
    dust_to_gas = density[1:] / density[0]
    stopping_time = density.sum() / drag_coeff

    return dust_to_gas, stopping_time


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

    dust_to_gas, stopping_time = _get_dust_properties(sim)

    # Snapshot times
    _time = list()
    for snap in sim.snaps:
        _time.append(snap.properties['time'])
    time = np.array(_time)

    # Velocity differential: simulation data
    data = np.zeros((len(time), n_dust))
    for idx, snap in enumerate(sim.snaps):
        species = [snap[snap['dust_id'] == idx] for idx in dust_ids]
        v_mean = np.array([specie['vx'].mean() for specie in species])
        data[idx, :] = v_mean[1:] - v_mean[0]

    # Velocity differential: analytical solutions
    delta_vx_init = data[0, :]
    exact1 = np.zeros((len(time), n_dust))
    exact2 = np.zeros((len(time), n_dust))
    for idxi, t in enumerate(time):
        exact1[idxi, :] = exact_solution.delta_vx(
            t, stopping_time, dust_to_gas, delta_vx_init
        )
        for idxj in range(n_dust):
            exact2[idxi, idxj] = exact_solution.delta_vx(
                t, stopping_time[idxj], dust_to_gas[idxj], delta_vx_init[idxj]
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


def plot_results(dataframes: Dict[str, DataFrame], fig: Any, axes: Any) -> None:
    """Plot results.

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
    return None
