"""
Dustybox analysis: compare simulations with analytic solutions.
"""

import argparse
import pathlib
from pathlib import Path
from typing import Tuple, Union

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import phantomconfig

import exact_solutions as exact
import plonk


def do_analysis(run_root_dir: Path, force_recompute: bool = False) -> None:

    print(72 * '=')
    print(f'>>>  Run directory -- {str(run_root_dir):45}  <<<')
    print(72 * '=')

    for directory in sorted(run_root_dir.iterdir()):

        print(72 * '-')
        print(f'--- Data from {directory.name} ---')
        print(72 * '-')

        processed_data_dir = directory / 'processed_data_dir'
        fig_filename = processed_data_dir / f'delta_vx_{directory.name}.pdf'

        if force_recompute:
            read_dumps_and_compute(
                prefix='dustybox', run_directory=directory, save_dir=processed_data_dir
            )
            time, rho_g, rho_d, delta_vx_mean, delta_vx_var = read_processed_data(
                processed_data_dir
            )
        else:
            if processed_data_dir.exists():
                time, rho_g, rho_d, delta_vx_mean, delta_vx_var = read_processed_data(
                    processed_data_dir
                )
            else:
                read_dumps_and_compute(
                    prefix='dustybox',
                    run_directory=directory,
                    save_dir=processed_data_dir,
                )
                time, rho_g, rho_d, delta_vx_mean, delta_vx_var = read_processed_data(
                    processed_data_dir
                )

        rho = rho_g + np.sum(rho_d)
        eps = rho_d / rho

        try:
            in_file = directory / 'dustybox.in'
            K = np.ones_like(rho_d)
            K *= phantomconfig.read_config(in_file).get_value('K_code')
        except KeyError:
            header = (
                plonk.Simulation(prefix='dustybox', directory=directory).dumps[0].header
            )
            gamma = header['gamma']
            c_s = np.sqrt(2 / 3 * header['RK2'])
            rho_m = header['graindens'][0]
            s = header['grainsize'][header['grainsize'] > 0]
            K = rho_g * rho_d * c_s / (np.sqrt(np.pi * gamma / 8) * rho_m * s)

        t_s = rho / K

        make_plot(fig_filename, time, eps, t_s, delta_vx_mean, delta_vx_var)


def read_dumps_and_compute(
    prefix: str, run_directory: Path, save_dir: Path = None
) -> None:
    """
    Read dumps, extract data, and compute quantities.

    It writes the following to CSV files:
        time : np.ndarray
            The simulation time of the dumps
        rho_gas : float
            The initial gas density.
        rho_dust : np.ndarray
            The initial dust densities.
        delta_vx_mean : np.ndarray
            The mean of the difference between the gas velocity and dust
            velocities at each time.
        delta_vx_var : np.ndarray
            The variance of the difference between the gas velocity and
            dust velocities at each time.

    Parameters
    ----------
    prefix
        The simulation prefix to pass to plonk.Simulation.
    run_directory
        The simulation directory to pass to plonk.Simulation.
    save_dir
        Directory to save output to.
    """

    # ------------------------------------------------------------------
    # Open dumps

    print('Loading simulation data with Plonk...')
    sim = plonk.Simulation(prefix=prefix, directory=run_directory)
    header = sim.dumps[0].header

    ntimes = len(sim.dumps)
    ndusttypes = header['ndustlarge']
    idust = header['idust']

    itype = sim.dumps[0].particles.arrays['itype'][:]

    npartoftype = list(header['npartoftype'])
    itypes = [idx + 1 for idx, np in enumerate(npartoftype) if np > 0]

    # ------------------------------------------------------------------
    # Calculate initial density

    rho = sim.dumps[0].density

    igas = 1
    rho_gas = rho[itype == igas].mean()
    rho_dust = np.array(
        [rho[itype == ipart].mean() for ipart in itypes if ipart >= idust]
    )

    # ------------------------------------------------------------------
    # Calculate mean and variance in delta x-velocity

    time = np.zeros((ntimes))

    delta_vx_mean = np.zeros((ntimes, ndusttypes))
    delta_vx_var = np.zeros((ntimes, ndusttypes))

    for idx, dump in enumerate(sim.dumps):

        print(f'  time: {dump.header["time"]}')
        time[idx] = dump.header['time']

        v = dump.particles.arrays['vxyz'][:]
        itype = dump.particles.arrays['itype'][:]

        vx_mean = np.array([v[:, 0][itype == ipart].mean() for ipart in itypes])
        vx_var = np.array([v[:, 0][itype == ipart].var() for ipart in itypes])

        delta_vx_mean[idx, :] = vx_mean[1:] - vx_mean[0]
        delta_vx_var[idx, :] = vx_var[1:] - vx_var[0]

    # ------------------------------------------------------------------
    # Save data as .csv files

    if save_dir is not None:
        if not save_dir.exists():
            save_dir.mkdir()
        np.savetxt(save_dir / 'time.csv', time, delimiter=',')
        np.savetxt(save_dir / 'rho_gas.csv', np.array([rho_gas]), delimiter=',')
        np.savetxt(save_dir / 'rho_dust.csv', rho_dust, delimiter=',')
        np.savetxt(save_dir / 'delta_vx_mean.csv', delta_vx_mean, delimiter=',')
        np.savetxt(save_dir / 'delta_vx_var.csv', delta_vx_var, delimiter=',')

    return


def read_processed_data(
    data_dir: Path
) -> Tuple[np.ndarray, float, np.ndarray, np.ndarray, np.ndarray]:
    """
    Parameters
    ----------
    data_dir
        Directory containing the processed data.

    Returns
    -------
    time : np.ndarray
        The simulation time of the dumps
    rho_gas : float
        The initial gas density.
    rho_dust : np.ndarray
        The initial dust densities.
    delta_vx_mean : np.ndarray
        The mean of the difference between the gas velocity and dust
        velocities at each time.
    delta_vx_var : np.ndarray
        The variance of the difference between the gas velocity and dust
        velocities at each time.
    """

    if not data_dir.exists():
        raise FileExistsError('Cannot find data_dir')

    print('Reading processed data...')

    time = np.loadtxt(data_dir / 'time.csv', delimiter=',')
    rho_gas = np.loadtxt(data_dir / 'rho_gas.csv', delimiter=',')
    rho_dust = np.loadtxt(data_dir / 'rho_dust.csv', delimiter=',')
    delta_vx_mean = np.loadtxt(data_dir / 'delta_vx_mean.csv', delimiter=',')
    delta_vx_var = np.loadtxt(data_dir / 'delta_vx_var.csv', delimiter=',')

    return time, rho_gas, rho_dust, delta_vx_mean, delta_vx_var


def make_plot(
    filename: Path,
    time: np.ndarray,
    eps: np.ndarray,
    t_s: np.ndarray,
    delta_vx_mean: np.ndarray,
    delta_vx_var: np.ndarray,
):
    """
    Parameters
    ----------
    filename
        The file name to save the figure to.
    time
        The simulation time of the dumps
    eps
        The dust-to-gas ratio for each dust species.
    t_s
        The stopping time for each dust species.
    delta_vx_mean
        The mean of the difference between the gas velocity and dust
        velocities at each time.
    delta_vx_var
        The variance of the difference between the gas velocity and dust
        velocities at each time.
    """

    print('Making plot...')

    prop_cycle = plt.rcParams['axes.prop_cycle']
    colors = prop_cycle.by_key()['color']

    fig, ax = plt.subplots()

    ndusttypes = delta_vx_mean.shape[1]
    delta_vx_init = delta_vx_mean[0, :]

    exact_solution_with_back_reaction = np.zeros((len(time), ndusttypes))
    exact_solution_without_back_reaction = np.zeros((len(time), ndusttypes))

    for idxi, t in enumerate(time):
        exact_solution_with_back_reaction[idxi, :] = exact.dustybox.delta_vx(
            t, t_s, eps, delta_vx_init
        )
        for idxj in range(ndusttypes):
            exact_solution_without_back_reaction[idxi, idxj] = exact.dustybox.delta_vx(
                t, t_s[idxj], eps[idxj], delta_vx_mean[0, idxj]
            )

    for idx, color in zip(range(ndusttypes), colors):

        ax.errorbar(
            time,
            delta_vx_mean[:, idx],
            yerr=delta_vx_var[:, idx],
            fmt='.',
            color=color,
            fillstyle='none',
        )

        ax.plot(time, exact_solution_with_back_reaction[:, idx], color=color)

        ax.plot(time, exact_solution_without_back_reaction[:, idx], '--', color=color)

    ax.set_xlabel(r'$t$')
    ax.set_ylabel(r'$\Delta v_x$')

    print(f'Writing figure to {filename.name}...')
    plt.savefig(filename)

    return fig, ax
