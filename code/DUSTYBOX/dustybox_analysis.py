"""
DUSTYBOX analysis: compare simulations with analytic solutions.

Daniel Mentiplay, 2019.
"""

import argparse
import pathlib
from pathlib import Path
from typing import Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import phantomconfig
import plonk

import exact_solutions as exact


def main():
    run_root_dir, force_recompute = get_command_line()
    do_analysis(run_root_dir, force_recompute)


def get_command_line() -> Tuple[Path, bool]:
    """
    Get command line options.

    Returns
    -------
    run_root_dir : Path
        The path to the root directory for the calculations.
    force_recompute : bool
        Whether to force recomputing quantities written to .csv files.
    """
    parser = argparse.ArgumentParser(description='Analyse dustybox calculations')
    parser.add_argument(
        '-r',
        '--run_root_dir',
        help='the root directory for the calculations',
        required=True,
    )
    parser.add_argument(
        '-f',
        '--force_recompute',
        help='force recomputing of quantities written to .csv files',
        required=True,
    )
    args = parser.parse_args()
    run_root_dir = pathlib.Path(args.run_root_dir).resolve()
    return run_root_dir, args.force_recompute


def do_analysis(run_root_dir: Path, force_recompute: bool = False):

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
            data = read_dumps_and_compute(
                prefix='dustybox', run_directory=directory, save_dir=processed_data_dir
            )
        else:
            if processed_data_dir.exists():
                data = read_processed_data(processed_data_dir)
            else:
                data = read_dumps_and_compute(
                    prefix='dustybox',
                    run_directory=directory,
                    save_dir=processed_data_dir,
                )

        try:
            K = phantomconfig.read_config(directory / 'dustybox.in').get_value('K_code')
        except KeyError:
            header = (
                plonk.Simulation(prefix='dustybox', directory=directory).dumps[0].header
            )
            gamma = header['gamma']
            sound_speed = np.sqrt(2 / 3 * header['RK2'])
            grain_density = header['graindens'][0]
            grain_size = header['grainsize'][header['grainsize'] > 0]
            K = (
                sound_speed
                / (grain_density * grain_size)
                * np.sqrt(8 / (np.pi * gamma))
            )
        make_plot(fig_filename, K, *data)


def read_dumps_and_compute(
    prefix: str, run_directory: Path, save_dir: Path = None
) -> Tuple[np.ndarray, float, np.ndarray, np.ndarray, np.ndarray]:
    """
    Read dumps, extract data, and compute quantities.

    Parameters
    ----------
    prefix
        The simulation prefix to pass to plonk.Simulation.
    run_directory
        The simulation directory to pass to plonk.Simulation.
    save_dir
        Directory to save output to.

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

    return time, rho_gas, rho_dust, delta_vx_mean, delta_vx_var


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
    K: Union[float, np.ndarray],
    time: np.ndarray,
    rho_gas: np.ndarray,
    rho_dust: np.ndarray,
    delta_vx_mean: np.ndarray,
    delta_vx_var: np.ndarray,
):
    """
    Parameters
    ----------
    filename : pathlib.Path
        The file name to save the figure to.
    K : float or np.ndarray
        The dust drag constant.
    time : np.ndarray
        The simulation time of the dumps
    rho_gas : float
        The initial gas density.
    rho_dust : np.ndarray
        The initial dust densities.
    delta_vx_mean : np.ndarray
        The mean of the difference between the gas velocity and dust
        velocities at each time.
    delta_vx_var: np.ndarray
        The variance of the difference between the gas velocity and dust
        velocities at each time.
    """

    print('Making plot...')

    prop_cycle = plt.rcParams['axes.prop_cycle']
    colors = prop_cycle.by_key()['color']

    fig, ax = plt.subplots()

    ndusttypes = delta_vx_mean.shape[1]

    rho = rho_gas + np.sum(rho_dust)
    eps = rho_dust / rho
    delta_vx_init = delta_vx_mean[0, :]

    if isinstance(K, float):
        K = np.full_like(eps, K)

    exact_solution_with_back_reaction = np.zeros((len(time), ndusttypes))
    exact_solution_without_back_reaction = np.zeros((len(time), ndusttypes))

    for idxi, t in enumerate(time):
        exact_solution_with_back_reaction[idxi, :] = exact.dustybox.delta_vx(
            t, K, rho, eps, delta_vx_init
        )
        for idxj in range(ndusttypes):
            exact_solution_without_back_reaction[idxi, idxj] = exact.dustybox.delta_vx(
                t, K[idxj], rho, eps[idxj], delta_vx_mean[0, idxj]
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


# ------------------------------------------------------------------------------------ #

if __name__ == '__main__':
    main()
