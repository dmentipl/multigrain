"""
DUSTYBOX analysis: compare simulations with analytic solutions.

Solutions from Laibe and Price (2011) MNRAS 418, 1491.

Daniel Mentiplay, 2019.
"""

import pathlib

import matplotlib.pyplot as plt
import numpy as np
import phantomconfig as pc
import plonk
from dustybox_exact_solutions import delta_vx_multiple_species, delta_vx_single_species

# ------------------------------------------------------------------------------------ #
# PARAMETERS

FORCE_RECOMPUTE = False
ROOT_RUN_DIR = pathlib.Path('~/runs/multigrain/dustybox').expanduser()
LABELS = ['K=0.1', 'K=1.0', 'K=10.0']

# ------------------------------------------------------------------------------------ #

I_GAS = 1
I_DUST = 7


class InFileError(Exception):
    pass


def read_dumps_and_compute(prefix, run_directory, save_dir=None):
    """
    Read dumps, extract data, and compute quantities.

    Parameters
    ----------
    prefix : str
        The simulation prefix to pass to plonk.Simulation.
    run_directory : pathlib.Path
        The simulation directory to pass to plonk.Simulation.
    save_dir : pathlib.Path
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
    delta_vx_var
        The variance of the difference between the gas velocity and dust
        velocities at each time.
    """

    # ------------------------------------------------------------------
    # Open dumps

    print('Loading simulation data with Plonk...')
    sim = plonk.Simulation(prefix=prefix, directory=run_directory)

    ntimes = len(sim.dumps)
    ndusttypes = sim.dumps[0].header['ndustlarge']

    itype = sim.dumps[0].particles.arrays['itype'][:]

    npartoftype = list(sim.dumps[0].header['npartoftype'])
    itypes = [idx + 1 for idx, np in enumerate(npartoftype) if np > 0]

    # ------------------------------------------------------------------
    # Calculate initial density

    rho = sim.dumps[0].density

    rho_gas = rho[itype == I_GAS].mean()
    rho_dust = np.array(
        [rho[itype == ipart].mean() for ipart in itypes if ipart >= I_DUST]
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


def read_processed_data(data_dir):
    """
    Parameters
    ----------
    data_dir : pathlib.Path
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
    delta_vx_var
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


def make_plot(filename, K, time, rho_gas, rho_dust, delta_vx_mean, delta_vx_var):
    """
    Parameters
    ----------
    filename : pathlib.Path
        The file name to save the figure to.
    K : float
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
    delta_vx_var
        The variance of the difference between the gas velocity and dust
        velocities at each time.
    """

    print('Making plot...')

    prop_cycle = plt.rcParams['axes.prop_cycle']
    colors = prop_cycle.by_key()['color']

    fig, ax = plt.subplots()

    ndusttypes = delta_vx_mean.shape[1]

    rho = rho_gas + np.sum(rho_dust)
    eps = rho_dust / rho_gas
    K_i = np.full_like(eps, K)
    delta_vx_init = delta_vx_mean[0, :]

    exact_solution_with_back_reaction = np.zeros((len(time), ndusttypes))
    exact_solution_without_back_reaction = np.zeros((len(time), ndusttypes))

    for idx, t in enumerate(time):
        exact_solution_with_back_reaction[idx, :] = delta_vx_multiple_species(
            t, K_i, rho, eps, delta_vx_init
        )
    for idx in range(ndusttypes):
        exact_solution_without_back_reaction[:, idx] = delta_vx_single_species(
            np.array(time),
            K=K,
            rho_g=rho_gas,
            rho_d=rho_dust[idx],
            delta_vx_init=delta_vx_mean[0, idx],
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

        ax.plot(
            time,
            exact_solution_with_back_reaction[:, idx],
            color=color,
        )

        ax.plot(
            time,
            exact_solution_without_back_reaction[:, idx],
            '--',
            color=color,
        )

    ax.set_xlabel(r'$t$')
    ax.set_ylabel(r'$-\Delta v_x$')
    ax.set_title(f'DUSTYBOX: K={K} drag')

    print(f'Writing figure to {filename.name}...')
    plt.savefig(filename)


# ------------------------------------------------------------------------------------ #

if __name__ == '__main__':

    print(72 * '=')
    print(f'>>>  Run directory -- {str(ROOT_RUN_DIR):45}  <<<')
    print(72 * '=')

    for label in LABELS:

        prefix = f'dustybox-{label}'

        run_directory = ROOT_RUN_DIR / label
        in_file = run_directory / f'dustybox-{label}.in'

        print(72 * '-')
        print(f'--- Data from {run_directory.name} ---')
        print(72 * '-')

        processed_data_dir = run_directory / 'processed_data_dir'
        fig_filename = processed_data_dir / f'delta_vx_{label}.pdf'

        if FORCE_RECOMPUTE:
            data = read_dumps_and_compute(
                prefix, run_directory, save_dir=processed_data_dir
            )
        else:
            if processed_data_dir.exists():
                data = read_processed_data(processed_data_dir)
            else:
                data = read_dumps_and_compute(
                    prefix, run_directory, save_dir=processed_data_dir
                )

        K = pc.read_config(in_file).config['K_code'].value
        make_plot(fig_filename, K, *data)
