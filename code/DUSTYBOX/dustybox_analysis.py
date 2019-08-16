"""
DUSTYBOX analysis: compare simulations with analytic solutions.

Solutions from Laibe and Price (2011) MNRAS 418, 1491.

Daniel Mentiplay, 2019.
"""

import argparse
import pathlib

import matplotlib.pyplot as plt
import numpy as np
import phantom_config as pc
import plonk

I_GAS = 1
I_DUST = 7


def delta_vx_exact(t, K, rho_g, rho_d, delta_vx_init):
    return delta_vx_init * np.exp(-K * (1 / rho_g + 1 / rho_d) * t)


def get_velocities():

    print('Getting velocity from dumps...')

    ndumps = len(sim.dumps)
    ndustlarge = sim.dumps[0].header['ndustlarge']
    npartoftype_gas = sim.dumps[0].header['npartoftype'][I_GAS - 1]
    npartoftype_dust = sim.dumps[0].header['npartoftype'][
        I_DUST - 1 : I_DUST + ndustlarge - 1
    ]

    time = np.zeros((ndumps))
    vx_gas = np.zeros((ndumps, npartoftype_gas))
    vx_dust = np.zeros((ndumps, ndustlarge, np.max(npartoftype_dust)))

    for index, dump in enumerate(sim.dumps):

        print(f'Time: {dump.header["time"]}')
        time[index] = dump.header['time']

        vx_gas[index, :] = dump.particles.arrays['vxyz'][:, 0][
            dump.particles.arrays['itype'][:] == I_GAS
        ]

        for idust in range(ndustlarge):
            vx_dust[index, idust, :] = dump.particles.arrays['vxyz'][:, 0][
                dump.particles.arrays['itype'][:] == I_DUST + idust
            ]

    return time, vx_gas, vx_dust


def make_plot(time, vx_gas, vx_dust):

    prop_cycle = plt.rcParams['axes.prop_cycle']
    colors = prop_cycle.by_key()['color']

    print('Making plot...')

    fig, ax = plt.subplots()

    dump_init = sim.dumps[0]
    ndustlarge = dump_init.header['ndustlarge']

    rho_g = dump_init.density[dump_init.particles.arrays['itype'][:] == I_GAS].mean()

    v_gas_init = dump_init.particles.arrays['vxyz'][:, 0][
        dump_init.particles.arrays['itype'][:] == I_GAS
    ].mean()

    for idust, color in zip(range(ndustlarge), colors):

        rho_d = dump_init.density[
            dump_init.particles.arrays['itype'][:] == I_DUST + idust
        ].mean()

        v_dust_init = dump_init.particles.arrays['vxyz'][:, 0][
            dump_init.particles.arrays['itype'][:] == I_DUST + idust
        ].mean()

        delta_vx_init = v_gas_init - v_dust_init

        exact_solution = -delta_vx_exact(
            time, K=K, rho_g=rho_g, rho_d=rho_d, delta_vx_init=delta_vx_init
        )

        delta_vx = -(vx_gas.mean(axis=1) - vx_dust[:, idust, :].mean(axis=1))
        delta_vx_err = -(vx_gas.var(axis=1) - vx_dust[:, idust, :].var(axis=1))

        ax.errorbar(
            time, delta_vx, yerr=delta_vx_err, fmt='.', color=color, fillstyle='none'
        )

        ax.plot(time, exact_solution, color=color, label=f'dust species: {idust}')

    ax.set_xlabel(r'$t$')
    ax.set_ylabel(r'$-\Delta v_x$')
    ax.set_title(f'DUSTYBOX: K={K} drag')
    ax.legend()

    plt.show()


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-d',
        '--directory',
        type=pathlib.Path,
        help='Path to the data directory',
        required=True,
    )

    p = parser.parse_args()
    run_directory = p.directory

    print(f'Reading data from {run_directory}')

    K = pc.read_config(run_directory / 'dustybox.in').to_ordered_dict()['K_code'][0]

    print('Loading simulation data with Plonk...')
    sim = plonk.Simulation(prefix='dustybox', directory=run_directory)

    time, vx_gas, vx_dust = get_velocities()

    make_plot(time, vx_gas, vx_dust)
