"""
Plot v_x on dust for DUSTYBOX test.
"""

import matplotlib.pyplot as plt
import numpy as np
import plonk

I_GAS = 1
I_DUST = 7


def v_x(t, K, rho_g, rho_d, delta_vx_0):
    return delta_vx_0 * np.exp(-K * (1 / rho_g + 1 / rho_d) * t)


def get_data():

    print('Loading simulation data...')
    sim = plonk.Simulation(prefix='dustybox')

    number_of_times = len(sim.dumps)
    number_of_dust_particles = (
        sim.dumps[0].particles.arrays['itype'][:] >= I_DUST
    ).sum()

    time = np.zeros((number_of_times))
    vx_dust = np.zeros((number_of_times, number_of_dust_particles))

    print('Getting velocity from dumps...')
    for index, dump in enumerate(sim.dumps):

        print(f'Time: {dump.header["time"]}')
        time[index] = dump.header['time']

        vx_dust[index, :] = dump.particles.arrays['vxyz'][:, 0][
            dump.particles.arrays['itype'][:] >= I_DUST
        ]

    return time, vx_dust


def make_plot(time, vx_dust):

    print('Making plot...')

    fig, ax = plt.subplots()

    ax.errorbar(
        time,
        vx_dust.mean(axis=1),
        yerr=vx_dust.var(axis=1),
        fmt='.',
        color='black',
        ecolor='lightgray',
    )

    ax.plot(time, v_x(time, K=1.0, rho_g=1.0, rho_d=0.01, delta_vx_0=1.0))

    ax.set_xlabel('time')
    ax.set_ylabel('dust x-velocity')

    plt.show()


if __name__ == '__main__':

    time, vx_dust = get_data()
    make_plot(time, vx_dust)
