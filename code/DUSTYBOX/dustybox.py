"""
Plot v_x on dust for DUSTYBOX test.
"""

import matplotlib.pyplot as plt
import numpy as np
import plonk

I_GAS = 1
I_DUST = 7

print('Loading simulation...')
sim = plonk.Simulation(prefix='dustybox')

number_of_times = len(sim.dumps)
number_of_dust_particles = (sim.dumps[0].particles.arrays['itype'][:] >= I_DUST).sum()

time = np.zeros((number_of_times))
vx_dust = np.zeros((number_of_times, number_of_dust_particles))

print('Getting velocity from dumps...')
for index, dump in enumerate(sim.dumps):

    print(f'Time: {dump.header["time"]}')
    time[index] = dump.header['time']

    vx_dust[index, :] = dump.particles.arrays['vxyz'][:, 0][
        dump.particles.arrays['itype'][:] >= I_DUST
    ]

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
ax.set_xlabel('time')
ax.set_ylabel('dust x-velocity')
plt.show()
