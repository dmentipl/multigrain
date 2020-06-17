"""Make animations of dusty-shock simulations."""

from pathlib import Path

import matplotlib.pyplot as plt
import plonk

PATH = Path('/fred/oz015/dmentipl/runs/multigrain/dustyshock2')


def particle_animation(name, filename, snaps, xlim=None, dlim=None, vlim=None):

    fig, axs = plt.subplots(nrows=2, sharex=True, figsize=(9, 6))
    fig.subplots_adjust(hspace=0.05)
    marker_style = {'marker': 'o', 'fillstyle': 'none'}

    plonk.visualize.particle_plot(
        snap=snaps[0], y='velocity_x', **marker_style, ax=axs[0]
    )
    plonk.visualize.particle_plot(snap=snaps[0], y='density', **marker_style, ax=axs[1])

    axs[0].set(xlim=xlim, ylim=vlim, ylabel='velocity', title=name)
    axs[1].set(xlim=xlim, ylim=dlim, ylabel='density', xlabel='x')

    text = [plonk.utils.time_string(snap, 'second', 's') for snap in snaps]

    plonk.visualize.animation_particles(
        filename=filename,
        snaps=snaps,
        x='x',
        y=['velocity_x', 'density'],
        fig=fig,
        adaptive_limits=False,
        text=text,
        save_kwargs={'fps': 10, 'dpi': 300},
    )


if __name__ == '__main__':

    for d in PATH.glob('N_3*'):
        print(f'Simulation: {d.name}')
        print('Loading simulation...')
        sim = plonk.load_sim(prefix='dustyshock', directory=d)
        print('Loading snaps...')
        print(sim.snaps)
        print('Animating...')
        particle_animation(
            name=d.name,
            filename=d / 'anim.mp4',
            snaps=sim.snaps,
            xlim=(-20, 40),
            dlim=(-1, 25),
            vlim=(-0.5, 2.5),
        )
        print('Finished!')
