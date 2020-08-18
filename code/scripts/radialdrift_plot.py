"""Radial drift velocity.

See the following reference for more details.

- Dipierro et al., 2018, MNRAS, 479, 3, p.4187-4206
"""

from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
import phantomconfig
import plonk
from plonk import Profile
from plonk import Snap
from plonk._units import Quantity


def calculate_profiles(
    snap: Snap,
    radius_min: Quantity,
    radius_max: Quantity,
    scale_height_fac: float,
    n_bins: int = 50,
) -> Dict[str, List[Profile]]:
    """Calculate radial drift velocity profiles.

    Parameters
    ----------
    snap
        The Snap object.
    radius_min
        The minimum radius for the profiles.
    radius_max
        The maximum radius for the profiles.
    scale_height_fac
        A factor of the scale height within which to average over.
    n_bins
        The number of bins in the profile. Default is 50.

    Returns
    -------
    Dict
        A dictionary of list of profiles. The keys are 'gas' and 'dust'
        and the values are lists of profiles, one per sub-type.
    """
    snap.set_gravitational_parameter(0)
    gamma = snap.properties['adiabatic_index']
    num_dust = snap.num_dust_species

    # Use particles in the midplane only
    # Choose particles such that they are within a factor of the gas scale height
    gas = snap.family('gas')
    prof = plonk.load_profile(snap=gas, cmin=radius_min, cmax=radius_max)
    scale_height = prof.to_function('scale_height')
    snap_midplane = snap[np.abs(snap['z']) < scale_height_fac * scale_height(snap['R'])]

    subsnaps = snap_midplane.subsnaps_as_dict()

    # Create radial profiles for the gas and each dust species
    cmin, cmax = radius_min, radius_max
    profs: Dict[str, List[Profile]] = {'gas': list(), 'dust': list()}
    profs['gas'] = [
        plonk.load_profile(subsnaps['gas'], cmin=cmin, cmax=cmax, n_bins=n_bins)
    ]
    for subsnap in subsnaps['dust']:
        profs['dust'].append(
            plonk.load_profile(subsnap, cmin=cmin, cmax=cmax, n_bins=n_bins)
        )

    p = profs['gas'][0]

    # velocity_pressure is (15) in Dipierro+2018
    p['velocity_pressure'] = np.gradient(p['pressure'], p['radius']) / (
        p['density'] * p['keplerian_frequency']
    )

    # shear_viscosity is between (16) an (17) in Dipierro+2018
    p['shear_viscosity'] = ALPHA * p['sound_speed'] * p['scale_height'] * p['density']

    # velocity_visc is (16) in Dipierro+2018
    p['velocity_visc'] = np.gradient(
        p['shear_viscosity']
        * p['radius'] ** 3
        * np.gradient(p['keplerian_frequency'], p['radius']),
        p['radius'],
    ) / (
        p['radius']
        * p['density']
        * np.gradient(p['radius'] ** 2 * p['keplerian_frequency'], p['radius'])
    )

    for idx, prof_dust in enumerate(profs['dust']):
        p[f'midplane_dust_to_gas_{idx+1:03}'] = prof_dust['density'] / p['density']
        p[f'midplane_stokes_number_{idx+1:03}'] = (
            np.sqrt(np.pi * gamma / 8)
            * snap.properties['grain_density'][idx]
            * snap.properties['grain_size'][idx]
            * p['keplerian_frequency']
            / (p['density'] * p['sound_speed'])
        )

    # lambda_0 and lambda_1 are (17) in Dipierro+2018
    l0 = np.zeros(len(p)) * plonk.units['dimensionless']
    l1 = np.zeros(len(p)) * plonk.units['dimensionless']
    for idx in range(num_dust):
        St = p[f'midplane_stokes_number_{idx+1:03}']
        eps = p[f'midplane_dust_to_gas_{idx+1:03}']
        l0 += 1 / (1 + St ** 2) * eps
        l1 += St / (1 + St ** 2) * eps
    p['lambda_0'] = l0
    p['lambda_1'] = l1

    v_P = p['velocity_pressure']
    v_visc = p['velocity_visc']
    l0 = p['lambda_0']
    l1 = p['lambda_1']

    # velocity_radial_gas is (11) in Dipierro+2018
    p['velocity_radial_gas'] = (-l1 * v_P + (1 + l0) * v_visc) / (
        (1 + l0) ** 2 + l1 ** 2
    )

    # velocity_azimuthal_gas is (12) in Dipierro+2018
    p['velocity_azimuthal_gas'] = (
        1 / 2 * (v_P * (1 + l0) + v_visc * l1) / ((1 + l0) ** 2 + l1 ** 2)
    )

    # velocity_radial_dust_i is (13) in Dipierro+2018
    # velocity_azimuthal_dust_i is (14) in Dipierro+2018
    for idx in range(num_dust):
        St = p[f'midplane_stokes_number_{idx+1:03}']
        eps = p[f'midplane_dust_to_gas_{idx+1:03}']
        numerator_R = v_P * ((1 + l0) * St - l1) + v_visc * (1 + l0 + St * l1)
        numerator_phi = 0.5 * v_P * (1 + l0 + St * l1) - v_visc * ((1 + l0) * St - l1)
        denominator = ((1 + l0) ** 2 + l1 ** 2) * (1 + St ** 2)
        p[f'velocity_radial_dust_{idx+1:03}'] = numerator_R / denominator
        p[f'velocity_azimuthal_dust_{idx+1:03}'] = numerator_phi / denominator

    # Divide by |v_P| for comparison with Figure B1 in Dipierro+2018
    # "Analytical" solution
    v_R = p['velocity_radial_gas']
    p['velocity_radial_gas_analytical'] = v_R / np.abs(v_P)
    for idx in range(num_dust):
        v_R = p[f'velocity_radial_dust_{idx+1:03}']
        p[f'velocity_radial_dust_{idx+1:03}_analytical'] = v_R / np.abs(v_P)

    # "Numerical" solution
    v_R = p['velocity_radial_cylindrical']
    v_R_std = p['velocity_radial_cylindrical_std']
    p['velocity_radial_numerical'] = v_R / np.abs(v_P)
    p['velocity_radial_numerical_std'] = v_R_std / np.abs(v_P)
    for prof in profs['dust']:
        v_R = prof['velocity_radial_cylindrical']
        v_R_std = prof['velocity_radial_cylindrical_std']
        prof['velocity_radial_numerical'] = v_R / np.abs(v_P)
        prof['velocity_radial_numerical_std'] = v_R_std / np.abs(v_P)

    return profs


def plot_profiles(snap, profs):
    """Plot radial drift velocity.

    Compares the numerical and analytical solutions.

    Parameters
    ----------
    snap
        The Snap object from which the profiles are generated.
    profs
        A dictionary of lists of profiles. See calculate_profiles.

    Returns
    -------
    AxesSubplot
        A matplotlib AxesSubplot object.
    """
    p = profs['gas'][0]

    if DEBUG:
        num_dust = snap.num_dust_species
        ax = p.plot(x='radius', y=['velocity_pressure', 'velocity_visc'], units=UNITS)
        y = ['velocity_radial_gas']
        y += [f'velocity_radial_dust_{idx+1:03}' for idx in range(num_dust)]
        ax = p.plot(x='radius', y=y, units=UNITS)
        ax.legend().remove()

    fig, ax = plt.subplots()

    # Plot "analytical" radial drift velocity / velocity pressure component
    p.plot(
        x='radius',
        y='velocity_radial_gas_analytical',
        units=UNITS,
        color='black',
        label='',
        ax=ax,
    )
    y = [f'velocity_radial_dust_{idx+1:03}_analytical' for idx in DUST_SPECIES_TO_PLOT]
    p.plot(x='radius', y=y, units=UNITS, label='', ax=ax)
    colors = [line.get_color() for line in ax.lines[1:]]

    # Plot "numerical" radial drift velocity / velocity pressure component
    p.plot(
        x='radius',
        y='velocity_radial_numerical',
        units=UNITS,
        color='black',
        linestyle='',
        marker='o',
        markersize=4,
        fillstyle='none',
        label='gas',
        std='shading',
        ax=ax,
    )
    profs_to_plot = [
        prof for idx, prof in enumerate(profs['dust']) if idx in DUST_SPECIES_TO_PLOT
    ]
    for species, prof, color in zip(DUST_SPECIES_TO_PLOT, profs_to_plot, colors):
        label = f'{snap.properties["grain_size"][species].to("cm"):.1f~P}'
        prof.plot(
            x='radius',
            y='velocity_radial_numerical',
            units=UNITS,
            color=color,
            linestyle='',
            marker='o',
            markersize=4,
            fillstyle='none',
            label=label,
            std='shading',
            ax=ax,
        )

    ax.set_ylabel(r'$v_R / |v_P|$')
    ax.grid()

    textstr = f't = {snap.properties["time"].to("years").m:.0f} years'
    bbox = dict(boxstyle='round', facecolor='white', edgecolor='grey', alpha=0.8)
    ax.text(
        0.97,
        0.97,
        textstr,
        transform=ax.transAxes,
        horizontalalignment='right',
        verticalalignment='top',
        bbox=bbox,
    )
    ax.legend(framealpha=0.8, edgecolor='grey')

    return ax


if __name__ == '__main__':

    DIRNAME = 'test1'
    FILENAME = 'radialdrift_00210.h5'

    DEBUG = False

    UNITS = plonk.units_defaults()
    UNITS['position'] = 'au'

    DUST_SPECIES_TO_PLOT = [1, 2, 3, 4]

    RADIUS_MIN = 25 * plonk.units['au']
    RADIUS_MAX = 125 * plonk.units['au']

    SCALE_HEIGHT_FAC = 0.05

    N_BINS = 25

    path = Path(DIRNAME).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError('DIRNAME does not exist')
    in_file = path / 'radialdrift.in'
    if not in_file.exists():
        raise FileNotFoundError('{DIRNAME}/radialdrift.in does not exist')
    snap_file = path / FILENAME
    if not snap_file.exists():
        raise FileNotFoundError('{DIRNAME}/{FILENAME} does not exist')

    ALPHA = phantomconfig.read_config(in_file).config['alpha'].value

    print(f'          file name = {DIRNAME}/{FILENAME}')
    print(f'         radius min = {RADIUS_MIN:~}')
    print(f'         radius max = {RADIUS_MAX:~}')
    print(f'scale height factor = {SCALE_HEIGHT_FAC}')
    print(f'             n_bins = {N_BINS}')
    print(f'              alpha = {ALPHA}')

    snap = plonk.load_snap(filename=snap_file)

    profs = calculate_profiles(
        snap=snap,
        radius_min=RADIUS_MIN,
        radius_max=RADIUS_MAX,
        scale_height_fac=SCALE_HEIGHT_FAC,
        n_bins=N_BINS,
    )

    ax = plot_profiles(snap=snap, profs=profs)
