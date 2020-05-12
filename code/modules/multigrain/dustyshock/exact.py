"""Exact solution for dusty shock."""

from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np
import plonk
from numpy import ndarray
from scipy.integrate import solve_ivp


def _derivative(
    t, y, dust_to_gas, mach_number, drag_coefficient, density_left, velocity_left
):
    w_g = _velocity_gas_normalized(dust_to_gas, mach_number, y)
    K, rho_0, v_0 = drag_coefficient, density_left, velocity_left
    return K / (rho_0 * v_0) * (w_g - y)


def _velocity_gas_normalized(dust_to_gas, mach_number, velocity_dust):
    eps, M, v_d = dust_to_gas, mach_number, velocity_dust
    a, c = 1, M ** -2
    b = np.sum(eps * (v_d - 1)) - M ** -2 - 1
    return (-b - np.sqrt(b ** 2 - 4 * a * c)) / (2 * a)


def _velocity_dust_normalized(
    x, dust_to_gas, mach_number, drag_coefficient, density_left, velocity_left,
):
    t_eval = x
    t_span = x[0], x[-1]
    y0 = velocity_left / velocity_left
    args = (dust_to_gas, mach_number, drag_coefficient, density_left, velocity_left)
    sol = solve_ivp(fun=_derivative, t_span=t_span, y0=y0, t_eval=t_eval, args=args)
    return sol.y.T


def velocity(
    x_shock: float,
    x_width: float,
    n_dust: int,
    dust_to_gas: float,
    drag_coefficient: ndarray,
    density_left: float,
    velocity_left: float,
    mach_number: float,
) -> Tuple[ndarray, ndarray, ndarray]:
    x_L = x_shock - x_width / 2
    x_R = x_shock + x_width / 2
    n_points = 500

    _dust_to_gas = dust_to_gas * np.ones(n_dust)
    _mach_number = mach_number * np.ones(n_dust)
    _drag_coefficient = drag_coefficient * np.ones(n_dust)
    _density_left = density_left * np.ones(n_dust)
    _velocity_left = velocity_left * np.ones(n_dust)

    position = np.linspace(x_L, x_R, n_points)

    wd_R = _velocity_dust_normalized(
        position[n_points // 2 :],
        _dust_to_gas,
        _mach_number,
        _drag_coefficient,
        _density_left,
        _velocity_left,
    )
    wg_R = np.array(
        [_velocity_gas_normalized(_dust_to_gas, _mach_number, _wd_R) for _wd_R in wd_R]
    )

    wg = np.ones((2 * wg_R.shape[0], wg_R.shape[1]))
    wd = np.ones((2 * wd_R.shape[0], wd_R.shape[1]))
    wg[n_points // 2 :, :] = wg_R
    wd[n_points // 2 :, :] = wd_R

    velocity_gas = velocity_left * wg[:, 0]
    velocity_dust = velocity_left * wd

    return position, velocity_gas, velocity_dust


if __name__ == '__main__':

    x_shock = 0.0
    x_width = 20.0
    n_dust = 1
    dust_to_gas = 1.0
    drag_coefficient = 1.0
    density_left = 1.0
    velocity_left = 2.0
    mach_number = 2.0

    x, vg, vd = velocity(
        x_shock,
        x_width,
        n_dust,
        dust_to_gas,
        drag_coefficient,
        density_left,
        velocity_left,
        mach_number,
    )

    fig, ax = plt.subplots()
    ax.plot(x, vg, label='Gas')
    for idx, _wd in enumerate(vd.T):
        ax.plot(x, _wd, label=f'Dust {idx}')
    ax.set_xlabel('Position')
    ax.set_ylabel('Velocity')
    ax.legend()
    ax.grid()
    plt.show()

    snap = plonk.load_snap('dustyshock_00200.h5')
    subsnaps = [snap['gas']] + snap['dust']
    colors = [line.get_color() for line in ax.lines]
    for subsnap, color in zip(subsnaps, colors):
        ax.plot(subsnap['x'], subsnap['velocity_x'], 'o', fillstyle='none', color=color)

    ax.set_xlim([x[0], x[-1]])
    ax.set_title(f'N=1; t={snap.properties["time"].m}')
