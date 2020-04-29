"""Exact solution for dusty shock."""

import matplotlib.pyplot as plt
import numpy as np
import plonk
from scipy.integrate import solve_ivp


def _derivative(
    t, y, dust_to_gas, mach_number, drag_coefficient, density_left, velocity_left
):
    w_g = velocity_gas_normalized(dust_to_gas, mach_number, y)
    K, rho_0, v_0 = drag_coefficient, density_left, velocity_left
    return K / (rho_0 * v_0) * (w_g - y)


def velocity_gas_normalized(dust_to_gas, mach_number, velocity_dust):
    eps, M, v_d = dust_to_gas, mach_number, velocity_dust
    a, c = 1, M ** -2
    b = np.sum(eps * (v_d - 1)) - M ** -2 - 1
    return (-b - np.sqrt(b ** 2 - 4 * a * c)) / (2 * a)


def velocity_dust_normalized(
    x, dust_to_gas, mach_number, drag_coefficient, density_left, velocity_left,
):
    t_eval = x
    t_span = x[0], x[-1]
    y0 = velocity_left / velocity_left
    args = (dust_to_gas, mach_number, drag_coefficient, density_left, velocity_left)
    sol = solve_ivp(fun=_derivative, t_span=t_span, y0=y0, t_eval=t_eval, args=args)
    return sol.y.T


if __name__ == '__main__':

    x_shock = 64.0
    x_L = x_shock - 25.0
    x_R = x_shock + 25.0
    n_points = 500

    # N = 1
    dust_to_gas = np.array([1.0])
    mach_number = np.array([2.0])
    drag_coefficient = np.array([1.0])
    density_left = np.array([1.0])
    velocity_left = np.array([2.0])
    x = np.linspace(x_L, x_R, n_points)

    _wd = velocity_dust_normalized(
        x[n_points // 2 :],
        dust_to_gas,
        mach_number,
        drag_coefficient,
        density_left,
        velocity_left,
    )
    _wg = np.array(
        [velocity_gas_normalized(dust_to_gas, mach_number, __wd) for __wd in _wd]
    )

    wg = np.ones((2 * _wg.shape[0], _wg.shape[1]))
    wd = np.ones((2 * _wd.shape[0], _wd.shape[1]))
    wg[n_points // 2 :, :] = _wg
    wd[n_points // 2 :, :] = _wd

    wg *= 2
    wd *= 2

    fig, ax = plt.subplots()
    ax.plot(x, wg, label='Gas')
    ax.plot(x, wd, label='Dust')
    ax.set_xlabel('Position')
    ax.set_ylabel('Velocity')
    ax.set_title('N=1')
    ax.legend()
    ax.grid()
    plt.show()

    snap = plonk.load_snap('dustyshock_00300.h5')
    subsnaps = [snap['gas']] + snap['dust']
    colors = [line.get_color() for line in ax.lines]
    for subsnap, color in zip(subsnaps, colors):
        ax.plot(subsnap['x'], subsnap['velocity_x'], 'o', fillstyle='none', color=color)
    ax.set_xlim([x_L, x_R])
