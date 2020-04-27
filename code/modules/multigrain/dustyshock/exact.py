"""Exact solution for dusty shock."""

import matplotlib.pyplot as plt
import numpy as np
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

    # N = 1
    dust_to_gas = np.array([1.0])
    mach_number = np.array([2.0])
    drag_coefficient = np.array([1.0])
    density_left = np.array([1.0])
    velocity_left = np.array([2.0])
    x = np.linspace(0, 25)

    wd = velocity_dust_normalized(
        x, dust_to_gas, mach_number, drag_coefficient, density_left, velocity_left
    )
    wg = np.array(
        [velocity_gas_normalized(dust_to_gas, mach_number, _wd) for _wd in wd]
    )

    fig, ax = plt.subplots()
    ax.plot(x, wg, label='w_gas')
    ax.plot(x, wd, label='w_dust')
    ax.legend()
    ax.grid()
    plt.show()
