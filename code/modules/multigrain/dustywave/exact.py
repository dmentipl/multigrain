"""Exact solutions to dusty wave problem.

The solutions give the magnitude of the perturbations. From
Benítez-Llambay et al. (2019). The domain width is set to 1. The sound
speed is set to 1.
"""

import numpy as np

L = 1.0
k = 2.0 * np.pi / L
c_s = 1.0

drho_g = 1.0
rho0_g = 1.0

omega_s = k * c_s

OMEGA = dict()
TSTOP = dict()

OMEGA[2] = 1.915896 - 4.410541j
TSTOP[2] = (0.4,)

OMEGA[5] = 0.912414 - 5.493800j
TSTOP[5] = (0.1, 0.215443, 0.464159, 1.0)


def rho_g(time, omega):
    """Return normalized gas density perturbation."""
    const = 1
    return (const * np.exp(-omega * time)).real


def v_g(time, omega):
    """Return normalized gas velocity perturbation."""
    const = -1j * omega / omega_s * drho_g / rho0_g
    return (const * np.exp(-omega * time)).real


def rho_d(time, omega, tstop):
    """Return normalized dust density perturbation."""
    const = 1 / (1 - omega * tstop) * drho_g / rho0_g
    return (const * np.exp(-omega * time)).real


def v_d(time, omega, tstop):
    """Return normalized dust velocity perturbation."""
    const = -1j * omega / omega_s / (1 - omega * tstop) * drho_g / rho0_g
    return (const * np.exp(-omega * time)).real


def density(time, n, n_species):
    if n == 0:
        return rho_g(time, OMEGA[n_species])
    if 0 < n <= n_species:
        return rho_d(time, OMEGA[n_species], TSTOP[n_species][n - 1])


def velocity_x(time, n, n_species):
    if n == 0:
        return v_g(time, OMEGA[n_species])
    if 0 < n <= n_species:
        return v_d(time, OMEGA[n_species], TSTOP[n_species][n - 1])
