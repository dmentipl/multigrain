"""Exact solutions to dusty wave problem.

The solutions give the magnitude of the perturbations. From
Benítez-Llambay et al. (2019).
"""

import numpy as np

L = 1.0
k = 2.0 * np.pi / L
c_s = 1.0

drho_g = 1.0
rho0_g = 1.0

omega_s = k * c_s

omega = dict()
tstop = dict()

omega[2] = 1.915896 - 4.410541j
tstop[2] = (0.4,)

omega[5] = 0.912414 - 5.493800j
tstop[5] = (0.1, 0.215443, 0.464159, 1.0)


def rho_g(time, omega):
    """Normalized gas density perturbation."""
    const = 1
    return (const * np.exp(-omega * time)).real


def v_g(time, omega):
    """Normalized gas velocity perturbation."""
    const = -1j * omega / omega_s * drho_g / rho0_g
    return (const * np.exp(-omega * time)).real


def rho_d(time, omega, tstop):
    """Normalized dust density perturbation."""
    const = 1 / (1 - omega * tstop) * drho_g / rho0_g
    return (const * np.exp(-omega * time)).real


def v_d(time, omega, tstop):
    """Normalized dust velocity perturbation."""
    const = -1j * omega / omega_s / (1 - omega * tstop) * drho_g / rho0_g
    return (const * np.exp(-omega * time)).real
