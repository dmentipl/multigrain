"""
DUSTYBOX exact solutions

This module provides functions to compute exact solutions to various
DUSTYBOX test problems.

See the following references:

- Laibe and Price (2011) MNRAS, 418, 1491
- Laibe and Price (2014) MNRAS, 444, 1940

Daniel Mentiplay, 2019.
"""

import numpy as np
import scipy.linalg


def delta_vx_single_species(t, K, rho_g, rho_d, delta_vx_init):
    """
    Differential velocity for linear drag on a single dust species.

    Parameters
    ----------
    t : float
        Time at which to evaluate the expression.
    K : float
        Drag constant.
    rho_g : float
        Gas density, assuming constant.
    rho_d : float
        Dust density, assuming constant.
    delta_vx_init : float
        Initial differential velocity between the dust and gas.

    References
    ----------
    See Table (1) in Laibe and Price (2011) MNRAS, 418, 1491.
    """

    return delta_vx_init * np.exp(-K * (1 / rho_g + 1 / rho_d) * t)


def drag_matrix(K, rho, eps):
    """
    Drag matrix for multiple dust species.

    Parameters
    ----------
    K : np.ndarray
        Drag constant for each dust species.
    rho : float
        Total density, assuming constant.
    eps : np.ndarray
        The dust-to-gas ratio for each dust species.

    References
    ----------
    See Equation (65) in Laibe and Price (2014) MNRAS, 444, 1940.
    """

    N = len(K)
    K = np.array(K)
    eps = np.array(eps)
    ts = rho / K

    omega = np.zeros((N, N))

    for idx in range(N):
        omega[idx, :] = 1.0 / (ts * (1 - np.sum(eps)))

    omega += np.diag(1 / (ts * eps))

    return omega


def delta_vx_multiple_species(t, K, rho, eps, delta_vx_init):
    """
    Differential velocity for linear drag on multiple dust species.

    Parameters
    ----------
    t : float
        Time at which to evaluate the expression.
    K : np.ndarray
        Drag constants for each dust species.
    rho : float
        Total density, assuming constant.
    eps : np.ndarray
        The dust-to-gas ratio for each dust species.
    delta_vx_init : float
        Initial differential velocity between the dust and gas.

    References
    ----------
    See Table (1) in Laibe and Price (2011) MNRAS, 418, 1491.
    """

    omega = drag_matrix(K, rho, eps)

    return scipy.linalg.expm(-omega * t) @ delta_vx_init
