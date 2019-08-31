"""
DUSTYBOX exact solution

See the following references:

- Laibe and Price (2011) MNRAS, 418, 1491
- Laibe and Price (2014) MNRAS, 444, 1940

Daniel Mentiplay, 2019.
"""

import numpy as np
import scipy.linalg


def delta_vx(time, K, rho, eps, delta_vx_init):
    """
    Differential velocity for linear drag on multiple dust species.

    Parameters
    ----------
    time : float
        Time at which to evaluate the expression.
    K : (N,) ndarray
        Drag constants for each dust species.
    rho : float
        Total density, assuming constant.
    eps : (N,) ndarray
        The dust fraction for each dust species.
    delta_vx_init : float
        Initial differential velocity between the dust and gas.

    Returns
    -------
    delta_vx : (N,) ndarray
        The differential velocity between the gas and dust.

    References
    ----------
    See Table (1) in Laibe and Price (2011) MNRAS, 418, 1491, and
    see Equation (64) in Laibe and Price (2014) MNRAS, 444, 1940.
    """

    omega = drag_matrix(K, rho, eps)

    if omega.size > 1:
        return scipy.linalg.expm(-omega * time) @ delta_vx_init
    return np.exp(-omega[0, 0] * time) * delta_vx_init


def drag_matrix(K, rho, eps):
    """
    Drag matrix for multiple dust species.

    Parameters
    ----------
    K : (N,) array_like
        Drag constant for each dust species.
    rho : float
        Total density, assuming constant.
    eps : (N,) array_like
        The dust fraction for each dust species.

    Returns
    -------
    omega : (N, N) ndarray
        The drag matrix.

    References
    ----------
    See Equation (65) in Laibe and Price (2014) MNRAS, 444, 1940.
    """

    K = np.array(K)
    eps = np.array(eps)
    ts = rho / K

    N = K.size
    omega = np.zeros((N, N))

    if N > 1:
        for idx in range(N):
            omega[idx, :] = 1.0 / (ts * (1.0 - np.sum(eps)))
        omega += np.diag(1.0 / (ts * eps))

    else:
        omega[:] = 1.0 / (ts * eps * (1 - eps))

    return omega
