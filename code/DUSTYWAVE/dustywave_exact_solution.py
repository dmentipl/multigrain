"""
Exact solution for DUSTYWAVE.

See the following references:
- Laibe and Price (2011) MNRAS, 418, 1491

Daniel Mentiplay, 2019.
"""

import numpy as np


def exact_dustywave(
    time, ampl, cs, Kdrag, wavelength, x0, rhog0, rhod0, xposition
):
    """
    Exact solution to DUSTYWAVE.

    Parameters
    ----------
    time : float
        Time to evaluate exact solution.
    ampl : float
        Amplitude of wave.
    cs : float
        Sound speed.
    Kdrag : float
        Drag constant.
    wavelength : float
        Wavelength of wave.
    x0 : float
        Phase of wave in space.
    rhog0 : float
        Initial gas density.
    rhod0 : float
        Initial dust density.
    xposition : (N,) ndarray
        Spatial domain on which to evaluate the solution.

    Returns
    -------
    rhogas : (N,) ndarray
        Gas density.
    rhodust : (N,) ndarray
        Dust density.
    vgas : (N,) ndarray
        Gas velocity.
    vdust : (N,) ndarray
        Dust velocity.

    References
    ----------
    See Laibe and Price (2011) MNRAS, 418, 1491, and see in Splash
    source code src/exact_dustywave.f90.
    """

    if ampl < 0.0:
        raise ValueError('amplitude < 0 on input')
    if wavelength <= 0.0:
        raise ValueError('error: wavelength <= 0 on input')
    if cs <= 0:
        raise ValueError('error: sound speed <= 0 on input')
    if rhog0 < 0:
        raise ValueError('error: gas density < 0 on input')
    if rhod0 < 0:
        raise ValueError('error: dust density < 0 on input')
    if Kdrag < 0:
        raise ValueError('error: drag coefficient < 0 on input')
    elif np.abs(Kdrag) < 1.0e-8:
        print('WARNING: Kdrag < 1.0e-8 on input; using zero to avoid divergence')
        Kdrag = 0.0

    # initial gas and dust density
    rhodeq = rhod0
    rhogeq = rhog0

    # amplitude of gas and dust density perturbation
    rhodsol = ampl * rhod0
    rhogsol = ampl * rhog0

    vdeq = 0.0
    vgeq = 0.0

    # amplitude of gas and dust velocity perturbation
    vgsol = ampl
    vdsol = ampl

    # wavenumber
    k = 2.0 * np.pi / wavelength

    vd1r = 0.0
    vd1i = 0.0
    vd2r = 0.0
    vd2i = 0.0
    vd3r = 0.0
    vd3i = 0.0
    rhod1r = 0.0
    rhod1i = 0.0
    rhod2r = 0.0
    rhod2i = 0.0
    rhod3r = 0.0
    rhod3i = 0.0

    # Solve cubic to get the 3 solutions for omega.
    # These each have both real and imaginary components,
    # labelled w1r, w1i etc.

    tdust1 = Kdrag / rhodeq
    tgas1 = Kdrag / rhogeq
    aa = tdust1 + tgas1
    bb = k ** 2 * cs ** 2
    cc = bb * tdust1

    xc = np.roots([1, aa, bb, cc])

    # get solutions for (w = iy instead of y)
    xc = xc * 1j

    w1r = xc[0].real
    w2r = xc[1].real
    w3r = xc[2].real

    w1i = xc[0].imag
    w2i = xc[1].imag
    w3i = xc[2].imag

    # ------------------------------------------------------------------
    # GAS VELOCITIES
    # ------------------------------------------------------------------
    vg3r = (
        (
            k * Kdrag * vdsol * w3r ** 2 * w2r
            + k * Kdrag * vdsol * w3r ** 2 * w1r
            - k * Kdrag * vdsol * w3r * w2r * w1r
            - k * Kdrag * vdsol * w3r * w3i ** 2
            + k * Kdrag * vdsol * w2i * w1i * w3r
            - k * Kdrag * vdsol * w2r * w1i * w3i
            + k * Kdrag * vdsol * w2r * w3i ** 2
            - k * Kdrag * vdsol * w2i * w3i * w1r
            + k * Kdrag * vdsol * w3i ** 2 * w1r
            - k * Kdrag * vgsol * w3r ** 2 * w2r
            - k * Kdrag * vgsol * w3r ** 2 * w1r
            + k * Kdrag * vgsol * w3r * w2r * w1r
            + k * Kdrag * vgsol * w3r * w3i ** 2
            - k * Kdrag * vgsol * w2i * w1i * w3r
            + k * Kdrag * vgsol * w2r * w1i * w3i
            - k * Kdrag * vgsol * w2r * w3i ** 2
            + k * Kdrag * vgsol * w2i * w3i * w1r
            - k * Kdrag * vgsol * w3i ** 2 * w1r
            - rhogsol * w3r * w3i ** 2 * w2r * w1i
            - rhogsol * w3r * w3i ** 2 * w2i * w1r
            + k * rhogeq * vgsol * w3r ** 3 * w1i
            + k * rhogeq * vgsol * w3r ** 3 * w2i
            - k * rhogeq * vgsol * w3r ** 2 * w2r * w3i
            - k * rhogeq * vgsol * w3r ** 2 * w3i * w1r
            - k * rhogeq * vgsol * w3r * w2r ** 2 * w1i
            - k * rhogeq * vgsol * w3r * w1i * w2i ** 2
            + k * rhogeq * vgsol * w3r * w3i ** 2 * w2i
            - k * rhogeq * vgsol * w2r * w3i ** 3
            - rhogsol * w3r ** 3 * w1i * w2r
            - rhogsol * w3r ** 3 * w2i * w1r
            + rhogsol * w3r ** 2 * w2r ** 2 * w1i
            + rhogsol * w3r ** 2 * w1i * w2i ** 2
            + rhogsol * w3r ** 2 * w2i * w1r ** 2
            + rhogsol * w3r ** 2 * w2i * w1i ** 2
            - rhogsol * w2r ** 2 * w1i ** 2 * w3i
            + rhogsol * w2r ** 2 * w1i * w3i ** 2
            - rhogsol * w3i * w1r ** 2 * w2r ** 2
            + rhogsol * w3i ** 3 * w1r * w2r
            + rhogsol * w3i ** 2 * w2i * w1r ** 2
            + rhogsol * w2i * w1i ** 2 * w3i ** 2
            - rhogsol * w1i ** 2 * w2i ** 2 * w3i
            - rhogsol * w2i ** 2 * w1r ** 2 * w3i
            + rhogsol * w2i ** 2 * w3i ** 2 * w1i
            - rhogsol * w2i * w3i ** 3 * w1i
            - rhogsol * k ** 2 * cs ** 2 * w3r ** 2 * w2i
            - rhogsol * k ** 2 * cs ** 2 * w3r ** 2 * w1i
            + rhogsol * k ** 2 * cs ** 2 * w3r * w1i * w2r
            + rhogsol * k ** 2 * cs ** 2 * w3r * w2i * w1r
            - rhogsol * k ** 2 * cs ** 2 * w3i * w2r * w1r
            + rhogsol * k ** 2 * cs ** 2 * w2i * w3i * w1i
            - rhogsol * k ** 2 * cs ** 2 * w3i ** 2 * w1i
            - rhogsol * k ** 2 * cs ** 2 * w3i ** 2 * w2i
            + rhogsol * k ** 2 * cs ** 2 * w3r ** 2 * w3i
            - k * rhogeq * vgsol * w3r * w2i * w1r ** 2
            + k * rhogeq * vgsol * w3r * w3i ** 2 * w1i
            - k * rhogeq * vgsol * w3r * w2i * w1i ** 2
            + k * rhogeq * vgsol * w3i * w1r * w2r ** 2
            + k * rhogeq * vgsol * w3i * w2r * w1r ** 2
            + k * rhogeq * vgsol * w3i * w2r * w1i ** 2
            + k * rhogeq * vgsol * w3i * w1r * w2i ** 2
            - k * rhogeq * vgsol * w3i ** 3 * w1r
            - k * Kdrag * vdsol * w3r ** 3
            + k * Kdrag * vgsol * w3r ** 3
            + rhogsol * k ** 2 * cs ** 2 * w3i ** 3
            + rhogsol * w3r ** 2 * w3i * w2r * w1r
            - rhogsol * w3r ** 2 * w2i * w3i * w1i
        )
        / rhogeq
        / k
        / (w2r ** 2 - 2 * w3r * w2r + w2i ** 2 + w3i ** 2 - 2 * w2i * w3i + w3r ** 2)
        / (w1i ** 2 - 2 * w3i * w1i + w3r ** 2 + w1r ** 2 + w3i ** 2 - 2 * w3r * w1r)
    )

    vg3i = (
        -1
        / rhogeq
        / k
        * (
            -w3r * w3i ** 2 * cs ** 2 * k ** 2 * rhogsol
            - w3r ** 3 * rhogsol * k ** 2 * cs ** 2
            + w3r ** 3 * w2i * rhogsol * w1i
            + w3r ** 2 * w3i * w2i * rhogeq * k * vgsol
            - w3r ** 2 * w2i * w3i * w1r * rhogsol
            - w2r * w3r ** 2 * w3i * rhogsol * w1i
            - w2r * w1i * w3i ** 3 * rhogsol
            - w2r * w3r ** 3 * w1r * rhogsol
            - w3r ** 2 * w1i * k * Kdrag * vgsol
            + w3r ** 2 * w2i * k * Kdrag * vdsol
            + w3r * w3i ** 2 * k * rhogeq * w1r * vgsol
            + w3r * w1i * w3i ** 2 * rhogsol * w2i
            - w3r ** 2 * w2i * k * Kdrag * vgsol
            + w3r ** 2 * w1i * k * Kdrag * vdsol
            - w3r ** 2 * w1i ** 2 * k * rhogeq * vgsol
            + w3r ** 3 * k * rhogeq * w1r * vgsol
            + w3r ** 2 * w3i * w1i * rhogeq * k * vgsol
            - w3r ** 2 * w1r ** 2 * rhogeq * k * vgsol
            + w3r ** 2 * w1r * cs ** 2 * k ** 2 * rhogsol
            + w3r ** 2 * w3i * k * Kdrag * vgsol
            - w3r ** 2 * w3i * k * Kdrag * vdsol
            + w2r * w3r * rhogeq * k * vgsol * w3i ** 2
            + w2r * w3r ** 2 * cs ** 2 * k ** 2 * rhogsol
            + w2r * w3r ** 3 * vgsol * k * rhogeq
            + w2r * w3i ** 2 * cs ** 2 * k ** 2 * rhogsol
            - w3i ** 2 * k * rhogeq * w1r ** 2 * vgsol
            + w3i ** 2 * cs ** 2 * k ** 2 * rhogsol * w1r
            - 2 * w3r ** 2 * k * w2i * rhogeq * w1i * vgsol
            - w1i ** 2 * rhogeq * k * vgsol * w3i ** 2
            - 2 * w3i ** 2 * k * w2i * rhogeq * w1i * vgsol
            - w3r ** 2 * w2i ** 2 * rhogeq * k * vgsol
            + w2i * w1i ** 2 * w3i * rhogeq * k * vgsol
            - w2i * w3i * w1i * k * Kdrag * vdsol
            + w2i * w3i * w1i * k * Kdrag * vgsol
            - w2i * w3i * w1r * cs ** 2 * k ** 2 * rhogsol
            + w2i * rhogeq * k * vgsol * w3i * w1r ** 2
            + w2r * w3i * w1r * k * Kdrag * vdsol
            - w2i * w3i ** 2 * k * Kdrag * vgsol
            - w2r * w3i * w1r * k * Kdrag * vgsol
            + w2i * w3i ** 2 * k * Kdrag * vdsol
            - 2 * w2r * w3i ** 2 * k * rhogeq * w1r * vgsol
            + w2i * w3i ** 3 * rhogeq * k * vgsol
            + w3r * w2i ** 2 * w1r * rhogeq * k * vgsol
            + w2r ** 2 * w3i ** 2 * w1r * rhogsol
            - w1i * w3i ** 2 * k * Kdrag * vgsol
            + w1i * rhogeq * k * vgsol * w3i ** 3
            + w1i * w3i ** 2 * k * Kdrag * vdsol
            - w2r * w3r * w3i ** 2 * w1r * rhogsol
            - w3i ** 3 * k * Kdrag * vdsol
            + w3i ** 3 * k * Kdrag * vgsol
            - w2r * w1i * w3i * cs ** 2 * k ** 2 * rhogsol
            - w1r * w2i * w3i ** 3 * rhogsol
            + w3i * w2r ** 2 * w1i * rhogeq * k * vgsol
            + w1i ** 2 * w3i ** 2 * w2r * rhogsol
            - w3i ** 2 * w2i ** 2 * rhogeq * k * vgsol
            + w2i ** 2 * w3i * w1i * rhogeq * k * vgsol
            + w3r * w2i * w1i * cs ** 2 * k ** 2 * rhogsol
            - w3r * w1i ** 2 * w2r ** 2 * rhogsol
            - w3r * w2i ** 2 * w1i ** 2 * rhogsol
            + w3r ** 2 * w1i ** 2 * w2r * rhogsol
            + w3r ** 2 * w2i ** 2 * w1r * rhogsol
            - w2r ** 2 * w3i ** 2 * k * rhogeq * vgsol
            + w3r ** 2 * w2r * w1r ** 2 * rhogsol
            - w3r ** 2 * w2r ** 2 * rhogeq * k * vgsol
            + w3r ** 2 * w2r ** 2 * w1r * rhogsol
            - w3r * w1r ** 2 * w2i ** 2 * rhogsol
            - w3r * w2r ** 2 * w1r ** 2 * rhogsol
            - w3r * w2i * w1r * k * Kdrag * vdsol
            + w3r * w2i * w1r * k * Kdrag * vgsol
            - w3r * w2r * w1r * cs ** 2 * k ** 2 * rhogsol
            - 2 * w3r ** 2 * w2r * k * rhogeq * w1r * vgsol
            + w3r * w2r ** 2 * k * rhogeq * w1r * vgsol
            + w3r * w2r * w1i * k * Kdrag * vgsol
            + w3r * w2r * w1i ** 2 * k * rhogeq * vgsol
            - w3r * w2r * w1i * k * Kdrag * vdsol
            + w3r * w2r * w1r ** 2 * rhogeq * k * vgsol
            + w2i ** 2 * w3i ** 2 * w1r * rhogsol
            + w2r * w3i ** 2 * w1r ** 2 * rhogsol
        )
        / (w2r ** 2 - 2 * w3r * w2r + w2i ** 2 + w3i ** 2 - 2 * w2i * w3i + w3r ** 2)
        / (w1i ** 2 - 2 * w3i * w1i + w3r ** 2 + w1r ** 2 + w3i ** 2 - 2 * w3r * w1r)
    )

    vg2r = (
        -(
            w2r ** 2 * rhogsol * w2i * w3i * w1i
            + w2r ** 2 * rhogsol * k ** 2 * cs ** 2 * w1i
            + k * Kdrag * vdsol * w3r * w2r * w1r
            + k * Kdrag * vdsol * w2i * w1i * w3r
            - k * Kdrag * vdsol * w2r * w1i * w3i
            + k * Kdrag * vdsol * w2i * w3i * w1r
            - k * Kdrag * vgsol * w3r * w2r * w1r
            - k * Kdrag * vgsol * w2i * w1i * w3r
            + k * Kdrag * vgsol * w2r * w1i * w3i
            - k * Kdrag * vgsol * w2i * w3i * w1r
            - rhogsol * w3r ** 2 * w2r ** 2 * w1i
            - rhogsol * w3r ** 2 * w1i * w2i ** 2
            + rhogsol * w3r ** 2 * w2i * w1r ** 2
            + rhogsol * w3r ** 2 * w2i * w1i ** 2
            - rhogsol * w2r ** 2 * w1i ** 2 * w3i
            - rhogsol * w2r ** 2 * w1i * w3i ** 2
            - rhogsol * w3i * w1r ** 2 * w2r ** 2
            + rhogsol * w3i ** 2 * w2i * w1r ** 2
            + rhogsol * w2i * w1i ** 2 * w3i ** 2
            - rhogsol * w1i ** 2 * w2i ** 2 * w3i
            - rhogsol * w2i ** 2 * w1r ** 2 * w3i
            - rhogsol * w2i ** 2 * w3i ** 2 * w1i
            + w2r ** 3 * rhogsol * w3i * w1r
            - rhogsol * k ** 2 * cs ** 2 * w3r * w1i * w2r
            + rhogsol * k ** 2 * cs ** 2 * w3r * w2i * w1r
            - rhogsol * k ** 2 * cs ** 2 * w3i * w2r * w1r
            - rhogsol * k ** 2 * cs ** 2 * w2i * w3i * w1i
            - k * rhogeq * vgsol * w3r * w2i * w1r ** 2
            - k * rhogeq * vgsol * w3r * w2i * w1i ** 2
            + k * rhogeq * vgsol * w3i * w2r * w1r ** 2
            + k * rhogeq * vgsol * w3i * w2r * w1i ** 2
            + w2r ** 2 * rhogsol * k ** 2 * cs ** 2 * w3i
            + w2i ** 3 * rhogsol * w3i * w1i
            - w2r ** 3 * k * Kdrag * vgsol
            + w2r ** 3 * k * Kdrag * vdsol
            - w2r ** 3 * vgsol * k * rhogeq * w1i
            + w2i * vgsol * k * rhogeq * w1r * w2r ** 2
            + w2i ** 3 * vgsol * k * rhogeq * w1r
            - w2i ** 2 * w2r * vgsol * k * rhogeq * w1i
            - w3r * w2i ** 3 * rhogsol * w1r
            + w3r * w2r ** 3 * rhogsol * w1i
            - w3r * w2r ** 2 * rhogsol * w2i * w1r
            - w3r * w2i ** 2 * k * Kdrag * vdsol
            + w3r * w2i ** 2 * rhogsol * w1i * w2r
            + w3r * w2i ** 2 * k * Kdrag * vgsol
            - w3r ** 2 * vgsol * k * rhogeq * w2i * w1r
            + w3r ** 2 * w2r * k * rhogeq * vgsol * w1i
            + w3r * w2r ** 2 * k * Kdrag * vgsol
            - w3r * w2r ** 2 * k * Kdrag * vdsol
            + w3r * w2r ** 2 * k * rhogeq * vgsol * w2i
            + w3r * w2i ** 3 * k * rhogeq * vgsol
            - w2r ** 3 * k * rhogeq * vgsol * w3i
            - w2i ** 3 * rhogsol * k ** 2 * cs ** 2
            - vgsol * k * rhogeq * w2i * w1r * w3i ** 2
            + w2i ** 2 * k * Kdrag * vgsol * w1r
            - w2i ** 2 * k * Kdrag * vgsol * w2r
            - w2i ** 2 * k * Kdrag * vdsol * w1r
            + w2i ** 2 * rhogsol * k ** 2 * cs ** 2 * w3i
            - w2i ** 2 * k * rhogeq * vgsol * w2r * w3i
            + w2i ** 2 * rhogsol * k ** 2 * cs ** 2 * w1i
            + w2i ** 2 * rhogsol * w3i * w2r * w1r
            - w2r ** 2 * k * Kdrag * vdsol * w1r
            + w2r ** 2 * k * Kdrag * vgsol * w1r
            + w2r * vgsol * k * rhogeq * w1i * w3i ** 2
            + w2i ** 2 * k * Kdrag * vdsol * w2r
            - w2r ** 2 * rhogsol * k ** 2 * cs ** 2 * w2i
        )
        / rhogeq
        / k
        / (w2r ** 2 - 2 * w3r * w2r + w2i ** 2 + w3i ** 2 - 2 * w2i * w3i + w3r ** 2)
        / (w2r ** 2 + w1r ** 2 + w2i ** 2 - 2 * w2i * w1i - 2 * w2r * w1r + w1i ** 2)
    )

    vg2i = (
        (
            w1r ** 2 * w2i ** 2 * k * rhogeq * vgsol
            - w2r ** 2 * w2i * k * Kdrag * vgsol
            + w2i ** 2 * w1i ** 2 * k * rhogeq * vgsol
            - w1i * w2i ** 3 * rhogeq * k * vgsol
            - w2i ** 3 * w3i * rhogeq * k * vgsol
            - w3r ** 2 * k * w2i * rhogeq * w1i * vgsol
            - w3i ** 2 * k * w2i * rhogeq * w1i * vgsol
            + w3r * w2i * w1i * w2r ** 2 * rhogsol
            + w3r ** 2 * w2i ** 2 * rhogeq * k * vgsol
            - w2i ** 2 * w1r * cs ** 2 * k ** 2 * rhogsol
            - w2i * w1i ** 2 * w3i * rhogeq * k * vgsol
            + w2i * w3i * w1i * k * Kdrag * vdsol
            - w2i * w3i * w1i * k * Kdrag * vgsol
            - w2r ** 2 * w3i * w2i * rhogeq * k * vgsol
            + w2i * w3i * w1r * cs ** 2 * k ** 2 * rhogsol
            - w2i * rhogeq * k * vgsol * w3i * w1r ** 2
            + w2r ** 3 * cs ** 2 * k ** 2 * rhogsol
            - w2r ** 2 * w1i * w2i * rhogeq * k * vgsol
            + w2r ** 2 * w2i * w3i * w1r * rhogsol
            + w2r * w3i * w1r * k * Kdrag * vdsol
            - w2i ** 2 * w1i * k * Kdrag * vdsol
            - w2r * w3i * w1r * k * Kdrag * vgsol
            - w2r * w3i ** 2 * k * rhogeq * w1r * vgsol
            - w2r * w1r * w2i ** 2 * rhogeq * k * vgsol
            + 2 * w3r * w2i ** 2 * w1r * rhogeq * k * vgsol
            + w2r ** 2 * w1i * k * Kdrag * vgsol
            - w2r ** 2 * w3i ** 2 * w1r * rhogsol
            + w3r * w2i ** 3 * w1i * rhogsol
            + w2i ** 2 * w1i * k * Kdrag * vgsol
            - w2r * w1i * w3i * cs ** 2 * k ** 2 * rhogsol
            + 2 * w3i * w2r ** 2 * w1i * rhogeq * k * vgsol
            + w1i ** 2 * w3i ** 2 * w2r * rhogsol
            + w3i ** 2 * w2i ** 2 * rhogeq * k * vgsol
            + 2 * w2i ** 2 * w3i * w1i * rhogeq * k * vgsol
            + w3r * w2i * w1i * cs ** 2 * k ** 2 * rhogsol
            - w3r * w1i ** 2 * w2r ** 2 * rhogsol
            - w3r * w2i ** 2 * w1i ** 2 * rhogsol
            + w3r ** 2 * w1i ** 2 * w2r * rhogsol
            - w3r ** 2 * w2i ** 2 * w1r * rhogsol
            + w2r ** 2 * w2i * k * Kdrag * vdsol
            - w2r ** 3 * k * rhogeq * w1r * vgsol
            + w2i ** 2 * w3i * k * Kdrag * vgsol
            - w2i ** 2 * w3i * k * Kdrag * vdsol
            + w2r * w2i ** 2 * cs ** 2 * k ** 2 * rhogsol
            - w2r * w2i ** 2 * w3i * rhogsol * w1i
            + w2r ** 2 * w3i ** 2 * k * rhogeq * vgsol
            + w2r ** 2 * w1r ** 2 * rhogeq * k * vgsol
            - w2r ** 2 * w1r * cs ** 2 * k ** 2 * rhogsol
            + w3r ** 2 * w2r * w1r ** 2 * rhogsol
            + w2r ** 2 * w3i * k * Kdrag * vgsol
            - w2r ** 2 * w3i * k * Kdrag * vdsol
            - w2r ** 2 * w1i * k * Kdrag * vdsol
            + w2r ** 2 * w1i ** 2 * k * rhogeq * vgsol
            - w3r * w2r ** 2 * cs ** 2 * k ** 2 * rhogsol
            + w3r * w2r * w2i ** 2 * w1r * rhogsol
            - w3r * w2i ** 2 * cs ** 2 * k ** 2 * rhogsol
            - w3r * w2r ** 3 * rhogeq * k * vgsol
            + w3r ** 2 * w2r ** 2 * rhogeq * k * vgsol
            - w3r ** 2 * w2r ** 2 * w1r * rhogsol
            - w2r ** 3 * w3i * rhogsol * w1i
            - w3r * w1r ** 2 * w2i ** 2 * rhogsol
            - w3r * w2r ** 2 * w1r ** 2 * rhogsol
            - w3r * w2i * w1r * k * Kdrag * vdsol
            + w3r * w2i * w1r * k * Kdrag * vgsol
            + w3r * w2r ** 3 * w1r * rhogsol
            + w3r * w2r * w1r * cs ** 2 * k ** 2 * rhogsol
            - w3r ** 2 * w2r * k * rhogeq * w1r * vgsol
            + 2 * w3r * w2r ** 2 * k * rhogeq * w1r * vgsol
            - w3r * w2r * w2i ** 2 * rhogeq * k * vgsol
            - w3r * w2r * w1i * k * Kdrag * vgsol
            - w3r * w2r * w1i ** 2 * k * rhogeq * vgsol
            + w3r * w2r * w1i * k * Kdrag * vdsol
            - w3r * w2r * w1r ** 2 * rhogeq * k * vgsol
            + w2i ** 3 * w3i * w1r * rhogsol
            - w2i ** 3 * k * Kdrag * vgsol
            - w2i ** 2 * w3i ** 2 * w1r * rhogsol
            + w2i ** 3 * k * Kdrag * vdsol
            + w2r * w3i ** 2 * w1r ** 2 * rhogsol
        )
        / rhogeq
        / k
        / (w2r ** 2 - 2 * w3r * w2r + w2i ** 2 + w3i ** 2 - 2 * w2i * w3i + w3r ** 2)
        / (w2r ** 2 + w1r ** 2 + w2i ** 2 - 2 * w2i * w1i - 2 * w2r * w1r + w1i ** 2)
    )

    vg1r = (
        (
            -rhogsol * w1i ** 3 * w3i * w2i
            + rhogsol * k ** 2 * cs ** 2 * w1i ** 3
            - rhogsol * w3i * w2r * w1r ** 3
            - k * Kdrag * vgsol * w2r * w1r ** 2
            + k * Kdrag * vdsol * w2r * w1r ** 2
            - k * Kdrag * vdsol * w1r * w1i ** 2
            - k * Kdrag * vgsol * w2r * w1i ** 2
            + k * Kdrag * vdsol * w2r * w1i ** 2
            + k * Kdrag * vgsol * w1r * w1i ** 2
            - k * Kdrag * vdsol * w1r ** 3
            + k * Kdrag * vgsol * w1r ** 3
            - w3r * rhogsol * w2i * w1r ** 3
            - w3r * Kdrag * w1i ** 2 * k * vgsol
            + w3r * k * Kdrag * vdsol * w1r ** 2
            + w3r * k * Kdrag * vdsol * w1i ** 2
            - w3r * Kdrag * w1r ** 2 * k * vgsol
            - k * Kdrag * vdsol * w3r * w2r * w1r
            - k * Kdrag * vdsol * w2i * w1i * w3r
            - k * Kdrag * vdsol * w2r * w1i * w3i
            + k * Kdrag * vdsol * w2i * w3i * w1r
            + k * Kdrag * vgsol * w3r * w2r * w1r
            + k * Kdrag * vgsol * w2i * w1i * w3r
            + k * Kdrag * vgsol * w2r * w1i * w3i
            - k * Kdrag * vgsol * w2i * w3i * w1r
            + k * rhogeq * vgsol * w3r * w2r ** 2 * w1i
            + k * rhogeq * vgsol * w3r * w1i * w2i ** 2
            - rhogsol * w3r ** 2 * w2r ** 2 * w1i
            - rhogsol * w3r ** 2 * w1i * w2i ** 2
            + rhogsol * w3r ** 2 * w2i * w1r ** 2
            + rhogsol * w3r ** 2 * w2i * w1i ** 2
            + rhogsol * w2r ** 2 * w1i ** 2 * w3i
            - rhogsol * w2r ** 2 * w1i * w3i ** 2
            + rhogsol * w3i * w1r ** 2 * w2r ** 2
            + rhogsol * w3i ** 2 * w2i * w1r ** 2
            + rhogsol * w2i * w1i ** 2 * w3i ** 2
            + rhogsol * w1i ** 2 * w2i ** 2 * w3i
            + rhogsol * w2i ** 2 * w1r ** 2 * w3i
            - rhogsol * w2i ** 2 * w3i ** 2 * w1i
            - rhogsol * k ** 2 * cs ** 2 * w3r * w1i * w2r
            + rhogsol * k ** 2 * cs ** 2 * w3r * w2i * w1r
            + rhogsol * k ** 2 * cs ** 2 * w3i * w2r * w1r
            + rhogsol * k ** 2 * cs ** 2 * w2i * w3i * w1i
            - k * rhogeq * vgsol * w3i * w1r * w2r ** 2
            - k * rhogeq * vgsol * w3i * w1r * w2i ** 2
            - w3r ** 2 * vgsol * k * rhogeq * w2i * w1r
            + w3r ** 2 * w2r * k * rhogeq * vgsol * w1i
            + w3r * rhogsol * w1i ** 3 * w2r
            - vgsol * k * rhogeq * w2i * w1r * w3i ** 2
            - w3r * w1r ** 2 * k * rhogeq * vgsol * w1i
            - w3r * w1i ** 3 * k * rhogeq * vgsol
            - w3r * w1r * rhogsol * w2i * w1i ** 2
            + w3r * rhogsol * w1i * w2r * w1r ** 2
            + w2r * vgsol * k * rhogeq * w1i * w3i ** 2
            - rhogsol * w3i * w2r * w1r * w1i ** 2
            - rhogeq * w2r * w1i * k * vgsol * w1r ** 2
            - rhogeq * w2r * w1i ** 3 * k * vgsol
            + k * rhogeq * vgsol * w2i * w1r ** 3
            - rhogsol * k ** 2 * cs ** 2 * w2i * w1r ** 2
            + k * rhogeq * vgsol * w3i * w1r * w1i ** 2
            + rhogsol * k ** 2 * cs ** 2 * w1i * w1r ** 2
            - rhogsol * k ** 2 * cs ** 2 * w3i * w1i ** 2
            + k * rhogeq * vgsol * w3i * w1r ** 3
            - rhogsol * k ** 2 * cs ** 2 * w1i ** 2 * w2i
            - rhogsol * k ** 2 * cs ** 2 * w3i * w1r ** 2
            - rhogsol * w3i * w1r ** 2 * w2i * w1i
            + w1r * k * rhogeq * vgsol * w2i * w1i ** 2
        )
        / (w1i ** 2 - 2 * w3i * w1i + w3r ** 2 + w1r ** 2 + w3i ** 2 - 2 * w3r * w1r)
        / (w2r ** 2 + w1r ** 2 + w2i ** 2 - 2 * w2i * w1i - 2 * w2r * w1r + w1i ** 2)
        / rhogeq
        / k
    )

    vg1i = (
        -(
            -w1r ** 2 * w2i ** 2 * k * rhogeq * vgsol
            - w2i ** 2 * w1i ** 2 * k * rhogeq * vgsol
            - w3r ** 2 * w1i ** 2 * k * rhogeq * vgsol
            - w3r ** 2 * w1r ** 2 * rhogeq * k * vgsol
            - w3r * w2r * w1r ** 3 * rhogsol
            - w3r * w1i ** 2 * w2r * rhogsol * w1r
            - w3i ** 2 * k * rhogeq * w1r ** 2 * vgsol
            + w3r ** 2 * k * w2i * rhogeq * w1i * vgsol
            - w1i ** 2 * rhogeq * k * vgsol * w3i ** 2
            + w3i ** 2 * k * w2i * rhogeq * w1i * vgsol
            + w1i ** 3 * k * rhogeq * vgsol * w3i
            - w3r * w2i * w1i * rhogsol * w1r ** 2
            - w1i ** 3 * k * Kdrag * vdsol
            - 2 * w2i * w1i ** 2 * w3i * rhogeq * k * vgsol
            + w2i * w3i * w1r * rhogsol * w1i ** 2
            - w1i ** 2 * k * Kdrag * vgsol * w3i
            - w1i * k * Kdrag * vdsol * w1r ** 2
            + w1i ** 2 * k * rhogeq * vgsol * w3r * w1r
            + w1i * k * Kdrag * vgsol * w1r ** 2
            - w1r ** 3 * cs ** 2 * k ** 2 * rhogsol
            + cs ** 2 * k ** 2 * rhogsol * w3r * w1r ** 2
            + w3i * k * Kdrag * vdsol * w1r ** 2
            - w2i * w3i * w1i * k * Kdrag * vdsol
            + w2i * w3i * w1i * k * Kdrag * vgsol
            + cs ** 2 * k ** 2 * rhogsol * w3r * w1i ** 2
            + w3i * k * Kdrag * vdsol * w1i ** 2
            + w1i ** 3 * w2i * rhogeq * k * vgsol
            + w2r * k * rhogeq * w1r * vgsol * w1i ** 2
            - w2i * k * Kdrag * vgsol * w1i ** 2
            + w2i * k * Kdrag * vdsol * w1i ** 2
            + w1r ** 2 * rhogeq * k * vgsol * w3i * w1i
            + w1r ** 3 * rhogeq * k * vgsol * w3r
            + w2i * k * Kdrag * vdsol * w1r ** 2
            + w2i * w3i * w1r * cs ** 2 * k ** 2 * rhogsol
            - 2 * w2i * rhogeq * k * vgsol * w3i * w1r ** 2
            - w3i * k * Kdrag * vgsol * w1r ** 2
            - w2r * w3i * w1r * k * Kdrag * vdsol
            + w2r * w3i * w1r * k * Kdrag * vgsol
            + w2r * w3i ** 2 * k * rhogeq * w1r * vgsol
            + w3r * w2i ** 2 * w1r * rhogeq * k * vgsol
            - w2r ** 2 * w3i ** 2 * w1r * rhogsol
            - w1i ** 3 * w2r * rhogsol * w3i
            - w2r * w1r ** 2 * rhogsol * w3i * w1i
            - w2i * k * Kdrag * vgsol * w1r ** 2
            - w2r * w1i * w3i * cs ** 2 * k ** 2 * rhogsol
            + w2i * w3i * w1r ** 3 * rhogsol
            + w3i * w2r ** 2 * w1i * rhogeq * k * vgsol
            + w1i ** 2 * w3i ** 2 * w2r * rhogsol
            + w1i * w2i * rhogeq * k * vgsol * w1r ** 2
            + w2i ** 2 * w3i * w1i * rhogeq * k * vgsol
            - w3r * w2i * w1i * cs ** 2 * k ** 2 * rhogsol
            + w3r * w1i ** 2 * w2r ** 2 * rhogsol
            + w3r * w2i ** 2 * w1i ** 2 * rhogsol
            + w3r ** 2 * w1i ** 2 * w2r * rhogsol
            - w3r ** 2 * w2i ** 2 * w1r * rhogsol
            - w2r ** 2 * w1r ** 2 * rhogeq * k * vgsol
            + w3r ** 2 * w2r * w1r ** 2 * rhogsol
            - w2r ** 2 * w1i ** 2 * k * rhogeq * vgsol
            - w1r * cs ** 2 * k ** 2 * rhogsol * w1i ** 2
            - w3r ** 2 * w2r ** 2 * w1r * rhogsol
            + w3r * w1r ** 2 * w2i ** 2 * rhogsol
            + w3r * w2r ** 2 * w1r ** 2 * rhogsol
            - w3r * w2i * w1i ** 3 * rhogsol
            + w1i ** 3 * k * Kdrag * vgsol
            - w3r * w2i * w1r * k * Kdrag * vdsol
            + w3r * w2i * w1r * k * Kdrag * vgsol
            - w3r * w2r * w1r * cs ** 2 * k ** 2 * rhogsol
            + w3r ** 2 * w2r * k * rhogeq * w1r * vgsol
            + w3r * w2r ** 2 * k * rhogeq * w1r * vgsol
            - w3r * w2r * w1i * k * Kdrag * vgsol
            - 2 * w3r * w2r * w1i ** 2 * k * rhogeq * vgsol
            + w3r * w2r * w1i * k * Kdrag * vdsol
            - 2 * w3r * w2r * w1r ** 2 * rhogeq * k * vgsol
            + w2r * k * rhogeq * w1r ** 3 * vgsol
            + w2r * w1r ** 2 * cs ** 2 * k ** 2 * rhogsol
            - w2i ** 2 * w3i ** 2 * w1r * rhogsol
            + rhogsol * k ** 2 * cs ** 2 * w1i ** 2 * w2r
            + w2r * w3i ** 2 * w1r ** 2 * rhogsol
        )
        / (w1i ** 2 - 2 * w3i * w1i + w3r ** 2 + w1r ** 2 + w3i ** 2 - 2 * w3r * w1r)
        / (w2r ** 2 + w1r ** 2 + w2i ** 2 - 2 * w2i * w1i - 2 * w2r * w1r + w1i ** 2)
        / rhogeq
        / k
    )

    # ------------------------------------------------------------------
    # DUST VELOCITIES
    # ------------------------------------------------------------------
    if Kdrag > 0.0:

        vd3r = (
            -(
                rhogeq * cs ** 2 * k ** 2 * w2r ** 2 * w1r ** 2 * rhogsol
                + rhogeq ** 2 * w3i ** 4 * k * w1r * vgsol
                - w3i * cs ** 2 * k ** 3 * rhogeq * Kdrag * vdsol * w2r
                - w3i * cs ** 2 * k ** 3 * rhogeq * Kdrag * vdsol * w1r
                + w3i * cs ** 2 * k ** 3 * rhogeq * Kdrag * vgsol * w2r
                + w3i * cs ** 2 * k ** 3 * rhogeq * Kdrag * vgsol * w1r
                - rhogeq ** 2 * cs ** 2 * k ** 3 * w2i ** 2 * w1r * vgsol
                + rhogeq * cs ** 4 * k ** 4 * w2r * w1r * rhogsol
                - rhogeq ** 2 * cs ** 2 * k ** 3 * w2r ** 2 * w1r * vgsol
                - rhogeq * cs ** 2 * k ** 3 * w2r * w1i * Kdrag * vgsol
                - rhogeq ** 2 * cs ** 2 * k ** 3 * w2r * w1i ** 2 * vgsol
                + rhogeq * cs ** 2 * k ** 3 * w2r * w1i * Kdrag * vdsol
                - rhogeq ** 2 * cs ** 2 * k ** 3 * w2r * w1r ** 2 * vgsol
                - rhogeq * w3i ** 4 * cs ** 2 * k ** 2 * rhogsol
                - rhogeq * cs ** 4 * k ** 4 * w2i * w1i * rhogsol
                + rhogeq * cs ** 2 * k ** 2 * w1i ** 2 * w2r ** 2 * rhogsol
                + rhogeq ** 2 * w3i ** 4 * w2r * k * vgsol
                - rhogeq * w3i ** 3 * k * Kdrag * vdsol * w1r
                + 2 * rhogeq * w3i ** 3 * k * Kdrag * vgsol * w2r
                + rhogeq * w2i ** 2 * w1i ** 2 * cs ** 2 * k ** 2 * rhogsol
                + rhogeq * w2i ** 2 * w1r ** 2 * cs ** 2 * k ** 2 * rhogsol
                - rhogeq * w3i ** 4 * w2r * w1r * rhogsol
                + rhogeq * w3i ** 4 * w1i * rhogsol * w2i
                + w3i * cs ** 4 * k ** 4 * rhogeq * rhogsol * w2i
                + w3i * cs ** 4 * k ** 4 * rhogeq * rhogsol * w1i
                + w3i ** 2 * Kdrag ** 2 * k * vgsol * w1r
                - w3i ** 3 * Kdrag * rhogsol * k ** 2 * cs ** 2
                - w3i ** 3 * Kdrag * rhogsol * w2r * w1r
                + w3i ** 3 * Kdrag * rhogsol * w2i * w1i
                - w3i * cs ** 2 * k ** 2 * rhogeq * rhogsol * w2r ** 2 * w1i
                - w3i * cs ** 2 * k ** 2 * rhogeq * rhogsol * w1i * w2i ** 2
                + rhogeq * w3i ** 3 * rhogsol * k ** 2 * cs ** 2 * w2i
                + rhogeq * w3i ** 3 * rhogsol * k ** 2 * cs ** 2 * w1i
                - rhogeq * w3i ** 3 * rhogsol * w2r ** 2 * w1i
                - rhogeq * w3i ** 3 * rhogsol * w1i * w2i ** 2
                - rhogeq * w3i ** 3 * rhogsol * w2i * w1r ** 2
                - w3i * cs ** 2 * k ** 2 * rhogeq * rhogsol * w2i * w1r ** 2
                - w3i * cs ** 2 * k ** 2 * rhogeq * rhogsol * w2i * w1i ** 2
                - rhogeq * w3i ** 3 * rhogsol * w2i * w1i ** 2
                - rhogeq * w3i ** 3 * k * Kdrag * vdsol * w2r
                + 2 * rhogeq * w3i ** 3 * k * Kdrag * vgsol * w1r
                + w3r * rhogeq ** 2 * cs ** 2 * k ** 3 * w2i ** 2 * vgsol
                + w3r * rhogeq * w2r * w3i ** 2 * cs ** 2 * k ** 2 * rhogsol
                - w3r * rhogeq ** 2 * w3i ** 2 * k * w1r ** 2 * vgsol
                + w3r * rhogeq * w3i ** 2 * cs ** 2 * k ** 2 * rhogsol * w1r
                + 2 * w3r * rhogeq * cs ** 2 * k ** 3 * w3i * Kdrag * vdsol
                - w3r * rhogeq * cs ** 4 * k ** 4 * w2r * rhogsol
                + w3r * rhogeq ** 2 * cs ** 2 * k ** 3 * w2r ** 2 * vgsol
                + w3r ** 4 * rhogeq ** 2 * w2r * vgsol * k
                - w3r ** 4 * rhogeq * w2r * w1r * rhogsol
                + w3r ** 4 * rhogeq ** 2 * k * w1r * vgsol
                - w3r ** 4 * rhogeq * rhogsol * k ** 2 * cs ** 2
                + w3r ** 4 * rhogeq * w2i * rhogsol * w1i
                + 2 * w3r * rhogeq * w2i * w3i * w1i * k * Kdrag * vgsol
                + 2 * w3r * rhogeq ** 2 * w2i * k * vgsol * w3i * w1r ** 2
                + 2 * w3r * rhogeq * w2r * w3i * w1r * k * Kdrag * vdsol
                - 2 * w3r * rhogeq * w2i * w3i ** 2 * k * Kdrag * vgsol
                - w3r * w3i ** 2 * Kdrag ** 2 * k * vgsol
                + w3r * w3i ** 2 * Kdrag ** 2 * k * vdsol
                - w3r * rhogeq ** 2 * w1i ** 2 * k * vgsol * w3i ** 2
                - 2 * w3r * rhogeq ** 2 * w3i ** 2 * k * w2i * w1i * vgsol
                + 2 * w3r * rhogeq ** 2 * w2i * w1i ** 2 * w3i * k * vgsol
                + w3r * w3i ** 2 * Kdrag * rhogsol * w1i * w2r
                + w3r * w3i ** 2 * Kdrag * rhogsol * w2i * w1r
                - w3r * Kdrag * rhogsol * k ** 2 * cs ** 2 * w1i * w2r
                - w3r * Kdrag * rhogsol * k ** 2 * cs ** 2 * w2i * w1r
                + w3r * Kdrag * k * rhogeq * vgsol * w2i * w1r ** 2
                + w3r * Kdrag * k * rhogeq * vgsol * w2i * w1i ** 2
                + w3r * Kdrag ** 2 * k * vdsol * w2r * w1r
                - w3r * Kdrag ** 2 * k * vdsol * w2i * w1i
                - w3r * Kdrag ** 2 * k * vgsol * w2r * w1r
                + w3r * Kdrag ** 2 * k * vgsol * w2i * w1i
                + w3r * Kdrag * k * rhogeq * vgsol * w2r ** 2 * w1i
                + w3r * Kdrag * k * rhogeq * vgsol * w1i * w2i ** 2
                - w3r * rhogeq ** 2 * w2r ** 2 * w3i ** 2 * k * vgsol
                - 2 * w3r * rhogeq * w2r * w3i * w1r * k * Kdrag * vgsol
                + w3r * rhogeq * w2i * w3i ** 2 * k * Kdrag * vdsol
                - 2 * w3r * rhogeq ** 2 * w2r * w3i ** 2 * k * w1r * vgsol
                + w3r * rhogeq * w2r ** 2 * w3i ** 2 * w1r * rhogsol
                - 2 * w3r * rhogeq * w1i * w3i ** 2 * k * Kdrag * vgsol
                + w3r * rhogeq * w1i * w3i ** 2 * k * Kdrag * vdsol
                + 2 * w3r * rhogeq ** 2 * w3i * w2r ** 2 * w1i * k * vgsol
                + w3r * rhogeq * w1i ** 2 * w3i ** 2 * w2r * rhogsol
                - w3r * rhogeq ** 2 * w3i ** 2 * w2i ** 2 * k * vgsol
                - w3r * rhogeq * cs ** 2 * k ** 2 * w1i ** 2 * w2r * rhogsol
                - w3r * rhogeq * cs ** 2 * k ** 2 * w2r * w1r ** 2 * rhogsol
                - w3r * rhogeq * cs ** 2 * k ** 2 * w2r ** 2 * w1r * rhogsol
                + w3r * rhogeq * w2i ** 2 * w3i ** 2 * w1r * rhogsol
                + w3r * rhogeq * w2r * w3i ** 2 * w1r ** 2 * rhogsol
                - 2 * w3r * rhogeq ** 2 * cs ** 2 * k ** 3 * w3i * w2i * vgsol
                - w3r * rhogeq * w2i ** 2 * w1r * cs ** 2 * k ** 2 * rhogsol
                + 2 * w3r * rhogeq ** 2 * w2i ** 2 * w3i * w1i * k * vgsol
                + w3r * rhogeq * cs ** 2 * k ** 3 * w1i * Kdrag * vgsol
                - w3r * rhogeq * cs ** 2 * k ** 3 * w2i * Kdrag * vdsol
                + w3r * rhogeq * cs ** 2 * k ** 3 * w2i * Kdrag * vgsol
                - w3r * rhogeq * cs ** 2 * k ** 3 * w1i * Kdrag * vdsol
                + w3r * rhogeq ** 2 * cs ** 2 * k ** 3 * w1i ** 2 * vgsol
                - 2 * w3r * rhogeq ** 2 * cs ** 2 * k ** 3 * w3i * w1i * vgsol
                + w3r * rhogeq ** 2 * cs ** 2 * k ** 3 * w1r ** 2 * vgsol
                - w3r * rhogeq * cs ** 4 * k ** 4 * w1r * rhogsol
                - 2 * w3r * rhogeq * cs ** 2 * k ** 3 * w3i * Kdrag * vgsol
                + 2 * w3r * rhogeq ** 2 * cs ** 2 * k ** 3 * w2r * w1r * vgsol
                - 2 * w3r * rhogeq * w2i * w3i * w1i * k * Kdrag * vdsol
                + 2 * w3r * rhogeq ** 2 * cs ** 2 * k ** 3 * w2i * w1i * vgsol
                + w3r ** 2 * w3i * rhogeq * rhogsol * k ** 2 * cs ** 2 * w1i
                - w3r ** 2 * w3i * rhogeq * rhogsol * w2r ** 2 * w1i
                - w3r ** 2 * w3i * rhogeq * rhogsol * w1i * w2i ** 2
                - w3r ** 2 * w3i * rhogeq * rhogsol * w2i * w1r ** 2
                - w3r ** 2 * w3i * rhogeq * rhogsol * w2i * w1i ** 2
                - w3r ** 2 * w3i * rhogeq * k * Kdrag * vdsol * w2r
                - w3r ** 2 * Kdrag * rhogsol * w3i * w2r * w1r
                + w3r ** 2 * Kdrag * rhogsol * w2i * w3i * w1i
                - w3r ** 2 * w3i * rhogeq * k * Kdrag * vdsol * w1r
                + w3r ** 2 * w3i * rhogeq * rhogsol * k ** 2 * cs ** 2 * w2i
                - w3r ** 2 * Kdrag * rhogsol * k ** 2 * cs ** 2 * w3i
                + w3r ** 2 * Kdrag * rhogsol * k ** 2 * cs ** 2 * w2i
                + w3r ** 2 * Kdrag * rhogsol * k ** 2 * cs ** 2 * w1i
                + 2 * w3r ** 2 * Kdrag * k * rhogeq * vgsol * w2r * w3i
                + 2 * w3r ** 2 * Kdrag * k * rhogeq * vgsol * w3i * w1r
                + 2 * w3r ** 2 * rhogeq ** 2 * w2r * k * vgsol * w3i ** 2
                - w3r ** 2 * rhogeq ** 2 * cs ** 2 * k ** 3 * w2r * vgsol
                + w3r ** 2 * rhogeq * w2r * w1i * k * Kdrag * vgsol
                + w3r ** 2 * rhogeq ** 2 * w2r * w1i ** 2 * k * vgsol
                - w3r ** 2 * rhogeq * w2r * w1i * k * Kdrag * vdsol
                + w3r ** 2 * rhogeq ** 2 * w2r * w1r ** 2 * k * vgsol
                - w3r ** 2 * rhogeq ** 2 * cs ** 2 * k ** 3 * w1r * vgsol
                - w3r ** 2 * Kdrag ** 2 * k * vdsol * w2r
                - w3r ** 2 * rhogeq * w2i ** 2 * w1i ** 2 * rhogsol
                - w3r ** 2 * rhogeq * w1r ** 2 * w2i ** 2 * rhogsol
                - w3r ** 2 * rhogeq * w2r ** 2 * w1r ** 2 * rhogsol
                + w3r ** 2 * rhogeq * cs ** 4 * k ** 4 * rhogsol
                - w3r ** 2 * rhogeq * w1i ** 2 * w2r ** 2 * rhogsol
                - w3r ** 2 * Kdrag * rhogsol * w1i * w2i ** 2
                - w3r ** 2 * Kdrag * rhogsol * w2i * w1r ** 2
                - w3r ** 2 * Kdrag * rhogsol * w2i * w1i ** 2
                - w3r ** 2 * Kdrag * rhogsol * w2r ** 2 * w1i
                - w3r ** 2 * Kdrag ** 2 * k * vdsol * w1r
                + w3r ** 2 * Kdrag ** 2 * k * vgsol * w2r
                + w3r ** 2 * Kdrag ** 2 * k * vgsol * w1r
                + w3r ** 3 * Kdrag * rhogsol * w2i * w1r
                + w3r ** 3 * rhogeq * w2i * k * Kdrag * vdsol
                - w3r ** 3 * rhogeq ** 2 * w2i ** 2 * k * vgsol
                + w3r ** 2 * rhogeq ** 2 * w2i ** 2 * w1r * k * vgsol
                - 2 * w3r ** 2 * rhogeq * w2r * w3i ** 2 * w1r * rhogsol
                + 2 * w3r ** 2 * rhogeq ** 2 * w3i ** 2 * k * w1r * vgsol
                + 2 * w3r ** 2 * rhogeq * w1i * w3i ** 2 * rhogsol * w2i
                - w3r ** 2 * rhogeq * w2i * w1r * k * Kdrag * vdsol
                + w3r ** 2 * rhogeq * w2i * w1r * k * Kdrag * vgsol
                - w3r ** 3 * rhogeq ** 2 * w1i ** 2 * k * vgsol
                - 2 * w3r ** 3 * rhogeq ** 2 * w2r * k * w1r * vgsol
                + w3r ** 3 * rhogeq * w2r ** 2 * w1r * rhogsol
                + w3r ** 3 * rhogeq * w2r * cs ** 2 * k ** 2 * rhogsol
                - 2 * w3r ** 3 * rhogeq * w2i * k * Kdrag * vgsol
                - w3r ** 3 * rhogeq ** 2 * w1r ** 2 * k * vgsol
                + w3r ** 3 * rhogeq * w1r * cs ** 2 * k ** 2 * rhogsol
                - w3r ** 3 * rhogeq ** 2 * w2r ** 2 * k * vgsol
                + w3r ** 3 * rhogeq * w1i * k * Kdrag * vdsol
                - 2 * w3r ** 3 * rhogeq ** 2 * k * w2i * w1i * vgsol
                + w3r ** 3 * rhogeq * w2r * w1r ** 2 * rhogsol
                + w3r ** 3 * rhogeq * w1i ** 2 * w2r * rhogsol
                + w3r ** 3 * rhogeq * w2i ** 2 * w1r * rhogsol
                - 2 * w3r ** 3 * rhogeq * w1i * k * Kdrag * vgsol
                + w3r ** 3 * Kdrag * rhogsol * w1i * w2r
                + w3r ** 2 * rhogeq ** 2 * w2r ** 2 * k * w1r * vgsol
                - 2 * w3r ** 2 * rhogeq * w3i ** 2 * cs ** 2 * k ** 2 * rhogsol
                + w3r ** 3 * Kdrag ** 2 * k * vdsol
                + w3i ** 2 * rhogeq ** 2 * cs ** 2 * k ** 3 * w2r * vgsol
                - w3r ** 3 * Kdrag ** 2 * k * vgsol
                + w3i ** 2 * rhogeq * w2i * w1r * k * Kdrag * vdsol
                - w3i ** 2 * Kdrag ** 2 * k * vdsol * w2r
                + w3i ** 2 * Kdrag ** 2 * k * vgsol * w2r
                - w3i ** 2 * Kdrag * rhogsol * w2r ** 2 * w1i
                - w3i ** 2 * Kdrag * rhogsol * w1i * w2i ** 2
                - w3i ** 2 * Kdrag * rhogsol * w2i * w1r ** 2
                - w3i ** 2 * Kdrag * rhogsol * w2i * w1i ** 2
                + w3i ** 2 * Kdrag * rhogsol * k ** 2 * cs ** 2 * w2i
                + w3i ** 2 * Kdrag * rhogsol * k ** 2 * cs ** 2 * w1i
                - Kdrag * k * rhogeq * vgsol * w3i * w1r * w2r ** 2
                - Kdrag * k * rhogeq * vgsol * w3i * w2r * w1r ** 2
                - Kdrag * k * rhogeq * vgsol * w3i * w2r * w1i ** 2
                - Kdrag * k * rhogeq * vgsol * w3i * w1r * w2i ** 2
                + Kdrag * rhogsol * k ** 2 * cs ** 2 * w3i * w2r * w1r
                - Kdrag * rhogsol * k ** 2 * cs ** 2 * w2i * w3i * w1i
                + Kdrag * rhogsol * w2r ** 2 * w1i ** 2 * w3i
                + Kdrag * rhogsol * w3i * w1r ** 2 * w2r ** 2
                + Kdrag * rhogsol * w1i ** 2 * w2i ** 2 * w3i
                + Kdrag * rhogsol * w2i ** 2 * w1r ** 2 * w3i
                + Kdrag ** 2 * k * vdsol * w2r * w1i * w3i
                + Kdrag ** 2 * k * vdsol * w2i * w3i * w1r
                - Kdrag ** 2 * k * vgsol * w2r * w1i * w3i
                - Kdrag ** 2 * k * vgsol * w2i * w3i * w1r
                - w3i ** 2 * Kdrag ** 2 * k * vdsol * w1r
                + rhogeq * cs ** 2 * k ** 3 * w2i * w1r * Kdrag * vdsol
                - rhogeq * cs ** 2 * k ** 3 * w2i * w1r * Kdrag * vgsol
                - w3i ** 2 * rhogeq ** 2 * w2i ** 2 * w1r * k * vgsol
                - w3i ** 2 * rhogeq ** 2 * w2r ** 2 * k * w1r * vgsol
                - w3i ** 2 * rhogeq * w2i * w1r * k * Kdrag * vgsol
                - w3i ** 2 * rhogeq * w2r * w1i * k * Kdrag * vgsol
                - w3i ** 2 * rhogeq ** 2 * w2r * w1i ** 2 * k * vgsol
                + w3i ** 2 * rhogeq * w2r * w1i * k * Kdrag * vdsol
                - w3i ** 2 * rhogeq ** 2 * w2r * w1r ** 2 * k * vgsol
                + w3i ** 2 * rhogeq ** 2 * cs ** 2 * k ** 3 * w1r * vgsol
                + w3i ** 2 * rhogeq * w2i ** 2 * w1i ** 2 * rhogsol
                + w3i ** 2 * rhogeq * w1r ** 2 * w2i ** 2 * rhogsol
                + w3i ** 2 * rhogeq * w2r ** 2 * w1r ** 2 * rhogsol
                - w3i ** 2 * rhogeq * cs ** 4 * k ** 4 * rhogsol
                + w3i ** 2 * rhogeq * w1i ** 2 * w2r ** 2 * rhogsol
            )
            / (
                w1i ** 2
                - 2 * w3i * w1i
                + w3r ** 2
                + w1r ** 2
                + w3i ** 2
                - 2 * w3r * w1r
            )
            / (
                w2r ** 2
                - 2 * w3r * w2r
                + w2i ** 2
                + w3i ** 2
                - 2 * w2i * w3i
                + w3r ** 2
            )
            / k
            / rhogeq
            / Kdrag
        )

        vd3i = (
            -(
                cs ** 2 * k ** 3 * rhogeq * w3i ** 2 * Kdrag * vgsol
                - cs ** 2 * k ** 3 * rhogeq * w3i ** 2 * Kdrag * vdsol
                - 2 * cs ** 4 * k ** 4 * rhogeq * w3r * w3i * rhogsol
                - 4 * cs ** 2 * k ** 2 * rhogeq * w3r * w3i * w2r * w1r * rhogsol
                + 4 * cs ** 2 * k ** 2 * rhogeq * w3r * w3i * w2i * rhogsol * w1i
                - cs ** 2 * k ** 3 * rhogeq * w3r ** 2 * Kdrag * vgsol
                + cs ** 2 * k ** 3 * rhogeq * w3r ** 2 * Kdrag * vdsol
                - 2 * cs ** 2 * k ** 3 * rhogeq ** 2 * w3i * w2r * w1r * vgsol
                + 2 * cs ** 2 * k ** 3 * rhogeq ** 2 * w3r * w3i * w2r * vgsol
                + 2 * cs ** 2 * k ** 3 * rhogeq ** 2 * w3r * w3i * w1r * vgsol
                - w3r ** 2 * w1i * k * Kdrag ** 2 * vgsol
                + w3r ** 2 * w1i * k * Kdrag ** 2 * vdsol
                + rhogeq * w3i * w3r ** 2 * w1i * k * Kdrag * vdsol
                - rhogeq ** 2 * w3i * w3r ** 2 * w1i ** 2 * k * vgsol
                + 2 * rhogeq ** 2 * w3i ** 2 * w3r ** 2 * w1i * k * vgsol
                - rhogeq ** 2 * w3i * w3r ** 2 * w1r ** 2 * k * vgsol
                + rhogeq * w3i * w3r ** 2 * w1r * cs ** 2 * k ** 2 * rhogsol
                + 2 * rhogeq * w3i ** 2 * w3r ** 2 * k * Kdrag * vgsol
                - 2 * rhogeq * w3i ** 2 * w3r ** 2 * k * Kdrag * vdsol
                + rhogeq * w3i * w2r * w3r ** 2 * cs ** 2 * k ** 2 * rhogsol
                + rhogeq * w3i ** 3 * w2r * cs ** 2 * k ** 2 * rhogsol
                - rhogeq ** 2 * w3i ** 3 * k * w1r ** 2 * vgsol
                + rhogeq * w3i ** 3 * cs ** 2 * k ** 2 * rhogsol * w1r
                - 2 * rhogeq ** 2 * w3i * w3r ** 2 * k * w2i * w1i * vgsol
                - rhogeq ** 2 * w3i ** 3 * w1i ** 2 * k * vgsol
                - 2 * rhogeq ** 2 * w3i ** 3 * k * w2i * w1i * vgsol
                - rhogeq ** 2 * w3i * w3r ** 2 * w2i ** 2 * k * vgsol
                + rhogeq ** 2 * w3i ** 2 * w2i * w1i ** 2 * k * vgsol
                - rhogeq * w3i ** 2 * w2i * w1i * k * Kdrag * vdsol
                + 2 * rhogeq ** 2 * w3i ** 2 * w3r ** 2 * w2i * k * vgsol
                - 2 * rhogeq * w3i ** 2 * w3r ** 2 * w2i * w1r * rhogsol
                - 2 * rhogeq * w3i ** 2 * w2r * w3r ** 2 * rhogsol * w1i
                - rhogeq * w3i ** 4 * w2r * w1i * rhogsol
                + rhogeq * w3i * w3r ** 2 * w2i * k * Kdrag * vdsol
                + w3r ** 2 * w3i * k * Kdrag ** 2 * vgsol
                - w3r ** 2 * w3i * k * Kdrag ** 2 * vdsol
                - w2i * w3i ** 2 * k * Kdrag ** 2 * vgsol
                + w2i * w3i ** 2 * k * Kdrag ** 2 * vdsol
                - w1i * w3i ** 2 * k * Kdrag ** 2 * vgsol
                + w1i * w3i ** 2 * k * Kdrag ** 2 * vdsol
                + rhogeq ** 2 * w3i ** 2 * w2i * k * vgsol * w1r ** 2
                + rhogeq * w3i ** 2 * w2r * w1r * k * Kdrag * vdsol
                + rhogeq * w3i ** 3 * w2i * k * Kdrag * vdsol
                - 2 * rhogeq ** 2 * w3i ** 3 * w2r * k * w1r * vgsol
                + rhogeq ** 2 * w3i ** 4 * w2i * k * vgsol
                + 2 * rhogeq ** 2 * w3i * w3r * w2i ** 2 * w1r * k * vgsol
                + rhogeq * w3i ** 3 * w1i ** 2 * w2r * rhogsol
                - rhogeq ** 2 * w3i ** 3 * w2i ** 2 * k * vgsol
                - 2 * rhogeq * w3i * w3r * w1i ** 2 * w2r ** 2 * rhogsol
                - 2 * rhogeq * w3i * w3r * w2i ** 2 * w1i ** 2 * rhogsol
                + rhogeq * w3i * w3r ** 2 * w1i ** 2 * w2r * rhogsol
                + rhogeq * w3i * w3r ** 2 * w2i ** 2 * w1r * rhogsol
                - rhogeq ** 2 * w3i ** 3 * w2r ** 2 * k * vgsol
                + rhogeq * w3i * w3r ** 2 * w2r * w1r ** 2 * rhogsol
                - 2 * rhogeq ** 2 * w3i * w3r ** 2 * w2r * k * w1r * vgsol
                + rhogeq * w3i ** 3 * w2r ** 2 * w1r * rhogsol
                + rhogeq ** 2 * w3i ** 4 * w1i * k * vgsol
                + rhogeq * w3i ** 3 * w1i * k * Kdrag * vdsol
                - rhogeq * w3i ** 4 * k * Kdrag * vdsol
                + rhogeq * w3i ** 4 * k * Kdrag * vgsol
                - rhogeq * w3i ** 4 * w1r * w2i * rhogsol
                - Kdrag * w2r * w3r * w3i ** 2 * w1r * rhogsol
                - Kdrag * w1r * w2i * w3i ** 3 * rhogsol
                + Kdrag * w2r * w3i ** 2 * cs ** 2 * k ** 2 * rhogsol
                - Kdrag * w3i ** 2 * k * rhogeq * w1r ** 2 * vgsol
                + Kdrag * w3i ** 2 * cs ** 2 * k ** 2 * rhogsol * w1r
                - Kdrag * w1i ** 2 * rhogeq * k * vgsol * w3i ** 2
                + Kdrag * w3r * w1i * w3i ** 2 * rhogsol * w2i
                - Kdrag * w3r ** 2 * w1i ** 2 * k * rhogeq * vgsol
                - Kdrag * w3r ** 2 * w1r ** 2 * rhogeq * k * vgsol
                + Kdrag * w3r ** 2 * w1r * cs ** 2 * k ** 2 * rhogsol
                + Kdrag * w2r * w3r ** 2 * cs ** 2 * k ** 2 * rhogsol
                - 2 * rhogeq * w3i * w3r * w2r * w1i * k * Kdrag * vdsol
                + 2 * rhogeq ** 2 * w3i * w3r * w2r * w1r ** 2 * k * vgsol
                + rhogeq * w3i ** 3 * w2i ** 2 * w1r * rhogsol
                + rhogeq * w3i ** 3 * w2r * w1r ** 2 * rhogsol
                - Kdrag * w3r * w3i ** 2 * cs ** 2 * k ** 2 * rhogsol
                - Kdrag * w3r ** 3 * rhogsol * k ** 2 * cs ** 2
                + Kdrag * w3r ** 3 * w2i * rhogsol * w1i
                - Kdrag * w3r ** 2 * w2i * w3i * w1r * rhogsol
                - Kdrag * w2r * w3r ** 2 * w3i * rhogsol * w1i
                - Kdrag * w2r * w1i * w3i ** 3 * rhogsol
                + 2 * rhogeq ** 2 * w3i * w3r * w2r * w1i ** 2 * k * vgsol
                - Kdrag * w2r * w3r ** 3 * w1r * rhogsol
                + rhogeq * w3r ** 3 * k * Kdrag * vdsol * w2r
                - rhogeq ** 2 * w3i * w3r ** 2 * w2r ** 2 * k * vgsol
                + rhogeq * w3i * w3r ** 2 * w2r ** 2 * w1r * rhogsol
                - 2 * rhogeq * w3i * w3r * w1r ** 2 * w2i ** 2 * rhogsol
                - 2 * rhogeq * w3i * w3r * w2r ** 2 * w1r ** 2 * rhogsol
                - 2 * rhogeq * w3i * w3r * w2i * w1r * k * Kdrag * vdsol
                + 2 * rhogeq * w3i * w3r * w2i * w1r * k * Kdrag * vgsol
                + 2 * rhogeq ** 2 * w3i * w3r * w2r ** 2 * k * w1r * vgsol
                + 2 * rhogeq * w3i * w3r * w2r * w1i * k * Kdrag * vgsol
                + rhogeq ** 2 * w3r ** 4 * k * vgsol * w2i
                - rhogeq * w3r ** 4 * rhogsol * w1i * w2r
                - rhogeq * w3r ** 4 * rhogsol * w2i * w1r
                + rhogeq * w3r ** 3 * rhogsol * w2r ** 2 * w1i
                + rhogeq * w3r ** 3 * rhogsol * w1i * w2i ** 2
                + rhogeq * w3r ** 3 * rhogsol * w2i * w1r ** 2
                + rhogeq * w3r ** 3 * rhogsol * w2i * w1i ** 2
                + rhogeq * w3r * rhogsol * w2r ** 2 * w1i * w3i ** 2
                - rhogeq * w3r ** 4 * k * Kdrag * vdsol
                - rhogeq * w3r ** 2 * k * Kdrag * vdsol * w2r * w1r
                + rhogeq * w3r ** 2 * k * Kdrag * vdsol * w2i * w1i
                + rhogeq * w3r * k * Kdrag * vdsol * w2r * w3i ** 2
                + rhogeq * w3r * k * Kdrag * vdsol * w3i ** 2 * w1r
                - rhogeq * w3r * rhogsol * k ** 2 * cs ** 2 * w3i ** 2 * w1i
                - rhogeq * w3r * rhogsol * k ** 2 * cs ** 2 * w3i ** 2 * w2i
                - rhogeq ** 2 * w3r ** 2 * k * vgsol * w2i * w1r ** 2
                - rhogeq ** 2 * w3r ** 2 * k * vgsol * w2i * w1i ** 2
                + rhogeq ** 2 * w3r ** 4 * k * vgsol * w1i
                + rhogeq * w3r ** 4 * k * Kdrag * vgsol
                + rhogeq * w3r * rhogsol * w3i ** 2 * w2i * w1r ** 2
                + rhogeq * w3r * rhogsol * w2i * w1i ** 2 * w3i ** 2
                + rhogeq * w3r * rhogsol * w2i ** 2 * w3i ** 2 * w1i
                - rhogeq * w3r ** 3 * rhogsol * k ** 2 * cs ** 2 * w2i
                - rhogeq * w3r ** 3 * rhogsol * k ** 2 * cs ** 2 * w1i
                + rhogeq * w3r ** 3 * k * Kdrag * vdsol * w1r
                + w3r ** 2 * w2i * k * Kdrag ** 2 * vdsol
                - w3r ** 2 * w2i * k * Kdrag ** 2 * vgsol
                - w3i ** 3 * k * Kdrag ** 2 * vdsol
                + w3i ** 3 * k * Kdrag ** 2 * vgsol
                - w3r * Kdrag * rhogsol * w2r ** 2 * w1r ** 2
                - w3r * Kdrag * rhogsol * w2r ** 2 * w1i ** 2
                + w3r * Kdrag * k * rhogeq * vgsol * w2r * w1i ** 2
                + w3r * Kdrag * k * rhogeq * vgsol * w2r * w1r ** 2
                - w3r * rhogeq * cs ** 2 * k ** 2 * rhogsol * w2r ** 2 * w1i
                + w3r * cs ** 2 * k ** 3 * rhogeq * Kdrag * vgsol * w2r
                - w3r * Kdrag * cs ** 2 * k ** 2 * rhogsol * w2r * w1r
                - w3r * cs ** 2 * k ** 3 * rhogeq * Kdrag * vdsol * w2r
                + w3r ** 2 * Kdrag * rhogsol * w2r * w1r ** 2
                + 2 * w3r ** 2 * cs ** 2 * k ** 2 * rhogeq * rhogsol * w2r * w1i
                - w3r ** 2 * rhogeq ** 2 * k * vgsol * w2r ** 2 * w1i
                - w3r ** 2 * Kdrag * rhogeq * k * vgsol * w2r * w1r
                - w3r ** 2 * Kdrag * rhogeq * k * vgsol * w2r ** 2
                - w3r ** 2 * Kdrag * rhogeq * k * vgsol * w2i ** 2
                - w3r ** 2 * rhogeq ** 2 * cs ** 2 * k ** 3 * vgsol * w1i
                + w3r * Kdrag ** 2 * k * vgsol * w2r * w1i
                + w3i ** 2 * rhogeq ** 2 * k * vgsol * w2r ** 2 * w1i
                + rhogeq * cs ** 2 * k ** 3 * vdsol * Kdrag * w2r * w1r
                + w3i * Kdrag * k * rhogeq * vgsol * w2r ** 2 * w1i
                + w3i * cs ** 2 * k ** 2 * rhogeq * rhogsol * w2r ** 2 * w1r
                + w3i * cs ** 2 * k ** 2 * rhogeq * rhogsol * w2r * w1r ** 2
                + w3i * cs ** 2 * k ** 2 * rhogeq * rhogsol * w2r * w1i ** 2
                - 3 * w3i ** 2 * Kdrag * rhogeq * k * vgsol * w2r * w1r
                - w3i ** 2 * Kdrag * rhogeq * k * vgsol * w2i ** 2
                - w3i * rhogeq ** 2 * cs ** 2 * k ** 3 * vgsol * w2r ** 2
                - 2 * w3i ** 2 * cs ** 2 * k ** 2 * rhogeq * rhogsol * w2r * w1i
                - w3i ** 2 * Kdrag * rhogeq * k * vgsol * w2r ** 2
                - rhogeq * cs ** 4 * k ** 4 * rhogsol * w2r * w1i
                + w3i * rhogeq * cs ** 4 * k ** 4 * rhogsol * w2r
                - w3i * Kdrag * cs ** 2 * k ** 2 * rhogsol * w2r * w1i
                - cs ** 2 * k ** 3 * rhogeq * Kdrag * vgsol * w2r * w1r
                + w3i ** 2 * cs ** 2 * k ** 3 * rhogeq ** 2 * vgsol * w1i
                + w3i ** 2 * Kdrag * rhogsol * w2r ** 2 * w1r
                + w3i * Kdrag ** 2 * k * vdsol * w2r * w1r
                + w3i ** 2 * Kdrag * rhogsol * w2r * w1r ** 2
                + w3i ** 2 * Kdrag * rhogsol * w2r * w1i ** 2
                - w3i * Kdrag ** 2 * k * vgsol * w2r * w1r
                + w3r ** 2 * Kdrag * rhogsol * w2r * w1i ** 2
                + w3r ** 2 * Kdrag * rhogsol * w2r ** 2 * w1r
                + rhogeq ** 2 * cs ** 2 * k ** 3 * vgsol * w1i * w2r ** 2
                + w3r * Kdrag * k * rhogeq * vgsol * w2r ** 2 * w1r
                - w3r * Kdrag ** 2 * k * vdsol * w2r * w1i
                - w3r * rhogeq * cs ** 2 * k ** 2 * rhogsol * w2i * w1r ** 2
                + w3r * cs ** 4 * k ** 4 * rhogeq * rhogsol * w1i
                + w3r * Kdrag * cs ** 2 * k ** 2 * rhogsol * w1i * w2i
                - w3r * cs ** 2 * k ** 3 * rhogeq * Kdrag * vdsol * w1r
                - w3i * Kdrag ** 2 * k * vdsol * w1i * w2i
                + w3i * Kdrag ** 2 * k * vgsol * w1i * w2i
                + w3r * Kdrag ** 2 * k * vgsol * w2i * w1r
                - cs ** 2 * k ** 3 * rhogeq * Kdrag * vdsol * w1i * w2i
                - w3i ** 2 * Kdrag * rhogeq * k * vgsol * w1i * w2i
                + w3r * Kdrag * k * rhogeq * vgsol * w2i ** 2 * w1r
                + w3i * rhogeq * k * Kdrag * vgsol * w2i ** 2 * w1i
                + w3i * rhogeq * k * Kdrag * vgsol * w2i * w1r ** 2
                + w3i * rhogeq * k * Kdrag * vgsol * w1i ** 2 * w2i
                - w3i * rhogeq ** 2 * cs ** 2 * k ** 3 * vgsol * w1r ** 2
                - w3i * rhogeq ** 2 * cs ** 2 * k ** 3 * vgsol * w1i ** 2
                - 3 * w3r ** 2 * Kdrag * k * rhogeq * vgsol * w1i * w2i
                + w3r * cs ** 2 * k ** 3 * rhogeq * Kdrag * vgsol * w1r
                - w3i * Kdrag * cs ** 2 * k ** 2 * rhogsol * w2i * w1r
                - rhogeq * cs ** 4 * k ** 4 * rhogsol * w2i * w1r
                + 2 * w3r ** 2 * rhogeq * cs ** 2 * k ** 2 * rhogsol * w2i * w1r
                - w3r ** 2 * vgsol * rhogeq ** 2 * k ** 3 * cs ** 2 * w2i
                + w3i * rhogeq * cs ** 2 * k ** 3 * vdsol * Kdrag * w2i
                + w3i * rhogeq * cs ** 2 * k ** 3 * vdsol * Kdrag * w1i
                + cs ** 2 * k ** 3 * rhogeq * Kdrag * vgsol * w1i * w2i
                - w3i * cs ** 2 * k ** 3 * rhogeq * Kdrag * vgsol * w1i
                - w3i * cs ** 2 * k ** 3 * rhogeq * Kdrag * vgsol * w2i
                + w3i * rhogeq * cs ** 4 * k ** 4 * rhogsol * w1r
                + w3i ** 2 * vgsol * rhogeq ** 2 * k ** 3 * cs ** 2 * w2i
                + w3i ** 2 * vgsol * k * rhogeq ** 2 * w2i ** 2 * w1i
                - 2 * w3i ** 2 * rhogeq * cs ** 2 * k ** 2 * rhogsol * w2i * w1r
                - w3r ** 2 * vgsol * k * rhogeq ** 2 * w2i ** 2 * w1i
                - w3i * rhogeq ** 2 * cs ** 2 * k ** 3 * vgsol * w2i ** 2
                - 2 * w3i * rhogeq ** 2 * cs ** 2 * k ** 3 * vgsol * w2i * w1i
                - w3r * Kdrag * rhogsol * w2i ** 2 * w1r ** 2
                + rhogeq ** 2 * cs ** 2 * k ** 3 * vgsol * w1i ** 2 * w2i
                + rhogeq ** 2 * cs ** 2 * k ** 3 * vgsol * w2i * w1r ** 2
                - w3r * Kdrag ** 2 * k * vdsol * w2i * w1r
                + w3r * cs ** 4 * k ** 4 * rhogeq * rhogsol * w2i
                - w3r * rhogeq * cs ** 2 * k ** 2 * rhogsol * w1i ** 2 * w2i
                - w3r * rhogeq * cs ** 2 * k ** 2 * rhogsol * w2i ** 2 * w1i
                - w3r * Kdrag * rhogsol * w2i ** 2 * w1i ** 2
                + w3i * rhogeq * cs ** 2 * k ** 2 * rhogsol * w2i ** 2 * w1r
                + w3i ** 2 * Kdrag * rhogsol * w2i ** 2 * w1r
                + w3r ** 2 * Kdrag * rhogsol * w2i ** 2 * w1r
                + rhogeq ** 2 * cs ** 2 * k ** 3 * vgsol * w1i * w2i ** 2
            )
            / Kdrag
            / rhogeq
            / k
            / (
                w2r ** 2
                - 2 * w3r * w2r
                + w2i ** 2
                + w3i ** 2
                - 2 * w2i * w3i
                + w3r ** 2
            )
            / (
                w1i ** 2
                - 2 * w3i * w1i
                + w3r ** 2
                + w1r ** 2
                + w3i ** 2
                - 2 * w3r * w1r
            )
        )

        vd2r = (
            -(
                -rhogeq * w1i * rhogsol * k ** 2 * cs ** 2 * w3i ** 2 * w2i
                + rhogeq * w1i ** 2 * rhogsol * k ** 2 * cs ** 2 * w3r ** 2
                + rhogeq * w1i ** 2 * rhogsol * k ** 2 * cs ** 2 * w3i ** 2
                - rhogeq * w1i * rhogsol * k ** 2 * cs ** 2 * w3r ** 2 * w2i
                + rhogeq ** 2 * w1i ** 2 * k * vgsol * w3r * w2r ** 2
                - rhogeq ** 2 * w1i ** 2 * k * vgsol * w3r * w2i ** 2
                - rhogeq ** 2 * w2r ** 3 * k * vgsol * w3i ** 2
                + rhogeq * w2r ** 3 * w3i ** 2 * w1r * rhogsol
                + 2 * rhogeq ** 2 * w1r * w2r ** 2 * w2i ** 2 * k * vgsol
                - rhogeq ** 2 * w1r ** 2 * w2r * k * vgsol * w2i ** 2
                - 2 * cs ** 2 * k ** 3 * rhogeq ** 2 * w2r * vgsol * w2i * w1i
                + cs ** 2 * k ** 2 * rhogeq * w2r * w1r * rhogsol * w2i ** 2
                + cs ** 2 * k ** 2 * rhogeq * w2r ** 3 * w1r * rhogsol
                - w3i * cs ** 2 * k ** 3 * rhogeq * Kdrag * vdsol * w2r
                + w3i * cs ** 2 * k ** 3 * rhogeq * Kdrag * vdsol * w1r
                + w3i * cs ** 2 * k ** 3 * rhogeq * Kdrag * vgsol * w2r
                - w3i * cs ** 2 * k ** 3 * rhogeq * Kdrag * vgsol * w1r
                + rhogeq ** 2 * cs ** 2 * k ** 3 * w2i ** 2 * w1r * vgsol
                - rhogeq * cs ** 4 * k ** 4 * w2r * w1r * rhogsol
                - rhogeq ** 2 * cs ** 2 * k ** 3 * w2r ** 2 * w1r * vgsol
                + rhogeq * cs ** 2 * k ** 3 * w2r * w1i * Kdrag * vgsol
                + rhogeq ** 2 * cs ** 2 * k ** 3 * w2r * w1i ** 2 * vgsol
                - rhogeq * cs ** 2 * k ** 3 * w2r * w1i * Kdrag * vdsol
                + rhogeq ** 2 * cs ** 2 * k ** 3 * w2r * w1r ** 2 * vgsol
                + rhogeq * cs ** 4 * k ** 4 * w2i * w1i * rhogsol
                + w3i * cs ** 4 * k ** 4 * rhogeq * rhogsol * w2i
                - w3i * cs ** 4 * k ** 4 * rhogeq * rhogsol * w1i
                - w3i * cs ** 2 * k ** 2 * rhogeq * rhogsol * w2i * w1r ** 2
                - w3i * cs ** 2 * k ** 2 * rhogeq * rhogsol * w2i * w1i ** 2
                + w3r * rhogeq ** 2 * cs ** 2 * k ** 3 * w2i ** 2 * vgsol
                - w3r * rhogeq * cs ** 4 * k ** 4 * w2r * rhogsol
                - w3r * rhogeq ** 2 * cs ** 2 * k ** 3 * w2r ** 2 * vgsol
                - w3r * Kdrag * rhogsol * k ** 2 * cs ** 2 * w1i * w2r
                + w3r * Kdrag * rhogsol * k ** 2 * cs ** 2 * w2i * w1r
                - w3r * Kdrag * k * rhogeq * vgsol * w2i * w1r ** 2
                - w3r * Kdrag * k * rhogeq * vgsol * w2i * w1i ** 2
                + w3r * Kdrag ** 2 * k * vdsol * w2r * w1r
                + w3r * Kdrag ** 2 * k * vdsol * w2i * w1i
                - w3r * Kdrag ** 2 * k * vgsol * w2r * w1r
                - w3r * Kdrag ** 2 * k * vgsol * w2i * w1i
                + w3r * Kdrag * k * rhogeq * vgsol * w2r ** 2 * w1i
                - w3r * Kdrag * k * rhogeq * vgsol * w1i * w2i ** 2
                - w3r * rhogeq * cs ** 2 * k ** 2 * w1i ** 2 * w2r * rhogsol
                - w3r * rhogeq * cs ** 2 * k ** 2 * w2r * w1r ** 2 * rhogsol
                - w3r * rhogeq * cs ** 2 * k ** 3 * w1i * Kdrag * vgsol
                - w3r * rhogeq * cs ** 2 * k ** 3 * w2i * Kdrag * vdsol
                + w3r * rhogeq * cs ** 2 * k ** 3 * w2i * Kdrag * vgsol
                + w3r * rhogeq * cs ** 2 * k ** 3 * w1i * Kdrag * vdsol
                - w3r * rhogeq ** 2 * cs ** 2 * k ** 3 * w1i ** 2 * vgsol
                - w3r * rhogeq ** 2 * cs ** 2 * k ** 3 * w1r ** 2 * vgsol
                + w3r * rhogeq * cs ** 4 * k ** 4 * w1r * rhogsol
                + 2 * w3r * rhogeq ** 2 * cs ** 2 * k ** 3 * w2r * w1r * vgsol
                + w3r ** 2 * rhogeq ** 2 * cs ** 2 * k ** 3 * w2r * vgsol
                + w3r ** 2 * rhogeq * w2r * w1i * k * Kdrag * vgsol
                - w3r ** 2 * rhogeq ** 2 * cs ** 2 * k ** 3 * w1r * vgsol
                + w3r ** 2 * rhogeq * w2i ** 2 * w1i ** 2 * rhogsol
                + w3r ** 2 * rhogeq * w1r ** 2 * w2i ** 2 * rhogsol
                - w3r ** 2 * rhogeq * w2r ** 2 * w1r ** 2 * rhogsol
                - w3r ** 2 * rhogeq * w1i ** 2 * w2r ** 2 * rhogsol
                - w3r ** 2 * Kdrag * rhogsol * w1i * w2i ** 2
                + w3r ** 2 * Kdrag * rhogsol * w2i * w1r ** 2
                + w3r ** 2 * Kdrag * rhogsol * w2i * w1i ** 2
                - w3r ** 2 * Kdrag * rhogsol * w2r ** 2 * w1i
                - w3r ** 2 * rhogeq ** 2 * w2i ** 2 * w1r * k * vgsol
                - w3r ** 2 * rhogeq * w2i * w1r * k * Kdrag * vgsol
                + w3r ** 2 * rhogeq ** 2 * w2r ** 2 * k * w1r * vgsol
                + w3i ** 2 * rhogeq ** 2 * cs ** 2 * k ** 3 * w2r * vgsol
                - w3i ** 2 * Kdrag * rhogsol * w2r ** 2 * w1i
                - w3i ** 2 * Kdrag * rhogsol * w1i * w2i ** 2
                + w3i ** 2 * Kdrag * rhogsol * w2i * w1r ** 2
                + w3i ** 2 * Kdrag * rhogsol * w2i * w1i ** 2
                + Kdrag * k * rhogeq * vgsol * w3i * w1r * w2r ** 2
                + Kdrag * k * rhogeq * vgsol * w3i * w2r * w1r ** 2
                + Kdrag * k * rhogeq * vgsol * w3i * w2r * w1i ** 2
                - Kdrag * k * rhogeq * vgsol * w3i * w1r * w2i ** 2
                - Kdrag * rhogsol * k ** 2 * cs ** 2 * w3i * w2r * w1r
                - Kdrag * rhogsol * k ** 2 * cs ** 2 * w2i * w3i * w1i
                - Kdrag * rhogsol * w2r ** 2 * w1i ** 2 * w3i
                - Kdrag * rhogsol * w3i * w1r ** 2 * w2r ** 2
                - Kdrag * rhogsol * w1i ** 2 * w2i ** 2 * w3i
                - Kdrag * rhogsol * w2i ** 2 * w1r ** 2 * w3i
                - Kdrag ** 2 * k * vdsol * w2r * w1i * w3i
                + Kdrag ** 2 * k * vdsol * w2i * w3i * w1r
                + Kdrag ** 2 * k * vgsol * w2r * w1i * w3i
                - Kdrag ** 2 * k * vgsol * w2i * w3i * w1r
                - rhogeq * cs ** 2 * k ** 3 * w2i * w1r * Kdrag * vdsol
                + rhogeq * cs ** 2 * k ** 3 * w2i * w1r * Kdrag * vgsol
                - w3i ** 2 * rhogeq ** 2 * w2i ** 2 * w1r * k * vgsol
                + w3i ** 2 * rhogeq ** 2 * w2r ** 2 * k * w1r * vgsol
                - w3i ** 2 * rhogeq * w2i * w1r * k * Kdrag * vgsol
                + w3i ** 2 * rhogeq * w2r * w1i * k * Kdrag * vgsol
                - w3i ** 2 * rhogeq ** 2 * cs ** 2 * k ** 3 * w1r * vgsol
                + w3i ** 2 * rhogeq * w2i ** 2 * w1i ** 2 * rhogsol
                + w3i ** 2 * rhogeq * w1r ** 2 * w2i ** 2 * rhogsol
                - w3i ** 2 * rhogeq * w2r ** 2 * w1r ** 2 * rhogsol
                - w3i ** 2 * rhogeq * w1i ** 2 * w2r ** 2 * rhogsol
                - 2 * w3i * rhogeq * k * Kdrag * vdsol * w2r * w2i * w1i
                + w3i * rhogeq * rhogsol * k ** 2 * cs ** 2 * w2i * w2r ** 2
                + 2 * rhogeq ** 2 * w1i ** 2 * k * vgsol * w2r * w3i * w2i
                - rhogeq * w2i * w1r * k * Kdrag * vdsol * w2r ** 2
                - 2 * vgsol * Kdrag * rhogeq * k ** 3 * cs ** 2 * w2r * w2i
                - 2 * cs ** 2 * k ** 2 * rhogeq * rhogsol * w2i ** 2 * w2r ** 2
                + 2 * vdsol * Kdrag * rhogeq * k ** 3 * cs ** 2 * w2r * w2i
                - cs ** 2 * k ** 2 * rhogeq * rhogsol * w2r ** 4
                - rhogeq ** 2 * w2r * k * vgsol * w3i ** 2 * w2i ** 2
                - w3i * rhogeq * k * Kdrag * vdsol * w1r * w2r ** 2
                + w3i * rhogeq * k * Kdrag * vdsol * w2r * w2i ** 2
                - w3i * rhogeq * rhogsol * w2i * w1r ** 2 * w2r ** 2
                + 2 * w3i * rhogeq * rhogsol * w2r ** 2 * w1i * w2i ** 2
                + rhogeq * cs ** 4 * k ** 4 * rhogsol * w2r ** 2
                + w3i * rhogeq * k * Kdrag * vdsol * w2r ** 3
                + w3i * rhogeq * rhogsol * w2r ** 4 * w1i
                - rhogeq * w1i * w3i ** 2 * rhogsol * w2i * w2r ** 2
                + rhogeq * w2r * w3i ** 2 * w1r * rhogsol * w2i ** 2
                + 2 * w3i * rhogeq ** 2 * k * vgsol * w2i * w2r * w1r ** 2
                - 2 * w3i * cs ** 2 * k ** 3 * rhogeq ** 2 * vgsol * w2r * w2i
                + 2 * w3i * cs ** 2 * k ** 3 * rhogeq ** 2 * vgsol * w2r * w1i
                - rhogeq * w3i ** 2 * cs ** 2 * k ** 2 * rhogsol * w2r * w1r
                + 2 * rhogeq ** 2 * w2r * k * vgsol * w3i ** 2 * w2i * w1i
                + 2 * rhogeq * w2r * w1i * k * Kdrag * vgsol * w2i * w3i
                + w3r * rhogeq * w2r ** 3 * cs ** 2 * k ** 2 * rhogsol
                + w3r * rhogeq * w2r ** 3 * w1r ** 2 * rhogsol
                + w3r * rhogeq ** 2 * w2r ** 4 * k * vgsol
                + w3r * rhogeq ** 2 * k * vgsol * w2i ** 4
                - w3r * vdsol * Kdrag * k * rhogeq * w2i ** 3
                - w3i * rhogeq * rhogsol * w2i ** 3 * w1r ** 2
                + w3i * rhogeq * rhogsol * w2i ** 4 * w1i
                - w3r ** 2 * rhogeq * rhogsol * w2i ** 3 * w1i
                - w3i ** 2 * rhogeq * rhogsol * w2i ** 3 * w1i
                - w3r * rhogeq ** 2 * k * vgsol * w2i ** 2 * w1r ** 2
                + w3i * vdsol * Kdrag * k * rhogeq * w2i ** 2 * w1r
                - rhogeq * rhogsol * k ** 2 * cs ** 2 * w2i ** 4
                + w3r ** 2 * cs ** 2 * k ** 2 * rhogeq * rhogsol * w1r ** 2
                - rhogeq * k * Kdrag * vdsol * w2i ** 3 * w1r
                + w3i ** 2 * cs ** 2 * k ** 2 * rhogeq * rhogsol * w1r ** 2
                + w3i * cs ** 2 * k ** 2 * rhogeq * rhogsol * w2i ** 3
                - w3r * rhogeq * rhogsol * w2i ** 4 * w1r
                - rhogeq ** 2 * w1i ** 2 * w2r ** 3 * vgsol * k
                - rhogeq * w1i ** 2 * w2i ** 3 * rhogsol * w3i
                - cs ** 4 * k ** 4 * rhogeq * rhogsol * w2i ** 2
                + w2i ** 2 * k * Kdrag ** 2 * vdsol * w2r
                + w2r ** 2 * k * Kdrag ** 2 * vgsol * w1r
                - w2r ** 2 * k * Kdrag ** 2 * vdsol * w1r
                - w2i ** 2 * k * Kdrag ** 2 * vgsol * w2r
                - w2i ** 2 * k * Kdrag ** 2 * vdsol * w1r
                + w2r ** 3 * k * Kdrag ** 2 * vdsol
                + w3r * w2r ** 2 * k * Kdrag ** 2 * vgsol
                - w3r * w2r ** 2 * k * Kdrag ** 2 * vdsol
                + w2i ** 2 * k * Kdrag ** 2 * vgsol * w1r
                + w3r * w2i ** 2 * k * Kdrag ** 2 * vgsol
                - w3r * w2i ** 2 * k * Kdrag ** 2 * vdsol
                - w2r ** 3 * k * Kdrag ** 2 * vgsol
                - 2 * Kdrag * w2r ** 3 * k * rhogeq * vgsol * w3i
                + Kdrag * w2i ** 2 * rhogsol * k ** 2 * cs ** 2 * w3i
                - 2 * Kdrag * w2i ** 2 * k * rhogeq * vgsol * w2r * w3i
                - 2 * Kdrag * w2r ** 3 * vgsol * k * rhogeq * w1i
                + 2 * Kdrag * w2i * vgsol * k * rhogeq * w1r * w2r ** 2
                + 2 * Kdrag * w2i ** 3 * vgsol * k * rhogeq * w1r
                - 2 * Kdrag * w2i ** 2 * w2r * vgsol * k * rhogeq * w1i
                + Kdrag * w2i ** 3 * rhogsol * w3i * w1i
                - Kdrag * w3r * w2i ** 3 * rhogsol * w1r
                + Kdrag * w3r * w2r ** 3 * rhogsol * w1i
                - Kdrag * w3r * w2r ** 2 * rhogsol * w2i * w1r
                + Kdrag * w3r * w2i ** 2 * rhogsol * w1i * w2r
                + Kdrag * w2r ** 3 * rhogsol * w3i * w1r
                + rhogeq * w1i * w2i ** 2 * k * Kdrag * vdsol * w2r
                + rhogeq * w1i * w2r ** 2 * rhogsol * k ** 2 * cs ** 2 * w2i
                + Kdrag * w2r ** 2 * rhogsol * w2i * w3i * w1i
                + Kdrag * w2r ** 2 * rhogsol * k ** 2 * cs ** 2 * w1i
                + Kdrag * w2r ** 2 * rhogsol * k ** 2 * cs ** 2 * w3i
                - 2 * rhogeq ** 2 * w1i * w2i ** 2 * k * vgsol * w2r * w3i
                - rhogeq * w1i * w3r * w2r ** 2 * k * Kdrag * vdsol
                - 2 * rhogeq ** 2 * w1i * w2r ** 3 * k * vgsol * w3i
                + rhogeq * w1i * w2i ** 3 * rhogsol * k ** 2 * cs ** 2
                + rhogeq * w1i * w3r * w2i ** 2 * k * Kdrag * vdsol
                + rhogeq * w1i ** 2 * w3r * w2i ** 2 * rhogsol * w2r
                + rhogeq * w1i * w2r ** 3 * k * Kdrag * vdsol
                + rhogeq * w1i ** 2 * w3r * w2r ** 3 * rhogsol
                - rhogeq * w1i ** 2 * w2r ** 2 * rhogsol * w2i * w3i
                - rhogeq ** 2 * w1i ** 2 * w2i ** 2 * w2r * vgsol * k
                + Kdrag * w2i ** 2 * rhogsol * k ** 2 * cs ** 2 * w1i
                + Kdrag * w2i ** 2 * rhogsol * w3i * w2r * w1r
                - Kdrag * w2r ** 2 * rhogsol * k ** 2 * cs ** 2 * w2i
                - Kdrag * w2i ** 3 * rhogsol * k ** 2 * cs ** 2
                + 2 * Kdrag * w3r * w2r ** 2 * k * rhogeq * vgsol * w2i
                + 2 * Kdrag * w3r * w2i ** 3 * k * rhogeq * vgsol
                - 2 * w3r * w2r * Kdrag * k * rhogeq * vgsol * w2i * w1r
                + 2 * w3r * rhogeq * w2i * k * Kdrag * vdsol * w2r * w1r
                - 2 * w3r * w2r * rhogeq ** 2 * vgsol * k * w2i ** 2 * w1r
                + w3r * rhogeq ** 2 * w1r ** 2 * k * vgsol * w2r ** 2
                + w3r * rhogeq * w2r * cs ** 2 * k ** 2 * rhogsol * w2i ** 2
                - w3r * rhogeq * w2r ** 4 * w1r * rhogsol
                + w3r * rhogeq * w2r * w1r ** 2 * rhogsol * w2i ** 2
                - 2 * w3r * rhogeq ** 2 * w2r ** 3 * k * w1r * vgsol
                - 2 * w3r * rhogeq * w2r ** 2 * w1r * rhogsol * w2i ** 2
                - w3r * rhogeq * w2i * k * Kdrag * vdsol * w2r ** 2
                + 2 * w3r * rhogeq ** 2 * w2i ** 2 * k * vgsol * w2r ** 2
                + w3r ** 2 * rhogeq * w2r * w1r * rhogsol * w2i ** 2
                - w3r ** 2 * rhogeq ** 2 * w2r * vgsol * k * w2i ** 2
                - w3r ** 2 * rhogeq * w2i * rhogsol * w1i * w2r ** 2
                - w3r ** 2 * rhogeq ** 2 * w2r ** 3 * vgsol * k
                + 2 * w3r ** 2 * rhogeq ** 2 * w2r * vgsol * k * w2i * w1i
                - w3r ** 2 * rhogeq * rhogsol * k ** 2 * cs ** 2 * w2r * w1r
                + w3r ** 2 * rhogeq * w2r ** 3 * w1r * rhogsol
                + rhogeq ** 2 * w1r * w2i ** 4 * k * vgsol
                + rhogeq ** 2 * w1r * w2r ** 4 * k * vgsol
                - rhogeq ** 2 * w1r ** 2 * w2r ** 3 * k * vgsol
            )
            / k
            / rhogeq
            / (
                w2r ** 2
                - 2 * w3r * w2r
                + w2i ** 2
                + w3i ** 2
                - 2 * w2i * w3i
                + w3r ** 2
            )
            / (
                w2r ** 2
                + w1r ** 2
                + w2i ** 2
                - 2 * w2i * w1i
                - 2 * w2r * w1r
                + w1i ** 2
            )
            / Kdrag
        )

        vd2i = (
            1
            / k
            * (
                w3r * rhogeq * k * Kdrag * vdsol * w2r ** 2 * w1r
                - w3r * rhogeq * k * Kdrag * vdsol * w2r * w2i ** 2
                - w3r * rhogeq * k * Kdrag * vdsol * w2r ** 3
                - w3r * Kdrag * rhogsol * w2r ** 2 * w1r ** 2
                - w3r * Kdrag * rhogsol * w2r ** 2 * w1i ** 2
                + w3r * rhogeq * rhogsol * w2r ** 4 * w1i
                - w3r * Kdrag * k * rhogeq * vgsol * w2r * w1i ** 2
                - w3r * Kdrag * k * rhogeq * vgsol * w2r * w1r ** 2
                - 2 * w3r * rhogeq * cs ** 2 * k ** 2 * rhogsol * w2r ** 2 * w1i
                - w3r * cs ** 2 * k ** 3 * rhogeq * Kdrag * vgsol * w2r
                - 2 * w3r * rhogeq ** 2 * k * vgsol * w2r * w2i * w1i ** 2
                - 2 * w3r * rhogeq ** 2 * k * vgsol * w2r * w2i * w1r ** 2
                + 2 * w3r * rhogeq ** 2 * k * vgsol * w2r ** 2 * w2i * w1r
                - w3r * rhogeq * cs ** 2 * k ** 2 * rhogsol * w2i * w2r ** 2
                + w3r * Kdrag * cs ** 2 * k ** 2 * rhogsol * w2r * w1r
                - w3r * Kdrag * cs ** 2 * k ** 2 * rhogsol * w2r ** 2
                + w3r * Kdrag * rhogsol * w2r ** 2 * w1i * w2i
                - 2 * w3r * cs ** 2 * k ** 3 * rhogeq ** 2 * vgsol * w2i * w2r
                + w3r * cs ** 2 * k ** 3 * rhogeq * Kdrag * vdsol * w2r
                + 4 * w3r * rhogeq * cs ** 2 * k ** 2 * rhogsol * w2r * w2i * w1r
                + w3r * Kdrag * rhogsol * w2r * w1r * w2i ** 2
                + 2 * w3r * rhogeq * rhogsol * w2i ** 2 * w2r ** 2 * w1i
                + w3r ** 2 * Kdrag * rhogsol * w2r * w1r ** 2
                + 2 * w3r ** 2 * rhogeq * rhogsol * w2r * w2i * w1i ** 2
                - w3r * rhogeq * rhogsol * w1i ** 2 * w2r ** 2 * w2i
                - w3r * rhogeq * rhogsol * w2i * w1r ** 2 * w2r ** 2
                - 2 * w3r * w2r * Kdrag * rhogeq * k * vgsol * w2i * w1i
                + w3r ** 2 * cs ** 2 * k ** 2 * rhogeq * rhogsol * w2r * w1i
                + w3r ** 2 * rhogeq ** 2 * k * vgsol * w2i * w2r ** 2
                + w3r ** 2 * rhogeq ** 2 * k * vgsol * w2r ** 2 * w1i
                - w3r ** 2 * Kdrag * rhogeq * k * vgsol * w2r * w1r
                + w3r ** 2 * Kdrag * rhogeq * k * vgsol * w2r ** 2
                - w3r ** 2 * rhogeq * rhogsol * w2r * w2i ** 2 * w1i
                - w3r ** 2 * rhogeq * rhogsol * w2r ** 2 * w2i * w1r
                + w3r ** 2 * Kdrag * rhogeq * k * vgsol * w2i ** 2
                - w3r ** 2 * rhogeq ** 2 * cs ** 2 * k ** 3 * vgsol * w1i
                + 2 * w3r ** 2 * rhogeq * rhogsol * w2r * w2i * w1r ** 2
                - 2 * w3r ** 2 * rhogeq ** 2 * k * vgsol * w2r * w2i * w1r
                - w3r * Kdrag ** 2 * k * vgsol * w2r * w1i
                + rhogeq * k * Kdrag * vdsol * w2r ** 4
                - w3i * rhogeq * rhogsol * w2r ** 3 * w1i ** 2
                + Kdrag ** 2 * k * vdsol * w2i * w2r ** 2
                - 2 * vgsol * k * rhogeq ** 2 * w2i ** 2 * w2r ** 2 * w1i
                + vgsol * k * rhogeq ** 2 * w2i * w1r ** 2 * w2r ** 2
                - 2 * vgsol * rhogeq ** 2 * k ** 3 * cs ** 2 * w2r * w2i * w1r
                - rhogeq * cs ** 2 * k ** 2 * rhogsol * w2r ** 2 * w2i * w1r
                + rhogeq * cs ** 2 * k ** 2 * rhogsol * w2r ** 3 * w1i
                + rhogeq * cs ** 2 * k ** 2 * rhogsol * w2r * w2i ** 2 * w1i
                + w3i * rhogeq ** 2 * k * vgsol * w2r ** 2 * w1r ** 2
                - w3i * rhogeq ** 2 * k * vgsol * w2r ** 4
                + w3i ** 2 * rhogeq ** 2 * k * vgsol * w2i * w2r ** 2
                - 2 * w3i ** 2 * rhogeq ** 2 * k * vgsol * w2r * w2i * w1r
                + w3i ** 2 * rhogeq ** 2 * k * vgsol * w2r ** 2 * w1i
                - rhogeq * cs ** 2 * k ** 3 * vdsol * Kdrag * w2r ** 2
                + rhogeq * cs ** 2 * k ** 3 * vdsol * Kdrag * w2r * w1r
                - Kdrag ** 2 * k * vdsol * w2r ** 2 * w1i
                + 3 * w3i * Kdrag * k * rhogeq * vgsol * w2r ** 2 * w1i
                - 2 * w3i * cs ** 2 * k ** 2 * rhogeq * rhogsol * w2r ** 2 * w1r
                + w3i * cs ** 2 * k ** 2 * rhogeq * rhogsol * w2r * w2i ** 2
                + w3i * cs ** 2 * k ** 2 * rhogeq * rhogsol * w2r ** 3
                + w3i * cs ** 2 * k ** 2 * rhogeq * rhogsol * w2r * w1r ** 2
                - 4 * w3i * cs ** 2 * k ** 2 * rhogeq * rhogsol * w2r * w2i * w1i
                + w3i * cs ** 2 * k ** 2 * rhogeq * rhogsol * w2r * w1i ** 2
                + w3i * rhogeq ** 2 * k * vgsol * w2r ** 2 * w1i ** 2
                - 2 * w3i * rhogeq ** 2 * k * vgsol * w2i ** 2 * w2r ** 2
                - w3i ** 2 * Kdrag * rhogeq * k * vgsol * w2r * w1r
                + w3i ** 2 * Kdrag * rhogeq * k * vgsol * w2i ** 2
                + Kdrag ** 2 * k * vgsol * w2r ** 2 * w1i
                + w3i * rhogeq ** 2 * cs ** 2 * k ** 3 * vgsol * w2r ** 2
                + Kdrag * k * rhogeq * vgsol * w2r ** 2 * w1r ** 2
                + Kdrag * k * rhogeq * vgsol * w2r ** 2 * w1i ** 2
                + w3i ** 2 * cs ** 2 * k ** 2 * rhogeq * rhogsol * w2r * w1i
                - Kdrag ** 2 * k * vgsol * w2i * w2r ** 2
                + 2 * rhogeq * cs ** 4 * k ** 4 * rhogsol * w2i * w2r
                + 2 * w3i * rhogeq * rhogsol * w2i ** 2 * w2r ** 2 * w1r
                + w3i ** 2 * Kdrag * rhogeq * k * vgsol * w2r ** 2
                + w3i * rhogeq * rhogsol * w2r ** 4 * w1r
                - w3i * rhogeq * rhogsol * w2r ** 3 * w1r ** 2
                - rhogeq * cs ** 4 * k ** 4 * rhogsol * w2r * w1i
                - w3i * rhogeq * cs ** 4 * k ** 4 * rhogsol * w2r
                - rhogeq * k * Kdrag * vdsol * w2r ** 2 * w1i * w2i
                + Kdrag * cs ** 2 * k ** 2 * rhogsol * w2r ** 3
                + 2 * rhogeq * k * Kdrag * vdsol * w2i ** 2 * w2r ** 2
                - rhogeq * k * Kdrag * vdsol * w2r ** 3 * w1r
                + Kdrag * cs ** 2 * k ** 2 * rhogsol * w2r * w2i ** 2
                - rhogeq * k * Kdrag * vdsol * w2r * w1r * w2i ** 2
                - Kdrag * cs ** 2 * k ** 2 * rhogsol * w2r ** 2 * w1r
                + w3i * Kdrag * rhogsol * w2r ** 2 * w2i * w1r
                - w3i * Kdrag * rhogsol * w2r * w2i ** 2 * w1i
                - w3i * Kdrag * rhogsol * w2r ** 3 * w1i
                - w3i * Kdrag * cs ** 2 * k ** 2 * rhogsol * w2r * w1i
                - w3i ** 2 * rhogeq * rhogsol * w2r ** 2 * w2i * w1r
                - w3i ** 2 * rhogeq * rhogsol * w2r * w2i ** 2 * w1i
                - w3i * rhogeq * rhogsol * w2r * w2i ** 2 * w1r ** 2
                + 2 * w3i ** 2 * rhogeq * rhogsol * w2r * w2i * w1r ** 2
                - w3i ** 2 * rhogeq * rhogsol * w2r ** 3 * w1i
                + 2 * w3i ** 2 * rhogeq * rhogsol * w2r * w2i * w1i ** 2
                - w3i * rhogeq * rhogsol * w2r * w2i ** 2 * w1i ** 2
                - cs ** 2 * k ** 3 * rhogeq * Kdrag * vgsol * w2r * w1r
                + cs ** 2 * k ** 3 * rhogeq * Kdrag * vgsol * w2r ** 2
                - w3i ** 2 * cs ** 2 * k ** 3 * rhogeq ** 2 * vgsol * w1i
                - w3i ** 2 * Kdrag * rhogsol * w2r ** 2 * w1r
                + w3i * Kdrag ** 2 * k * vdsol * w2r * w1r
                + w3i ** 2 * Kdrag * rhogsol * w2r * w1r ** 2
                + w3i ** 2 * Kdrag * rhogsol * w2r * w1i ** 2
                - w3i * Kdrag ** 2 * k * vdsol * w2r ** 2
                + w3i * Kdrag ** 2 * k * vgsol * w2r ** 2
                - w3i * Kdrag ** 2 * k * vgsol * w2r * w1r
                + 2 * w3i * rhogeq * k * Kdrag * vdsol * w2r * w2i * w1r
                - w3i * rhogeq * k * Kdrag * vdsol * w2r ** 2 * w1i
                - w3i * rhogeq * k * Kdrag * vdsol * w2i * w2r ** 2
                + w3r ** 2 * Kdrag * rhogsol * w2r * w1i ** 2
                - w3r ** 2 * Kdrag * rhogsol * w2r ** 2 * w1r
                + w3r * Kdrag * rhogsol * w2r ** 3 * w1r
                - w3r ** 2 * rhogeq * rhogsol * w2r ** 3 * w1i
                + Kdrag ** 2 * k * vdsol * w2i ** 3
                - Kdrag ** 2 * k * vgsol * w2i ** 3
                - Kdrag * rhogeq * k * vgsol * w2r ** 4
                + rhogeq ** 2 * cs ** 2 * k ** 3 * vgsol * w1i * w2r ** 2
                - 2 * Kdrag * rhogeq * k * vgsol * w2i ** 2 * w2r ** 2
                - 2 * Kdrag * rhogeq * k * vgsol * w2r * w1r * w2i * w3i
                + 2 * rhogeq ** 2 * k * vgsol * w2r ** 2 * w1i * w2i * w3i
                + 2 * w3r * rhogeq * k * Kdrag * vdsol * w2r * w2i * w1i
                + w3r * Kdrag * k * rhogeq * vgsol * w2r ** 2 * w1r
                + w3r * Kdrag ** 2 * k * vdsol * w2r * w1i
                - rhogeq ** 2 * k * vgsol * w2r ** 4 * w1i
                - w3r * rhogeq * k * Kdrag * vdsol * w2i ** 2 * w1r
                - w3r * rhogeq * cs ** 2 * k ** 2 * rhogsol * w2i * w1r ** 2
                - w3r * rhogeq * cs ** 2 * k ** 2 * rhogsol * w2i ** 3
                + 2 * w3r * vgsol * k ** 3 * cs ** 2 * rhogeq ** 2 * w2i * w1r
                + w3r * cs ** 4 * k ** 4 * rhogeq * rhogsol * w1i
                + w3r * Kdrag * cs ** 2 * k ** 2 * rhogsol * w1i * w2i
                - w3r * cs ** 2 * k ** 3 * rhogeq * Kdrag * vdsol * w1r
                + w3i * Kdrag ** 2 * k * vdsol * w1i * w2i
                - w3i * Kdrag ** 2 * k * vdsol * w2i ** 2
                + rhogeq * k * Kdrag * vdsol * w2i ** 4
                - w3i * Kdrag ** 2 * k * vgsol * w1i * w2i
                + w3i * Kdrag ** 2 * k * vgsol * w2i ** 2
                + w3r * Kdrag ** 2 * k * vgsol * w2i * w1r
                + 2 * w3r * rhogeq ** 2 * k * vgsol * w2i ** 3 * w1r
                - cs ** 2 * k ** 3 * rhogeq * Kdrag * vdsol * w1i * w2i
                - w3i ** 2 * Kdrag * rhogeq * k * vgsol * w1i * w2i
                - Kdrag * rhogeq * k * vgsol * w2i ** 4
                + Kdrag * rhogeq * k * vgsol * w2i ** 2 * w1r ** 2
                + Kdrag * rhogeq * k * vgsol * w2i ** 2 * w1i ** 2
                + 3 * w3r * Kdrag * k * rhogeq * vgsol * w2i ** 2 * w1r
                + w3i * rhogeq * k * Kdrag * vgsol * w2i ** 2 * w1i
                - w3i * rhogeq * k * Kdrag * vgsol * w2i * w1r ** 2
                - w3i * rhogeq * k * Kdrag * vgsol * w1i ** 2 * w2i
                - w3i * rhogeq * k * Kdrag * vdsol * w2i ** 3
                + w3i * rhogeq * k * Kdrag * vdsol * w2i ** 2 * w1i
                - w3i * rhogeq ** 2 * k * vgsol * w2i ** 2 * w1i ** 2
                + 2 * w3i * rhogeq ** 2 * k * vgsol * w2i ** 3 * w1i
                - w3i * rhogeq ** 2 * k * vgsol * w2i ** 4
                - w3i * rhogeq ** 2 * cs ** 2 * k ** 3 * vgsol * w1r ** 2
                - w3i * rhogeq ** 2 * cs ** 2 * k ** 3 * vgsol * w1i ** 2
                - w3r ** 2 * Kdrag * k * rhogeq * vgsol * w1i * w2i
                + w3r * cs ** 2 * k ** 3 * rhogeq * Kdrag * vgsol * w1r
                - rhogeq * cs ** 2 * k ** 2 * rhogsol * w2i ** 3 * w1r
                + w3i * Kdrag * cs ** 2 * k ** 2 * rhogsol * w2i * w1r
                - rhogeq * cs ** 4 * k ** 4 * rhogsol * w2i * w1r
                - w3r ** 2 * rhogeq * cs ** 2 * k ** 2 * rhogsol * w2i * w1r
                + w3r ** 2 * vgsol * rhogeq ** 2 * k ** 3 * cs ** 2 * w2i
                - w3i * rhogeq * cs ** 2 * k ** 3 * vdsol * Kdrag * w2i
                + w3i * rhogeq * cs ** 2 * k ** 3 * vdsol * Kdrag * w1i
                + cs ** 2 * k ** 3 * rhogeq * Kdrag * vgsol * w1i * w2i
                - w3i * cs ** 2 * k ** 3 * rhogeq * Kdrag * vgsol * w1i
                + w3i * cs ** 2 * k ** 3 * rhogeq * Kdrag * vgsol * w2i
                + w3i * rhogeq * cs ** 4 * k ** 4 * rhogsol * w1r
                + w3i ** 2 * vgsol * rhogeq ** 2 * k ** 3 * cs ** 2 * w2i
                - w3i ** 2 * vgsol * k * rhogeq ** 2 * w2i ** 2 * w1i
                - w3i ** 2 * rhogeq * cs ** 2 * k ** 2 * rhogsol * w2i * w1r
                + w3i ** 2 * vgsol * k * rhogeq ** 2 * w2i ** 3
                - w3r ** 2 * vgsol * k * rhogeq ** 2 * w2i ** 2 * w1i
                + w3r ** 2 * vgsol * k * rhogeq ** 2 * w2i ** 3
                - w3i * rhogeq ** 2 * cs ** 2 * k ** 3 * vgsol * w2i ** 2
                + 2 * w3i * rhogeq ** 2 * cs ** 2 * k ** 3 * vgsol * w2i * w1i
                - cs ** 2 * k ** 3 * rhogeq * Kdrag * vgsol * w2i ** 2
                + cs ** 2 * k ** 3 * rhogeq * Kdrag * vdsol * w2i ** 2
                - w3i * rhogeq ** 2 * k * vgsol * w2i ** 2 * w1r ** 2
                - rhogeq * k * Kdrag * vdsol * w2i ** 3 * w1i
                - Kdrag * cs ** 2 * k ** 2 * rhogsol * w2i ** 2 * w1r
                + w3r * Kdrag * rhogsol * w2i ** 3 * w1i
                - w3r * Kdrag * rhogsol * w2i ** 2 * w1r ** 2
                + w3i * Kdrag * rhogsol * w2i ** 3 * w1r
                + rhogeq ** 2 * cs ** 2 * k ** 3 * vgsol * w1i ** 2 * w2i
                + rhogeq ** 2 * cs ** 2 * k ** 3 * vgsol * w2i * w1r ** 2
                - w3r * Kdrag ** 2 * k * vdsol * w2i * w1r
                + vgsol * k * rhogeq ** 2 * w1i ** 2 * w2r ** 2 * w2i
                - w3r * cs ** 4 * k ** 4 * rhogeq * rhogsol * w2i
                - w3r * rhogeq * cs ** 2 * k ** 2 * rhogsol * w1i ** 2 * w2i
                + 2 * w3r * rhogeq * cs ** 2 * k ** 2 * rhogsol * w2i ** 2 * w1i
                - w3r * Kdrag * rhogsol * w2i ** 2 * w1i ** 2
                + 2 * w3i * rhogeq * cs ** 2 * k ** 2 * rhogsol * w2i ** 2 * w1r
                - w3r ** 2 * rhogeq * rhogsol * w2i ** 3 * w1r
                - w3i ** 2 * rhogeq * rhogsol * w2i ** 3 * w1r
                - w3i ** 2 * Kdrag * rhogsol * w2i ** 2 * w1r
                - w3r ** 2 * Kdrag * rhogsol * w2i ** 2 * w1r
                - rhogeq ** 2 * k * vgsol * w2i ** 4 * w1i
                + rhogeq ** 2 * k * vgsol * w2i ** 3 * w1r ** 2
                - w3r * rhogeq * rhogsol * w2i ** 3 * w1i ** 2
                + w3r * rhogeq * rhogsol * w2i ** 4 * w1i
                - w3r * rhogeq * rhogsol * w2i ** 3 * w1r ** 2
                + rhogeq ** 2 * k * vgsol * w2i ** 3 * w1i ** 2
                - Kdrag ** 2 * k * vdsol * w2i ** 2 * w1i
                + w3i * rhogeq * rhogsol * w2i ** 4 * w1r
                + Kdrag ** 2 * k * vgsol * w2i ** 2 * w1i
                - w3r * Kdrag * cs ** 2 * k ** 2 * rhogsol * w2i ** 2
                - rhogeq ** 2 * cs ** 2 * k ** 3 * vgsol * w1i * w2i ** 2
            )
            / (
                w2r ** 2
                - 2 * w3r * w2r
                + w2i ** 2
                + w3i ** 2
                - 2 * w2i * w3i
                + w3r ** 2
            )
            / (
                w2r ** 2
                + w1r ** 2
                + w2i ** 2
                - 2 * w2i * w1i
                - 2 * w2r * w1r
                + w1i ** 2
            )
            / rhogeq
            / Kdrag
        )

        vd1r = (
            (
                w3r * Kdrag * rhogsol * w1i ** 3 * w2r
                - w3r * w1r ** 2 * k * Kdrag ** 2 * vgsol
                - w3r * w1r ** 3 * rhogeq * rhogsol * w2i ** 2
                - w3r * rhogeq * w2r ** 2 * w1r ** 3 * rhogsol
                - w3r * w1i ** 2 * k * Kdrag ** 2 * vgsol
                + w3r * rhogeq * w1i ** 4 * w2r * rhogsol
                + w3r * Kdrag * rhogsol * w1i * w2r * w1r ** 2
                - 2 * w3r * rhogeq * w1i * k * Kdrag * vdsol * w2r * w1r
                + 2 * w3r * rhogeq * w2r * w1i * k * Kdrag * vgsol * w1r
                - w3r * rhogeq * w1i ** 2 * w2r ** 2 * rhogsol * w1r
                + w3r * w1i ** 2 * k * Kdrag ** 2 * vdsol
                + w3r * rhogeq * w2i * w1r ** 2 * k * Kdrag * vdsol
                + w3r * w1r ** 2 * k * Kdrag ** 2 * vdsol
                - w3r * Kdrag * rhogsol * w2i * w1r ** 3
                + w3r ** 2 * w1i ** 2 * rhogeq ** 2 * k * w1r * vgsol
                + w3r ** 2 * w1r ** 3 * rhogeq ** 2 * k * vgsol
                - 2 * w3r ** 2 * w2i * w1i * rhogeq ** 2 * k * w1r * vgsol
                - w3r ** 2 * rhogeq * rhogsol * k ** 2 * cs ** 2 * w2r ** 2
                - w3r ** 2 * rhogeq * rhogsol * k ** 2 * cs ** 2 * w2i ** 2
                - w3r ** 2 * rhogeq * w2r * w1r * rhogsol * w1i ** 2
                + w3r ** 2 * w2i * w1i * rhogeq * w1r ** 2 * rhogsol
                + 2 * w3r * w2r * w1i ** 2 * rhogeq ** 2 * k * w1r * vgsol
                + 2 * w3r * w2r * w1r ** 3 * rhogeq ** 2 * k * vgsol
                - w3r * rhogeq * w1i ** 2 * rhogsol * k ** 2 * cs ** 2 * w1r
                - w3r * cs ** 2 * k ** 2 * rhogeq * rhogsol * w1r ** 3
                - 2 * w3r * w1i ** 2 * rhogeq ** 2 * w1r ** 2 * k * vgsol
                + w3r * w1i ** 3 * rhogeq * k * Kdrag * vdsol
                - 2 * w3r * w1i ** 3 * Kdrag * k * rhogeq * vgsol
                - w3r * w1i ** 2 * rhogeq * w1r * rhogsol * w2i ** 2
                + w3r * w1r ** 2 * rhogeq * w1i * k * Kdrag * vdsol
                - w3r * w1r * Kdrag * rhogsol * w2i * w1i ** 2
                - w3r * w2i * w1i ** 2 * rhogeq * k * Kdrag * vdsol
                - 2 * w3r * w1r ** 2 * Kdrag * k * rhogeq * vgsol * w1i
                + 2 * w3r * rhogeq * w1i ** 2 * w2r * rhogsol * w1r ** 2
                + rhogeq * w1i * rhogsol * k ** 2 * cs ** 2 * w3i ** 2 * w2i
                + rhogeq * w1i * rhogsol * k ** 2 * cs ** 2 * w3r ** 2 * w2i
                + rhogeq ** 2 * w1i ** 2 * k * vgsol * w3r * w2r ** 2
                + rhogeq ** 2 * w1i ** 2 * k * vgsol * w3r * w2i ** 2
                - w3i * cs ** 2 * k ** 3 * rhogeq * Kdrag * vdsol * w2r
                + w3i * cs ** 2 * k ** 3 * rhogeq * Kdrag * vdsol * w1r
                + w3i * cs ** 2 * k ** 3 * rhogeq * Kdrag * vgsol * w2r
                - w3i * cs ** 2 * k ** 3 * rhogeq * Kdrag * vgsol * w1r
                - rhogeq ** 2 * cs ** 2 * k ** 3 * w2i ** 2 * w1r * vgsol
                + rhogeq * cs ** 4 * k ** 4 * w2r * w1r * rhogsol
                - rhogeq ** 2 * cs ** 2 * k ** 3 * w2r ** 2 * w1r * vgsol
                - rhogeq * cs ** 2 * k ** 3 * w2r * w1i * Kdrag * vgsol
                - rhogeq ** 2 * cs ** 2 * k ** 3 * w2r * w1i ** 2 * vgsol
                + rhogeq * cs ** 2 * k ** 3 * w2r * w1i * Kdrag * vdsol
                + rhogeq ** 2 * cs ** 2 * k ** 3 * w2r * w1r ** 2 * vgsol
                - rhogeq * cs ** 4 * k ** 4 * w2i * w1i * rhogsol
                + w3i * cs ** 4 * k ** 4 * rhogeq * rhogsol * w2i
                - w3i * cs ** 4 * k ** 4 * rhogeq * rhogsol * w1i
                + w3i * cs ** 2 * k ** 2 * rhogeq * rhogsol * w2r ** 2 * w1i
                + w3i * cs ** 2 * k ** 2 * rhogeq * rhogsol * w1i * w2i ** 2
                + w3r * rhogeq ** 2 * cs ** 2 * k ** 3 * w2i ** 2 * vgsol
                - w3r * rhogeq * cs ** 4 * k ** 4 * w2r * rhogsol
                + w3r * rhogeq ** 2 * cs ** 2 * k ** 3 * w2r ** 2 * vgsol
                - w3r * Kdrag * rhogsol * k ** 2 * cs ** 2 * w1i * w2r
                + w3r * Kdrag * rhogsol * k ** 2 * cs ** 2 * w2i * w1r
                - w3r * Kdrag * k * rhogeq * vgsol * w2i * w1r ** 2
                + w3r * Kdrag * k * rhogeq * vgsol * w2i * w1i ** 2
                - w3r * Kdrag ** 2 * k * vdsol * w2r * w1r
                - w3r * Kdrag ** 2 * k * vdsol * w2i * w1i
                + w3r * Kdrag ** 2 * k * vgsol * w2r * w1r
                + w3r * Kdrag ** 2 * k * vgsol * w2i * w1i
                + w3r * Kdrag * k * rhogeq * vgsol * w2r ** 2 * w1i
                + w3r * Kdrag * k * rhogeq * vgsol * w1i * w2i ** 2
                + w3r * rhogeq * cs ** 2 * k ** 2 * w2r ** 2 * w1r * rhogsol
                + w3r * rhogeq * w2i ** 2 * w1r * cs ** 2 * k ** 2 * rhogsol
                - w3r * rhogeq * cs ** 2 * k ** 3 * w1i * Kdrag * vgsol
                - w3r * rhogeq * cs ** 2 * k ** 3 * w2i * Kdrag * vdsol
                + w3r * rhogeq * cs ** 2 * k ** 3 * w2i * Kdrag * vgsol
                + w3r * rhogeq * cs ** 2 * k ** 3 * w1i * Kdrag * vdsol
                - w3r * rhogeq ** 2 * cs ** 2 * k ** 3 * w1i ** 2 * vgsol
                + w3r * rhogeq ** 2 * cs ** 2 * k ** 3 * w1r ** 2 * vgsol
                + w3r * rhogeq * cs ** 4 * k ** 4 * w1r * rhogsol
                - 2 * w3r * rhogeq ** 2 * cs ** 2 * k ** 3 * w2r * w1r * vgsol
                + w3r ** 2 * rhogeq ** 2 * cs ** 2 * k ** 3 * w2r * vgsol
                + w3r ** 2 * rhogeq * w2r * w1i * k * Kdrag * vgsol
                + w3r ** 2 * rhogeq ** 2 * w2r * w1i ** 2 * k * vgsol
                - w3r ** 2 * rhogeq ** 2 * w2r * w1r ** 2 * k * vgsol
                - w3r ** 2 * rhogeq ** 2 * cs ** 2 * k ** 3 * w1r * vgsol
                - w3r ** 2 * rhogeq * w2i ** 2 * w1i ** 2 * rhogsol
                + w3r ** 2 * rhogeq * w1r ** 2 * w2i ** 2 * rhogsol
                + w3r ** 2 * rhogeq * w2r ** 2 * w1r ** 2 * rhogsol
                - w3r ** 2 * rhogeq * w1i ** 2 * w2r ** 2 * rhogsol
                - w3r ** 2 * Kdrag * rhogsol * w1i * w2i ** 2
                + w3r ** 2 * Kdrag * rhogsol * w2i * w1r ** 2
                + w3r ** 2 * Kdrag * rhogsol * w2i * w1i ** 2
                - w3r ** 2 * Kdrag * rhogsol * w2r ** 2 * w1i
                - w3r ** 2 * rhogeq * w2i * w1r * k * Kdrag * vgsol
                + w3i ** 2 * rhogeq ** 2 * cs ** 2 * k ** 3 * w2r * vgsol
                - w3i ** 2 * Kdrag * rhogsol * w2r ** 2 * w1i
                - w3i ** 2 * Kdrag * rhogsol * w1i * w2i ** 2
                + w3i ** 2 * Kdrag * rhogsol * w2i * w1r ** 2
                + w3i ** 2 * Kdrag * rhogsol * w2i * w1i ** 2
                - Kdrag * k * rhogeq * vgsol * w3i * w1r * w2r ** 2
                - Kdrag * k * rhogeq * vgsol * w3i * w2r * w1r ** 2
                + Kdrag * k * rhogeq * vgsol * w3i * w2r * w1i ** 2
                - Kdrag * k * rhogeq * vgsol * w3i * w1r * w2i ** 2
                + Kdrag * rhogsol * k ** 2 * cs ** 2 * w3i * w2r * w1r
                + Kdrag * rhogsol * k ** 2 * cs ** 2 * w2i * w3i * w1i
                + Kdrag * rhogsol * w2r ** 2 * w1i ** 2 * w3i
                + Kdrag * rhogsol * w3i * w1r ** 2 * w2r ** 2
                + Kdrag * rhogsol * w1i ** 2 * w2i ** 2 * w3i
                + Kdrag * rhogsol * w2i ** 2 * w1r ** 2 * w3i
                - Kdrag ** 2 * k * vdsol * w2r * w1i * w3i
                + Kdrag ** 2 * k * vdsol * w2i * w3i * w1r
                + Kdrag ** 2 * k * vgsol * w2r * w1i * w3i
                - Kdrag ** 2 * k * vgsol * w2i * w3i * w1r
                + rhogeq * cs ** 2 * k ** 3 * w2i * w1r * Kdrag * vdsol
                - rhogeq * cs ** 2 * k ** 3 * w2i * w1r * Kdrag * vgsol
                - w1r ** 3 * vdsol * Kdrag * k * rhogeq * w2i
                - Kdrag * rhogsol * k ** 2 * cs ** 2 * w2i * w1r ** 2
                + 2 * Kdrag * k * rhogeq * vgsol * w2i * w1r ** 3
                - w3i ** 2 * rhogeq * w2i * w1r * k * Kdrag * vgsol
                + w3i ** 2 * rhogeq * w2r * w1i * k * Kdrag * vgsol
                + w3i ** 2 * rhogeq ** 2 * w2r * w1i ** 2 * k * vgsol
                - w3i ** 2 * rhogeq ** 2 * w2r * w1r ** 2 * k * vgsol
                - w3i ** 2 * rhogeq ** 2 * cs ** 2 * k ** 3 * w1r * vgsol
                - w3i ** 2 * rhogeq * w2i ** 2 * w1i ** 2 * rhogsol
                + w3i ** 2 * rhogeq * w1r ** 2 * w2i ** 2 * rhogsol
                + w3i ** 2 * rhogeq * w2r ** 2 * w1r ** 2 * rhogsol
                - w3i ** 2 * rhogeq * w1i ** 2 * w2r ** 2 * rhogsol
                + rhogeq * w1i ** 3 * w2r ** 2 * rhogsol * w3i
                - 2 * rhogeq * w2r * w1i ** 3 * k * Kdrag * vgsol
                + rhogeq * w2r ** 2 * w1r ** 2 * rhogsol * w3i * w1i
                + rhogeq ** 2 * w2r ** 2 * k * w1r * vgsol * w1i ** 2
                + Kdrag ** 2 * k * vdsol * w2r * w1r ** 2
                - Kdrag ** 2 * k * vgsol * w2r * w1r ** 2
                - rhogeq * cs ** 2 * k ** 2 * w2r * w1r ** 3 * rhogsol
                + Kdrag ** 2 * k * vgsol * w1r ** 3
                - Kdrag ** 2 * k * vdsol * w1r ** 3
                + rhogeq ** 2 * w2r ** 2 * k * w1r ** 3 * vgsol
                - 2 * rhogeq ** 2 * w2r ** 2 * k * w1r * vgsol * w3i * w1i
                + rhogeq * w3i ** 2 * cs ** 2 * k ** 2 * rhogsol * w2r * w1r
                - 2 * rhogeq * w2r * w1i * k * Kdrag * vgsol * w1r ** 2
                - rhogeq * cs ** 2 * k ** 2 * w1i ** 2 * w2r * rhogsol * w1r
                - w3r * rhogeq ** 2 * k * vgsol * w2i ** 2 * w1r ** 2
                - rhogeq ** 2 * w2r * w1r ** 4 * k * vgsol
                - rhogeq * w2r * w3i ** 2 * w1r ** 3 * rhogsol
                + w3i * rhogeq * k * Kdrag * vdsol * w2r * w1r ** 2
                + rhogeq * w2r * w1i * k * Kdrag * vdsol * w1r ** 2
                - Kdrag * rhogsol * w3i * w2r * w1r ** 3
                - 2 * rhogeq ** 2 * w2r * w1i ** 2 * k * vgsol * w1r ** 2
                - rhogeq * w3i ** 2 * cs ** 2 * k ** 2 * rhogsol * w2r ** 2
                - rhogeq * w3i ** 2 * cs ** 2 * k ** 2 * rhogsol * w2i ** 2
                - Kdrag * rhogsol * w3i * w2r * w1r * w1i ** 2
                - w3i * rhogeq * k * Kdrag * vdsol * w2r * w1i ** 2
                + Kdrag ** 2 * k * vdsol * w2r * w1i ** 2
                - rhogeq * w2r * w3i ** 2 * w1r * rhogsol * w1i ** 2
                - Kdrag ** 2 * k * vgsol * w2r * w1i ** 2
                + rhogeq * w2r * w1i ** 3 * k * Kdrag * vdsol
                - rhogeq ** 2 * w2r * w1i ** 4 * k * vgsol
                + cs ** 2 * k ** 2 * rhogeq * rhogsol * w1r ** 4
                + w1r ** 2 * w3i * rhogeq * rhogsol * w2i ** 2 * w1i
                + 2 * rhogeq * w1i ** 2 * rhogsol * k ** 2 * cs ** 2 * w1r ** 2
                + rhogeq * w1i ** 4 * rhogsol * k ** 2 * cs ** 2
                - w2i * w1i ** 3 * rhogeq * rhogsol * k ** 2 * cs ** 2
                - w1r ** 4 * w3i * rhogeq * rhogsol * w2i
                - w1r ** 2 * rhogeq * cs ** 4 * k ** 4 * rhogsol
                + w2i * w1i ** 3 * rhogeq * rhogsol * w3i ** 2
                - w3i * cs ** 2 * k ** 2 * rhogeq * rhogsol * w1i * w1r ** 2
                + w2i * w1i * rhogeq * w1r ** 2 * rhogsol * w3i ** 2
                - w3i * cs ** 2 * k ** 2 * rhogeq * rhogsol * w1i ** 3
                + w1i ** 3 * w3i * rhogeq * rhogsol * w2i ** 2
                - w2i * w1i * cs ** 2 * k ** 2 * rhogeq * rhogsol * w1r ** 2
                + w1i ** 2 * rhogeq ** 2 * k * vgsol * w2i ** 2 * w1r
                - 2 * w2i * w1i * rhogeq ** 2 * k * w1r * vgsol * w3i ** 2
                - 2 * vgsol * k * rhogeq * Kdrag * w2i * w1r * w3i * w1i
                + w1r ** 3 * rhogeq ** 2 * k * vgsol * w2i ** 2
                - w1i ** 4 * w3i * rhogeq * rhogsol * w2i
                + 2 * rhogeq * cs ** 2 * k ** 3 * w1i * Kdrag * vgsol * w1r
                + rhogeq * cs ** 4 * k ** 4 * rhogsol * w1i ** 2
                - 2 * rhogeq * cs ** 2 * k ** 3 * w1i * Kdrag * vdsol * w1r
                + Kdrag * rhogsol * k ** 2 * cs ** 2 * w1i ** 3
                - Kdrag ** 2 * k * vdsol * w1r * w1i ** 2
                + rhogeq ** 2 * w3i ** 2 * k * w1r ** 3 * vgsol
                + Kdrag ** 2 * k * vgsol * w1r * w1i ** 2
                - Kdrag * rhogsol * w1i ** 3 * w3i * w2i
                + 2 * w1r * Kdrag * k * rhogeq * vgsol * w2i * w1i ** 2
                + 2 * w2i * w1i * rhogeq ** 2 * cs ** 2 * k ** 3 * w1r * vgsol
                - Kdrag * rhogsol * w3i * w1r ** 2 * w2i * w1i
                + rhogeq ** 2 * w3i ** 2 * k * w1r * vgsol * w1i ** 2
                + 2 * w3i * rhogeq * k * Kdrag * vdsol * w1r * w2i * w1i
                - Kdrag * rhogsol * k ** 2 * cs ** 2 * w3i * w1r ** 2
                - w3i * rhogeq * k * Kdrag * vdsol * w1r ** 3
                + 2 * rhogeq ** 2 * cs ** 2 * k ** 3 * w1r * vgsol * w3i * w1i
                + 2 * Kdrag * k * rhogeq * vgsol * w3i * w1r ** 3
                - Kdrag * rhogsol * k ** 2 * cs ** 2 * w1i ** 2 * w2i
                - Kdrag * rhogsol * k ** 2 * cs ** 2 * w3i * w1i ** 2
                + Kdrag * rhogsol * k ** 2 * cs ** 2 * w1i * w1r ** 2
                - 2 * rhogeq ** 2 * k * vgsol * w2i ** 2 * w1r * w3i * w1i
                - 2 * w1r ** 2 * w3i * rhogeq * rhogsol * w1i ** 2 * w2i
                - w1r * w1i ** 2 * vdsol * Kdrag * k * rhogeq * w2i
                - 2 * rhogeq ** 2 * cs ** 2 * k ** 3 * w1r * vgsol * w2i * w3i
                - w3i * rhogeq * k * Kdrag * vdsol * w1r * w1i ** 2
                + 2 * Kdrag * k * rhogeq * vgsol * w3i * w1r * w1i ** 2
                - w3r ** 2 * rhogeq * w2r * w1r ** 3 * rhogsol
                - w3r * w1i ** 4 * rhogeq ** 2 * k * vgsol
                + w3r * rhogeq * w2r * w1r ** 4 * rhogsol
                - w3r * rhogeq ** 2 * w1r ** 4 * k * vgsol
                + 2 * w1i ** 2 * rhogeq ** 2 * k * w1r * vgsol * w2i * w3i
                + w3r ** 2 * w2i * w1i ** 3 * rhogeq * rhogsol
                - w3r * rhogeq ** 2 * w1r ** 2 * k * vgsol * w2r ** 2
                + 2 * w1r ** 3 * rhogeq ** 2 * k * vgsol * w2i * w3i
                + w3r ** 2 * rhogeq * rhogsol * k ** 2 * cs ** 2 * w2r * w1r
            )
            / (
                w1i ** 2
                - 2 * w3i * w1i
                + w3r ** 2
                + w1r ** 2
                + w3i ** 2
                - 2 * w3r * w1r
            )
            / Kdrag
            / (
                w2r ** 2
                + w1r ** 2
                + w2i ** 2
                - 2 * w2i * w1i
                - 2 * w2r * w1r
                + w1i ** 2
            )
            / rhogeq
            / k
        )

        vd1i = (
            -(
                w3r ** 2 * rhogeq * rhogsol * w1i ** 3 * w2r
                + w3r * Kdrag * rhogsol * k ** 2 * cs ** 2 * w1i ** 2
                + w3r * rhogeq * rhogsol * k ** 2 * cs ** 2 * w1i ** 3
                - 2 * w3r * rhogeq * rhogsol * w2i * w1i ** 2 * w1r ** 2
                + w3r * rhogeq * rhogsol * w1i * w2i ** 2 * w1r ** 2
                + w3r * rhogeq * rhogsol * w2r ** 2 * w1i * w1r ** 2
                - 2 * w3r * w2r * rhogeq ** 2 * k * vgsol * w1i * w1r ** 2
                - 2 * w3r * w2r * rhogeq ** 2 * cs ** 2 * k ** 3 * vgsol * w1i
                - 2 * w3r * w2r * rhogeq ** 2 * k * vgsol * w1i ** 3
                - w3r * Kdrag * rhogsol * w2r * w1i ** 2 * w1r
                + w3r * rhogeq * rhogsol * k ** 2 * cs ** 2 * w1i * w1r ** 2
                + w3r * rhogeq * k * Kdrag * vdsol * w2r * w1i ** 2
                + w3r * Kdrag * rhogsol * k ** 2 * cs ** 2 * w1r ** 2
                - w3r * Kdrag * w2i * rhogsol * w1i * w1r ** 2
                - w3r * rhogeq * k * Kdrag * vdsol * w2r * w1r ** 2
                + w3r * rhogeq * k * Kdrag * vdsol * w1r ** 3
                - 2 * w3r * rhogeq * k * Kdrag * vdsol * w1r * w2i * w1i
                + w3r * rhogeq * k * Kdrag * vdsol * w1r * w1i ** 2
                + 2 * w3r * vgsol * k * rhogeq ** 2 * w2i ** 2 * w1i * w1r
                + 2 * w3r * Kdrag * k * rhogeq * vgsol * w1i * w2i * w1r
                + 2 * w3r * rhogeq ** 2 * cs ** 2 * k ** 3 * vgsol * w1i * w1r
                - 2 * w3r ** 2 * rhogeq * rhogsol * w1i * w2r ** 2 * w1r
                - w3r ** 2 * rhogeq ** 2 * k * vgsol * w1i * w1r ** 2
                - 2 * w3r ** 2 * rhogeq * rhogsol * w2i ** 2 * w1r * w1i
                + 2 * w3r ** 2 * rhogeq ** 2 * k * vgsol * w1i * w2r * w1r
                + w3r ** 2 * rhogeq * rhogsol * w1i * w2r * w1r ** 2
                + rhogeq ** 2 * w3i ** 2 * w2i * w1i ** 2 * k * vgsol
                - rhogeq ** 2 * w3i ** 2 * w2i * k * vgsol * w1r ** 2
                - Kdrag * w3i ** 2 * k * rhogeq * w1r ** 2 * vgsol
                - Kdrag * w1i ** 2 * rhogeq * k * vgsol * w3i ** 2
                - Kdrag * w3r ** 2 * w1i ** 2 * k * rhogeq * vgsol
                - Kdrag * w3r ** 2 * w1r ** 2 * rhogeq * k * vgsol
                - rhogeq ** 2 * w3r ** 2 * k * vgsol * w2i * w1r ** 2
                + rhogeq ** 2 * w3r ** 2 * k * vgsol * w2i * w1i ** 2
                + w3r * Kdrag * rhogsol * w2r ** 2 * w1r ** 2
                + w3r * Kdrag * rhogsol * w2r ** 2 * w1i ** 2
                - 3 * w3r * Kdrag * k * rhogeq * vgsol * w2r * w1i ** 2
                - w3r * Kdrag * k * rhogeq * vgsol * w2r * w1r ** 2
                + w3r * rhogeq * cs ** 2 * k ** 2 * rhogsol * w2r ** 2 * w1i
                - w3r * cs ** 2 * k ** 3 * rhogeq * Kdrag * vgsol * w2r
                - w3r * Kdrag * cs ** 2 * k ** 2 * rhogsol * w2r * w1r
                + w3r * cs ** 2 * k ** 3 * rhogeq * Kdrag * vdsol * w2r
                + w3r ** 2 * Kdrag * rhogsol * w2r * w1r ** 2
                + w3r ** 2 * cs ** 2 * k ** 2 * rhogeq * rhogsol * w2r * w1i
                + w3r ** 2 * Kdrag * rhogeq * k * vgsol * w2r * w1r
                - w3r ** 2 * rhogeq ** 2 * cs ** 2 * k ** 3 * vgsol * w1i
                - w3r * Kdrag ** 2 * k * vgsol * w2r * w1i
                - 4 * w3r * cs ** 2 * k ** 2 * rhogeq * rhogsol * w2r * w1i * w1r
                + 2 * w3r * rhogeq ** 2 * k * vgsol * w2r ** 2 * w1i * w1r
                - Kdrag * rhogsol * w2r * w1i ** 3 * w3i
                + cs ** 2 * k ** 2 * rhogeq * rhogsol * w2r * w1i ** 3
                + 2 * Kdrag * rhogeq * k * vgsol * w2r * w1r * w3i * w1i
                - rhogeq ** 2 * k * vgsol * w2r ** 2 * w1i ** 3
                - 2 * rhogeq * k * Kdrag * vdsol * w1r ** 2 * w1i ** 2
                - rhogeq * k * Kdrag * vdsol * w1r ** 4
                - Kdrag * rhogsol * w2r * w1r ** 2 * w3i * w1i
                + Kdrag * cs ** 2 * k ** 2 * rhogsol * w2r * w1r ** 2
                + cs ** 2 * k ** 2 * rhogeq * rhogsol * w2r * w1i * w1r ** 2
                - rhogeq ** 2 * k * vgsol * w2r ** 2 * w1i * w1r ** 2
                - vgsol * k * rhogeq ** 2 * w2i ** 2 * w1i ** 3
                - rhogeq * k * Kdrag * vdsol * w1i ** 4
                - 2 * rhogeq ** 2 * k * vgsol * w1i ** 3 * w2i * w3i
                - cs ** 2 * k ** 3 * rhogeq * Kdrag * vgsol * w1r ** 2
                + 4 * rhogeq * cs ** 2 * k ** 2 * rhogsol * w2i * w1r * w3i * w1i
                - rhogeq * cs ** 2 * k ** 2 * rhogsol * w2i * w1r * w1i ** 2
                + w3r ** 2 * rhogeq * rhogsol * w2i * w1r * w1i ** 2
                - rhogeq * cs ** 2 * k ** 2 * rhogsol * w2i * w1r ** 3
                + cs ** 2 * k ** 3 * rhogeq * Kdrag * vdsol * w1r ** 2
                - 2 * cs ** 4 * k ** 4 * rhogeq * rhogsol * w1i * w1r
                + w1i ** 3 * k * Kdrag ** 2 * vgsol
                - vgsol * k * rhogeq ** 2 * w2i ** 2 * w1i * w1r ** 2
                - Kdrag ** 2 * k * vgsol * w2i * w1r ** 2
                + rhogeq * k * Kdrag * vdsol * w2r * w1r ** 3
                + rhogeq * k * Kdrag * vdsol * w2i * w1i * w1r ** 2
                + cs ** 2 * k ** 3 * rhogeq * Kdrag * vgsol * w1i ** 2
                - cs ** 2 * k ** 3 * rhogeq * Kdrag * vdsol * w1i ** 2
                + w1i * k * Kdrag ** 2 * vgsol * w1r ** 2
                + rhogeq * w3i * w1i * k * Kdrag * vdsol * w1r ** 2
                + 2 * rhogeq ** 2 * w3i * w1i ** 2 * k * vgsol * w1r ** 2
                - rhogeq ** 2 * w3i ** 2 * w1i * k * vgsol * w1r ** 2
                - rhogeq * w3i * w1r ** 3 * cs ** 2 * k ** 2 * rhogsol
                + rhogeq * w3i * w2i * k * Kdrag * vdsol * w1r ** 2
                + rhogeq * w3i ** 2 * w2r * rhogsol * w1i * w1r ** 2
                + rhogeq * w3i ** 2 * w2i * w1r ** 3 * rhogsol
                + rhogeq * k * Kdrag * vdsol * w2r * w1r * w1i ** 2
                + rhogeq * w3i * w2i ** 2 * w1r * rhogsol * w1i ** 2
                - Kdrag * w1r * cs ** 2 * k ** 2 * rhogsol * w1i ** 2
                + Kdrag * w2i * w3i * w1r * rhogsol * w1i ** 2
                + Kdrag * w2r * cs ** 2 * k ** 2 * rhogsol * w1i ** 2
                + rhogeq * w3i ** 2 * w2i * w1r * rhogsol * w1i ** 2
                + rhogeq * w3i * w1i ** 2 * w2r ** 2 * rhogsol * w1r
                + 2 * rhogeq ** 2 * w3i ** 2 * w1i * k * vgsol * w2r * w1r
                - 2 * rhogeq * w3i * w1i * k * Kdrag * vdsol * w2r * w1r
                + rhogeq * w3i * w1i ** 3 * k * Kdrag * vdsol
                - rhogeq * w3i * w1i ** 2 * k * Kdrag * vdsol * w2i
                - rhogeq * w3i * w1r * cs ** 2 * k ** 2 * rhogsol * w1i ** 2
                - 2 * rhogeq * w3i ** 2 * w2r ** 2 * rhogsol * w1i * w1r
                + rhogeq * w3i ** 2 * w2r * rhogsol * w1i ** 3
                - 2 * rhogeq * w3i ** 2 * w2i ** 2 * w1r * rhogsol * w1i
                - 2 * rhogeq * w3i * w1i ** 2 * w2r * rhogsol * w1r ** 2
                + 2 * Kdrag * w1i ** 2 * k * rhogeq * vgsol * w1r ** 2
                - w1i ** 3 * k * Kdrag ** 2 * vdsol
                + Kdrag * w1r ** 4 * rhogeq * k * vgsol
                + rhogeq * k * Kdrag * vdsol * w2i * w1i ** 3
                + 2 * rhogeq ** 2 * k * vgsol * w2i * w1i ** 2 * w1r ** 2
                + 2 * rhogeq ** 2 * cs ** 2 * k ** 3 * vgsol * w1i * w2r * w1r
                + Kdrag ** 2 * k * vdsol * w2i * w1r ** 2
                - 2 * rhogeq ** 2 * k * vgsol * w1i * w1r ** 2 * w2i * w3i
                + w3i * k * Kdrag ** 2 * vdsol * w1r ** 2
                + rhogeq * w3i * w2i ** 2 * w1r ** 3 * rhogsol
                - w3i * k * Kdrag ** 2 * vgsol * w1r ** 2
                + rhogeq ** 2 * w3i * w1r ** 4 * k * vgsol
                - w1i * k * Kdrag ** 2 * vdsol * w1r ** 2
                - w3i * rhogeq ** 2 * k * vgsol * w2r ** 2 * w1r ** 2
                - rhogeq * cs ** 2 * k ** 3 * vdsol * Kdrag * w2r * w1r
                + w3i * Kdrag * k * rhogeq * vgsol * w2r ** 2 * w1i
                - w3i * cs ** 2 * k ** 2 * rhogeq * rhogsol * w2r ** 2 * w1r
                + 2 * w3i * cs ** 2 * k ** 2 * rhogeq * rhogsol * w2r * w1r ** 2
                - 2 * w3i * cs ** 2 * k ** 2 * rhogeq * rhogsol * w2r * w1i ** 2
                + w3i * rhogeq ** 2 * k * vgsol * w2r ** 2 * w1i ** 2
                + w3i ** 2 * Kdrag * rhogeq * k * vgsol * w2r * w1r
                - w1i ** 2 * k * Kdrag ** 2 * vgsol * w2i
                + w3i * rhogeq ** 2 * cs ** 2 * k ** 3 * vgsol * w2r ** 2
                - Kdrag * k * rhogeq * vgsol * w2r ** 2 * w1r ** 2
                - Kdrag * k * rhogeq * vgsol * w2r ** 2 * w1i ** 2
                + w3i ** 2 * cs ** 2 * k ** 2 * rhogeq * rhogsol * w2r * w1i
                + rhogeq * cs ** 4 * k ** 4 * rhogsol * w2r * w1i
                - w3i * rhogeq * cs ** 4 * k ** 4 * rhogsol * w2r
                - w3i * Kdrag * cs ** 2 * k ** 2 * rhogsol * w2r * w1i
                + w1i ** 2 * k * Kdrag ** 2 * vdsol * w2i
                - rhogeq ** 2 * w3i ** 2 * w1i ** 3 * k * vgsol
                + rhogeq ** 2 * w3i * w1i ** 4 * k * vgsol
                - rhogeq * w3i * w2r * w1r ** 4 * rhogsol
                - Kdrag * w1r ** 3 * cs ** 2 * k ** 2 * rhogsol
                + Kdrag * w2i * w3i * w1r ** 3 * rhogsol
                + rhogeq * w3i * w2r ** 2 * w1r ** 3 * rhogsol
                - w3i * k * Kdrag ** 2 * vgsol * w1i ** 2
                + w3i * k * Kdrag ** 2 * vdsol * w1i ** 2
                + rhogeq ** 2 * k * vgsol * w2i * w1r ** 4
                - rhogeq * w3i * w1i ** 4 * w2r * rhogsol
                + Kdrag * w1i ** 4 * k * rhogeq * vgsol
                + cs ** 2 * k ** 3 * rhogeq * Kdrag * vgsol * w2r * w1r
                - w3i ** 2 * cs ** 2 * k ** 3 * rhogeq ** 2 * vgsol * w1i
                - w3i ** 2 * Kdrag * rhogsol * w2r ** 2 * w1r
                - w3i * Kdrag ** 2 * k * vdsol * w2r * w1r
                + w3i ** 2 * Kdrag * rhogsol * w2r * w1r ** 2
                + w3i ** 2 * Kdrag * rhogsol * w2r * w1i ** 2
                + w3i * Kdrag ** 2 * k * vgsol * w2r * w1r
                + rhogeq ** 2 * k * vgsol * w2i * w1i ** 4
                - w3r * Kdrag * rhogsol * w2r * w1r ** 3
                + w3r * rhogeq * rhogsol * w2r ** 2 * w1i ** 3
                - w3r ** 2 * rhogeq ** 2 * k * vgsol * w1i ** 3
                + w3r ** 2 * rhogeq * rhogsol * w2i * w1r ** 3
                + w3r ** 2 * Kdrag * rhogsol * w2r * w1i ** 2
                - w3r ** 2 * Kdrag * rhogsol * w2r ** 2 * w1r
                - w3r * rhogeq * rhogsol * w2i * w1i ** 4
                + w3r * rhogeq * rhogsol * w1i ** 3 * w2i ** 2
                - w3r * Kdrag * w2i * rhogsol * w1i ** 3
                - w3r * rhogeq * rhogsol * w2i * w1r ** 4
                - rhogeq ** 2 * cs ** 2 * k ** 3 * vgsol * w1i * w2r ** 2
                + w3r * Kdrag * k * rhogeq * vgsol * w2r ** 2 * w1r
                + w3r * Kdrag ** 2 * k * vdsol * w2r * w1i
                + 2 * w3r * rhogeq * cs ** 2 * k ** 2 * rhogsol * w2i * w1r ** 2
                + w3r * cs ** 4 * k ** 4 * rhogeq * rhogsol * w1i
                - w3r * Kdrag * cs ** 2 * k ** 2 * rhogsol * w1i * w2i
                - w3r * cs ** 2 * k ** 3 * rhogeq * Kdrag * vdsol * w1r
                - w3i * Kdrag ** 2 * k * vdsol * w1i * w2i
                + w3i * Kdrag ** 2 * k * vgsol * w1i * w2i
                + w3r * Kdrag ** 2 * k * vgsol * w2i * w1r
                + cs ** 2 * k ** 3 * rhogeq * Kdrag * vdsol * w1i * w2i
                + w3i ** 2 * Kdrag * rhogeq * k * vgsol * w1i * w2i
                - Kdrag * rhogeq * k * vgsol * w2i ** 2 * w1r ** 2
                - Kdrag * rhogeq * k * vgsol * w2i ** 2 * w1i ** 2
                + w3r * Kdrag * k * rhogeq * vgsol * w2i ** 2 * w1r
                + w3i * rhogeq * k * Kdrag * vgsol * w2i ** 2 * w1i
                - 3 * w3i * rhogeq * k * Kdrag * vgsol * w2i * w1r ** 2
                - w3i * rhogeq * k * Kdrag * vgsol * w1i ** 2 * w2i
                + w3i * rhogeq ** 2 * k * vgsol * w2i ** 2 * w1i ** 2
                - w3i * rhogeq ** 2 * cs ** 2 * k ** 3 * vgsol * w1r ** 2
                + w3i * rhogeq ** 2 * cs ** 2 * k ** 3 * vgsol * w1i ** 2
                + w3r ** 2 * Kdrag * k * rhogeq * vgsol * w1i * w2i
                + w3r * cs ** 2 * k ** 3 * rhogeq * Kdrag * vgsol * w1r
                + w3i * Kdrag * cs ** 2 * k ** 2 * rhogsol * w2i * w1r
                + rhogeq * cs ** 4 * k ** 4 * rhogsol * w2i * w1r
                - w3r ** 2 * rhogeq * cs ** 2 * k ** 2 * rhogsol * w2i * w1r
                + w3r ** 2 * vgsol * rhogeq ** 2 * k ** 3 * cs ** 2 * w2i
                - w3i * rhogeq * cs ** 2 * k ** 3 * vdsol * Kdrag * w2i
                + w3i * rhogeq * cs ** 2 * k ** 3 * vdsol * Kdrag * w1i
                - cs ** 2 * k ** 3 * rhogeq * Kdrag * vgsol * w1i * w2i
                - w3i * cs ** 2 * k ** 3 * rhogeq * Kdrag * vgsol * w1i
                + w3i * cs ** 2 * k ** 3 * rhogeq * Kdrag * vgsol * w2i
                + w3i * rhogeq * cs ** 4 * k ** 4 * rhogsol * w1r
                + w3i ** 2 * vgsol * rhogeq ** 2 * k ** 3 * cs ** 2 * w2i
                - w3i ** 2 * rhogeq * cs ** 2 * k ** 2 * rhogsol * w2i * w1r
                + w3i * rhogeq ** 2 * cs ** 2 * k ** 3 * vgsol * w2i ** 2
                - 2 * w3i * rhogeq ** 2 * cs ** 2 * k ** 3 * vgsol * w2i * w1i
                - w3i * rhogeq ** 2 * k * vgsol * w2i ** 2 * w1r ** 2
                + w3r * Kdrag * rhogsol * w2i ** 2 * w1r ** 2
                + rhogeq ** 2 * cs ** 2 * k ** 3 * vgsol * w1i ** 2 * w2i
                - rhogeq ** 2 * cs ** 2 * k ** 3 * vgsol * w2i * w1r ** 2
                - w3r * Kdrag ** 2 * k * vdsol * w2i * w1r
                - w3r * cs ** 4 * k ** 4 * rhogeq * rhogsol * w2i
                - 2 * w3r * rhogeq * cs ** 2 * k ** 2 * rhogsol * w1i ** 2 * w2i
                + w3r * rhogeq * cs ** 2 * k ** 2 * rhogsol * w2i ** 2 * w1i
                + w3r * Kdrag * rhogsol * w2i ** 2 * w1i ** 2
                - w3i * rhogeq * cs ** 2 * k ** 2 * rhogsol * w2i ** 2 * w1r
                - w3i ** 2 * Kdrag * rhogsol * w2i ** 2 * w1r
                - w3r ** 2 * Kdrag * rhogsol * w2i ** 2 * w1r
                - rhogeq ** 2 * cs ** 2 * k ** 3 * vgsol * w1i * w2i ** 2
            )
            / (
                w1i ** 2
                - 2 * w3i * w1i
                + w3r ** 2
                + w1r ** 2
                + w3i ** 2
                - 2 * w3r * w1r
            )
            / Kdrag
            / (
                w2r ** 2
                + w1r ** 2
                + w2i ** 2
                - 2 * w2i * w1i
                - 2 * w2r * w1r
                + w1i ** 2
            )
            / rhogeq
            / k
        )

    # ------------------------------------------------------------------
    # GAS DENSITIES
    # ------------------------------------------------------------------
    rhog3r = (
        (
            w3r ** 2 * k * rhogeq * vgsol * w1i
            + w3r ** 2 * w2i * rhogeq * k * vgsol
            - w3r ** 2 * w2r * w1i * rhogsol
            - w3i ** 2 * w2i * rhogeq * k * vgsol
            + w3i ** 2 * w2r * w1i * rhogsol
            - w3i * w1i ** 2 * w2r * rhogsol
            - w3i * w2i ** 2 * w1r * rhogsol
            - w3i * w2r ** 2 * w1r * rhogsol
            - w3i * w2r * w1r ** 2 * rhogsol
            + w3i ** 2 * rhogsol * w2i * w1r
            - w3i * w1i * k * Kdrag * vdsol
            + k * Kdrag * vdsol * w2i * w1i
            - k * Kdrag * vgsol * w2i * w1i
            + w3i * w1i ** 2 * k * rhogeq * vgsol
            + w3i * w1r ** 2 * rhogeq * k * vgsol
            - w3i * w1r * cs ** 2 * k ** 2 * rhogsol
            - w3i * w2i * k * Kdrag * vdsol
            + w3i * w1i * k * Kdrag * vgsol
            + w3i * w2i * k * Kdrag * vgsol
            + w3i * w2r ** 2 * rhogeq * k * vgsol
            - w3i ** 2 * k * Kdrag * vgsol
            - k * Kdrag * vdsol * w2r * w1r
            + w3i ** 2 * k * Kdrag * vdsol
            - k * rhogeq * vgsol * w2r ** 2 * w1i
            - w3i * w2r * cs ** 2 * k ** 2 * rhogsol
            + w3i * w2i ** 2 * rhogeq * k * vgsol
            + k * Kdrag * vgsol * w2r * w1r
            - k * rhogeq * vgsol * w2i * w1r ** 2
            + rhogsol * k ** 2 * cs ** 2 * w2i * w1r
            + rhogsol * k ** 2 * cs ** 2 * w1i * w2r
            - k * rhogeq * vgsol * w1i * w2i ** 2
            - w3i ** 2 * k * rhogeq * vgsol * w1i
            - k * rhogeq * vgsol * w2i * w1i ** 2
            + w3r * k * Kdrag * vdsol * w2r
            + 2 * w3r * w3i * rhogsol * k ** 2 * cs ** 2
            - w3r * rhogsol * k ** 2 * cs ** 2 * w1i
            + w3r * k * Kdrag * vdsol * w1r
            - w3r * rhogsol * k ** 2 * cs ** 2 * w2i
            + 2 * w3r * w3i * w2r * w1r * rhogsol
            - 2 * w3r * w3i * w2i * rhogsol * w1i
            - w3r * k * Kdrag * vgsol * w2r
            - w3r * k * Kdrag * vgsol * w1r
            + w3r * rhogsol * w2i * w1i ** 2
            + w3r * rhogsol * w2i * w1r ** 2
            + w3r * rhogsol * w2r ** 2 * w1i
            + w3r * rhogsol * w1i * w2i ** 2
            + w3r ** 2 * k * Kdrag * vgsol
            - w3r ** 2 * k * Kdrag * vdsol
            - w3r ** 2 * rhogsol * w2i * w1r
            + 2 * w3i * w2r * k * rhogeq * w1r * vgsol
            + 2 * w3i * k * w2i * rhogeq * w1i * vgsol
            - 2 * w3r * w3i * w2r * vgsol * k * rhogeq
            - 2 * w3r * w3i * k * rhogeq * w1r * vgsol
        )
        / (w2r ** 2 - 2 * w3r * w2r + w2i ** 2 + w3i ** 2 - 2 * w2i * w3i + w3r ** 2)
        / (w1i ** 2 - 2 * w3i * w1i + w3r ** 2 + w1r ** 2 + w3i ** 2 - 2 * w3r * w1r)
    )

    rhog3i = (
        -(
            -rhogsol * w1i ** 2 * w2r ** 2
            - rhogsol * w1i ** 2 * w2i ** 2
            - rhogsol * w1r ** 2 * w2r ** 2
            - rhogsol * w1r ** 2 * w2i ** 2
            - 2 * w3r * rhogeq * k * vgsol * w2i * w1i
            + rhogsol * w2i ** 2 * w3i * w1i
            + rhogsol * k ** 2 * cs ** 2 * w1r * w3r
            - rhogsol * k ** 2 * cs ** 2 * w3i * w1i
            - vgsol * k * rhogeq * w3r * w2i ** 2
            - vgsol * Kdrag * k * w3r * w1i
            - vgsol * Kdrag * k * w3r * w2i
            + 2 * w3r * rhogeq * k * vgsol * w3i * w1i
            + rhogsol * w1i ** 2 * w2i * w3i
            + rhogsol * w1i * w2i * w3r ** 2
            + rhogsol * w2i * w3i * w1r ** 2
            - rhogsol * w1i * w2i * w3i ** 2
            + rhogsol * w2i ** 2 * w3r * w1r
            - rhogsol * k ** 2 * cs ** 2 * w2i * w3i
            - vgsol * k * rhogeq * w3r * w1i ** 2
            + w3r * rhogsol * k ** 2 * cs ** 2 * w2r
            - 2 * w3r * w2r * k * rhogeq * w1r * vgsol
            + w3r ** 2 * w2r * vgsol * k * rhogeq
            - w3r * w2r ** 2 * rhogeq * k * vgsol
            - vgsol * k * rhogeq * w3r * w1r ** 2
            + vgsol * k * rhogeq * w3r ** 2 * w1r
            + vdsol * Kdrag * k * w3r * w1i
            - vdsol * Kdrag * k * w2i * w1r
            + vdsol * Kdrag * k * w3r * w2i
            + vdsol * Kdrag * k * w3i * w1r
            + rhogsol * k ** 2 * cs ** 2 * w1i * w2i
            + vgsol * Kdrag * k * w2i * w1r
            - vgsol * Kdrag * k * w3i * w1r
            - vgsol * k * rhogeq * w1r * w3i ** 2
            + vgsol * k * rhogeq * w2i ** 2 * w1r
            - rhogsol * k ** 2 * cs ** 2 * w2r * w1r
            - w3r ** 2 * w2r * w1r * rhogsol
            + w3r * w2r * rhogsol * w1i ** 2
            + w3r * w2r * w1r ** 2 * rhogsol
            + w3r * w2r ** 2 * w1r * rhogsol
            - 2 * w3r * w1r * rhogsol * w2i * w3i
            + 2 * vgsol * Kdrag * k * w3i * w3r
            - 2 * vdsol * Kdrag * k * w3i * w3r
            - w3r ** 2 * rhogsol * k ** 2 * cs ** 2
            + 2 * w3r * rhogeq * k * vgsol * w2i * w3i
            + rhogsol * k ** 2 * cs ** 2 * w3i ** 2
            - 2 * w2r * rhogsol * w3i * w1i * w3r
            + vgsol * k * rhogeq * w2r * w1r ** 2
            + vgsol * k * rhogeq * w1r * w2r ** 2
            + vgsol * k * rhogeq * w2r * w1i ** 2
            - vgsol * k * rhogeq * w2r * w3i ** 2
            + vgsol * Kdrag * k * w1i * w2r
            - vgsol * Kdrag * k * w2r * w3i
            + rhogsol * w2r ** 2 * w3i * w1i
            + rhogsol * w2r * w1r * w3i ** 2
            + vdsol * Kdrag * k * w2r * w3i
            - vdsol * Kdrag * k * w1i * w2r
        )
        / (w2r ** 2 - 2 * w3r * w2r + w2i ** 2 + w3i ** 2 - 2 * w2i * w3i + w3r ** 2)
        / (w1i ** 2 - 2 * w3i * w1i + w3r ** 2 + w1r ** 2 + w3i ** 2 - 2 * w3r * w1r)
    )

    rhog2r = (
        (
            -w3r ** 2 * k * rhogeq * vgsol * w1i
            + w3r ** 2 * w2i * rhogeq * k * vgsol
            + w3r ** 2 * w2r * w1i * rhogsol
            + w3i ** 2 * w2i * rhogeq * k * vgsol
            + w3i ** 2 * w2r * w1i * rhogsol
            + w3i * w1i ** 2 * w2r * rhogsol
            + w3i * w2i ** 2 * w1r * rhogsol
            - w3i * w2r ** 2 * w1r * rhogsol
            + w3i * w2r * w1r ** 2 * rhogsol
            + 2 * rhogsol * w3r * w2i * w1r * w2r
            + 2 * cs ** 2 * k ** 2 * rhogsol * w2r * w2i
            - k * Kdrag * vgsol * w2i ** 2
            + k * Kdrag * vgsol * w2r ** 2
            - 2 * rhogsol * w2r * w2i * w3i * w1i
            + k * Kdrag * vdsol * w2i ** 2
            - k * Kdrag * vdsol * w2r ** 2
            - 2 * vgsol * rhogeq * k * w3r * w2r * w2i
            - 2 * vgsol * rhogeq * k * w2i * w1r * w2r
            + 2 * vgsol * rhogeq * k * w3r * w2i * w1r
            - w3i ** 2 * rhogsol * w2i * w1r
            + w3i * w1i * k * Kdrag * vdsol
            - k * Kdrag * vdsol * w2i * w1i
            + k * Kdrag * vgsol * w2i * w1i
            - w3i * w1i ** 2 * k * rhogeq * vgsol
            - w3i * w1r ** 2 * rhogeq * k * vgsol
            + w3i * w1r * cs ** 2 * k ** 2 * rhogsol
            - w3i * w2i * k * Kdrag * vdsol
            - w3i * w1i * k * Kdrag * vgsol
            + w3i * w2i * k * Kdrag * vgsol
            + w3i * w2r ** 2 * rhogeq * k * vgsol
            + k * Kdrag * vdsol * w2r * w1r
            + k * rhogeq * vgsol * w2r ** 2 * w1i
            - w3i * w2r * cs ** 2 * k ** 2 * rhogsol
            - w3i * w2i ** 2 * rhogeq * k * vgsol
            - k * Kdrag * vgsol * w2r * w1r
            + k * rhogeq * vgsol * w2i * w1r ** 2
            - rhogsol * k ** 2 * cs ** 2 * w2i * w1r
            - rhogsol * k ** 2 * cs ** 2 * w1i * w2r
            - k * rhogeq * vgsol * w1i * w2i ** 2
            - w3i ** 2 * k * rhogeq * vgsol * w1i
            + k * rhogeq * vgsol * w2i * w1i ** 2
            + w3r * k * Kdrag * vdsol * w2r
            + w3r * rhogsol * k ** 2 * cs ** 2 * w1i
            - w3r * k * Kdrag * vdsol * w1r
            - w3r * rhogsol * k ** 2 * cs ** 2 * w2i
            - w3r * k * Kdrag * vgsol * w2r
            + w3r * k * Kdrag * vgsol * w1r
            - w3r * rhogsol * w2i * w1i ** 2
            - w3r * rhogsol * w2i * w1r ** 2
            - w3r * rhogsol * w2r ** 2 * w1i
            + w3r * rhogsol * w1i * w2i ** 2
            - w3r ** 2 * rhogsol * w2i * w1r
            + 2 * w3i * k * w2i * rhogeq * w1i * vgsol
        )
        / (w2r ** 2 - 2 * w3r * w2r + w2i ** 2 + w3i ** 2 - 2 * w2i * w3i + w3r ** 2)
        / (w2r ** 2 + w1r ** 2 + w2i ** 2 - 2 * w2i * w1i - 2 * w2r * w1r + w1i ** 2)
    )

    rhog2i = (
        (
            -2 * w2i * w2r * vgsol * k * rhogeq * w1i
            + 2 * w3r * w2i * rhogsol * w1i * w2r
            - 2 * w2i * k * rhogeq * vgsol * w2r * w3i
            + rhogsol * w2i ** 2 * w3i * w1i
            + rhogsol * k ** 2 * cs ** 2 * w1r * w3r
            - rhogsol * k ** 2 * cs ** 2 * w3i * w1i
            + vgsol * k * rhogeq * w3r * w2i ** 2
            - vgsol * Kdrag * k * w3r * w1i
            + vgsol * Kdrag * k * w3r * w2i
            - rhogsol * w1i ** 2 * w2i * w3i
            - rhogsol * w1i * w2i * w3r ** 2
            - rhogsol * w2i * w3i * w1r ** 2
            - rhogsol * w1i * w2i * w3i ** 2
            - rhogsol * w2i ** 2 * w3r * w1r
            + rhogsol * k ** 2 * cs ** 2 * w2i * w3i
            - vgsol * k * rhogeq * w3r * w1i ** 2
            - w3r * rhogsol * k ** 2 * cs ** 2 * w2r
            + 2 * w3r * w2r * k * rhogeq * w1r * vgsol
            + w3r ** 2 * w2r * vgsol * k * rhogeq
            - w3r * w2r ** 2 * rhogeq * k * vgsol
            - vgsol * k * rhogeq * w3r * w1r ** 2
            - vgsol * k * rhogeq * w3r ** 2 * w1r
            + vdsol * Kdrag * k * w3r * w1i
            - vdsol * Kdrag * k * w2i * w1r
            - vdsol * Kdrag * k * w3r * w2i
            + vdsol * Kdrag * k * w3i * w1r
            + rhogsol * k ** 2 * cs ** 2 * w1i * w2i
            + vgsol * Kdrag * k * w2i * w1r
            - vgsol * Kdrag * k * w3i * w1r
            - vgsol * k * rhogeq * w1r * w3i ** 2
            + vgsol * k * rhogeq * w2i ** 2 * w1r
            - rhogsol * k ** 2 * cs ** 2 * w2r * w1r
            - w3r ** 2 * w2r * w1r * rhogsol
            - w3r * w2r * rhogsol * w1i ** 2
            - w3r * w2r * w1r ** 2 * rhogsol
            + w3r * w2r ** 2 * w1r * rhogsol
            + rhogsol * w3r ** 2 * w1i ** 2
            + rhogsol * w3r ** 2 * w1r ** 2
            + rhogsol * w1i ** 2 * w3i ** 2
            + vgsol * k * rhogeq * w2r * w1r ** 2
            - vgsol * k * rhogeq * w1r * w2r ** 2
            + vgsol * k * rhogeq * w2r * w1i ** 2
            + rhogsol * w3i ** 2 * w1r ** 2
            - w2i ** 2 * rhogsol * k ** 2 * cs ** 2
            + vgsol * k * rhogeq * w2r * w3i ** 2
            + vgsol * Kdrag * k * w1i * w2r
            + vgsol * Kdrag * k * w2r * w3i
            + w2r ** 2 * rhogsol * k ** 2 * cs ** 2
            + 2 * w2i * k * Kdrag * vdsol * w2r
            + 2 * w2i * rhogsol * w3i * w2r * w1r
            - 2 * w2i * k * Kdrag * vgsol * w2r
            + 2 * w2r * w3i * k * rhogeq * w1i * vgsol
            - rhogsol * w2r ** 2 * w3i * w1i
            - rhogsol * w2r * w1r * w3i ** 2
            - vdsol * Kdrag * k * w2r * w3i
            - vdsol * Kdrag * k * w1i * w2r
        )
        / (w2r ** 2 - 2 * w3r * w2r + w2i ** 2 + w3i ** 2 - 2 * w2i * w3i + w3r ** 2)
        / (w2r ** 2 + w1r ** 2 + w2i ** 2 - 2 * w2i * w1i - 2 * w2r * w1r + w1i ** 2)
    )

    rhog1r = (
        -(
            -w3r ** 2 * k * rhogeq * vgsol * w1i
            - 2 * rhogsol * k ** 2 * cs ** 2 * w1i * w1r
            + 2 * k * rhogeq * vgsol * w1i * w2r * w1r
            + 2 * rhogsol * w2i * w1r * w3i * w1i
            + 2 * w3r * k * rhogeq * vgsol * w1i * w1r
            - 2 * w3r * w2r * k * rhogeq * vgsol * w1i
            - 2 * w3r * w2r * w1i * rhogsol * w1r
            + w3r ** 2 * w2i * rhogeq * k * vgsol
            + w3r ** 2 * w2r * w1i * rhogsol
            + w3i ** 2 * w2i * rhogeq * k * vgsol
            + w3i ** 2 * w2r * w1i * rhogsol
            - w3i * w1i ** 2 * w2r * rhogsol
            - w3i * w2i ** 2 * w1r * rhogsol
            - w3i * w2r ** 2 * w1r * rhogsol
            + w3i * w2r * w1r ** 2 * rhogsol
            - w3i ** 2 * rhogsol * w2i * w1r
            + w3i * w1i * k * Kdrag * vdsol
            + k * Kdrag * vdsol * w2i * w1i
            - k * Kdrag * vgsol * w2i * w1i
            + w3i * w1i ** 2 * k * rhogeq * vgsol
            - w3i * w1r ** 2 * rhogeq * k * vgsol
            + w3i * w1r * cs ** 2 * k ** 2 * rhogsol
            - w3i * w2i * k * Kdrag * vdsol
            - w3i * w1i * k * Kdrag * vgsol
            + w3i * w2i * k * Kdrag * vgsol
            + w3i * w2r ** 2 * rhogeq * k * vgsol
            - k * Kdrag * vdsol * w2r * w1r
            - k * rhogeq * vgsol * w2r ** 2 * w1i
            - w3i * w2r * cs ** 2 * k ** 2 * rhogsol
            + w3i * w2i ** 2 * rhogeq * k * vgsol
            + k * Kdrag * vgsol * w2r * w1r
            - k * rhogeq * vgsol * w2i * w1r ** 2
            + rhogsol * k ** 2 * cs ** 2 * w2i * w1r
            + rhogsol * k ** 2 * cs ** 2 * w1i * w2r
            - k * rhogeq * vgsol * w1i * w2i ** 2
            - w3i ** 2 * k * rhogeq * vgsol * w1i
            + k * rhogeq * vgsol * w2i * w1i ** 2
            + w3r * k * Kdrag * vdsol * w2r
            + w3r * rhogsol * k ** 2 * cs ** 2 * w1i
            - w3r * k * Kdrag * vdsol * w1r
            - w3r * rhogsol * k ** 2 * cs ** 2 * w2i
            - w3r * k * Kdrag * vgsol * w2r
            + w3r * k * Kdrag * vgsol * w1r
            - Kdrag * w1r ** 2 * k * vgsol
            - w3r * rhogsol * w2i * w1i ** 2
            + w3r * rhogsol * w2i * w1r ** 2
            + Kdrag * w1i ** 2 * k * vgsol
            + w3r * rhogsol * w2r ** 2 * w1i
            + w3r * rhogsol * w1i * w2i ** 2
            + k * Kdrag * vdsol * w1r ** 2
            - w3r ** 2 * rhogsol * w2i * w1r
            - k * Kdrag * vdsol * w1i ** 2
            - 2 * w3i * k * w2i * rhogeq * w1i * vgsol
        )
        / (w1i ** 2 - 2 * w3i * w1i + w3r ** 2 + w1r ** 2 + w3i ** 2 - 2 * w3r * w1r)
        / (w2r ** 2 + w1r ** 2 + w2i ** 2 - 2 * w2i * w1i - 2 * w2r * w1r + w1i ** 2)
    )

    rhog1i = (
        -(
            -2 * vdsol * Kdrag * k * w1i * w1r
            + rhogsol * w2i ** 2 * w3i * w1i
            + rhogsol * k ** 2 * cs ** 2 * w1i ** 2
            - rhogsol * k ** 2 * cs ** 2 * w1r ** 2
            + rhogsol * k ** 2 * cs ** 2 * w1r * w3r
            - rhogsol * k ** 2 * cs ** 2 * w3i * w1i
            + vgsol * k * rhogeq * w3r * w2i ** 2
            + 2 * vgsol * Kdrag * k * w1i * w1r
            - vgsol * Kdrag * k * w3r * w1i
            + vgsol * Kdrag * k * w3r * w2i
            - rhogsol * w1i ** 2 * w2i * w3i
            + rhogsol * w1i * w2i * w3r ** 2
            - 2 * rhogsol * w1i * w2i * w3r * w1r
            + rhogsol * w2i * w3i * w1r ** 2
            + rhogsol * w1i * w2i * w3i ** 2
            + rhogsol * w2i ** 2 * w3r * w1r
            + rhogsol * k ** 2 * cs ** 2 * w2i * w3i
            - vgsol * k * rhogeq * w3r * w1i ** 2
            - w3r * rhogsol * k ** 2 * cs ** 2 * w2r
            - 2 * w3r * w2r * k * rhogeq * w1r * vgsol
            + w3r ** 2 * w2r * vgsol * k * rhogeq
            + w3r * w2r ** 2 * rhogeq * k * vgsol
            + vgsol * k * rhogeq * w3r * w1r ** 2
            + 2 * vgsol * k * rhogeq * w1r * w2i * w1i
            - vgsol * k * rhogeq * w3r ** 2 * w1r
            + vdsol * Kdrag * k * w3r * w1i
            + vdsol * Kdrag * k * w2i * w1r
            - vdsol * Kdrag * k * w3r * w2i
            + vdsol * Kdrag * k * w3i * w1r
            - w3r ** 2 * rhogsol * w2r ** 2
            - rhogsol * k ** 2 * cs ** 2 * w1i * w2i
            - vgsol * Kdrag * k * w2i * w1r
            - vgsol * Kdrag * k * w3i * w1r
            + 2 * vgsol * k * rhogeq * w1i * w1r * w3i
            - vgsol * k * rhogeq * w1r * w3i ** 2
            - 2 * vgsol * k * rhogeq * w2i * w1r * w3i
            - vgsol * k * rhogeq * w2i ** 2 * w1r
            + rhogsol * k ** 2 * cs ** 2 * w2r * w1r
            + w3r ** 2 * w2r * w1r * rhogsol
            + w3r * w2r * rhogsol * w1i ** 2
            - w3r * w2r * w1r ** 2 * rhogsol
            + w3r * w2r ** 2 * w1r * rhogsol
            + vgsol * k * rhogeq * w2r * w1r ** 2
            - vgsol * k * rhogeq * w1r * w2r ** 2
            - vgsol * k * rhogeq * w2r * w1i ** 2
            - rhogsol * w2i ** 2 * w3i ** 2
            - rhogsol * w2i ** 2 * w3r ** 2
            + vgsol * k * rhogeq * w2r * w3i ** 2
            - vgsol * Kdrag * k * w1i * w2r
            + vgsol * Kdrag * k * w2r * w3i
            - 2 * rhogsol * w2r * w1r * w3i * w1i
            + rhogsol * w2r ** 2 * w3i * w1i
            + rhogsol * w2r * w1r * w3i ** 2
            - vdsol * Kdrag * k * w2r * w3i
            + vdsol * Kdrag * k * w1i * w2r
            - rhogsol * w2r ** 2 * w3i ** 2
        )
        / (w1i ** 2 - 2 * w3i * w1i + w3r ** 2 + w1r ** 2 + w3i ** 2 - 2 * w3r * w1r)
        / (w2r ** 2 + w1r ** 2 + w2i ** 2 - 2 * w2i * w1i - 2 * w2r * w1r + w1i ** 2)
    )

    # ------------------------------------------------------------------
    # DUST DENSITIES
    # ------------------------------------------------------------------
    if Kdrag > 0.0:

        rhod3r = -rhodeq * (
            -w3r ** 3 * rhogeq * w1i ** 2 * w2r ** 2 * rhogsol
            - w3r ** 2 * Kdrag ** 2 * k * vgsol * w2r * w1r
            + w3r ** 2 * Kdrag ** 2 * k * vgsol * w2i * w1i
            - w3r ** 2 * Kdrag ** 2 * k * vdsol * w2i * w1i
            + w3r ** 2 * Kdrag * k * rhogeq * vgsol * w2i * w1i ** 2
            + w3r ** 2 * Kdrag ** 2 * k * vdsol * w2r * w1r
            - w3r ** 2 * Kdrag * rhogsol * k ** 2 * cs ** 2 * w2i * w1r
            + w3r ** 2 * Kdrag * k * rhogeq * vgsol * w2i * w1r ** 2
            - w3r ** 2 * Kdrag * rhogsol * k ** 2 * cs ** 2 * w1i * w2r
            + w3r ** 2 * rhogeq ** 2 * w2i * w1i ** 2 * w3i * k * vgsol
            - 4 * w3r ** 2 * rhogeq ** 2 * w3i ** 2 * k * w2i * w1i * vgsol
            - 2 * w3r ** 2 * rhogeq * w2i * w3i ** 2 * k * Kdrag * vgsol
            + w3r ** 2 * rhogeq * w2r * w3i * w1r * k * Kdrag * vdsol
            - 2 * w3r ** 2 * rhogeq ** 2 * w1i ** 2 * k * vgsol * w3i ** 2
            - w3r ** 2 * rhogeq * w2i * w3i * w1i * k * Kdrag * vgsol
            + w3r ** 5 * rhogeq * w2i * rhogsol * w1i
            + w3r ** 2 * rhogeq ** 2 * w2i * k * vgsol * w3i * w1r ** 2
            - w3r ** 5 * rhogeq * rhogsol * k ** 2 * cs ** 2
            - w3r ** 5 * rhogeq * w2r * w1r * rhogsol
            + w3r ** 5 * rhogeq ** 2 * k * w1r * vgsol
            + 3 * w3r ** 2 * rhogeq * cs ** 2 * k ** 3 * w3i * Kdrag * vdsol
            + w3r ** 5 * rhogeq ** 2 * w2r * vgsol * k
            + 2 * w3r ** 2 * rhogeq * w3i ** 2 * cs ** 2 * k ** 2 * rhogsol * w1r
            - w3r ** 2 * rhogeq * cs ** 4 * k ** 4 * w2r * rhogsol
            - 2 * w3r ** 2 * rhogeq ** 2 * w3i ** 2 * k * w1r ** 2 * vgsol
            + 2 * w3r ** 2 * rhogeq * w2r * w3i ** 2 * cs ** 2 * k ** 2 * rhogsol
            - w3r * rhogeq ** 2 * cs ** 2 * k ** 3 * w1r * vgsol * w2i ** 2
            - w3r ** 3 * Kdrag * rhogsol * w2i * w1i ** 2
            - w3r ** 3 * Kdrag * rhogsol * w2r ** 2 * w1i
            - w3r ** 3 * Kdrag * rhogsol * w1i * w2i ** 2
            - w3r ** 3 * Kdrag * rhogsol * w2i * w1r ** 2
            - w3r ** 3 * rhogeq * w2r * w1i * k * Kdrag * vdsol
            + w3r ** 3 * rhogeq ** 2 * w2r * w1r ** 2 * k * vgsol
            + w3r ** 3 * rhogeq ** 2 * w2r * w1i ** 2 * k * vgsol
            - w3r ** 3 * rhogeq ** 2 * cs ** 2 * k ** 3 * w1r * vgsol
            + w3r ** 3 * rhogeq * w2r * w1i * k * Kdrag * vgsol
            - w3r ** 3 * Kdrag ** 2 * k * vdsol * w2r
            + 2 * w3r ** 3 * Kdrag * k * rhogeq * vgsol * w3i * w1r
            + 2 * w3r ** 3 * rhogeq ** 2 * w2r * k * vgsol * w3i ** 2
            + 2 * w3r ** 3 * Kdrag * k * rhogeq * vgsol * w2r * w3i
            + w3r ** 3 * Kdrag * rhogsol * k ** 2 * cs ** 2 * w1i
            - w3r ** 3 * rhogeq ** 2 * cs ** 2 * k ** 3 * w2r * vgsol
            - 2 * w3r ** 3 * Kdrag * rhogsol * k ** 2 * cs ** 2 * w3i
            + w3r ** 3 * Kdrag * rhogsol * k ** 2 * cs ** 2 * w2i
            + 2 * w3r ** 3 * Kdrag * rhogsol * w2i * w3i * w1i
            - w3r ** 3 * rhogeq * w2i ** 2 * w1i ** 2 * rhogsol
            - 2 * w3r ** 3 * Kdrag * rhogsol * w3i * w2r * w1r
            - w3r ** 2 * rhogeq * w2i * w3i * w1i * k * Kdrag * vdsol
            - w3r ** 3 * rhogeq * w1r ** 2 * w2i ** 2 * rhogsol
            - w3r ** 2 * rhogeq * cs ** 4 * k ** 4 * w1r * rhogsol
            - 3 * w3r ** 2 * rhogeq ** 2 * cs ** 2 * k ** 3 * w3i * w1i * vgsol
            - 3 * w3r ** 2 * rhogeq * cs ** 2 * k ** 3 * w3i * Kdrag * vgsol
            - w3r ** 2 * rhogeq * cs ** 2 * k ** 3 * w1i * Kdrag * vdsol
            - w3r ** 2 * rhogeq * cs ** 2 * k ** 3 * w2i * Kdrag * vdsol
            + w3r ** 2 * rhogeq * cs ** 2 * k ** 3 * w2i * Kdrag * vgsol
            + w3r ** 2 * rhogeq * cs ** 2 * k ** 3 * w1i * Kdrag * vgsol
            - w3r ** 3 * rhogeq * w2r ** 2 * w1r ** 2 * rhogsol
            + w3r ** 2 * rhogeq ** 2 * w2i ** 2 * w3i * w1i * k * vgsol
            + w3r ** 2 * rhogeq ** 2 * cs ** 2 * k ** 3 * w1i ** 2 * vgsol
            + w3r ** 3 * rhogeq * cs ** 4 * k ** 4 * rhogsol
            - 3 * w3r ** 2 * rhogeq ** 2 * cs ** 2 * k ** 3 * w3i * w2i * vgsol
            + 2 * w3r ** 2 * rhogeq * w2r * w3i ** 2 * w1r ** 2 * rhogsol
            + w3r ** 2 * rhogeq ** 2 * cs ** 2 * k ** 3 * w1r ** 2 * vgsol
            - 2 * w3r ** 2 * rhogeq ** 2 * w3i ** 2 * w2i ** 2 * k * vgsol
            + 2 * w3r ** 2 * rhogeq * w2i ** 2 * w3i ** 2 * w1r * rhogsol
            + w3r ** 2 * rhogeq ** 2 * w3i * w2r ** 2 * w1i * k * vgsol
            + 2 * w3r ** 2 * rhogeq * w1i * w3i ** 2 * k * Kdrag * vdsol
            + 2 * w3r ** 2 * rhogeq * w1i ** 2 * w3i ** 2 * w2r * rhogsol
            - 2 * w3r ** 2 * rhogeq * w1i * w3i ** 2 * k * Kdrag * vgsol
            + 2 * w3r ** 2 * rhogeq * w2r ** 2 * w3i ** 2 * w1r * rhogsol
            - 4 * w3r ** 2 * rhogeq ** 2 * w2r * w3i ** 2 * k * w1r * vgsol
            - 3 * w3r ** 2 * rhogeq * w2r * w3i * w1r * k * Kdrag * vgsol
            - 2 * w3r ** 2 * rhogeq ** 2 * w2r ** 2 * w3i ** 2 * k * vgsol
            + 2 * w3r ** 2 * rhogeq * w2i * w3i ** 2 * k * Kdrag * vdsol
            + w3r ** 3 * rhogeq ** 2 * w2r ** 2 * k * w1r * vgsol
            - 2 * w3r ** 3 * rhogeq * w3i ** 2 * cs ** 2 * k ** 2 * rhogsol
            + w3r ** 4 * rhogeq * w2i ** 2 * w1r * rhogsol
            + w3r ** 4 * Kdrag * rhogsol * w1i * w2r
            + w3r ** 4 * rhogeq * w2r * w1r ** 2 * rhogsol
            + w3r ** 4 * rhogeq * w1i ** 2 * w2r * rhogsol
            - 2 * w3r ** 4 * rhogeq ** 2 * k * w2i * w1i * vgsol
            - 2 * w3r ** 4 * rhogeq * w1i * k * Kdrag * vgsol
            - w3r ** 4 * rhogeq ** 2 * w2r ** 2 * k * vgsol
            + w3r ** 4 * rhogeq * w1i * k * Kdrag * vdsol
            - w3r ** 4 * rhogeq ** 2 * w1r ** 2 * k * vgsol
            + w3r ** 4 * rhogeq * w1r * cs ** 2 * k ** 2 * rhogsol
            + w3r ** 4 * rhogeq * w2r * cs ** 2 * k ** 2 * rhogsol
            - 2 * w3r ** 4 * rhogeq * w2i * k * Kdrag * vgsol
            - w3r ** 4 * rhogeq ** 2 * w1i ** 2 * k * vgsol
            + w3r ** 4 * rhogeq * w2r ** 2 * w1r * rhogsol
            + w3r ** 3 * rhogeq * w2i * w1r * k * Kdrag * vgsol
            + w3r ** 4 * Kdrag ** 2 * k * vdsol
            + 2 * w3r ** 3 * rhogeq * w1i * w3i ** 2 * rhogsol * w2i
            - 2 * w3r ** 4 * rhogeq ** 2 * w2r * k * w1r * vgsol
            + 2 * w3r ** 3 * rhogeq ** 2 * w3i ** 2 * k * w1r * vgsol
            - w3r ** 3 * rhogeq * w2i * w1r * k * Kdrag * vdsol
            + w3r ** 3 * rhogeq ** 2 * w2i ** 2 * w1r * k * vgsol
            - 2 * w3r ** 3 * rhogeq * w2r * w3i ** 2 * w1r * rhogsol
            + w3r ** 4 * rhogeq * w2i * k * Kdrag * vdsol
            - w3r ** 4 * rhogeq ** 2 * w2i ** 2 * k * vgsol
            + w3r ** 3 * Kdrag ** 2 * k * vgsol * w1r
            + w3r ** 4 * Kdrag * rhogsol * w2i * w1r
            - w3r ** 4 * Kdrag ** 2 * k * vgsol
            - w3r ** 3 * Kdrag ** 2 * k * vdsol * w1r
            + w3r ** 3 * Kdrag ** 2 * k * vgsol * w2r
            + w1i * w3i ** 3 * k * Kdrag ** 2 * vdsol
            + 2 * w3r * w3i * cs ** 2 * k ** 3 * rhogeq * Kdrag * vgsol * w1r
            + w3r * rhogeq ** 2 * w3i ** 4 * w2r * k * vgsol
            + 2 * w3r * w3i * cs ** 2 * k ** 3 * rhogeq * Kdrag * vgsol * w2r
            - 2 * w3r * w3i * cs ** 2 * k ** 3 * rhogeq * Kdrag * vdsol * w1r
            - 2 * w3r * w3i * cs ** 2 * k ** 3 * rhogeq * Kdrag * vdsol * w2r
            + w3r * rhogeq ** 2 * w3i ** 4 * k * w1r * vgsol
            - w3r * rhogeq * w3i ** 4 * w2r * w1r * rhogsol
            + 2 * w3r * w3i * cs ** 4 * k ** 4 * rhogeq * rhogsol * w2i
            + w3r * rhogeq * w3i ** 4 * w1i * rhogsol * w2i
            - w2i * w3i ** 3 * k * Kdrag ** 2 * vgsol
            + 2 * w3r * rhogeq * w3i ** 3 * k * Kdrag * vgsol * w2r
            + w2i * w3i ** 3 * k * Kdrag ** 2 * vdsol
            - w1i * w3i ** 3 * k * Kdrag ** 2 * vgsol
            - w3r * rhogeq * w3i ** 4 * cs ** 2 * k ** 2 * rhogsol
            - 2 * w3r * w3i * cs ** 2 * k ** 2 * rhogeq * rhogsol * w1i * w2i ** 2
            - 2 * w3r * w3i * cs ** 2 * k ** 2 * rhogeq * rhogsol * w2r ** 2 * w1i
        )

        rhod3r = rhod3r - rhodeq * (
            2 * w3r * w3i ** 3 * Kdrag * rhogsol * w2i * w1i
            - 2 * w3r * w3i ** 3 * Kdrag * rhogsol * w2r * w1r
            - 2 * w3r * w3i ** 3 * Kdrag * rhogsol * k ** 2 * cs ** 2
            + w3r * w3i ** 2 * Kdrag ** 2 * k * vgsol * w1r
            + 2 * w3r * w3i * cs ** 4 * k ** 4 * rhogeq * rhogsol * w1i
            + w3r * rhogeq * cs ** 2 * k ** 2 * w1i ** 2 * w2r ** 2 * rhogsol
            + w3r * rhogeq * cs ** 2 * k ** 2 * w2r ** 2 * w1r ** 2 * rhogsol
            - w3r * rhogeq ** 2 * cs ** 2 * k ** 3 * w2r ** 2 * w1r * vgsol
            + w3r ** 2 * rhogeq ** 2 * cs ** 2 * k ** 3 * w2r ** 2 * vgsol
            + w3r ** 2 * rhogeq * w2r ** 2 * w1i * k * Kdrag * vgsol
            - 2 * w3r * w3i * cs ** 2 * k ** 2 * rhogeq * rhogsol * w2i * w1i ** 2
            - 2 * w3r * w3i * cs ** 2 * k ** 2 * rhogeq * rhogsol * w2i * w1r ** 2
            + w1r * w3i ** 2 * cs ** 2 * k ** 2 * rhogeq * rhogsol * w2i ** 2
            - w1r * w3r ** 2 * cs ** 2 * k ** 2 * rhogeq * rhogsol * w2i ** 2
            - w2r * rhogeq * w1i ** 2 * rhogsol * k ** 2 * cs ** 2 * w3r ** 2
            - w2r * w3r * rhogeq * cs ** 2 * k ** 3 * w1i * Kdrag * vgsol
            + w2r * w3r * rhogeq * cs ** 2 * k ** 3 * w1i * Kdrag * vdsol
            - w2r * w3r * rhogeq ** 2 * cs ** 2 * k ** 3 * w1i ** 2 * vgsol
            - w2r * w3r * rhogeq ** 2 * cs ** 2 * k ** 3 * w1r ** 2 * vgsol
            + w2r * w3r * rhogeq * cs ** 4 * k ** 4 * w1r * rhogsol
            + 2 * w2r * w3r ** 2 * rhogeq ** 2 * cs ** 2 * k ** 3 * w1r * vgsol
            + rhogeq * cs ** 2 * k ** 3 * w3i * Kdrag * vdsol * w2r * w1r
            - w3r * w3i ** 2 * rhogeq * w2i * w1r * k * Kdrag * vdsol
            + 3 * w3r * w3i ** 2 * rhogeq ** 2 * cs ** 2 * k ** 3 * w2r * vgsol
            + w3r * w3i ** 2 * Kdrag * rhogsol * k ** 2 * cs ** 2 * w1i
            + 2 * w3r * rhogeq * w3i ** 3 * k * Kdrag * vgsol * w1r
            + rhogeq * w3i ** 2 * cs ** 2 * k ** 2 * rhogsol * w1r * w2r ** 2
            - w3r ** 2 * rhogeq * rhogsol * k ** 2 * cs ** 2 * w2r ** 2 * w1r
            - w2r * w3r ** 2 * cs ** 2 * k ** 2 * rhogeq * rhogsol * w1r ** 2
            + rhogeq * w2r * w3i ** 2 * cs ** 2 * k ** 2 * rhogsol * w1r ** 2
            - w3r * w3i ** 2 * Kdrag * rhogsol * w2r ** 2 * w1i
            - w3r * w3i ** 2 * Kdrag ** 2 * k * vdsol * w2r
            + w3r * w3i ** 2 * Kdrag ** 2 * k * vgsol * w2r
            - rhogeq * cs ** 2 * k ** 3 * w3i * Kdrag * vgsol * w2r * w1r
            + rhogeq ** 2 * cs ** 2 * k ** 3 * w3i * w1i * vgsol * w2r ** 2
            - w3r * w3i ** 2 * Kdrag * rhogsol * w2i * w1r ** 2
            + w3r * w3i ** 2 * Kdrag * rhogsol * k ** 2 * cs ** 2 * w2i
            - w3r * w3i ** 2 * Kdrag * rhogsol * w1i * w2i ** 2
            - 2 * w3i ** 2 * cs ** 2 * k ** 3 * rhogeq ** 2 * vgsol * w2r * w1r
            - w3i ** 2 * cs ** 2 * k ** 3 * rhogeq ** 2 * vgsol * w2r ** 2
            - w3r * w3i ** 2 * Kdrag * rhogsol * w2i * w1i ** 2
            - w3i * rhogsol * k ** 4 * cs ** 4 * rhogeq * w2r * w1i
            + rhogeq * cs ** 2 * k ** 2 * w1i ** 2 * w2r * rhogsol * w3i ** 2
            + Kdrag * k * rhogeq * vgsol * w2r ** 2 * w1i * w3i ** 2
            - w3r * w3i ** 2 * Kdrag ** 2 * k * vdsol * w1r
            - rhogeq * w3i ** 5 * w2r * w1i * rhogsol
            + w3r * w3i ** 2 * rhogeq * w2r * w1i * k * Kdrag * vgsol
            + w3r * w3i ** 2 * rhogeq ** 2 * w2r * w1i ** 2 * k * vgsol
            + w3r * w3i ** 2 * rhogeq * w2i * w1r * k * Kdrag * vgsol
            - w3r * w3i ** 2 * rhogeq * w2i ** 2 * w1i ** 2 * rhogsol
            + w3r * w3i ** 2 * rhogeq ** 2 * w2r ** 2 * k * w1r * vgsol
            - w3r * w3i ** 2 * rhogeq * w1r ** 2 * w2i ** 2 * rhogsol
            - w3r * w3i ** 2 * rhogeq * w2r ** 2 * w1r ** 2 * rhogsol
            - 3 * w3r * w3i ** 2 * rhogeq * cs ** 4 * k ** 4 * rhogsol
            + w3r * w3i ** 2 * rhogeq ** 2 * w2i ** 2 * w1r * k * vgsol
            - 4 * cs ** 2 * k ** 2 * rhogeq * w3r * w3i ** 2 * w2r * w1r * rhogsol
            - rhogeq ** 2 * w3i ** 4 * w1i ** 2 * k * vgsol
            - cs ** 2 * k ** 3 * rhogeq * w3i ** 3 * Kdrag * vdsol
            - w3r * w3i ** 2 * rhogeq * w1i ** 2 * w2r ** 2 * rhogsol
            + cs ** 2 * k ** 3 * rhogeq * w3i ** 3 * Kdrag * vgsol
            + 3 * w3r * w3i ** 2 * rhogeq ** 2 * cs ** 2 * k ** 3 * w1r * vgsol
            - w3r * w3i ** 2 * rhogeq * w2r * w1i * k * Kdrag * vdsol
            + w3r * w3i ** 2 * rhogeq ** 2 * w2r * w1r ** 2 * k * vgsol
            + 4 * cs ** 2 * k ** 2 * rhogeq * w3r * w3i ** 2 * w2i * rhogsol * w1i
            - rhogeq ** 2 * w3i ** 4 * k * w1r ** 2 * vgsol
            + 2 * rhogeq * w3i ** 3 * w3r ** 2 * k * Kdrag * vgsol
            + 2 * rhogeq ** 2 * w3i ** 3 * w3r ** 2 * w1i * k * vgsol
            + rhogeq * w3i ** 4 * cs ** 2 * k ** 2 * rhogsol * w1r
            - 2 * rhogeq * w3i ** 3 * w3r ** 2 * k * Kdrag * vdsol
            + rhogeq * w3i ** 4 * w2r * cs ** 2 * k ** 2 * rhogsol
            - 2 * rhogeq ** 2 * w3i ** 4 * k * w2i * w1i * vgsol
            - rhogeq * w3i ** 3 * w2i * w1i * k * Kdrag * vdsol
            + rhogeq ** 2 * w3i ** 3 * w2i * w1i ** 2 * k * vgsol
            + Kdrag * w2i ** 2 * vgsol * k * rhogeq * w1i * w3r ** 2
            - w3i ** 2 * rhogeq ** 2 * cs ** 2 * k ** 3 * vgsol * w2i ** 2
            + 2 * w3r ** 2 * rhogeq ** 2 * cs ** 2 * k ** 3 * vgsol * w2i * w1i
            - w3i * cs ** 2 * k ** 3 * rhogeq * Kdrag * vdsol * w2i * w1i
            - rhogeq ** 2 * w3i ** 4 * w2r ** 2 * k * vgsol
            + w3i * cs ** 2 * k ** 3 * rhogeq ** 2 * vgsol * w2i ** 2 * w1i
            + w3i * cs ** 2 * k ** 3 * rhogeq ** 2 * vgsol * w2i * w1r ** 2
            + w3i * cs ** 2 * k ** 3 * rhogeq ** 2 * vgsol * w2i * w1i ** 2
            + rhogeq * w3i ** 4 * w1i ** 2 * w2r * rhogsol
            - rhogeq ** 2 * w3i ** 4 * w2i ** 2 * k * vgsol
        )

        rhod3r = rhod3r - rhodeq * (
            rhogeq ** 2 * w3i ** 5 * w2i * k * vgsol
            + rhogeq * w3i ** 3 * w2r * w1r * k * Kdrag * vdsol
            + rhogeq * w3i ** 4 * w2i * k * Kdrag * vdsol
            + rhogeq ** 2 * w3i ** 3 * w2i * k * vgsol * w1r ** 2
            - 2 * rhogeq ** 2 * w3i ** 4 * w2r * k * w1r * vgsol
            + w3r * rhogeq * cs ** 2 * k ** 2 * w1r ** 2 * rhogsol * w2i ** 2
            + w3r * rhogeq * cs ** 2 * k ** 2 * w1i ** 2 * rhogsol * w2i ** 2
            - 2 * rhogeq * w3i ** 3 * w3r ** 2 * w2i * w1r * rhogsol
            - 2 * rhogeq * w3i ** 3 * w2r * w3r ** 2 * rhogsol * w1i
            + w3r ** 2 * rhogeq ** 2 * cs ** 2 * k ** 3 * vgsol * w2i ** 2
            - w3i * cs ** 4 * k ** 4 * rhogeq * rhogsol * w2i * w1r
            - w3r * rhogeq * cs ** 4 * k ** 4 * rhogsol * w2i * w1i
            + 2 * rhogeq ** 2 * w3i ** 3 * w3r ** 2 * w2i * k * vgsol
            + w3r * rhogeq * cs ** 2 * k ** 3 * w2i * Kdrag * vdsol * w1r
            - w3r * rhogeq * cs ** 2 * k ** 3 * w2i * Kdrag * vgsol * w1r
            + w3i * cs ** 2 * k ** 3 * rhogeq * Kdrag * vgsol * w2i * w1i
            + Kdrag * w3i ** 3 * cs ** 2 * k ** 2 * rhogsol * w1r
            + rhogeq * w3i ** 4 * w2i ** 2 * w1r * rhogsol
            - Kdrag * w3i ** 3 * k * rhogeq * w1r ** 2 * vgsol
            + Kdrag * w2r * w3i ** 3 * cs ** 2 * k ** 2 * rhogsol
            - rhogeq * w3i ** 5 * w1r * w2i * rhogsol
            - Kdrag * w1r * w2i * w3i ** 4 * rhogsol
            - rhogeq * w3i ** 5 * k * Kdrag * vdsol
            + rhogeq * w3i ** 5 * k * Kdrag * vgsol
            - 2 * w3i ** 2 * rhogeq ** 2 * cs ** 2 * k ** 3 * vgsol * w2i * w1i
            + rhogeq ** 2 * w3i ** 5 * w1i * k * vgsol
            + rhogeq * w3i ** 4 * w2r ** 2 * w1r * rhogsol
            + rhogeq * w3i ** 4 * w1i * k * Kdrag * vdsol
            - Kdrag * w1i ** 2 * rhogeq * k * vgsol * w3i ** 3
            + rhogeq * w3i ** 4 * w2r * w1r ** 2 * rhogsol
            - Kdrag * w2r * w1i * w3i ** 4 * rhogsol
            + w3i ** 4 * k * Kdrag ** 2 * vgsol
            - w3i ** 4 * k * Kdrag ** 2 * vdsol
            + w3i ** 3 * rhogeq ** 2 * k * vgsol * w2r ** 2 * w1i
            - w3i ** 2 * Kdrag ** 2 * k * vgsol * w2r * w1r
            + w3i ** 3 * Kdrag * rhogsol * w2r * w1r ** 2
            + w3i ** 3 * Kdrag * rhogsol * w2r * w1i ** 2
            + w3i ** 2 * Kdrag ** 2 * k * vdsol * w2r * w1r
            + w3i ** 3 * cs ** 2 * k ** 3 * rhogeq ** 2 * vgsol * w1i
            - w3i ** 2 * Kdrag * cs ** 2 * k ** 2 * rhogsol * w2r * w1i
            + w3i ** 3 * Kdrag * rhogsol * w2r ** 2 * w1r
            + w3i ** 2 * rhogeq * cs ** 4 * k ** 4 * rhogsol * w2r
            - 2 * w3i ** 3 * cs ** 2 * k ** 2 * rhogeq * rhogsol * w2r * w1i
            - w3i ** 3 * Kdrag * rhogeq * k * vgsol * w2r ** 2
            - 3 * w3i ** 3 * Kdrag * rhogeq * k * vgsol * w2r * w1r
            - w3i ** 3 * Kdrag * rhogeq * k * vgsol * w2i ** 2
            - w3i ** 2 * rhogeq ** 2 * cs ** 2 * k ** 3 * vgsol * w1i ** 2
            + w3i ** 2 * rhogeq * k * Kdrag * vgsol * w1i ** 2 * w2i
            - w3i ** 2 * rhogeq ** 2 * cs ** 2 * k ** 3 * vgsol * w1r ** 2
            + w3i ** 2 * rhogeq * k * Kdrag * vgsol * w2i * w1r ** 2
            - w3i ** 3 * Kdrag * rhogeq * k * vgsol * w1i * w2i
            - w3i ** 2 * Kdrag ** 2 * k * vdsol * w1i * w2i
            + w3i ** 2 * Kdrag ** 2 * k * vgsol * w1i * w2i
            - w3i ** 2 * cs ** 2 * k ** 3 * rhogeq * Kdrag * vgsol * w2i
            - w3i ** 2 * cs ** 2 * k ** 3 * rhogeq * Kdrag * vgsol * w1i
            + w3i ** 3 * Kdrag * rhogsol * w2i ** 2 * w1r
            + w3i ** 2 * rhogeq * cs ** 2 * k ** 3 * vdsol * Kdrag * w1i
            + w3i ** 2 * rhogeq * cs ** 2 * k ** 3 * vdsol * Kdrag * w2i
            - w3i ** 2 * Kdrag * cs ** 2 * k ** 2 * rhogsol * w2i * w1r
            - 2 * w3i ** 3 * rhogeq * cs ** 2 * k ** 2 * rhogsol * w2i * w1r
            + w3i ** 3 * vgsol * k * rhogeq ** 2 * w2i ** 2 * w1i
            + w3i ** 3 * vgsol * rhogeq ** 2 * k ** 3 * cs ** 2 * w2i
            + w3i ** 2 * rhogeq * cs ** 4 * k ** 4 * rhogsol * w1r
            - w3i * Kdrag * w3r ** 2 * w1r ** 2 * rhogeq * k * vgsol
            - w3i * Kdrag * w3r ** 2 * w1i ** 2 * k * rhogeq * vgsol
            - w3i * w3r ** 2 * w1i * k * Kdrag ** 2 * vgsol
            + w3i * w3r ** 2 * w1i * k * Kdrag ** 2 * vdsol
            + w3i * Kdrag * w3r ** 2 * w1r * cs ** 2 * k ** 2 * rhogsol
            + w3i * Kdrag * w2r * w3r ** 2 * cs ** 2 * k ** 2 * rhogsol
            + Kdrag * w2i ** 2 * k * rhogeq * vgsol * w3i ** 2 * w1i
            - w3i * rhogeq * w3r ** 4 * k * Kdrag * vdsol
            - w3i * rhogeq * w3r ** 4 * rhogsol * w2i * w1r
            - w3i * rhogeq * w3r ** 4 * rhogsol * w1i * w2r
            + w3i * rhogeq ** 2 * w3r ** 4 * k * vgsol * w2i
            - w3i * w3r ** 2 * Kdrag * rhogeq * k * vgsol * w2i ** 2
            + w3i * w3r ** 2 * Kdrag * rhogsol * w2r * w1r ** 2
            - w3i * w3r ** 2 * w2i * k * Kdrag ** 2 * vgsol
            + w3i * w3r ** 2 * w2i * k * Kdrag ** 2 * vdsol
            + w3i * rhogeq * w3r ** 4 * k * Kdrag * vgsol
            + w3i * rhogeq ** 2 * w3r ** 4 * k * vgsol * w1i
            + w3i * w3r ** 2 * Kdrag * rhogsol * w2r ** 2 * w1r
            + w3i * w3r ** 2 * Kdrag * rhogsol * w2r * w1i ** 2
            + w3i * w3r ** 2 * Kdrag * rhogsol * w2i ** 2 * w1r
            + 2 * w3i * w3r ** 2 * cs ** 2 * k ** 2 * rhogeq * rhogsol * w2r * w1i
            - w3i * w3r ** 2 * Kdrag * rhogeq * k * vgsol * w2r ** 2
            + 2 * w3i * w3r ** 2 * rhogeq * cs ** 2 * k ** 2 * rhogsol * w2i * w1r
        )
        rhod3r = (
            rhod3r
            / (w3r ** 2 + w3i ** 2)
            / (
                w1i ** 2
                - 2 * w3i * w1i
                + w3r ** 2
                + w1r ** 2
                + w3i ** 2
                - 2 * w3r * w1r
            )
            / (
                w2r ** 2
                - 2 * w3r * w2r
                + w2i ** 2
                + w3i ** 2
                - 2 * w2i * w3i
                + w3r ** 2
            )
            / rhogeq
            / Kdrag
        )

        rhod3i = -rhodeq * (
            w3r * cs ** 2 * k ** 3 * rhogeq ** 2 * vgsol * w2i ** 2 * w1i
            + w2r * w3i * cs ** 2 * k ** 3 * rhogeq ** 2 * vgsol * w1i ** 2
            + w2r * w3i * cs ** 2 * k ** 3 * rhogeq ** 2 * vgsol * w1r ** 2
            - w3r ** 2 * cs ** 2 * k ** 2 * rhogeq * rhogsol * w2r ** 2 * w1i
            + rhogeq * w2r ** 2 * w3i ** 2 * cs ** 2 * k ** 2 * rhogsol * w1i
            - w3i ** 2 * Kdrag * rhogsol * w2i ** 2 * w1r ** 2
            - w3i ** 2 * Kdrag * rhogsol * w2i ** 2 * w1i ** 2
            - w3r ** 2 * Kdrag * rhogsol * w2i ** 2 * w1i ** 2
            + rhogeq * w1i * rhogsol * k ** 2 * cs ** 2 * w3i ** 2 * w2i ** 2
            - w3i * cs ** 2 * k ** 2 * rhogeq * rhogsol * w2r ** 2 * w1i ** 2
            - w3i * cs ** 2 * k ** 2 * rhogeq * rhogsol * w2r ** 2 * w1r ** 2
            - w2r * w3i * cs ** 4 * k ** 4 * rhogeq * rhogsol * w1r
            - w2r * w3r * rhogeq * cs ** 4 * k ** 4 * rhogsol * w1i
            + w2r * w3i * cs ** 2 * k ** 3 * rhogeq * Kdrag * vgsol * w1i
            + rhogeq * w1i ** 2 * rhogsol * k ** 2 * cs ** 2 * w3i ** 2 * w2i
            - rhogeq * w1i ** 2 * rhogsol * k ** 2 * cs ** 2 * w3r ** 2 * w2i
            + w2r * w3r * rhogeq * cs ** 2 * k ** 3 * Kdrag * vdsol * w1r
            - w2r * w3r * rhogeq * cs ** 2 * k ** 3 * Kdrag * vgsol * w1r
            + w3i * cs ** 4 * k ** 4 * rhogeq * rhogsol * w1i * w2i
            - w3i * cs ** 2 * k ** 2 * rhogeq * rhogsol * w2i ** 2 * w1r ** 2
            + w3i * cs ** 2 * k ** 3 * rhogeq * Kdrag * vgsol * w1r * w2i
            - w3i * cs ** 2 * k ** 3 * rhogeq * Kdrag * vdsol * w1r * w2i
            - rhogeq * w1i * rhogsol * k ** 2 * cs ** 2 * w3r ** 2 * w2i ** 2
            - w3i * cs ** 2 * k ** 2 * rhogeq * rhogsol * w2i ** 2 * w1i ** 2
            + w3r * rhogeq * cs ** 2 * k ** 3 * w1i * Kdrag * vgsol * w2i
            + w3r * rhogeq ** 2 * cs ** 2 * k ** 3 * w1i ** 2 * vgsol * w2i
            + w3r * rhogeq ** 2 * cs ** 2 * k ** 3 * w1r ** 2 * vgsol * w2i
            - w3r * rhogeq * cs ** 4 * k ** 4 * w1r * rhogsol * w2i
            - w3r * rhogeq * cs ** 2 * k ** 3 * w1i * Kdrag * vdsol * w2i
            + w3r ** 2 * rhogeq * w2i ** 2 * w1r * k * Kdrag * vgsol
            + w3r ** 2 * rhogeq * w1r * k * Kdrag * vgsol * w2r ** 2
            - w3r ** 2 * Kdrag * rhogsol * w1i ** 2 * w2r ** 2
            - w3r ** 2 * Kdrag * rhogsol * w2i ** 2 * w1r ** 2
            - w2r * w3i * cs ** 2 * k ** 3 * rhogeq * Kdrag * vdsol * w1i
            - w3i ** 2 * Kdrag * rhogsol * w1r ** 2 * w2r ** 2
            - w3i ** 2 * Kdrag * rhogsol * w1i ** 2 * w2r ** 2
            - w3r ** 2 * Kdrag * rhogsol * w1r ** 2 * w2r ** 2
            + w3i ** 2 * rhogeq * w2i ** 2 * w1r * k * Kdrag * vgsol
            + w3i ** 2 * rhogeq * w1r * k * Kdrag * vgsol * w2r ** 2
            - w3r ** 2 * cs ** 2 * k ** 2 * rhogeq * rhogsol * w1r ** 2 * w2i
            + w3i ** 2 * cs ** 2 * k ** 2 * rhogeq * rhogsol * w1r ** 2 * w2i
            + rhogeq * w3r ** 4 * rhogsol * w2r ** 2 * w1i
            + rhogeq * w3r ** 4 * rhogsol * w1i * w2i ** 2
            + rhogeq * w3r ** 4 * rhogsol * w2i * w1i ** 2
            + rhogeq * w3r ** 4 * rhogsol * w2i * w1r ** 2
            - w3r * rhogeq * w2r * w3i ** 2 * w1r * k * Kdrag * vgsol
            - rhogeq ** 2 * w3i ** 5 * w2r * k * vgsol
            - rhogeq ** 2 * w3i ** 5 * k * w1r * vgsol
            + rhogeq * w3i ** 5 * w2r * w1r * rhogsol
            - w3i ** 2 * cs ** 4 * k ** 4 * rhogeq * rhogsol * w2i
            + 2 * w3r ** 2 * rhogeq * w3i ** 3 * cs ** 2 * k ** 2 * rhogsol
            - 3 * w3r * rhogeq * w2i * w3i ** 2 * w1i * k * Kdrag * vgsol
            - w3r * rhogeq * w2r * w3i ** 2 * w1r * k * Kdrag * vdsol
            - 2 * w3r ** 2 * Kdrag * k * rhogeq * vgsol * w3i ** 2 * w1r
            - 2 * w3r ** 2 * Kdrag * k * rhogeq * vgsol * w2r * w3i ** 2
            - 2 * w3r ** 2 * rhogeq ** 2 * w2r * k * vgsol * w3i ** 3
            + 3 * w3r * rhogeq ** 2 * cs ** 2 * k ** 3 * w3i ** 2 * w1i * vgsol
            + 3 * w3r * rhogeq * cs ** 2 * k ** 3 * w3i ** 2 * Kdrag * vgsol
            - w3r * rhogeq ** 2 * w2i ** 2 * w3i ** 2 * w1i * k * vgsol
            + 3 * w3r * rhogeq ** 2 * cs ** 2 * k ** 3 * w3i ** 2 * w2i * vgsol
            - w3r * rhogeq ** 2 * w3i ** 2 * w2r ** 2 * w1i * k * vgsol
            + 2 * w3r * rhogeq * w1i * w3i ** 3 * k * Kdrag * vgsol
            - w3r * rhogeq ** 2 * w2i * w1i ** 2 * w3i ** 2 * k * vgsol
            + 2 * w3r * rhogeq * w2i * w3i ** 3 * k * Kdrag * vgsol
            - w3r * rhogeq ** 2 * w2i * k * vgsol * w3i ** 2 * w1r ** 2
            - 3 * w3r * rhogeq * cs ** 2 * k ** 3 * w3i ** 2 * Kdrag * vdsol
            - w3i ** 3 * rhogeq * w2i ** 2 * w1i ** 2 * rhogsol
            + w3i ** 2 * cs ** 2 * k ** 3 * rhogeq * Kdrag * vdsol * w1r
            - w3i ** 3 * rhogeq * w1r ** 2 * w2i ** 2 * rhogsol
            + w3i ** 3 * rhogeq ** 2 * w2r ** 2 * k * w1r * vgsol
            + w3i ** 3 * rhogeq ** 2 * w2i ** 2 * w1r * k * vgsol
            + w3i ** 3 * rhogeq * cs ** 4 * k ** 4 * rhogsol
            - rhogeq * w3i ** 5 * w1i * rhogsol * w2i
            - w3i ** 4 * Kdrag * rhogsol * w2i * w1i
            + rhogeq * w3i ** 5 * cs ** 2 * k ** 2 * rhogsol
            + w3i ** 4 * Kdrag * rhogsol * w2r * w1r
            + w3i ** 2 * cs ** 2 * k ** 3 * rhogeq * Kdrag * vdsol * w2r
            + w3i ** 4 * Kdrag * rhogsol * k ** 2 * cs ** 2
            - w3i ** 2 * cs ** 4 * k ** 4 * rhogeq * rhogsol * w1i
            - w3i ** 3 * Kdrag ** 2 * k * vgsol * w1r
            - w3i ** 3 * rhogeq ** 2 * cs ** 2 * k ** 3 * w2r * vgsol
            + w3i ** 3 * Kdrag * rhogsol * w2r ** 2 * w1i
            - w3i ** 3 * Kdrag * rhogsol * k ** 2 * cs ** 2 * w1i
            - 2 * rhogeq * w3i ** 4 * k * Kdrag * vgsol * w1r
            - 2 * w3r ** 2 * rhogeq * w1i * w3i ** 3 * rhogsol * w2i
            - 2 * w3r ** 2 * rhogeq ** 2 * w3i ** 3 * k * w1r * vgsol
            + 2 * w3r ** 2 * rhogeq * w2r * w3i ** 3 * w1r * rhogsol
            + w3r * rhogeq * w2i * w3i ** 2 * w1i * k * Kdrag * vdsol
            - w3i ** 3 * rhogeq * w2i * w1r * k * Kdrag * vdsol
            - w3i ** 3 * Kdrag * rhogsol * k ** 2 * cs ** 2 * w2i
            - w3i ** 2 * cs ** 2 * k ** 3 * rhogeq * Kdrag * vgsol * w1r
        )

        rhod3i = rhod3i - rhodeq * (
            w3i ** 3 * Kdrag * rhogsol * w2i * w1i ** 2
            - w3i ** 2 * cs ** 2 * k ** 3 * rhogeq * Kdrag * vgsol * w2r
            + w3i ** 3 * Kdrag ** 2 * k * vdsol * w1r
            + w3i ** 3 * rhogeq ** 2 * w2r * w1i ** 2 * k * vgsol
            - w3i ** 3 * rhogeq * w2r ** 2 * w1r ** 2 * rhogsol
            - 2 * rhogeq * w3i ** 4 * k * Kdrag * vgsol * w2r
            - Kdrag * w3r ** 4 * rhogsol * k ** 2 * cs ** 2
            - w3i ** 3 * Kdrag ** 2 * k * vgsol * w2r
            - rhogeq * w3r ** 5 * rhogsol * w2i * w1r
            + w3i ** 3 * Kdrag * rhogsol * w1i * w2i ** 2
            + w3i ** 3 * rhogeq * w2r * w1i * k * Kdrag * vgsol
            + Kdrag * w3r ** 4 * w2i * rhogsol * w1i
            + w3i ** 3 * rhogeq * w2i * w1r * k * Kdrag * vgsol
            + rhogeq ** 2 * w3i ** 4 * w3r * w1i * k * vgsol
            - rhogeq * w3i ** 4 * w3r * k * Kdrag * vdsol
            + w3i ** 3 * rhogeq ** 2 * w2r * w1r ** 2 * k * vgsol
            + w3i ** 3 * Kdrag ** 2 * k * vdsol * w2r
            + rhogeq * w3i ** 4 * w3r * k * Kdrag * vgsol
            - w3i ** 3 * rhogeq ** 2 * cs ** 2 * k ** 3 * w1r * vgsol
            - rhogeq * w3i ** 4 * w2r * w3r * rhogsol * w1i
            - w3i ** 3 * rhogeq * w1i ** 2 * w2r ** 2 * rhogsol
            - rhogeq * w3i ** 4 * w3r * w2i * w1r * rhogsol
            - w3i ** 3 * rhogeq * w2r * w1i * k * Kdrag * vdsol
            - 2 * w3r * w3i ** 3 * k * Kdrag ** 2 * vdsol
            + 2 * w3i ** 2 * rhogeq ** 2 * w3r ** 3 * k * vgsol * w1i
            + 2 * w3i ** 2 * rhogeq * w3r ** 3 * k * Kdrag * vgsol
            + w3i ** 2 * w3r * Kdrag * rhogsol * w2r * w1i ** 2
            + w3i ** 2 * w3r * Kdrag * rhogsol * w2r ** 2 * w1r
            - w3i ** 2 * w3r * Kdrag * rhogeq * k * vgsol * w2r ** 2
            + w3i ** 2 * w3r * Kdrag * rhogsol * w2i ** 2 * w1r
            - 2 * w3i ** 2 * w3r * cs ** 2 * k ** 2 * rhogeq * rhogsol * w2r * w1i
            - 2 * w3i ** 2 * w3r * rhogeq * cs ** 2 * k ** 2 * rhogsol * w2i * w1r
            - w3i * w3r ** 2 * rhogeq * w1i ** 2 * w2r ** 2 * rhogsol
            + 2 * w3i * w3r * rhogeq * cs ** 4 * k ** 4 * w2r * rhogsol
            - w3i * w3r ** 4 * rhogeq * w2i * rhogsol * w1i
            + w3i * w3r ** 4 * rhogeq * rhogsol * k ** 2 * cs ** 2
            + w3i * w3r ** 4 * rhogeq * w2r * w1r * rhogsol
            - w3i * w3r ** 4 * rhogeq ** 2 * w2r * vgsol * k
            + rhogeq ** 2 * w3i ** 4 * w3r * w2i * k * vgsol
            - w3i * w3r ** 4 * rhogeq ** 2 * k * w1r * vgsol
            + w3i * w3r ** 2 * Kdrag * rhogsol * w2r ** 2 * w1i
            + w3i * w3r ** 2 * Kdrag * rhogsol * w2i * w1r ** 2
            + w3i * w3r ** 2 * Kdrag * rhogsol * w1i * w2i ** 2
            + w3i * w3r ** 2 * Kdrag ** 2 * k * vdsol * w2r
            + w3i * w3r ** 2 * rhogeq ** 2 * w2r * w1r ** 2 * k * vgsol
            + w3i * w3r ** 2 * rhogeq ** 2 * w2r * w1i ** 2 * k * vgsol
            + 3 * w3i * w3r ** 2 * rhogeq ** 2 * cs ** 2 * k ** 3 * w1r * vgsol
            + 2 * w3i * w3r * rhogeq * cs ** 2 * k ** 3 * w1i * Kdrag * vdsol
            - w3i * w3r ** 2 * Kdrag * rhogsol * k ** 2 * cs ** 2 * w1i
            + 3 * w3i * w3r ** 2 * rhogeq ** 2 * cs ** 2 * k ** 3 * w2r * vgsol
            + 2 * w3i * w3r * rhogeq * cs ** 4 * k ** 4 * w1r * rhogsol
            + 2 * w3i * w3r * rhogeq * cs ** 2 * k ** 3 * w2i * Kdrag * vdsol
            - w3i * w3r ** 2 * Kdrag * rhogsol * k ** 2 * cs ** 2 * w2i
            - w3i * w3r ** 2 * rhogeq * w1r ** 2 * w2i ** 2 * rhogsol
            - w3i * w3r ** 2 * rhogeq * w2i ** 2 * w1i ** 2 * rhogsol
            + w3i ** 2 * w3r * w1i * k * Kdrag ** 2 * vdsol
            - w3i ** 2 * Kdrag * w3r * w1r ** 2 * rhogeq * k * vgsol
            - w3i ** 2 * Kdrag * w3r * w1i ** 2 * k * rhogeq * vgsol
            + w3i ** 2 * Kdrag * w3r * w1r * cs ** 2 * k ** 2 * rhogsol
            + w3i ** 2 * Kdrag * w2r * w3r * cs ** 2 * k ** 2 * rhogsol
            - w3i ** 2 * w3r * w1i * k * Kdrag ** 2 * vgsol
            - 2 * w3i ** 2 * rhogeq * w3r ** 3 * k * Kdrag * vdsol
            - 2 * w3i ** 2 * rhogeq * w3r ** 3 * rhogsol * w2i * w1r
            - 2 * w3i ** 2 * rhogeq * w3r ** 3 * rhogsol * w1i * w2r
            + w3i ** 2 * w3r * Kdrag * rhogsol * w2r * w1r ** 2
            - w3i ** 2 * w3r * Kdrag * rhogeq * k * vgsol * w2i ** 2
            - 2 * w3i * w3r ** 3 * Kdrag ** 2 * k * vdsol
            - w3i ** 2 * w3r * w2i * k * Kdrag ** 2 * vgsol
            + w3i ** 2 * w3r * w2i * k * Kdrag ** 2 * vdsol
            - w3i * w3r ** 2 * Kdrag ** 2 * k * vgsol * w1r
            + 2 * w3i * w3r ** 3 * rhogeq * w2i * k * Kdrag * vgsol
            - 2 * w3i * w3r ** 3 * Kdrag * rhogsol * w2i * w1r
            + w3i * w3r ** 2 * Kdrag ** 2 * k * vdsol * w1r
            - 2 * w3i * w3r * rhogeq ** 2 * cs ** 2 * k ** 3 * vgsol * w2i ** 2
            - 2 * w3i * w3r * rhogeq ** 2 * cs ** 2 * k ** 3 * w2r ** 2 * vgsol
            - w3i * w3r ** 2 * rhogeq * w2r ** 2 * w1r ** 2 * rhogsol
            - 2 * w3i * w3r * rhogeq ** 2 * cs ** 2 * k ** 3 * w1i ** 2 * vgsol
            - 2 * w3i * w3r * rhogeq ** 2 * cs ** 2 * k ** 3 * w1r ** 2 * vgsol
            - 3 * w3i * w3r ** 2 * rhogeq * cs ** 4 * k ** 4 * rhogsol
            + w3i * w3r ** 2 * rhogeq ** 2 * w2r ** 2 * k * w1r * vgsol
            - w3i * w3r ** 2 * rhogeq * w2r * w1i * k * Kdrag * vdsol
            - 2 * w3i * w3r ** 3 * Kdrag * rhogsol * w1i * w2r
            + 2 * w3i * w3r ** 3 * rhogeq * w1i * k * Kdrag * vgsol
        )

        rhod3i = rhod3i - rhodeq * (
            w3i * w3r ** 2 * rhogeq * w2r * w1i * k * Kdrag * vgsol
            + w3i * w3r ** 2 * Kdrag * rhogsol * w2i * w1i ** 2
            + rhogeq * rhogsol * w2r ** 2 * w1i * w3i ** 4
            - rhogeq * w3r ** 5 * k * Kdrag * vdsol
            + rhogeq * w3r ** 3 * k * Kdrag * vdsol * w2i * w1i
            - rhogeq * w3r ** 3 * k * Kdrag * vdsol * w2r * w1r
            + 2 * rhogeq * w3r ** 2 * rhogsol * w2r ** 2 * w1i * w3i ** 2
            + w3i * w3r ** 2 * rhogeq * w2i * w1r * k * Kdrag * vgsol
            + 2 * rhogeq * w3r ** 2 * rhogsol * w1i * w2i ** 2 * w3i ** 2
            + 2 * rhogeq * w3r ** 2 * rhogsol * w2i * w1r ** 2 * w3i ** 2
            + 2 * rhogeq * w3r ** 2 * rhogsol * w2i * w1i ** 2 * w3i ** 2
            - 2 * Kdrag * w2r * w3r * w3i ** 3 * rhogsol * w1i
            + rhogeq * w3r ** 4 * k * Kdrag * vdsol * w2r
            + 2 * rhogeq * w3r ** 2 * k * Kdrag * vdsol * w2r * w3i ** 2
            - Kdrag * w3r ** 3 * w1r ** 2 * rhogeq * k * vgsol
            - Kdrag * w3r ** 3 * w1i ** 2 * k * rhogeq * vgsol
            + Kdrag * w3r ** 3 * w1r * cs ** 2 * k ** 2 * rhogsol
            + Kdrag * w2r * w3r ** 3 * cs ** 2 * k ** 2 * rhogsol
            + cs ** 2 * k ** 3 * rhogeq * w3r ** 3 * Kdrag * vdsol
            + w3i * w3r ** 2 * rhogeq ** 2 * w2i ** 2 * w1r * k * vgsol
            - rhogeq ** 2 * w3r ** 3 * k * vgsol * w2i * w1r ** 2
            - rhogeq * rhogsol * k ** 2 * cs ** 2 * w3i ** 4 * w2i
            + rhogeq ** 2 * w3r ** 5 * k * vgsol * w1i
            + rhogeq * w3r ** 5 * k * Kdrag * vgsol
            - w3i * w3r ** 2 * rhogeq * w2i * w1r * k * Kdrag * vdsol
            - w3i * w3r ** 2 * Kdrag ** 2 * k * vgsol * w2r
            + rhogeq * rhogsol * w3i ** 4 * w2i * w1r ** 2
            + 2 * w3i * w3r ** 3 * Kdrag ** 2 * k * vgsol
            + w3i * rhogeq ** 2 * cs ** 2 * k ** 3 * w2r ** 2 * w1r * vgsol
            + rhogeq * rhogsol * w2i * w1i ** 2 * w3i ** 4
            + rhogeq * k * Kdrag * vdsol * w2r * w3i ** 4
            + 2 * w3i * w1r * w3r * cs ** 2 * k ** 2 * rhogeq * rhogsol * w2i ** 2
            + rhogeq ** 2 * w3r ** 5 * k * vgsol * w2i
            + 2 * rhogeq * w3r ** 2 * k * Kdrag * vdsol * w3i ** 2 * w1r
            - 2 * w3i * w3r * rhogeq * cs ** 2 * k ** 3 * w2i * Kdrag * vgsol
            - rhogeq * w3r ** 5 * rhogsol * w1i * w2r
            + rhogeq * k * Kdrag * vdsol * w3i ** 4 * w1r
            - 2 * rhogeq * w3r ** 2 * rhogsol * k ** 2 * cs ** 2 * w3i ** 2 * w1i
            - 2 * w3i * w3r * rhogeq * cs ** 2 * k ** 3 * w1i * Kdrag * vgsol
            - Kdrag * w2r * w3r ** 4 * w1r * rhogsol
            + w3r ** 3 * Kdrag * rhogsol * w2r ** 2 * w1r
            + Kdrag ** 2 * k * vgsol * w2r * w1i * w3i ** 2
            + w3r ** 2 * Kdrag * k * rhogeq * vgsol * w2r * w1i ** 2
            + w3r ** 2 * Kdrag * k * rhogeq * vgsol * w2r * w1r ** 2
            + w3r ** 2 * cs ** 2 * k ** 3 * rhogeq * Kdrag * vgsol * w2r
            - w3r ** 2 * Kdrag * cs ** 2 * k ** 2 * rhogsol * w2r * w1r
            - w3r ** 2 * cs ** 2 * k ** 3 * rhogeq * Kdrag * vdsol * w2r
            + rhogeq ** 2 * cs ** 2 * k ** 3 * w1r * vgsol * w2i ** 2 * w3i
            + 2 * w3r ** 3 * cs ** 2 * k ** 2 * rhogeq * rhogsol * w2r * w1i
            + 2 * w3i ** 2 * rhogeq ** 2 * w3r ** 3 * k * vgsol * w2i
            + w3r ** 3 * Kdrag * rhogsol * w2r * w1r ** 2
            - w3r ** 3 * Kdrag * rhogeq * k * vgsol * w2i ** 2
            - w3r ** 3 * rhogeq ** 2 * cs ** 2 * k ** 3 * vgsol * w1i
            + 2 * w3i * w2r * w3r * cs ** 2 * k ** 2 * rhogeq * rhogsol * w1r ** 2
            - w3r ** 3 * rhogeq ** 2 * k * vgsol * w2r ** 2 * w1i
            - w3r ** 3 * Kdrag * rhogeq * k * vgsol * w2r ** 2
            - rhogeq * w3r ** 4 * rhogsol * k ** 2 * cs ** 2 * w2i
            + rhogeq * rhogsol * w2i ** 2 * w3i ** 4 * w1i
            - rhogeq * w3r ** 4 * rhogsol * k ** 2 * cs ** 2 * w1i
            - 4 * w3i * w3r * rhogeq ** 2 * cs ** 2 * k ** 3 * vgsol * w2i * w1i
            + w3r ** 3 * w1i * k * Kdrag ** 2 * vdsol
            - w3r ** 3 * w2i * k * Kdrag ** 2 * vgsol
            + rhogeq * w3r ** 4 * k * Kdrag * vdsol * w1r
            + w3r ** 3 * w2i * k * Kdrag ** 2 * vdsol
            - rhogeq * rhogsol * k ** 2 * cs ** 2 * w3i ** 4 * w1i
            - rhogeq ** 2 * w3r ** 3 * k * vgsol * w2i * w1i ** 2
            - 2 * rhogeq * w3r ** 2 * rhogsol * k ** 2 * cs ** 2 * w3i ** 2 * w2i
            - w3r ** 3 * w1i * k * Kdrag ** 2 * vgsol
            - 4 * w3i * w2r * w3r * rhogeq ** 2 * cs ** 2 * k ** 3 * w1r * vgsol
            + w3r ** 2 * cs ** 4 * k ** 4 * rhogeq * rhogsol * w2i
            + 2 * w3i * w3r * rhogeq * rhogsol * k ** 2 * cs ** 2 * w2r ** 2 * w1r
            + 2 * w3r * w3i ** 3 * k * Kdrag ** 2 * vgsol
            - w3r ** 2 * Kdrag ** 2 * k * vdsol * w2i * w1r
            - Kdrag ** 2 * k * vdsol * w2i * w1r * w3i ** 2
            - w3r ** 3 * vgsol * k * rhogeq ** 2 * w2i ** 2 * w1i
            - w3r ** 3 * Kdrag * rhogeq * k * vgsol * w2r * w1r
            + 2 * w3r ** 3 * rhogeq * cs ** 2 * k ** 2 * rhogsol * w2i * w1r
            - 3 * w3r ** 3 * Kdrag * k * rhogeq * vgsol * w1i * w2i
            + w3r ** 2 * cs ** 2 * k ** 3 * rhogeq * Kdrag * vgsol * w1r
            - w3r ** 2 * cs ** 2 * k ** 3 * rhogeq * Kdrag * vdsol * w1r
            + Kdrag ** 2 * k * vgsol * w2i * w1r * w3i ** 2
            + rhogeq ** 2 * cs ** 2 * k ** 3 * vgsol * w1i * w2r ** 2 * w3r
            - w3r ** 2 * Kdrag ** 2 * k * vdsol * w2r * w1i
            - Kdrag ** 2 * k * vdsol * w2r * w1i * w3i ** 2
            + w3r ** 2 * cs ** 4 * k ** 4 * rhogeq * rhogsol * w1i
            + w3r ** 2 * Kdrag * cs ** 2 * k ** 2 * rhogsol * w1i * w2i
            + w3r ** 3 * Kdrag * rhogsol * w2r * w1i ** 2
            - cs ** 2 * k ** 3 * rhogeq * w3r ** 3 * Kdrag * vgsol
            + 2 * w3i * w2r * rhogeq * w1i ** 2 * rhogsol * k ** 2 * cs ** 2 * w3r
            + 4 * cs ** 2 * k ** 2 * rhogeq * w3r ** 2 * w3i * w2i * rhogsol * w1i
            - 4 * cs ** 2 * k ** 2 * rhogeq * w3r ** 2 * w3i * w2r * w1r * rhogsol
            - 2 * Kdrag * w1r * w2i * w3i ** 3 * rhogsol * w3r
            + Kdrag * k * rhogeq * vgsol * w2r * w1r ** 2 * w3i ** 2
            + w3r ** 2 * Kdrag ** 2 * k * vgsol * w2r * w1i
            - Kdrag * cs ** 2 * k ** 2 * rhogsol * w2r * w1r * w3i ** 2
            + w3i ** 3 * Kdrag * rhogsol * w2i * w1r ** 2
            + Kdrag * k * rhogeq * vgsol * w2r * w1i ** 2 * w3i ** 2
            + w3r ** 2 * Kdrag ** 2 * k * vgsol * w2i * w1r
            + Kdrag * cs ** 2 * k ** 2 * rhogsol * w1i * w2i * w3i ** 2
            + w3r ** 3 * Kdrag * rhogsol * w2i ** 2 * w1r
            - w3r ** 3 * vgsol * rhogeq ** 2 * k ** 3 * cs ** 2 * w2i
        )
        rhod3i = (
            rhod3i
            / (w3r ** 2 + w3i ** 2)
            / (
                w1i ** 2
                - 2 * w3i * w1i
                + w3r ** 2
                + w1r ** 2
                + w3i ** 2
                - 2 * w3r * w1r
            )
            / (
                w2r ** 2
                - 2 * w3r * w2r
                + w2i ** 2
                + w3i ** 2
                - 2 * w2i * w3i
                + w3r ** 2
            )
            / rhogeq
            / Kdrag
        )

        rhod2r = -rhodeq * (
            k * Kdrag ** 2 * vdsol * w2i ** 3 * w1i
            - k * Kdrag ** 2 * vgsol * w2i ** 3 * w1i
            - 2 * w3r * rhogeq ** 2 * cs ** 2 * k ** 3 * w1r * vgsol * w2i ** 2
            + w3r * Kdrag * rhogsol * w2i ** 3 * w1i ** 2
            - Kdrag * k * rhogeq * vgsol * w2i ** 3 * w1i ** 2
            + w3r * Kdrag * rhogsol * w2i ** 3 * w1r ** 2
            - Kdrag * k * rhogeq * vgsol * w2i ** 3 * w1r ** 2
            - 2 * w2i * w3r ** 2 * cs ** 2 * k ** 2 * rhogeq * rhogsol * w2r * w1i
            + w2i * w3r ** 2 * rhogeq ** 2 * k * vgsol * w2r ** 2 * w1i
            - w2i * w3r ** 2 * Kdrag * rhogeq * k * vgsol * w2r ** 2
            - w3r * rhogeq ** 2 * cs ** 2 * k ** 3 * w2r ** 3 * vgsol
            - w3r * Kdrag * rhogsol * k ** 2 * cs ** 2 * w1i * w2r ** 2
            + w3r * Kdrag ** 2 * k * vdsol * w2r ** 2 * w1r
            - w3r * Kdrag ** 2 * k * vgsol * w2r ** 2 * w1r
            + w3r * Kdrag * k * rhogeq * vgsol * w2r ** 3 * w1i
            - w3r * rhogeq * cs ** 2 * k ** 2 * w1i ** 2 * w2r ** 2 * rhogsol
            - w3r * rhogeq * cs ** 2 * k ** 2 * w2r ** 2 * w1r ** 2 * rhogsol
            + 2 * w3r * rhogeq ** 2 * cs ** 2 * k ** 3 * w2r ** 2 * w1r * vgsol
            + w3r ** 2 * rhogeq ** 2 * cs ** 2 * k ** 3 * w2r ** 2 * vgsol
            + w3r ** 2 * rhogeq * w2r ** 2 * w1i * k * Kdrag * vgsol
            - w3r ** 2 * Kdrag * rhogsol * w2r ** 3 * w1i
            + w3r ** 2 * rhogeq ** 2 * w2r ** 3 * k * w1r * vgsol
            + w2i * w3r * rhogeq * k * Kdrag * vdsol * w2r ** 2 * w1r
            + w2i * w3r * Kdrag * rhogsol * w2r ** 2 * w1r ** 2
            + w2i * w3r * Kdrag * rhogsol * w2r ** 2 * w1i ** 2
            - w2i * w3r * rhogeq * rhogsol * w2r ** 4 * w1i
            + 2 * w2i * w3r * rhogeq * cs ** 2 * k ** 2 * rhogsol * w2r ** 2 * w1i
            + 2 * w2i * w3r * cs ** 2 * k ** 3 * rhogeq * Kdrag * vgsol * w2r
            + w2i * w3r * Kdrag * cs ** 2 * k ** 2 * rhogsol * w2r ** 2
            - 2 * w2i * w3r * cs ** 2 * k ** 3 * rhogeq * Kdrag * vdsol * w2r
            - 4 * w3r * rhogeq * cs ** 2 * k ** 2 * rhogsol * w2r * w2i ** 2 * w1r
            - w3r * rhogeq * k * Kdrag * vdsol * w2r * w2i ** 2 * w1i
            + rhogeq ** 2 * k * vgsol * w2i ** 5 * w1i
            - w3r * rhogeq * rhogsol * w2i ** 5 * w1i
            + 4 * w3i * cs ** 2 * k ** 2 * rhogeq * rhogsol * w2r * w2i ** 2 * w1i
            + 2 * w3r ** 2 * rhogeq * rhogsol * w2r ** 2 * w2i ** 2 * w1r
            - w3r ** 2 * rhogeq * rhogsol * w2r * w2i ** 2 * w1r ** 2
            + w3r ** 2 * rhogeq ** 2 * k * vgsol * w2r * w2i ** 2 * w1r
            + 2 * vgsol * k * rhogeq ** 2 * w2i ** 3 * w2r ** 2 * w1i
            + 2 * w3r * rhogeq * cs ** 2 * k ** 2 * rhogsol * w2i ** 2 * w2r ** 2
            + 3 * w3r * cs ** 2 * k ** 3 * rhogeq ** 2 * vgsol * w2i ** 2 * w2r
            - 2 * w3r * Kdrag * rhogsol * w2r * w1r * w2i ** 3
            - 2 * w3r * rhogeq * rhogsol * w2i ** 3 * w2r ** 2 * w1i
            - w3r ** 2 * rhogeq * rhogsol * w2r * w2i ** 2 * w1i ** 2
            + 2 * w3r * rhogeq * rhogsol * w1i ** 2 * w2r ** 2 * w2i ** 2
            + 2 * w3r * rhogeq * rhogsol * w2i ** 2 * w1r ** 2 * w2r ** 2
            + w3r * w2r * Kdrag * rhogeq * k * vgsol * w2i ** 2 * w1i
            - 2 * w3r ** 2 * rhogeq ** 2 * k * vgsol * w2i ** 2 * w2r ** 2
            + w3r * rhogeq ** 2 * k * vgsol * w2r * w2i ** 2 * w1i ** 2
            + w3r * rhogeq ** 2 * k * vgsol * w2r * w2i ** 2 * w1r ** 2
            - 4 * w3r * rhogeq ** 2 * k * vgsol * w2r ** 2 * w2i ** 2 * w1r
            - w3r * rhogeq * cs ** 4 * k ** 4 * w2r ** 2 * rhogsol
            - w3r ** 2 * rhogeq * w2r ** 3 * w1r ** 2 * rhogsol
            + w1r * Kdrag * rhogsol * k ** 2 * cs ** 2 * w2i ** 3
            + w2i * w3r ** 2 * Kdrag * rhogsol * w2r ** 2 * w1r
            - 2 * w2i * w3r * Kdrag * rhogsol * w2r ** 3 * w1r
            - 3 * w2i * w3r * Kdrag * k * rhogeq * vgsol * w2r ** 2 * w1r
            + w2i * rhogeq ** 2 * k * vgsol * w2r ** 4 * w1i
            + rhogeq ** 2 * w1i ** 2 * k * vgsol * w3r * w2r ** 3
            + 2 * w2i * w3i * cs ** 2 * k ** 2 * rhogeq * rhogsol * w2r ** 2 * w1r
            + w1r * w3i ** 2 * cs ** 2 * k ** 2 * rhogeq * rhogsol * w2i ** 2
            + w1r * w3r ** 2 * cs ** 2 * k ** 2 * rhogeq * rhogsol * w2i ** 2
            + 2 * Kdrag * w3r * w2r ** 3 * k * rhogeq * vgsol * w2i
            + w3r * rhogeq ** 2 * w1r ** 2 * k * vgsol * w2r ** 3
            - w3r * rhogeq * w2r ** 5 * w1r * rhogsol
            - rhogeq * w1i * w3r * w2r ** 3 * k * Kdrag * vdsol
            + rhogeq * w1i ** 2 * w3r * w2r ** 4 * rhogsol
            + w3r * rhogeq * w2r ** 4 * w1r ** 2 * rhogsol
            + w3r * rhogeq ** 2 * w2r ** 5 * k * vgsol
            + w3r * w2r ** 3 * k * Kdrag ** 2 * vgsol
            - w3r * w2r ** 3 * k * Kdrag ** 2 * vdsol
            + Kdrag * w3r * w2r ** 4 * rhogsol * w1i
            + w3r * rhogeq * w2r ** 4 * cs ** 2 * k ** 2 * rhogsol
            + w2r * rhogeq * w1i ** 2 * rhogsol * k ** 2 * cs ** 2 * w3r ** 2
            - w2r * w3r * rhogeq * cs ** 2 * k ** 3 * w1i * Kdrag * vgsol
            + w2r * w3r * rhogeq * cs ** 2 * k ** 3 * w1i * Kdrag * vdsol
            - w2r * w3r * rhogeq ** 2 * cs ** 2 * k ** 3 * w1i ** 2 * vgsol
            - w2r * w3r * rhogeq ** 2 * cs ** 2 * k ** 3 * w1r ** 2 * vgsol
            + w2r * w3r * rhogeq * cs ** 4 * k ** 4 * w1r * rhogsol
            - w2r * w3r ** 2 * rhogeq ** 2 * cs ** 2 * k ** 3 * w1r * vgsol
            - w2r * w3r ** 2 * Kdrag * rhogsol * w1i * w2i ** 2
            - rhogeq * w2r * w3i * w1r * k * Kdrag * vdsol * w2i ** 2
            + rhogeq * cs ** 2 * k ** 3 * w3i * Kdrag * vdsol * w2r * w1r
            - rhogeq * w2i * w3i * w1i * k * Kdrag * vdsol * w2r ** 2
            - rhogeq * w3i ** 2 * cs ** 2 * k ** 2 * rhogsol * w1r * w2r ** 2
            - 2 * w3r * rhogeq ** 2 * w2r ** 4 * k * w1r * vgsol
            - 2 * w3r * rhogeq * w2r ** 3 * w1r * rhogsol * w2i ** 2
        )

        rhod2r = rhod2r - rhodeq * (
            2 * w3r * rhogeq ** 2 * w2i ** 2 * k * vgsol * w2r ** 3
            - w3r ** 2 * rhogeq ** 2 * w2r ** 4 * vgsol * k
            - w3r ** 2 * rhogeq * rhogsol * k ** 2 * cs ** 2 * w2r ** 2 * w1r
            + w3r ** 2 * rhogeq * w2r ** 4 * w1r * rhogsol
            + w2r * w3r ** 2 * cs ** 2 * k ** 2 * rhogeq * rhogsol * w1r ** 2
            - w2r * w3r * rhogeq * rhogsol * w2i ** 4 * w1r
            + Kdrag * rhogsol * k ** 2 * cs ** 2 * w1i * w2r ** 3
            - rhogeq ** 2 * w2r ** 4 * w3i ** 2 * k * vgsol
            + rhogeq * w2r * w3i ** 2 * cs ** 2 * k ** 2 * rhogsol * w1r ** 2
            + rhogeq * cs ** 4 * k ** 4 * w2r ** 3 * rhogsol
            + w2r * w3r * rhogeq ** 2 * k * vgsol * w2i ** 4
            + rhogeq * w2r ** 4 * w3i ** 2 * w1r * rhogsol
            + rhogeq ** 2 * cs ** 2 * k ** 3 * w2r ** 2 * vgsol * w1r ** 2
            - 3 * rhogeq * cs ** 4 * k ** 4 * w2r * rhogsol * w2i ** 2
            - 2 * rhogeq * cs ** 2 * k ** 3 * w2i * Kdrag * vdsol * w2r * w1r
            + 2 * rhogeq * cs ** 2 * k ** 3 * w2i * Kdrag * vgsol * w2r * w1r
            - rhogeq * cs ** 2 * k ** 3 * w3i * Kdrag * vdsol * w2r ** 2
            - 2 * rhogeq * w2r * w3i ** 2 * cs ** 2 * k ** 2 * rhogsol * w2i * w1i
            - rhogeq * cs ** 2 * k ** 3 * w3i * Kdrag * vgsol * w2r * w1r
            - rhogeq ** 2 * w1i ** 2 * k * vgsol * w2r ** 4
            + rhogeq * cs ** 2 * k ** 3 * w3i * Kdrag * vgsol * w2r ** 2
            + 2 * rhogeq ** 2 * cs ** 2 * k ** 3 * w3i * w1i * vgsol * w2r ** 2
            - 3 * rhogeq ** 2 * cs ** 2 * k ** 3 * w2r ** 2 * vgsol * w2i * w1i
            + 2 * rhogeq * cs ** 4 * k ** 4 * w2r * rhogsol * w2i * w1i
            - rhogeq * cs ** 2 * k ** 3 * w1i * Kdrag * vdsol * w2r ** 2
            - 3 * rhogeq * cs ** 2 * k ** 3 * w2i * Kdrag * vgsol * w2r ** 2
            - 2 * rhogeq ** 2 * w1i ** 2 * k * vgsol * w2r ** 2 * w2i ** 2
            + rhogeq * cs ** 2 * k ** 3 * w1i * Kdrag * vgsol * w2r ** 2
            + rhogeq ** 2 * cs ** 2 * k ** 3 * w1i ** 2 * vgsol * w2r ** 2
            + 3 * rhogeq * cs ** 2 * k ** 3 * w2i * Kdrag * vdsol * w2r ** 2
            - 4 * rhogeq ** 2 * w3i * w2r ** 2 * w1i * k * vgsol * w2i ** 2
            - 2 * rhogeq ** 2 * w3i * w2r ** 4 * w1i * k * vgsol
            + 2 * rhogeq * w2r ** 2 * w3i ** 2 * w1r * rhogsol * w2i ** 2
            + 2 * w2r * Kdrag * w3r * w2i ** 3 * k * rhogeq * vgsol
            + rhogeq * cs ** 2 * k ** 2 * w2r ** 4 * w1r * rhogsol
            + rhogeq ** 2 * w2r ** 3 * w3i ** 2 * k * w1r * vgsol
            - rhogeq ** 2 * cs ** 2 * k ** 3 * w2r ** 3 * w1r * vgsol
            - rhogeq * cs ** 4 * k ** 4 * w1r * rhogsol * w2r ** 2
            + w2r * w3r * w2i ** 2 * k * Kdrag ** 2 * vgsol
            - w2r * w3r * w2i ** 2 * k * Kdrag ** 2 * vdsol
            - rhogeq * w2r * w3i ** 2 * w1r ** 2 * rhogsol * w2i ** 2
            + 2 * rhogeq * cs ** 2 * k ** 2 * w2r ** 2 * w1r * rhogsol * w2i ** 2
            + rhogeq ** 2 * w2i * k * vgsol * w3i * w1r ** 2 * w2r ** 2
            - 3 * rhogeq ** 2 * cs ** 2 * k ** 3 * w3i * w2i * vgsol * w2r ** 2
            - 2 * Kdrag * k * rhogeq * vgsol * w2r ** 4 * w1i
            - rhogeq * w2r ** 3 * w3i ** 2 * w1r ** 2 * rhogsol
            + w3i * vdsol * Kdrag * k * rhogeq * w2r ** 4
            - vdsol * Kdrag * k * rhogeq * w2i * w2r ** 4
            - 2 * w3i * rhogeq * rhogsol * w2r ** 2 * w2i ** 3 * w1r
            + w3i * rhogeq * rhogsol * w2r * w2i ** 4 * w1i
            - w3i * rhogeq * rhogsol * w2i ** 5 * w1r
            + w3i * rhogeq * rhogsol * w2r ** 5 * w1i
            - w3i * rhogeq * rhogsol * w2i * w2r ** 4 * w1r
            + 2 * w3i * rhogeq * rhogsol * w2i ** 2 * w2r ** 3 * w1i
            - w3i ** 2 * cs ** 2 * k ** 3 * rhogeq ** 2 * vgsol * w2r * w1r
            + w3i ** 2 * cs ** 2 * k ** 3 * rhogeq ** 2 * vgsol * w2r ** 2
            - 2 * rhogsol * k ** 2 * cs ** 2 * Kdrag * w2i * w2r ** 3
            + w3i * rhogsol * k ** 2 * cs ** 2 * Kdrag * w2r ** 3
            + w3i * rhogsol * k ** 2 * cs ** 2 * Kdrag * w2r * w2i ** 2
            - 2 * rhogsol * k ** 2 * cs ** 2 * Kdrag * w2r * w2i ** 3
            - w3i * vgsol * k * Kdrag ** 2 * w2i * w2r ** 2
            + vdsol * k * Kdrag ** 2 * w2r ** 4
            - cs ** 2 * k ** 2 * rhogeq * rhogsol * w2r * w2i ** 4
            + 2 * w3i * rhogeq ** 2 * k * vgsol * w2r ** 2 * w2i ** 3
            - 2 * cs ** 2 * k ** 2 * rhogeq * rhogsol * w2r ** 3 * w2i ** 2
            + w3i * rhogeq ** 2 * k * vgsol * w2i * w2r ** 4
            + 2 * w3i * Kdrag * rhogsol * w2r * w2i ** 3 * w1i
            - cs ** 2 * k ** 2 * rhogeq * rhogsol * w2r ** 5
            + w3i * rhogeq ** 2 * k * vgsol * w2i ** 5
            + 2 * rhogeq ** 2 * k * vgsol * w2i ** 2 * w2r ** 3 * w1r
            + w3i * Kdrag * rhogsol * w2r ** 4 * w1r
            + 2 * w3i * Kdrag * rhogsol * w2i * w2r ** 3 * w1i
            + rhogeq * w2r ** 2 * w3i * w1r ** 2 * k * Kdrag * vgsol
            + w3i * vdsol * k * Kdrag ** 2 * w2i * w2r ** 2
            - vgsol * k * Kdrag ** 2 * w2r ** 4
            + 2 * vdsol * Kdrag * k * rhogeq * w2i ** 2 * w2r ** 2 * w1i
            - vdsol * Kdrag * k * rhogeq * w2i ** 5
            + vdsol * Kdrag * k * rhogeq * w2r ** 4 * w1i
            - 2 * vdsol * Kdrag * k * rhogeq * w2r ** 2 * w2i ** 3
            - w3i * rhogsol * k ** 4 * cs ** 4 * rhogeq * w2r * w1i
            + 2 * vgsol * k * rhogeq * Kdrag * w2i * w2r ** 3 * w1r
            + 2 * vgsol * k * rhogeq * Kdrag * w2r ** 2 * w2i ** 3
            + 2 * vgsol * k * rhogeq * Kdrag * w2r * w2i ** 3 * w1r
            + vgsol * k * rhogeq * Kdrag * w2i * w2r ** 4
            - 2 * w3i * vgsol * k * rhogeq * Kdrag * w2r ** 4
            + vgsol * k * rhogeq * Kdrag * w2i ** 5
            - Kdrag * w2r ** 3 * rhogsol * w1i ** 2 * w3i
            - Kdrag * w2r ** 2 * k * rhogeq * vgsol * w2i * w3i ** 2
            - Kdrag * w2i ** 2 * rhogsol * w3i * w2r * w1r ** 2
            + Kdrag * w2r ** 2 * rhogsol * k ** 2 * cs ** 2 * w2i * w1r
            - Kdrag * w2i ** 2 * rhogsol * w1i ** 2 * w2r * w3i
            + w2i ** 2 * k * Kdrag ** 2 * vgsol * w2r * w1r
            - w2r ** 3 * k * Kdrag ** 2 * vdsol * w1r
            - w2i ** 2 * k * Kdrag ** 2 * vdsol * w2r * w1r
            + rhogeq ** 2 * w1i ** 2 * w2r ** 2 * k * vgsol * w2i * w3i
            + rhogeq * cs ** 2 * k ** 2 * w1i ** 2 * w2r * rhogsol * w3i ** 2
            + Kdrag * k * rhogeq * vgsol * w2r ** 2 * w1i * w3i ** 2
            + Kdrag * k * rhogeq * vgsol * w2r ** 2 * w1i ** 2 * w3i
            + w2r ** 2 * k * Kdrag ** 2 * vgsol * w3i * w1i
            - w2r ** 2 * k * Kdrag ** 2 * vdsol * w3i * w1i
            + rhogeq ** 2 * k * vgsol * w2r ** 5 * w1r
            - 2 * w3i * cs ** 2 * k ** 2 * rhogeq * rhogsol * w2i * w2r * w1i ** 2
            - 2 * w3i * cs ** 2 * k ** 2 * rhogeq * rhogsol * w2i * w2r * w1r ** 2
            - 2 * rhogeq ** 2 * k * vgsol * w2i ** 2 * w2r ** 2 * w1r ** 2
        )

        rhod2r = rhod2r - rhodeq * (
            rhogeq ** 2 * k * vgsol * w2r * w2i ** 4 * w1r
            - Kdrag * w2r ** 2 * k * rhogeq * vgsol * w2i * w3i * w1i
            - Kdrag * w2r ** 2 * k * rhogeq * vgsol * w2i * w1i ** 2
            + Kdrag * w2r ** 3 * k * rhogeq * vgsol * w3i * w1r
            - Kdrag * w2i * vgsol * k * rhogeq * w1r ** 2 * w2r ** 2
            + Kdrag * w2i ** 2 * k * rhogeq * vgsol * w2r * w3i * w1r
            + w2r ** 3 * k * Kdrag ** 2 * vgsol * w1r
            - Kdrag * w2r ** 3 * rhogsol * w1i * w3i ** 2
            - Kdrag * w2r ** 2 * rhogsol * k ** 2 * cs ** 2 * w3i * w1r
            + rhogeq ** 2 * w1i * w2r ** 2 * k * vgsol * w2i * w3i ** 2
            + Kdrag * w2r ** 2 * rhogsol * w2i * w1r * w3i ** 2
            - rhogeq * w1i ** 2 * w2r ** 3 * rhogsol * w3i ** 2
            - Kdrag * w2r ** 3 * rhogsol * w3i * w1r ** 2
            - rhogeq * w1i ** 2 * w2i ** 2 * rhogsol * w2r * w3i ** 2
            - Kdrag * w2i ** 2 * rhogsol * w1i * w2r * w3i ** 2
            - w3r ** 2 * rhogeq * w1i ** 2 * w2r ** 3 * rhogsol
            + 2 * rhogsol * k ** 4 * cs ** 4 * rhogeq * w2r * w2i * w3i
            + 3 * cs ** 2 * k ** 3 * rhogeq ** 2 * vgsol * w2r * w1r * w2i ** 2
            + 2 * rhogeq * w2i ** 2 * k * Kdrag * vdsol * w2r ** 2 * w3i
            - rhogeq * w2r ** 3 * w3i * w1r * k * Kdrag * vdsol
            + Kdrag * rhogsol * k ** 2 * cs ** 2 * w1i * w2r * w2i ** 2
            - 2 * Kdrag * k * rhogeq * vgsol * w2r ** 2 * w1i * w2i ** 2
            - 2 * rhogeq ** 2 * w2r ** 2 * w3i ** 2 * k * vgsol * w2i ** 2
            - Kdrag ** 2 * k * vgsol * w2i * w1i * w2r ** 2
            + Kdrag ** 2 * k * vdsol * w2i * w1i * w2r ** 2
            + w2r * rhogeq ** 2 * vgsol * k * w2i ** 2 * w1r * w3i ** 2
            - 2 * Kdrag * w2r ** 2 * k * rhogeq * vgsol * w2i ** 2 * w3i
            - k * Kdrag ** 2 * vdsol * w2i ** 4
            + k * Kdrag ** 2 * vgsol * w2i ** 4
            + Kdrag * w2i ** 2 * vgsol * k * rhogeq * w1i * w3r ** 2
            - w3i ** 2 * rhogeq ** 2 * cs ** 2 * k ** 3 * vgsol * w2i ** 2
            + w2i ** 2 * k * Kdrag ** 2 * vdsol * w3r * w1r
            - w2i ** 2 * k * Kdrag ** 2 * vdsol * w3i * w1i
            + w2i ** 2 * k * Kdrag ** 2 * vgsol * w3i * w1i
            - Kdrag * rhogsol * k ** 2 * cs ** 2 * w3i * w1r * w2i ** 2
            - w2i ** 2 * k * Kdrag ** 2 * vgsol * w3r * w1r
            + w3r ** 2 * rhogeq ** 2 * cs ** 2 * k ** 3 * vgsol * w2i * w1i
            - w3i * cs ** 2 * k ** 3 * rhogeq * Kdrag * vdsol * w2i * w1i
            - rhogeq ** 2 * vgsol * k * w2i ** 4 * w1r ** 2
            - rhogeq ** 2 * vgsol * k * w2i ** 4 * w1i ** 2
            - w3r ** 2 * vgsol * k * rhogeq * Kdrag * w2i ** 3
            - w3r * Kdrag * rhogsol * w2i ** 4 * w1i
            + cs ** 2 * k ** 3 * rhogeq * vgsol * Kdrag * w2i ** 3
            + w3r ** 2 * rhogeq ** 2 * vgsol * k * w2i ** 3 * w1i
            + cs ** 2 * k ** 3 * rhogeq ** 2 * vgsol * w2i ** 3 * w1i
            + w3r * vdsol * Kdrag * k * rhogeq * w2i ** 3 * w1r
            - w3i * vgsol * k * Kdrag ** 2 * w2i ** 3
            + w3i * vdsol * k * Kdrag ** 2 * w2i ** 3
            - w3i ** 2 * vgsol * k * rhogeq * Kdrag * w2i ** 3
            - w3i * vgsol * k * rhogeq * Kdrag * w2i ** 3 * w1i
            - 3 * w3r * vgsol * k * rhogeq * Kdrag * w2i ** 3 * w1r
            - w3i * Kdrag * rhogsol * w2i ** 4 * w1r
            + w3r * rhogsol * k ** 2 * cs ** 2 * Kdrag * w2i ** 3
            + w3r ** 2 * rhogeq * rhogsol * w2i ** 4 * w1r
            + w3i ** 2 * rhogeq * rhogsol * w2i ** 4 * w1r
            + w3i ** 2 * Kdrag * rhogsol * w2i ** 3 * w1r
            + w3r ** 2 * Kdrag * rhogsol * w2i ** 3 * w1r
            - 2 * w3i * cs ** 2 * k ** 3 * rhogeq ** 2 * vgsol * w2i ** 2 * w1i
            + w3i * cs ** 2 * k ** 3 * rhogeq ** 2 * vgsol * w2i ** 3
            + w3i * cs ** 2 * k ** 3 * rhogeq ** 2 * vgsol * w2i * w1r ** 2
            + w3i * cs ** 2 * k ** 3 * rhogeq ** 2 * vgsol * w2i * w1i ** 2
            - cs ** 2 * k ** 3 * rhogeq * vdsol * Kdrag * w2i ** 3
            - 2 * w3r * rhogeq ** 2 * k * vgsol * w2i ** 4 * w1r
            - w3r ** 2 * rhogeq ** 2 * vgsol * k * w2i ** 4
            + w3r * cs ** 2 * k ** 2 * rhogeq * rhogsol * w2i ** 4
            - w3i ** 2 * rhogeq ** 2 * vgsol * k * w2i ** 4
            - w3i * vdsol * Kdrag * k * rhogeq * w2i ** 3 * w1i
            + w3i * vdsol * Kdrag * k * rhogeq * w2i ** 4
            + rhogeq * rhogsol * k ** 2 * cs ** 2 * w2i ** 4 * w1r
            + rhogeq * k * Kdrag * vdsol * w2i ** 4 * w1i
            - 2 * w3i * rhogeq ** 2 * k * vgsol * w2i ** 4 * w1i
            + w3i * rhogeq ** 2 * k * vgsol * w2i ** 3 * w1i ** 2
            + w3i * rhogeq ** 2 * k * vgsol * w2i ** 3 * w1r ** 2
            - 2 * w3r * cs ** 2 * k ** 2 * rhogeq * rhogsol * w2i ** 3 * w1i
            + w3i ** 2 * rhogeq ** 2 * vgsol * k * w2i ** 3 * w1i
            + w3r * rhogeq * cs ** 2 * k ** 2 * w1r ** 2 * rhogsol * w2i ** 2
            + w3r * rhogeq * cs ** 2 * k ** 2 * w1i ** 2 * rhogsol * w2i ** 2
            + w3r * rhogeq * cs ** 4 * k ** 4 * rhogsol * w2i ** 2
            - w3r * Kdrag * rhogsol * k ** 2 * cs ** 2 * w1i * w2i ** 2
            - w3r ** 2 * rhogeq ** 2 * cs ** 2 * k ** 3 * vgsol * w2i ** 2
            - w3i * cs ** 4 * k ** 4 * rhogeq * rhogsol * w2i * w1r
            - w3r * rhogeq * cs ** 4 * k ** 4 * rhogsol * w2i * w1i
            + rhogeq * cs ** 4 * k ** 4 * w1r * rhogsol * w2i ** 2
            - w3i * cs ** 2 * k ** 3 * rhogeq * Kdrag * vgsol * w2i ** 2
            + w3i * cs ** 2 * k ** 3 * rhogeq * Kdrag * vdsol * w2i ** 2
            - rhogeq * cs ** 2 * k ** 3 * w1i * Kdrag * vgsol * w2i ** 2
            + rhogeq * cs ** 2 * k ** 3 * w1i * Kdrag * vdsol * w2i ** 2
            + w3r * rhogeq * cs ** 2 * k ** 3 * w2i * Kdrag * vdsol * w1r
            - rhogeq ** 2 * cs ** 2 * k ** 3 * w1r ** 2 * vgsol * w2i ** 2
            - w3r * rhogeq * cs ** 2 * k ** 3 * w2i * Kdrag * vgsol * w1r
            - rhogeq ** 2 * cs ** 2 * k ** 3 * w1i ** 2 * vgsol * w2i ** 2
            + w3i * cs ** 2 * k ** 3 * rhogeq * Kdrag * vgsol * w2i * w1i
            + Kdrag * w2i ** 2 * k * rhogeq * vgsol * w3i * w1i ** 2
            + w3i ** 2 * rhogeq ** 2 * cs ** 2 * k ** 3 * vgsol * w2i * w1i
            - 2 * w3i * cs ** 2 * k ** 2 * rhogeq * rhogsol * w2i ** 3 * w1r
            + w3r * rhogeq * rhogsol * w2i ** 4 * w1r ** 2
            + w3r * rhogeq * rhogsol * w2i ** 4 * w1i ** 2
            - rhogeq ** 2 * k * vgsol * w2r ** 4 * w1r ** 2
            + Kdrag * w2i ** 2 * k * rhogeq * vgsol * w3i ** 2 * w1i
            + Kdrag * w2i ** 2 * k * rhogeq * vgsol * w3i * w1r ** 2
        )
        rhod2r = (
            rhod2r
            / (w2i ** 2 + w2r ** 2)
            / rhogeq
            / (
                w2r ** 2
                - 2 * w3r * w2r
                + w2i ** 2
                + w3i ** 2
                - 2 * w2i * w3i
                + w3r ** 2
            )
            / (
                w2r ** 2
                + w1r ** 2
                + w2i ** 2
                - 2 * w2i * w1i
                - 2 * w2r * w1r
                + w1i ** 2
            )
            / Kdrag
        )

        rhod2i = rhodeq * (
            -w3r ** 2 * Kdrag * rhogsol * w1i * w2i ** 3
            - w2r * w3i * vdsol * k * Kdrag ** 2 * w2i ** 2
            + w2r * w3i ** 2 * vgsol * k * rhogeq * Kdrag * w2i ** 2
            + 3 * w2r * w3i * vgsol * k * rhogeq * Kdrag * w2i ** 2 * w1i
            + w2r * w3r * vgsol * k * rhogeq * Kdrag * w2i ** 2 * w1r
            + w2r * w3r ** 2 * vgsol * k * rhogeq * Kdrag * w2i ** 2
            - 3 * w2r * cs ** 2 * k ** 3 * rhogeq * vgsol * Kdrag * w2i ** 2
            - w2r * w3r * rhogsol * k ** 2 * cs ** 2 * Kdrag * w2i ** 2
            + 2 * w2r * w3i * Kdrag * rhogsol * w2i ** 3 * w1r
            - w2r * w3i * cs ** 2 * k ** 3 * rhogeq ** 2 * vgsol * w1i ** 2
            + 3 * w2r * cs ** 2 * k ** 3 * rhogeq * vdsol * Kdrag * w2i ** 2
            - w2r * w3r ** 2 * Kdrag * rhogsol * w2i ** 2 * w1r
            - 3 * w2r * w3i * cs ** 2 * k ** 3 * rhogeq ** 2 * vgsol * w2i ** 2
            - 2 * Kdrag * k * rhogeq * vgsol * w2r ** 3 * w1i * w2i
            + 4 * w2r * w3i * cs ** 2 * k ** 3 * rhogeq ** 2 * vgsol * w2i * w1i
            - w2r * w3i ** 2 * Kdrag * rhogsol * w2i ** 2 * w1r
            - w2r * w3i * cs ** 2 * k ** 3 * rhogeq ** 2 * vgsol * w1r ** 2
            - w2r * w3i * vdsol * Kdrag * k * rhogeq * w2i ** 2 * w1i
            + w2r * w3i * vgsol * k * Kdrag ** 2 * w2i ** 2
            + Kdrag * vgsol * k * rhogeq * w1r ** 2 * w2r ** 3
            - Kdrag * w2i * k * rhogeq * vgsol * w2r ** 2 * w3i * w1r
            - Kdrag * w2r ** 3 * rhogsol * w1r * w3i ** 2
            - 3 * cs ** 2 * k ** 3 * rhogeq ** 2 * vgsol * w2r ** 2 * w1r * w2i
            + Kdrag * rhogsol * k ** 2 * cs ** 2 * w1i * w2r ** 2 * w2i
            - w3i ** 2 * Kdrag * rhogsol * w1i * w2i ** 3
            + 4 * w2r * w3r * rhogeq ** 2 * cs ** 2 * k ** 3 * w1r * vgsol * w2i
            - w2r * w3r * Kdrag * rhogsol * w2i ** 2 * w1i ** 2
            - Kdrag ** 2 * k * vdsol * w1i * w2r ** 3
            - w2r ** 2 * rhogeq ** 2 * vgsol * k * w2i * w1r * w3i ** 2
            - 2 * w2r * w1r * w3r ** 2 * cs ** 2 * k ** 2 * rhogeq * rhogsol * w2i
            - w2r * k * Kdrag ** 2 * vdsol * w2i ** 2 * w1i
            + w2r * w3r * rhogeq * rhogsol * w2i ** 4 * w1i
            - 2 * w2r * w1r * w3i ** 2 * cs ** 2 * k ** 2 * rhogeq * rhogsol * w2i
            + w2r * Kdrag * k * rhogeq * vgsol * w2i ** 2 * w1i ** 2
            - w2r * rhogeq ** 2 * k * vgsol * w2i ** 4 * w1i
            - w2r * w3r * Kdrag * rhogsol * w2i ** 2 * w1r ** 2
            + w2r * k * Kdrag ** 2 * vgsol * w2i ** 2 * w1i
            - 2 * Kdrag * w2r ** 3 * k * rhogeq * vgsol * w2i * w3i
            + Kdrag ** 2 * k * vgsol * w1i * w2r ** 3
            + rhogeq * w1i ** 2 * w2i * rhogsol * w2r ** 2 * w3i ** 2
            - Kdrag * w2i * rhogsol * w1i * w2r ** 2 * w3i ** 2
            - rhogsol * k ** 4 * cs ** 4 * rhogeq * w2r ** 2 * w3i
            + rhogeq ** 2 * w1i * w2r ** 3 * k * vgsol * w3i ** 2
            + w3r ** 2 * rhogeq ** 2 * k * vgsol * w2r ** 3 * w1i
            + w3r ** 2 * Kdrag * rhogeq * k * vgsol * w2r ** 3
            + w2r * w3i * rhogeq * rhogsol * w2i ** 4 * w1r
            - w2r * w3i * rhogeq ** 2 * k * vgsol * w2i ** 4
            + w3r ** 2 * cs ** 2 * k ** 2 * rhogeq * rhogsol * w2r ** 2 * w1i
            + w2r * vdsol * Kdrag * k * rhogeq * w2i ** 4
            - w2r * vgsol * k * rhogeq * Kdrag * w2i ** 4
            + 2 * w2r * w3i ** 2 * rhogeq ** 2 * cs ** 2 * k ** 3 * vgsol * w2i
            + w2r * Kdrag * k * rhogeq * vgsol * w2i ** 2 * w1r ** 2
            - w2r * w1r * Kdrag * rhogsol * k ** 2 * cs ** 2 * w2i ** 2
            - 2 * vgsol * k * rhogeq ** 2 * w2i ** 2 * w2r ** 3 * w1i
            - 3 * w3r * cs ** 2 * k ** 3 * rhogeq ** 2 * vgsol * w2i * w2r ** 2
            + w3r * rhogeq * k * Kdrag * vdsol * w2r ** 3 * w1r
            - w3r * Kdrag * rhogsol * w2r ** 3 * w1r ** 2
            - w3r * Kdrag * rhogsol * w2r ** 3 * w1i ** 2
            + w3r * rhogeq * rhogsol * w2r ** 5 * w1i
            - 4 * w3i * cs ** 2 * k ** 2 * rhogeq * rhogsol * w2r ** 2 * w2i * w1i
            + w3r ** 2 * rhogeq * rhogsol * w2r ** 2 * w2i * w1r ** 2
            - w3r * cs ** 2 * k ** 3 * rhogeq * Kdrag * vgsol * w2r ** 2
            + w3r * cs ** 2 * k ** 3 * rhogeq * Kdrag * vdsol * w2r ** 2
            + w3r * rhogeq * k * Kdrag * vdsol * w2r ** 2 * w2i * w1i
            - w3r * Kdrag * cs ** 2 * k ** 2 * rhogsol * w2r ** 3
            - 2 * w3r * rhogeq * cs ** 2 * k ** 2 * rhogsol * w2r ** 3 * w1i
            + 4 * w3r * rhogeq * cs ** 2 * k ** 2 * rhogsol * w2r ** 2 * w2i * w1r
            - w3r ** 2 * rhogeq ** 2 * k * vgsol * w2r ** 2 * w2i * w1r
            - w3r * rhogeq ** 2 * k * vgsol * w2r ** 2 * w2i * w1r ** 2
            - w3r ** 2 * Kdrag * rhogsol * w2r ** 3 * w1r
            + w3r * Kdrag * rhogsol * w2r ** 4 * w1r
            + w3r * Kdrag * k * rhogeq * vgsol * w2r ** 3 * w1r
            - 2 * w2r * k * Kdrag ** 2 * vgsol * w2i ** 3
            + rhogeq * w2r ** 2 * w3i * w1r * k * Kdrag * vdsol * w2i
            - rhogeq * w3i * w1i * k * Kdrag * vdsol * w2r ** 3
            + 2 * w2r ** 2 * w3r * rhogeq ** 2 * k * vgsol * w2i ** 3
            - 2 * w3i * cs ** 2 * k ** 2 * rhogeq * rhogsol * w2r ** 3 * w1r
            - w3r * rhogeq ** 2 * k * vgsol * w2r ** 2 * w2i * w1i ** 2
            - rhogeq ** 2 * k * vgsol * w2r ** 5 * w1i
            - w2r ** 2 * w3r ** 2 * Kdrag * rhogsol * w1i * w2i
            + 2 * w3r * rhogeq * rhogsol * w2i ** 2 * w2r ** 3 * w1i
            + w3r ** 2 * rhogeq * rhogsol * w2r ** 2 * w2i * w1i ** 2
            - w3r * w2r ** 2 * Kdrag * rhogeq * k * vgsol * w2i * w1i
            + rhogeq ** 2 * cs ** 2 * k ** 3 * w2r ** 3 * vgsol * w1i
            + 3 * rhogeq * cs ** 4 * k ** 4 * w2r ** 2 * rhogsol * w2i
            + rhogeq * cs ** 2 * k ** 3 * Kdrag * vdsol * w2r ** 2 * w1r
            - rhogeq * cs ** 2 * k ** 3 * Kdrag * vgsol * w2r ** 2 * w1r
            - 2 * w2r ** 2 * w3r * rhogeq * rhogsol * w2i ** 3 * w1r
            - w3r * rhogeq * w2r ** 4 * w1r * rhogsol * w2i
            + w3r * rhogeq ** 2 * w2i * k * vgsol * w2r ** 4
            + 2 * w2r * k * Kdrag ** 2 * vdsol * w2i ** 3
            - rhogeq * cs ** 2 * k ** 3 * Kdrag * vdsol * w2r ** 3
            - w2r ** 2 * w3r * w2i * k * Kdrag ** 2 * vdsol
            + 2 * w2r ** 2 * Kdrag * w3r * w2i ** 2 * k * rhogeq * vgsol
            + rhogeq * w2r ** 2 * w3i ** 2 * w1r ** 2 * rhogsol * w2i
            + rhogeq ** 2 * k * vgsol * w3i * w1r ** 2 * w2r ** 3
            - rhogeq * cs ** 4 * k ** 4 * w2r ** 2 * rhogsol * w1i
        )

        rhod2i = rhod2i + rhodeq * (
            rhogeq * cs ** 2 * k ** 3 * Kdrag * vgsol * w2r ** 3
            + rhogeq * w2r ** 2 * w3i ** 2 * cs ** 2 * k ** 2 * rhogsol * w1i
            + w3i ** 2 * Kdrag * rhogsol * w2i ** 2 * w1r ** 2
            + rhogeq ** 2 * cs ** 2 * k ** 3 * w3i * vgsol * w2r ** 3
            + 2 * w3i * rhogeq * rhogsol * w2r ** 2 * w2i ** 3 * w1i
            + w3i * rhogeq * rhogsol * w2r ** 5 * w1r
            + w3i * rhogeq * rhogsol * w2i * w2r ** 4 * w1i
            + w2r ** 2 * w3r * w2i * k * Kdrag ** 2 * vgsol
            + rhogsol * k ** 2 * cs ** 2 * Kdrag * w2r ** 4
            - 2 * w3i * rhogeq ** 2 * k * vgsol * w2r ** 3 * w2i ** 2
            - cs ** 2 * k ** 2 * rhogeq * rhogsol * w2r ** 4 * w2i
            - 2 * cs ** 2 * k ** 2 * rhogeq * rhogsol * w2r ** 2 * w2i ** 3
            - w3i * rhogeq ** 2 * k * vgsol * w2r ** 5
            + rhogeq ** 2 * k * vgsol * w2i * w2r ** 4 * w1r
            + vdsol * Kdrag * k * rhogeq * w2r ** 5
            + 2 * w3i * rhogeq * rhogsol * w2r ** 3 * w2i ** 2 * w1r
            + w3i ** 2 * Kdrag * rhogsol * w2i ** 2 * w1i ** 2
            + 2 * vdsol * Kdrag * k * rhogeq * w2r ** 3 * w2i ** 2
            + 2 * vgsol * k * rhogeq * Kdrag * w2r ** 2 * w2i ** 2 * w1r
            + w3i * rhogsol * k ** 2 * cs ** 2 * Kdrag * w2r ** 2 * w2i
            + w3i * vgsol * k * Kdrag ** 2 * w2r ** 3
            - w2i * k * Kdrag ** 2 * vdsol * w2r ** 2 * w1r
            + rhogeq ** 2 * w1i ** 2 * w2r ** 3 * k * vgsol * w3i
            - w3i * vdsol * k * Kdrag ** 2 * w2r ** 3
            - w3i * Kdrag * rhogsol * w2r ** 4 * w1i
            - 2 * vgsol * k * rhogeq * Kdrag * w2r ** 3 * w2i ** 2
            - Kdrag * w2i * rhogsol * w1i ** 2 * w2r ** 2 * w3i
            + w2i * k * Kdrag ** 2 * vgsol * w2r ** 2 * w1r
            + 3 * Kdrag * w2r ** 3 * k * rhogeq * vgsol * w3i * w1i
            + Kdrag * w2r ** 3 * k * rhogeq * vgsol * w1i ** 2
            + w3r ** 2 * Kdrag * rhogsol * w2i ** 2 * w1i ** 2
            - rhogeq * w1i * rhogsol * k ** 2 * cs ** 2 * w3i ** 2 * w2i ** 2
            + w3i * cs ** 2 * k ** 2 * rhogeq * rhogsol * w2r ** 2 * w1i ** 2
            + w3i * cs ** 2 * k ** 2 * rhogeq * rhogsol * w2r ** 2 * w1r ** 2
            + 2 * rhogeq ** 2 * k * vgsol * w2r ** 2 * w2i ** 3 * w1r
            - vgsol * k * rhogeq * Kdrag * w2r ** 5
            + Kdrag * w2r ** 3 * k * rhogeq * vgsol * w3i ** 2
            - Kdrag * w2i * rhogsol * w3i * w2r ** 2 * w1r ** 2
            - Kdrag * w2r ** 3 * rhogsol * k ** 2 * cs ** 2 * w1r
            + 2 * Kdrag * w2i ** 4 * vgsol * k * rhogeq * w1r
            - 2 * Kdrag * w2i ** 3 * w2r * vgsol * k * rhogeq * w1i
            + Kdrag * w2i ** 4 * rhogsol * w3i * w1i
            + rhogeq * w1i * w2r ** 4 * rhogsol * k ** 2 * cs ** 2
            + 2 * rhogeq * w1i * w2r ** 2 * rhogsol * k ** 2 * cs ** 2 * w2i ** 2
            - Kdrag * w3r * w2i ** 4 * rhogsol * w1r
            + w2r * w3i * cs ** 4 * k ** 4 * rhogeq * rhogsol * w1r
            + rhogeq * w1i * w3r * w2i ** 3 * k * Kdrag * vdsol
            + 2 * Kdrag * w2r ** 3 * rhogsol * w3i * w1r * w2i
            - 2 * w2r * w3r * rhogeq * cs ** 4 * k ** 4 * rhogsol * w2i
            + 2 * w2r * w3r ** 2 * rhogeq ** 2 * cs ** 2 * k ** 3 * vgsol * w2i
            + rhogeq * w1i * w2i ** 4 * rhogsol * k ** 2 * cs ** 2
            + w2r * w3i * rhogeq ** 2 * k * vgsol * w2i ** 2 * w1r ** 2
            + 2 * w2r * w3r * cs ** 2 * k ** 2 * rhogeq * rhogsol * w2i ** 2 * w1i
            - 2 * w2r * w3i * cs ** 2 * k ** 3 * rhogeq * Kdrag * vdsol * w2i
            + 2 * w2r * rhogeq * cs ** 2 * k ** 3 * w1i * Kdrag * vgsol * w2i
            - 2 * w2r * rhogeq * cs ** 2 * k ** 3 * w1i * Kdrag * vdsol * w2i
            + w2r * w3r * rhogeq * cs ** 4 * k ** 4 * rhogsol * w1i
            + 2 * w2r * w3i * cs ** 2 * k ** 3 * rhogeq * Kdrag * vgsol * w2i
            - w2r * w3i * cs ** 2 * k ** 3 * rhogeq * Kdrag * vgsol * w1i
            - 2 * w2r * rhogeq * cs ** 4 * k ** 4 * w1r * rhogsol * w2i
            + w2r * w3i * rhogeq ** 2 * k * vgsol * w2i ** 2 * w1i ** 2
            + w2r * w3i ** 2 * rhogeq ** 2 * vgsol * k * w2i ** 2 * w1i
            - 2 * w2r * w3r * rhogeq * cs ** 2 * k ** 2 * w1r ** 2 * rhogsol * w2i
            - 2 * w2r * w3r * rhogeq * cs ** 2 * k ** 2 * w1i ** 2 * rhogsol * w2i
            - w2r * w3i ** 2 * rhogeq ** 2 * cs ** 2 * k ** 3 * vgsol * w1i
            + 2 * w2r * w3i * cs ** 2 * k ** 2 * rhogeq * rhogsol * w2i ** 2 * w1r
            - rhogeq ** 2 * w1i ** 2 * k * vgsol * w3r * w2i ** 3
            + rhogeq * w1i ** 2 * rhogsol * k ** 2 * cs ** 2 * w3i ** 2 * w2i
            + rhogeq * w1i ** 2 * rhogsol * k ** 2 * cs ** 2 * w3r ** 2 * w2i
            - w2r * w3r * rhogeq * cs ** 2 * k ** 3 * Kdrag * vdsol * w1r
            + 2 * w2r * rhogeq ** 2 * cs ** 2 * k ** 3 * w1r ** 2 * vgsol * w2i
            + w2r * w3r * rhogeq * cs ** 2 * k ** 3 * Kdrag * vgsol * w1r
            + 2 * w2r * rhogeq ** 2 * cs ** 2 * k ** 3 * w1i ** 2 * vgsol * w2i
            - w3i * cs ** 4 * k ** 4 * rhogeq * rhogsol * w1i * w2i
            - w3i * cs ** 2 * k ** 2 * rhogeq * rhogsol * w2i ** 2 * w1r ** 2
            + w3r * rhogeq ** 2 * cs ** 2 * k ** 3 * w2i ** 3 * vgsol
            - w3i * cs ** 2 * k ** 3 * rhogeq * Kdrag * vgsol * w1r * w2i
            + rhogeq ** 2 * cs ** 2 * k ** 3 * w2i ** 3 * w1r * vgsol
            + w3i * cs ** 2 * k ** 3 * rhogeq * Kdrag * vdsol * w1r * w2i
            - rhogeq * w1i * rhogsol * k ** 2 * cs ** 2 * w3r ** 2 * w2i ** 2
            - w3r * w2i ** 3 * k * Kdrag ** 2 * vdsol
            + w3r * Kdrag * rhogsol * k ** 2 * cs ** 2 * w1r * w2r ** 2
            - w3r * Kdrag * k * rhogeq * vgsol * w2i ** 2 * w1r ** 2
            - w3r * Kdrag * k * rhogeq * vgsol * w1r ** 2 * w2r ** 2
            - w3i * cs ** 2 * k ** 2 * rhogeq * rhogsol * w2i ** 2 * w1i ** 2
        )

        rhod2i = rhod2i + rhodeq * (
            w3i * cs ** 4 * k ** 4 * rhogeq * rhogsol * w2i ** 2
            + rhogeq * cs ** 4 * k ** 4 * w2i ** 2 * w1i * rhogsol
            - w3r * Kdrag * k * rhogeq * vgsol * w1i ** 2 * w2r ** 2
            + w3r * Kdrag ** 2 * k * vdsol * w2i ** 2 * w1i
            + w3r * Kdrag ** 2 * k * vdsol * w1i * w2r ** 2
            + w3r * Kdrag * rhogsol * k ** 2 * cs ** 2 * w2i ** 2 * w1r
            - w3r * rhogeq * cs ** 2 * k ** 3 * w1i * Kdrag * vgsol * w2i
            - w3r * rhogeq * cs ** 2 * k ** 3 * w2i ** 2 * Kdrag * vdsol
            + w3r ** 2 * rhogeq * w2i ** 3 * w1i ** 2 * rhogsol
            - 2 * w2i * k * Kdrag ** 2 * vgsol * w2r ** 3
            - w3r * rhogeq ** 2 * cs ** 2 * k ** 3 * w1i ** 2 * vgsol * w2i
            - w3r * Kdrag ** 2 * k * vgsol * w2i ** 2 * w1i
            - w3r * Kdrag ** 2 * k * vgsol * w1i * w2r ** 2
            - w3r * Kdrag * k * rhogeq * vgsol * w1i * w2i ** 3
            - w3r * Kdrag * k * rhogeq * vgsol * w2i ** 2 * w1i ** 2
            + w3r ** 2 * rhogeq * w1r ** 2 * w2i ** 3 * rhogsol
            - w3r * rhogeq ** 2 * cs ** 2 * k ** 3 * w1r ** 2 * vgsol * w2i
            + w3r * rhogeq * cs ** 4 * k ** 4 * w1r * rhogsol * w2i
            - w3r ** 2 * rhogeq ** 2 * w2i ** 3 * w1r * k * vgsol
            - w3r ** 2 * rhogeq ** 2 * cs ** 2 * k ** 3 * w1r * vgsol * w2i
            + w3r * rhogeq * cs ** 2 * k ** 3 * w2i ** 2 * Kdrag * vgsol
            + w3r * rhogeq * cs ** 2 * k ** 3 * w1i * Kdrag * vdsol * w2i
            - w3r ** 2 * rhogeq * w2i ** 2 * w1r * k * Kdrag * vgsol
            - w3r ** 2 * rhogeq * w1r * k * Kdrag * vgsol * w2r ** 2
            + w3r ** 2 * Kdrag * rhogsol * w1i ** 2 * w2r ** 2
            + w3r ** 2 * Kdrag * rhogsol * w2i ** 2 * w1r ** 2
            - Kdrag * rhogsol * k ** 2 * cs ** 2 * w3i * w1i * w2r ** 2
            - Kdrag * k * rhogeq * vgsol * w3i * w1r * w2i ** 3
            - Kdrag * rhogsol * w1i ** 2 * w2i ** 3 * w3i
            + w2r * w3i * cs ** 2 * k ** 3 * rhogeq * Kdrag * vdsol * w1i
            - Kdrag * rhogsol * k ** 2 * cs ** 2 * w2i ** 2 * w3i * w1i
            - Kdrag ** 2 * k * vgsol * w2i ** 2 * w3i * w1r
            + w3i ** 2 * Kdrag * rhogsol * w1r ** 2 * w2r ** 2
            + w3i ** 2 * Kdrag * rhogsol * w1i ** 2 * w2r ** 2
            + w3r ** 2 * Kdrag * rhogsol * w1r ** 2 * w2r ** 2
            + w2r * w3r ** 2 * rhogeq ** 2 * vgsol * k * w2i ** 2 * w1i
            - 3 * w2r * cs ** 2 * k ** 3 * rhogeq ** 2 * vgsol * w2i ** 2 * w1i
            + w2r * w3r * vdsol * Kdrag * k * rhogeq * w2i ** 2 * w1r
            - w2r * w3r ** 2 * rhogeq ** 2 * cs ** 2 * k ** 3 * vgsol * w1i
            + 2 * w2r * w3r * Kdrag * rhogsol * w2i ** 3 * w1i
            - w3i ** 2 * rhogeq ** 2 * cs ** 2 * k ** 3 * w1r * vgsol * w2i
            - rhogeq * cs ** 2 * k ** 3 * w2i ** 2 * w1r * Kdrag * vdsol
            - Kdrag * rhogsol * w2i ** 3 * w1r ** 2 * w3i
            - w3i ** 2 * rhogeq * w2i ** 2 * w1r * k * Kdrag * vgsol
            - w3i ** 2 * rhogeq * w1r * k * Kdrag * vgsol * w2r ** 2
            - rhogeq * w1r * k * Kdrag * vdsol * w2r ** 4
            - 2 * rhogeq * w2i ** 2 * w1r * k * Kdrag * vdsol * w2r ** 2
            - 2 * rhogeq * w1i * w3i ** 2 * rhogsol * w2i ** 2 * w2r ** 2
            - rhogeq * w1i * w3i ** 2 * rhogsol * w2r ** 4
            - 2 * w3i * rhogeq * rhogsol * w2i ** 2 * w1r ** 2 * w2r ** 2
            - w3r * rhogeq ** 2 * k * vgsol * w2i ** 3 * w1r ** 2
            - w3i * rhogeq * rhogsol * w1r ** 2 * w2r ** 4
            + 2 * w3i * rhogeq * rhogsol * k ** 2 * cs ** 2 * w2i ** 2 * w2r ** 2
            + w3i * rhogeq * rhogsol * k ** 2 * cs ** 2 * w2r ** 4
            + w3i ** 2 * rhogeq * w1r ** 2 * w2i ** 3 * rhogsol
            - w3r ** 2 * rhogeq * rhogsol * w2i ** 4 * w1i
            - 2 * w3r ** 2 * rhogeq * rhogsol * w2i ** 2 * w1i * w2r ** 2
            - w3i ** 2 * rhogeq * rhogsol * w2i ** 4 * w1i
            + w3i ** 2 * rhogeq * w2i ** 3 * w1i ** 2 * rhogsol
            - rhogeq * rhogsol * k ** 2 * cs ** 2 * w2i ** 5
            + w3r ** 2 * cs ** 2 * k ** 2 * rhogeq * rhogsol * w1r ** 2 * w2i
            - w3i * rhogeq * rhogsol * w2i ** 4 * w1r ** 2
            + w3i ** 2 * cs ** 2 * k ** 2 * rhogeq * rhogsol * w1r ** 2 * w2i
            + w3i * cs ** 2 * k ** 2 * rhogeq * rhogsol * w2i ** 4
            + 2 * w2i * k * Kdrag ** 2 * vdsol * w2r ** 3
            - w3r * rhogeq * rhogsol * w2i ** 5 * w1r
            - rhogeq * w1i ** 2 * w2i ** 4 * rhogsol * w3i
            - cs ** 4 * k ** 4 * rhogeq * rhogsol * w2i ** 3
            + w3r * w2i ** 3 * k * Kdrag ** 2 * vgsol
            - rhogeq * k * Kdrag * vdsol * w2i ** 4 * w1r
            + w3i * vdsol * Kdrag * k * rhogeq * w2i ** 3 * w1r
            + w3i * rhogeq * rhogsol * w2i ** 5 * w1i
            + w3r * rhogeq ** 2 * k * vgsol * w2i ** 5
            - w3r * vdsol * Kdrag * k * rhogeq * w2i ** 4
            - 2 * w3r * vdsol * Kdrag * k * rhogeq * w2i ** 2 * w2r ** 2
            - 2 * Kdrag * w2i ** 3 * k * rhogeq * vgsol * w2r * w3i
            - w2i ** 3 * k * Kdrag ** 2 * vdsol * w1r
            + w2i ** 3 * k * Kdrag ** 2 * vgsol * w1r
            + 2 * Kdrag * w3r * w2r ** 3 * rhogsol * w1i * w2i
            + Kdrag * w2i ** 3 * rhogsol * k ** 2 * cs ** 2 * w3i
            - 2 * rhogeq * w1i ** 2 * w2i ** 2 * rhogsol * w3i * w2r ** 2
            - rhogeq * w1i ** 2 * w2r ** 4 * rhogsol * w3i
            - Kdrag * w2i ** 4 * rhogsol * k ** 2 * cs ** 2
            + Kdrag * w2i ** 3 * rhogsol * k ** 2 * cs ** 2 * w1i
            - w3r ** 2 * rhogeq * rhogsol * w1i * w2r ** 4
            - w3r * rhogeq * k * Kdrag * vdsol * w2r ** 4
            + 2 * Kdrag * w3r * w2i ** 4 * k * rhogeq * vgsol
            - w3i ** 2 * rhogeq ** 2 * w2i ** 3 * w1r * k * vgsol
            + Kdrag ** 2 * k * vdsol * w2i ** 2 * w3i * w1r
            + Kdrag ** 2 * k * vdsol * w3i * w1r * w2r ** 2
            - Kdrag ** 2 * k * vgsol * w3i * w1r * w2r ** 2
            + rhogeq * cs ** 2 * k ** 3 * w2i ** 2 * w1r * Kdrag * vgsol
            + rhogeq ** 2 * w1r * w2i ** 5 * k * vgsol
        )
        rhod2i = (
            rhod2i
            / (w2i ** 2 + w2r ** 2)
            / rhogeq
            / (
                w2r ** 2
                - 2 * w3r * w2r
                + w2i ** 2
                + w3i ** 2
                - 2 * w2i * w3i
                + w3r ** 2
            )
            / (
                w2r ** 2
                + w1r ** 2
                + w2i ** 2
                - 2 * w2i * w1i
                - 2 * w2r * w1r
                + w1i ** 2
            )
            / Kdrag
        )

        rhod1r = (
            2 * w2i * w3r * cs ** 2 * k ** 3 * rhogeq * Kdrag * vgsol * w2r * w3i ** 2
            - w3r * rhogeq * cs ** 2 * k ** 3 * w2i * Kdrag * vgsol * w1r * w3i ** 2
            + rhogeq * w2r * w3i * w1r * k * Kdrag * vdsol * w2i ** 2 * w3r ** 2
            + w3r * rhogeq * cs ** 2 * k ** 3 * w2i * Kdrag * vdsol * w1r * w3i ** 2
            - w3i * cs ** 2 * k ** 3 * rhogeq * Kdrag * vdsol * w2i * w1i * w3r ** 2
            - Kdrag * w2i ** 2 * k * rhogeq * vgsol * w2r * w3i * w1r * w3r ** 2
            + Kdrag * w2r ** 2 * k * rhogeq * vgsol * w2i * w3i * w1i * w3r ** 2
            + 4
            * w3i
            * cs ** 2
            * k ** 2
            * rhogeq
            * rhogsol
            * w2i
            * w2r
            * w1i ** 2
            * w3r ** 2
            - rhogeq * cs ** 2 * k ** 3 * w3i * Kdrag * vgsol * w2r * w1r * w3r ** 2
            + rhogeq * w2i * w3i * w1i * k * Kdrag * vdsol * w2r ** 2 * w3r ** 2
            + w2i * w3r * rhogeq * k * Kdrag * vdsol * w2r ** 2 * w1r * w3i ** 2
            - 4 * rhogeq * cs ** 2 * k ** 3 * w2i * Kdrag * vgsol * w2r * w1r * w3r ** 2
            + 4 * rhogeq * cs ** 2 * k ** 3 * w2i * Kdrag * vdsol * w2r * w1r * w3r ** 2
            + w2r * w3r * rhogeq * cs ** 2 * k ** 3 * w1i * Kdrag * vdsol * w3i ** 2
            + 2 * rhogeq * cs ** 2 * k ** 3 * w3i ** 2 * Kdrag * vgsol * w2r * w1r * w1i
            - 2
            * w2i
            * w3i ** 2
            * cs ** 2
            * k ** 2
            * rhogeq
            * rhogsol
            * w2r ** 2
            * w1r
            * w1i
            + w1i ** 2 * rhogeq * cs ** 2 * k ** 3 * w3i * Kdrag * vdsol * w2r * w1r
            - w1i ** 2 * rhogeq * cs ** 2 * k ** 3 * w3i * Kdrag * vgsol * w2r * w1r
            + w3i * cs ** 2 * k ** 3 * rhogeq * Kdrag * vgsol * w2i * w1i * w3r ** 2
            - w3r * rhogeq * k * Kdrag * vdsol * w2r * w2i ** 2 * w1i * w3i ** 2
            + 4
            * w3r
            * w2i
            * cs ** 2
            * k ** 3
            * rhogeq
            * Kdrag
            * vdsol
            * w2r
            * w3i
            * w1i
            - 2 * rhogeq * cs ** 2 * k ** 3 * w3i ** 2 * Kdrag * vdsol * w2r * w1r * w1i
            - 2
            * w2r
            * rhogeq ** 2
            * cs ** 2
            * k ** 3
            * w1r
            * vgsol
            * w3i
            * w1i
            * w2i ** 2
            - 4
            * w2i
            * w3r ** 2
            * cs ** 2
            * k ** 2
            * rhogeq
            * rhogsol
            * w2r
            * w1i
            * w3i ** 2
            + 3 * w3r * w2r * Kdrag * rhogeq * k * vgsol * w2i ** 2 * w1i * w3i ** 2
            - w2i * w3r * Kdrag * k * rhogeq * vgsol * w2r ** 2 * w1r * w3i ** 2
            - w2r * w3r * rhogeq * cs ** 2 * k ** 3 * w1i * Kdrag * vgsol * w3i ** 2
            + rhogeq * cs ** 2 * k ** 3 * w3i * Kdrag * vdsol * w2r * w1r * w3r ** 2
            + 2 * w3r * w3i * cs ** 4 * k ** 4 * rhogeq * rhogsol * w2i * w2r ** 2
            - 2 * w3r * w3i * cs ** 2 * k ** 3 * rhogeq * Kdrag * vdsol * w2r * w2i ** 2
            - 2 * w3r * w3i * cs ** 2 * k ** 3 * rhogeq * Kdrag * vdsol * w2r ** 3
            - 4 * w3r * w3i * cs ** 2 * k ** 3 * rhogeq * Kdrag * vgsol * w1r * w2r ** 2
            + 2 * w3r * w3i * cs ** 2 * k ** 3 * rhogeq * Kdrag * vgsol * w2r ** 3
            + 2 * w3r * w3i * cs ** 2 * k ** 3 * rhogeq * Kdrag * vgsol * w2r * w2i ** 2
            + 4 * w3r * w3i * cs ** 2 * k ** 3 * rhogeq * Kdrag * vdsol * w1r * w2r ** 2
            + 2 * w3r * w3i * cs ** 4 * k ** 4 * rhogeq * rhogsol * w2i ** 3
            + 4
            * w2r
            * rhogeq ** 2
            * cs ** 2
            * k ** 3
            * vgsol
            * w2i
            * w1i
            * w1r
            * w3i ** 2
            - 4 * rhogeq ** 2 * k * vgsol * w2r * w2i ** 2 * w1r * w3i ** 3 * w1i
            - w3r * rhogeq ** 2 * cs ** 2 * k ** 3 * w1r * vgsol * w2i ** 4
            - 2
            * w3r
            * rhogeq ** 2
            * cs ** 2
            * k ** 3
            * w1r
            * vgsol
            * w2i ** 2
            * w2r ** 2
            - 2 * w3r ** 3 * Kdrag * rhogsol * w2r ** 2 * w1i * w2i ** 2
            - rhogeq * w3i ** 3 * w2i * w1r * rhogsol * w2r ** 2 * w1i ** 2
            + 2 * w1r ** 2 * rhogsol * k ** 4 * cs ** 4 * rhogeq * w2r * w2i * w3i
            - rhogeq * w3i ** 3 * k * Kdrag * vdsol * w2r ** 2 * w1i ** 2
            + w1i ** 2 * rhogeq ** 2 * k * vgsol * w2r * w2i ** 2 * w1r * w3i ** 2
            + w1i ** 3 * w2i * rhogeq ** 2 * k * vgsol * w2r ** 2 * w3i ** 2
            - 2 * w2r * rhogeq * w1i ** 3 * rhogsol * k ** 2 * cs ** 2 * w3i ** 2 * w2i
            + w1i ** 2 * rhogeq ** 2 * w2r ** 3 * k * w1r * vgsol * w3i ** 2
            - 2
            * w2r
            * cs ** 2
            * k ** 2
            * rhogeq
            * rhogsol
            * w1r ** 2
            * w3i ** 2
            * w1i
            * w2i
            - 2 * Kdrag * k * rhogeq * vgsol * w2r ** 3 * w1i * w1r * w3i ** 2
            - 2 * w2r * Kdrag * rhogeq * k * vgsol * w2i ** 2 * w1i * w1r * w3i ** 2
            + w1r ** 3 * rhogeq ** 2 * k * vgsol * w2r * w2i ** 2 * w3i ** 2
            - 2 * w1r ** 3 * w2r * rhogeq ** 2 * cs ** 2 * k ** 3 * vgsol * w2i * w3i
            - 2
            * w3i ** 2
            * cs ** 2
            * k ** 3
            * rhogeq ** 2
            * vgsol
            * w2r ** 2
            * w2i ** 2
            - w1r ** 3 * w2r * rhogeq ** 2 * cs ** 2 * k ** 3 * vgsol * w3i ** 2
            - w3i * rhogsol * k ** 4 * cs ** 4 * rhogeq * w2r * w1i * w2i ** 2
            - w3i * rhogsol * k ** 4 * cs ** 4 * rhogeq * w2r ** 3 * w1i
            - w2i * rhogeq * k * Kdrag * vdsol * w2r ** 2 * w1r ** 2 * w3i ** 2
            + w1r ** 3 * rhogeq ** 2 * w2r ** 3 * k * vgsol * w3i ** 2
            - w3i ** 2 * rhogeq * k * Kdrag * vdsol * w2r ** 2 * w1i ** 2 * w2i
            - w3i ** 2 * cs ** 2 * k ** 3 * rhogeq ** 2 * vgsol * w2r ** 4
            + 2
            * rhogeq ** 2
            * cs ** 2
            * k ** 3
            * w3i
            * w1i
            * vgsol
            * w2r ** 2
            * w2i ** 2
            - rhogeq * cs ** 2 * k ** 3 * w3i * Kdrag * vgsol * w2r ** 3 * w1r
            - 2 * w3r * w3i ** 2 * Kdrag * rhogsol * w2r ** 2 * w1i * w2i ** 2
            - rhogeq * cs ** 2 * k ** 3 * w3i * Kdrag * vgsol * w2r * w1r * w2i ** 2
            + rhogeq ** 2 * cs ** 2 * k ** 3 * w3i * w1i * vgsol * w2r ** 4
        )

        rhod1r = rhod1r + (
            2
            * w2r
            * w3r ** 2
            * cs ** 2
            * k ** 2
            * rhogeq
            * rhogsol
            * w1r ** 2
            * w2i ** 2
            + 2 * w2r ** 3 * w3r ** 2 * cs ** 2 * k ** 2 * rhogeq * rhogsol * w1r ** 2
            - 2
            * rhogeq
            * w2r
            * w3i ** 2
            * cs ** 2
            * k ** 2
            * rhogsol
            * w1r ** 2
            * w2i ** 2
            - 2 * rhogeq * w2r ** 3 * w3i ** 2 * cs ** 2 * k ** 2 * rhogsol * w1r ** 2
            + rhogeq * cs ** 2 * k ** 3 * w3i * Kdrag * vdsol * w2r * w1r * w2i ** 2
            + rhogeq * cs ** 2 * k ** 3 * w3i * Kdrag * vdsol * w2r ** 3 * w1r
            + w2r * w3r * rhogeq * cs ** 4 * k ** 4 * w1r * rhogsol * w2i ** 2
            + w2r ** 3 * w3r * rhogeq * cs ** 4 * k ** 4 * w1r * rhogsol
            + w2r * w3r * rhogeq ** 2 * cs ** 2 * k ** 3 * w1r ** 2 * vgsol * w2i ** 2
            + w2r ** 3 * w3r * rhogeq ** 2 * cs ** 2 * k ** 3 * w1r ** 2 * vgsol
            + w2r * w3r * rhogeq * cs ** 2 * k ** 3 * w1i * Kdrag * vdsol * w2i ** 2
            + w2r ** 3 * w3r * rhogeq * cs ** 2 * k ** 3 * w1i * Kdrag * vdsol
            - w2r ** 3 * w3r * rhogeq * cs ** 2 * k ** 3 * w1i * Kdrag * vgsol
            - w2r * w3r * rhogeq ** 2 * cs ** 2 * k ** 3 * w1i ** 2 * vgsol * w2i ** 2
            - w2r ** 3 * w3r * rhogeq ** 2 * cs ** 2 * k ** 3 * w1i ** 2 * vgsol
            - w2r * w3r * rhogeq * cs ** 2 * k ** 3 * w1i * Kdrag * vgsol * w2i ** 2
            + 4 * w3r * w3i * cs ** 2 * k ** 2 * rhogeq * rhogsol * w2i ** 3 * w1i ** 2
            + 4
            * w3r
            * w3i
            * cs ** 2
            * k ** 2
            * rhogeq
            * rhogsol
            * w2i
            * w1i ** 2
            * w2r ** 2
            - w3r * rhogeq ** 2 * cs ** 2 * k ** 3 * w2r ** 4 * w1r * vgsol
            + 2
            * w3r ** 2
            * rhogeq ** 2
            * cs ** 2
            * k ** 3
            * w2r ** 2
            * vgsol
            * w2i ** 2
            + w3r ** 2 * rhogeq ** 2 * cs ** 2 * k ** 3 * w2r ** 4 * vgsol
            + w3r * rhogeq * cs ** 2 * k ** 2 * w2r ** 4 * w1r ** 2 * rhogsol
            + 2
            * w3r
            * rhogeq
            * cs ** 2
            * k ** 2
            * w1i ** 2
            * w2r ** 2
            * rhogsol
            * w2i ** 2
            - 4 * w3r * w3i * cs ** 4 * k ** 4 * rhogeq * rhogsol * w1i * w2i ** 2
            + w3r * rhogeq * cs ** 2 * k ** 2 * w1i ** 2 * w2r ** 4 * rhogsol
            + 2
            * w3r
            * rhogeq
            * cs ** 2
            * k ** 2
            * w2r ** 2
            * w1r ** 2
            * rhogsol
            * w2i ** 2
            - 2 * w3r * w3i * cs ** 2 * k ** 2 * rhogeq * rhogsol * w2r ** 4 * w1i
            - 2 * w3r * w3i * cs ** 2 * k ** 2 * rhogeq * rhogsol * w1i * w2i ** 4
            - 4
            * w3r
            * w3i
            * cs ** 2
            * k ** 2
            * rhogeq
            * rhogsol
            * w1i
            * w2i ** 2
            * w2r ** 2
            - w3i ** 3 * Kdrag * rhogeq * k * vgsol * w2i ** 4
            + 2 * w3i ** 3 * Kdrag * rhogsol * w2r ** 2 * w1r * w2i ** 2
            + w3i * cs ** 2 * k ** 3 * rhogeq * Kdrag * vgsol * w2i ** 3 * w1i
            + w3i * cs ** 2 * k ** 3 * rhogeq * Kdrag * vgsol * w2i * w1i * w2r ** 2
            - w3r * rhogeq * cs ** 2 * k ** 3 * w2i * Kdrag * vgsol * w1r * w2r ** 2
            - w3r * rhogeq * cs ** 4 * k ** 4 * rhogsol * w2i ** 3 * w1i
            + w3r * rhogeq * cs ** 2 * k ** 3 * w2i ** 3 * Kdrag * vdsol * w1r
            + w3r * rhogeq * cs ** 2 * k ** 3 * w2i * Kdrag * vdsol * w1r * w2r ** 2
            - w3r * rhogeq * cs ** 2 * k ** 3 * w2i ** 3 * Kdrag * vgsol * w1r
            + w3r * rhogeq * cs ** 2 * k ** 2 * w1i ** 2 * rhogsol * w2i ** 4
            + w3r ** 2 * rhogeq ** 2 * cs ** 2 * k ** 3 * vgsol * w2i ** 4
            - w3i * cs ** 4 * k ** 4 * rhogeq * rhogsol * w2i ** 3 * w1r
            - w3i * cs ** 4 * k ** 4 * rhogeq * rhogsol * w2i * w1r * w2r ** 2
            - w3r * rhogeq * cs ** 4 * k ** 4 * rhogsol * w2i * w1i * w2r ** 2
            + w3r * rhogeq * cs ** 2 * k ** 2 * w1r ** 2 * rhogsol * w2i ** 4
            + w3i * cs ** 2 * k ** 3 * rhogeq ** 2 * vgsol * w2i ** 4 * w1i
            + w3i * cs ** 2 * k ** 3 * rhogeq ** 2 * vgsol * w2i ** 3 * w1r ** 2
            + w3i * cs ** 2 * k ** 3 * rhogeq ** 2 * vgsol * w2i * w1r ** 2 * w2r ** 2
            - w3i * cs ** 2 * k ** 3 * rhogeq ** 2 * vgsol * w2i ** 3 * w1i ** 2
            - w3i * cs ** 2 * k ** 3 * rhogeq ** 2 * vgsol * w2i * w1i ** 2 * w2r ** 2
            - w3i ** 2 * rhogeq ** 2 * cs ** 2 * k ** 3 * vgsol * w2i ** 4
            + Kdrag * w2i ** 4 * vgsol * k * rhogeq * w1i * w3r ** 2
            - w3i * cs ** 2 * k ** 3 * rhogeq * Kdrag * vdsol * w2i ** 3 * w1i
            - w3i * cs ** 2 * k ** 3 * rhogeq * Kdrag * vdsol * w2i * w1i * w2r ** 2
            - 2 * w3i ** 3 * cs ** 2 * k ** 3 * rhogeq ** 2 * vgsol * w2r * w1r * w2i
            - rhogeq * w3i ** 3 * w2i * w1r ** 3 * rhogsol * w2r ** 2
            + 2 * w3i * w3r ** 2 * Kdrag * rhogsol * w2r ** 2 * w1r * w2i ** 2
            - w3i * w3r ** 2 * Kdrag * rhogeq * k * vgsol * w2i ** 4
            + Kdrag * w2i ** 4 * k * rhogeq * vgsol * w3i ** 2 * w1i
            + 2 * w1r ** 2 * rhogeq * cs ** 2 * k ** 3 * w3i * Kdrag * vgsol * w2r ** 2
            - 2 * w1r ** 2 * rhogeq * cs ** 2 * k ** 3 * w3i * Kdrag * vdsol * w2r ** 2
            + w1r ** 4 * w2r * cs ** 2 * k ** 2 * rhogeq * rhogsol * w3i ** 2
            + w3r * rhogeq ** 2 * cs ** 2 * k ** 3 * w2r ** 3 * vgsol * w3i ** 2
            + 2 * w2i * w3r ** 2 * rhogeq ** 2 * k * vgsol * w2r ** 2 * w1i * w3i ** 2
            - w2i * w3r ** 4 * Kdrag * rhogeq * k * vgsol * w2r ** 2
            + w2i * w3r ** 4 * rhogeq ** 2 * k * vgsol * w2r ** 2 * w1i
            - 2 * w2i * w3r ** 4 * cs ** 2 * k ** 2 * rhogeq * rhogsol * w2r * w1i
            - w3r * Kdrag * rhogsol * w2i ** 3 * w1r ** 2 * w3i ** 2
            - Kdrag * w3r ** 3 * w2r ** 4 * rhogsol * w1i
            + w3r * Kdrag * rhogsol * w2i ** 3 * w1i ** 2 * w3i ** 2
            + 2 * Kdrag * k * rhogeq * vgsol * w2i ** 3 * w1r ** 2 * w3i ** 2
            - 2
            * w1i ** 2
            * w2r
            * rhogeq ** 2
            * cs ** 2
            * k ** 3
            * w1r
            * vgsol
            * w2i
            * w3i
            - w1i ** 2 * w2r * rhogeq ** 2 * cs ** 2 * k ** 3 * w1r * vgsol * w3i ** 2
            - w3r ** 3 * w2r ** 3 * k * Kdrag ** 2 * vdsol
            + 2 * Kdrag * k * rhogeq * vgsol * w2i ** 3 * w1r ** 2 * w3r ** 2
            + w1r ** 2 * w2i * rhogeq ** 2 * k * vgsol * w2r ** 2 * w1i * w3i ** 2
            + k * Kdrag ** 2 * vgsol * w2i ** 3 * w1i * w3i ** 2
        )

        rhod1r = rhod1r + (
            k * Kdrag ** 2 * vgsol * w2i ** 3 * w1i * w3r ** 2
            + w3r ** 3 * w2r ** 3 * k * Kdrag ** 2 * vgsol
            + 2
            * w1i ** 2
            * w2r
            * cs ** 2
            * k ** 2
            * rhogeq
            * rhogsol
            * w1r ** 2
            * w3i ** 2
            - w3r ** 3 * rhogeq * w2r ** 4 * w1r ** 2 * rhogsol
            - rhogeq * w1i ** 2 * w3r ** 3 * w2r ** 4 * rhogsol
            - w3r ** 4 * rhogeq * w2r ** 3 * w1r ** 2 * rhogsol
            - k * Kdrag ** 2 * vdsol * w2i ** 3 * w1i * w3i ** 2
            - w3r ** 4 * Kdrag * rhogsol * w2r ** 3 * w1i
            - w3r ** 3 * Kdrag * rhogsol * w2i ** 3 * w1r ** 2
            + w3r ** 3 * Kdrag * rhogsol * w2i ** 3 * w1i ** 2
            - k * Kdrag ** 2 * vdsol * w2i ** 3 * w1i * w3r ** 2
            - rhogeq ** 2 * w3i ** 2 * k * w1r ** 4 * vgsol * w2r ** 2
            + Kdrag * w2r ** 3 * rhogsol * w1i ** 2 * w3i ** 3
            + 2 * w3r ** 2 * rhogeq ** 2 * w2r ** 3 * k * w1r * vgsol * w3i ** 2
            + w2i * w3r ** 3 * Kdrag * rhogsol * w2r ** 2 * w1i ** 2
            + w3r ** 4 * rhogeq * w2r ** 2 * w1i * k * Kdrag * vgsol
            - w2i * w3r ** 3 * Kdrag * rhogsol * w2r ** 2 * w1r ** 2
            + w3r ** 4 * rhogeq ** 2 * w2r ** 3 * k * w1r * vgsol
            - 2
            * w3r
            * rhogeq ** 2
            * cs ** 2
            * k ** 3
            * w2r ** 2
            * w1r
            * vgsol
            * w3i ** 2
            - 2 * w3r ** 2 * Kdrag * rhogsol * w2r ** 3 * w1i * w3i ** 2
            + 2 * w2r * rhogeq ** 2 * cs ** 2 * k ** 3 * w1r * vgsol * w3i ** 3 * w1i
            + w3r ** 4 * rhogeq ** 2 * cs ** 2 * k ** 3 * w2r ** 2 * vgsol
            + 2
            * w3r
            * rhogeq
            * cs ** 2
            * k ** 2
            * w2r ** 2
            * w1r ** 2
            * rhogsol
            * w3i ** 2
            + 2
            * w3r ** 2
            * rhogeq ** 2
            * cs ** 2
            * k ** 3
            * w2r ** 2
            * vgsol
            * w3i ** 2
            + rhogeq ** 2 * cs ** 2 * k ** 3 * w2r ** 2 * vgsol * w1r ** 2 * w3i * w1i
            - w1i ** 3 * w3i * rhogsol * k ** 4 * cs ** 4 * rhogeq * w2r
            - 2 * w2r * cs ** 2 * k ** 2 * rhogeq * rhogsol * w1r ** 2 * w3i ** 3 * w1i
            + 3 * w3r * Kdrag * k * rhogeq * vgsol * w2r ** 3 * w1i * w3i ** 2
            + 2 * rhogeq * w3i ** 3 * cs ** 2 * k ** 2 * rhogsol * w1r * w2r ** 2 * w1i
            + w3i ** 3 * Kdrag * rhogsol * w2r ** 4 * w1r
            - 2 * w3r ** 3 * rhogeq ** 2 * cs ** 2 * k ** 3 * w2r ** 2 * w1r * vgsol
            + 2 * w3r ** 3 * rhogeq * cs ** 2 * k ** 2 * w2r ** 2 * w1r ** 2 * rhogsol
            - rhogeq * w2r ** 3 * w3i ** 4 * w1r ** 2 * rhogsol
            - w3r ** 3 * Kdrag ** 2 * k * vgsol * w2r ** 2 * w1r
            + w3r ** 3 * Kdrag ** 2 * k * vdsol * w2r ** 2 * w1r
            + 3 * w3r ** 3 * Kdrag * k * rhogeq * vgsol * w2r ** 3 * w1i
            + 2 * rhogeq * cs ** 4 * k ** 4 * w2r * rhogsol * w2i * w1i ** 2 * w3i
            - w3r * Kdrag ** 2 * k * vgsol * w2r ** 2 * w1r * w3i ** 2
            + rhogeq * w2r ** 4 * w3i ** 4 * w1r * rhogsol
            - rhogeq ** 2 * w2r ** 4 * w3i ** 4 * k * vgsol
            + w3r ** 4 * rhogeq * w2r ** 4 * w1r * rhogsol
            + w3r * Kdrag ** 2 * k * vdsol * w2r ** 2 * w1r * w3i ** 2
            - w3r * Kdrag * rhogsol * k ** 2 * cs ** 2 * w1i * w2r ** 2 * w3i ** 2
            - w3r ** 4 * rhogeq ** 2 * w2r ** 4 * vgsol * k
            + w3r ** 3 * rhogeq ** 2 * cs ** 2 * k ** 3 * w2r ** 3 * vgsol
            - w3r ** 3 * Kdrag * rhogsol * k ** 2 * cs ** 2 * w1i * w2r ** 2
            - 2 * w2i * w3r ** 2 * Kdrag * rhogeq * k * vgsol * w2r ** 2 * w3i ** 2
            + 2 * rhogeq * cs ** 4 * k ** 4 * w1r * rhogsol * w2r ** 2 * w3i * w1i
            - w3r ** 4 * rhogeq * w1i ** 2 * w2r ** 3 * rhogsol
            - 2 * w2i * w3r ** 3 * cs ** 2 * k ** 3 * rhogeq * Kdrag * vdsol * w2r
            + 4 * w3r ** 2 * rhogeq * rhogsol * w2r ** 2 * w2i ** 2 * w1r * w3i ** 2
            + w2i * w3r ** 3 * Kdrag * cs ** 2 * k ** 2 * rhogsol * w2r ** 2
            + w2i * w3r * Kdrag * cs ** 2 * k ** 2 * rhogsol * w2r ** 2 * w3i ** 2
            - 2 * rhogeq ** 2 * cs ** 2 * k ** 3 * w2r ** 3 * w1r * vgsol * w3i * w1i
            - Kdrag * w2r ** 3 * rhogsol * w3i ** 3 * w1r ** 2
            - w1r ** 3 * rhogeq * cs ** 2 * k ** 3 * w3i * Kdrag * vgsol * w2r
            - w3r ** 4 * rhogeq * rhogsol * w2r * w2i ** 2 * w1r ** 2
            + w2i * w3r * Kdrag * rhogsol * w2r ** 2 * w1i ** 2 * w3i ** 2
            + 2 * w3r ** 4 * rhogeq * rhogsol * w2r ** 2 * w2i ** 2 * w1r
            + 2 * w2i * w3r ** 3 * cs ** 2 * k ** 3 * rhogeq * Kdrag * vgsol * w2r
            - rhogeq * w1i ** 2 * w2r ** 3 * rhogsol * w3i ** 4
            - Kdrag * w2r ** 3 * rhogsol * w1i * w3i ** 4
            - w2i * w3r * Kdrag * rhogsol * w2r ** 2 * w1r ** 2 * w3i ** 2
            + w2i * w3r ** 3 * rhogeq * k * Kdrag * vdsol * w2r ** 2 * w1r
            + 2 * w3r ** 2 * rhogeq * w2r ** 2 * w1i * k * Kdrag * vgsol * w3i ** 2
            - 4 * w3r ** 2 * rhogeq ** 2 * k * vgsol * w2i ** 2 * w2r ** 2 * w3i ** 2
            + 2 * w3r ** 2 * rhogeq ** 2 * k * vgsol * w2r * w2i ** 2 * w1r * w3i ** 2
            - 2 * w3r * rhogeq * rhogsol * w2i ** 2 * w1r ** 2 * w2r ** 2 * w3i ** 2
            - w3r ** 4 * rhogeq * rhogsol * w2r * w2i ** 2 * w1i ** 2
            + w3i ** 3 * Kdrag * rhogsol * w2i ** 4 * w1r
            - 2 * w3r * rhogeq * rhogsol * w1i ** 2 * w2r ** 2 * w2i ** 2 * w3i ** 2
            - 2 * w3r ** 2 * rhogeq * rhogsol * w2r * w2i ** 2 * w1i ** 2 * w3i ** 2
            + w3r ** 4 * rhogeq * rhogsol * w2i ** 4 * w1r
            + w3i ** 3 * vdsol * k * Kdrag ** 2 * w2i ** 3
            + rhogeq * w1i * w3i ** 2 * k * Kdrag * vdsol * w2r ** 2 * w1r ** 2
            + w3r ** 3 * cs ** 2 * k ** 3 * rhogeq ** 2 * vgsol * w2i ** 2 * w2r
            - w3i ** 3 * vgsol * k * Kdrag ** 2 * w2i ** 3
            + w3r ** 4 * rhogeq ** 2 * k * vgsol * w2r * w2i ** 2 * w1r
            - w3r ** 3 * Kdrag * rhogsol * w2i ** 4 * w1i
            - 2 * w3r ** 2 * rhogeq * rhogsol * w2r * w2i ** 2 * w1r ** 2 * w3i ** 2
            - w3r ** 3 * rhogeq * k * Kdrag * vdsol * w2r * w2i ** 2 * w1i
            - w3r ** 3 * rhogeq * rhogsol * w2i ** 4 * w1i ** 2
            + 2 * w3i ** 2 * rhogsol * k ** 4 * cs ** 4 * rhogeq * w2r * w1i ** 2
            + 2 * w2i * w3r ** 2 * Kdrag * rhogsol * w2r ** 2 * w1r * w3i ** 2
            - w1r * Kdrag * rhogsol * k ** 2 * cs ** 2 * w2i ** 3 * w3i ** 2
            + 2 * w3r * rhogeq ** 2 * k * vgsol * w2r ** 2 * w2i ** 2 * w1r * w3i ** 2
            - w3r ** 3 * rhogeq * rhogsol * w2i ** 4 * w1r ** 2
            - w1r * Kdrag * rhogsol * k ** 2 * cs ** 2 * w2i ** 3 * w3r ** 2
            - 2 * w3r * rhogeq ** 2 * k * vgsol * w2r * w2i ** 2 * w1r ** 2 * w3i ** 2
            - w3i ** 4 * rhogeq ** 2 * vgsol * k * w2i ** 4
            + w3r ** 4 * Kdrag * rhogsol * w2i ** 3 * w1r
            - w3r ** 4 * rhogeq ** 2 * vgsol * k * w2i ** 4
            + w3i ** 4 * Kdrag * rhogsol * w2i ** 3 * w1r
            - w3r * rhogeq * cs ** 4 * k ** 4 * w2r ** 2 * rhogsol * w3i ** 2
        )

        rhod1r = rhod1r + (
            2 * w3r * rhogeq ** 2 * k * vgsol * w2r * w2i ** 2 * w1i ** 2 * w3i ** 2
            + w3i ** 4 * rhogeq * rhogsol * w2i ** 4 * w1r
            + 3 * w3r ** 3 * w2r * Kdrag * rhogeq * k * vgsol * w2i ** 2 * w1i
            + 2 * w3r ** 3 * rhogeq ** 2 * k * vgsol * w2r ** 2 * w2i ** 2 * w1r
            - 2 * w3r ** 3 * rhogeq ** 2 * k * vgsol * w2r * w2i ** 2 * w1r ** 2
            + 2 * w3r ** 3 * rhogeq ** 2 * k * vgsol * w2r * w2i ** 2 * w1i ** 2
            - 2 * w3r ** 4 * rhogeq ** 2 * k * vgsol * w2i ** 2 * w2r ** 2
            - 2 * w3r ** 3 * rhogeq * rhogsol * w2i ** 2 * w1r ** 2 * w2r ** 2
            + w3r * cs ** 2 * k ** 3 * rhogeq ** 2 * vgsol * w2i ** 2 * w2r * w3i ** 2
            - 2 * w3r ** 3 * rhogeq * rhogsol * w1i ** 2 * w2r ** 2 * w2i ** 2
            + w1r * w3r ** 4 * cs ** 2 * k ** 2 * rhogeq * rhogsol * w2i ** 2
            + w1r * w3i ** 4 * cs ** 2 * k ** 2 * rhogeq * rhogsol * w2i ** 2
            + 2 * rhogeq ** 2 * w1i ** 2 * k * vgsol * w3r * w2r ** 3 * w3i ** 2
            + w2i * w3r ** 4 * Kdrag * rhogsol * w2r ** 2 * w1r
            - 2 * w3r ** 2 * rhogeq * w2r ** 3 * w1r ** 2 * rhogsol * w3i ** 2
            - w3r ** 3 * rhogeq * cs ** 4 * k ** 4 * w2r ** 2 * rhogsol
            + w1r ** 3 * rhogeq * cs ** 2 * k ** 3 * w3i * Kdrag * vdsol * w2r
            - 2 * w3r ** 3 * rhogeq ** 2 * w1r ** 2 * k * vgsol * w2r ** 3
            + 2 * rhogeq ** 2 * w1i ** 2 * k * vgsol * w3r ** 3 * w2r ** 3
            - w2i * w3r ** 3 * Kdrag * k * rhogeq * vgsol * w2r ** 2 * w1r
            - rhogeq * w1i * w3r ** 3 * w2r ** 3 * k * Kdrag * vdsol
            - 2 * w3r * rhogeq ** 2 * w1r ** 2 * k * vgsol * w2r ** 3 * w3i ** 2
            + w2r * w3r ** 3 * rhogeq ** 2 * cs ** 2 * k ** 3 * w1r ** 2 * vgsol
            - w2r * w3r ** 3 * rhogeq ** 2 * cs ** 2 * k ** 3 * w1i ** 2 * vgsol
            - w1i ** 3 * rhogeq * w2r ** 2 * k * Kdrag * vgsol * w3i ** 2
            + 2
            * w1r
            * w3i ** 2
            * cs ** 2
            * k ** 2
            * rhogeq
            * rhogsol
            * w2i ** 2
            * w3r ** 2
            + w2r * rhogeq * w1i ** 2 * rhogsol * k ** 2 * cs ** 2 * w3r ** 4
            - rhogeq * w3i ** 4 * cs ** 2 * k ** 2 * rhogsol * w1r * w2r ** 2
            - 2 * w2r * w3r ** 2 * Kdrag * rhogsol * w1i * w2i ** 2 * w3i ** 2
            - w2r * w3r ** 4 * rhogeq ** 2 * cs ** 2 * k ** 3 * w1r * vgsol
            - rhogeq * w1i * w3r * w2r ** 3 * k * Kdrag * vdsol * w3i ** 2
            + w2r * w3r ** 3 * rhogeq * cs ** 4 * k ** 4 * w1r * rhogsol
            - Kdrag * w3r * w2r ** 4 * rhogsol * w1i * w3i ** 2
            - w3r * w2r ** 3 * k * Kdrag ** 2 * vdsol * w3i ** 2
            + w3r * w2r ** 3 * k * Kdrag ** 2 * vgsol * w3i ** 2
            + rhogeq * rhogsol * w2r * w2i ** 2 * w1i ** 3 * w3i ** 3
            - 4 * rhogeq ** 2 * w2r ** 3 * k * w1r * vgsol * w3i ** 3 * w1i
            + w2r ** 2 * k * Kdrag ** 2 * vdsol * w3i ** 2 * w1i ** 2
            - w3r * rhogeq * w2r ** 4 * w1r ** 2 * rhogsol * w3i ** 2
            - rhogeq * w1i ** 2 * w3r * w2r ** 4 * rhogsol * w3i ** 2
            + w3r * rhogeq ** 2 * w2r ** 4 * k * w1r * vgsol * w3i ** 2
            - w2r ** 2 * k * Kdrag ** 2 * vgsol * w3i ** 2 * w1i ** 2
            - Kdrag * rhogsol * k ** 2 * cs ** 2 * w1i * w2r ** 3 * w3r ** 2
            + w2r * w3r ** 3 * rhogeq * cs ** 2 * k ** 3 * w1i * Kdrag * vdsol
            - w2r * w3r ** 3 * rhogeq * cs ** 2 * k ** 3 * w1i * Kdrag * vgsol
            + w2r * w3r ** 4 * cs ** 2 * k ** 2 * rhogeq * rhogsol * w1r ** 2
            + 2
            * w2r
            * rhogeq
            * w1i ** 2
            * rhogsol
            * k ** 2
            * cs ** 2
            * w3r ** 2
            * w3i ** 2
            - w3r ** 4 * rhogeq * rhogsol * k ** 2 * cs ** 2 * w2r ** 2 * w1r
            - rhogeq ** 2 * w1i ** 4 * k * vgsol * w3i ** 2 * w2r ** 2
            - Kdrag * rhogsol * k ** 2 * cs ** 2 * w1i * w2r ** 3 * w3i ** 2
            + w2r * w3r * rhogeq ** 2 * cs ** 2 * k ** 3 * w1r ** 2 * vgsol * w3i ** 2
            - w2r * w3r * rhogeq ** 2 * cs ** 2 * k ** 3 * w1i ** 2 * vgsol * w3i ** 2
            - w2r * w3r ** 4 * Kdrag * rhogsol * w1i * w2i ** 2
            + rhogeq * w2r * w3i ** 3 * w1r * k * Kdrag * vdsol * w2i ** 2
            - 2
            * w2r
            * w3r ** 2
            * rhogeq ** 2
            * cs ** 2
            * k ** 3
            * w1r
            * vgsol
            * w3i ** 2
            + rhogeq ** 2 * w3i ** 3 * w1i ** 3 * k * vgsol * w2r ** 2
            + rhogeq * w2r * w3i ** 4 * cs ** 2 * k ** 2 * rhogsol * w1r ** 2
            + w2r * w3r * rhogeq * cs ** 4 * k ** 4 * w1r * rhogsol * w3i ** 2
            - 2 * w2r * rhogeq * w1i ** 3 * rhogsol * k ** 2 * cs ** 2 * w3i ** 3
            - rhogeq * cs ** 2 * k ** 3 * w3i ** 3 * Kdrag * vdsol * w2r ** 2
            + rhogeq * cs ** 4 * k ** 4 * w2r * rhogsol * w2i ** 2 * w3i ** 2
            + w3r ** 3 * rhogeq ** 2 * w2r ** 4 * k * w1r * vgsol
            - w1r ** 2 * rhogeq * w2r ** 2 * w1i * k * Kdrag * vgsol * w3i ** 2
            - rhogeq * cs ** 4 * k ** 4 * w2r * rhogsol * w2i ** 2 * w3r ** 2
            + 2
            * rhogeq ** 2
            * cs ** 2
            * k ** 3
            * w2r ** 2
            * vgsol
            * w1r ** 2
            * w3i ** 2
            + rhogeq * cs ** 2 * k ** 3 * w3i ** 3 * Kdrag * vdsol * w2r * w1r
            + 2 * rhogeq ** 2 * w3i ** 3 * w2r ** 2 * w1i * k * vgsol * w2i ** 2
            - rhogeq * cs ** 2 * k ** 3 * w3i * Kdrag * vdsol * w2r ** 2 * w3r ** 2
            - w1r ** 2 * w3i * rhogsol * k ** 4 * cs ** 4 * rhogeq * w2r * w1i
            - 2
            * rhogeq ** 2
            * cs ** 2
            * k ** 3
            * w1i ** 2
            * vgsol
            * w2r ** 2
            * w3i ** 2
            + rhogeq * cs ** 4 * k ** 4 * w2r ** 3 * rhogsol * w3i ** 2
            - rhogeq * cs ** 4 * k ** 4 * w2r ** 3 * rhogsol * w3r ** 2
            + 2
            * w2r
            * w3r ** 2
            * cs ** 2
            * k ** 2
            * rhogeq
            * rhogsol
            * w1r ** 2
            * w3i ** 2
            + 2 * w3r ** 2 * rhogeq * w2r ** 4 * w1r * rhogsol * w3i ** 2
            - 2 * w3r ** 2 * rhogeq ** 2 * w2r ** 4 * vgsol * k * w3i ** 2
            - 2
            * rhogeq
            * w3i ** 2
            * cs ** 2
            * k ** 2
            * rhogsol
            * w1r
            * w2r ** 2
            * w3r ** 2
            + rhogeq * w2i * w3i ** 3 * w1i * k * Kdrag * vdsol * w2r ** 2
            + rhogeq * cs ** 2 * k ** 3 * w3i ** 3 * Kdrag * vgsol * w2r ** 2
            + rhogeq * cs ** 2 * k ** 2 * w2r ** 4 * w1r * rhogsol * w3i ** 2
            - rhogeq * cs ** 2 * k ** 2 * w2r ** 4 * w1r * rhogsol * w3r ** 2
            + 2
            * rhogeq ** 2
            * cs ** 2
            * k ** 3
            * w2r ** 2
            * vgsol
            * w2i
            * w1i
            * w3i ** 2
            - 2 * w2i * w3r * cs ** 2 * k ** 3 * rhogeq * Kdrag * vdsol * w2r * w3i ** 2
            + rhogeq * cs ** 2 * k ** 3 * w3i * Kdrag * vgsol * w2r ** 2 * w3r ** 2
            - rhogeq * cs ** 2 * k ** 3 * w3i ** 3 * Kdrag * vgsol * w2r * w1r
            + w2r * w3r ** 3 * w2i ** 2 * k * Kdrag ** 2 * vgsol
        )

        rhod1r = rhod1r + (
            rhogeq ** 2 * w3i * w2r ** 4 * w1i * k * vgsol * w3r ** 2
            + rhogeq ** 2 * w2r ** 3 * w3i ** 4 * k * w1r * vgsol
            + 2 * rhogeq * w2r ** 2 * w3i ** 4 * w1r * rhogsol * w2i ** 2
            + rhogeq ** 2 * w3i ** 3 * w2r ** 4 * w1i * k * vgsol
            - 2 * rhogeq * w2r * w3i ** 4 * cs ** 2 * k ** 2 * rhogsol * w2i * w1i
            - w2r * w3r ** 3 * w2i ** 2 * k * Kdrag ** 2 * vdsol
            - rhogeq * cs ** 2 * k ** 3 * w1i * Kdrag * vgsol * w2r ** 2 * w3r ** 2
            - rhogeq * cs ** 2 * k ** 3 * w2i * Kdrag * vgsol * w2r ** 2 * w3i ** 2
            + rhogeq * cs ** 2 * k ** 3 * w2i * Kdrag * vgsol * w2r ** 2 * w3r ** 2
            + rhogeq * cs ** 2 * k ** 3 * w1i * Kdrag * vdsol * w2r ** 2 * w3i ** 2
            + w2r * w3r * w2i ** 2 * k * Kdrag ** 2 * vgsol * w3i ** 2
            - rhogeq * cs ** 4 * k ** 4 * w1r * rhogsol * w2r ** 2 * w3i ** 2
            + 3 * rhogeq * cs ** 4 * k ** 4 * w1r * rhogsol * w2r ** 2 * w3r ** 2
            + rhogeq * cs ** 2 * k ** 3 * w1i * Kdrag * vdsol * w2r ** 2 * w3r ** 2
            - rhogeq * w2r * w3i ** 4 * w1r ** 2 * rhogsol * w2i ** 2
            - 4 * rhogeq * cs ** 4 * k ** 4 * w2r * rhogsol * w2i * w1i * w3i ** 2
            - 2 * rhogeq ** 2 * cs ** 2 * k ** 3 * w2r ** 3 * w1r * vgsol * w3r ** 2
            + rhogeq * cs ** 2 * k ** 3 * w2i * Kdrag * vdsol * w2r ** 2 * w3i ** 2
            + Kdrag * k * rhogeq * vgsol * w2r ** 4 * w1i * w3i ** 2
            + Kdrag * k * rhogeq * vgsol * w2r ** 4 * w1i * w3r ** 2
            - rhogeq ** 2 * cs ** 2 * k ** 3 * w3i ** 3 * w2i * vgsol * w2r ** 2
            + w1i ** 3 * w3r ** 3 * rhogeq * rhogsol * w2i ** 3
            + 2 * rhogeq ** 2 * w2i * k * vgsol * w3i ** 3 * w1r ** 2 * w2r ** 2
            - w2r * w3r * w2i ** 2 * k * Kdrag ** 2 * vdsol * w3i ** 2
            - rhogeq * cs ** 2 * k ** 3 * w2i * Kdrag * vdsol * w2r ** 2 * w3r ** 2
            - rhogeq * cs ** 2 * k ** 3 * w1i * Kdrag * vgsol * w2r ** 2 * w3i ** 2
            - w3i ** 4 * cs ** 2 * k ** 3 * rhogeq ** 2 * vgsol * w2r * w1r
            + 2
            * rhogeq
            * cs ** 2
            * k ** 2
            * w2r ** 2
            * w1r
            * rhogsol
            * w2i ** 2
            * w3i ** 2
            - 2
            * rhogeq
            * cs ** 2
            * k ** 2
            * w2r ** 2
            * w1r
            * rhogsol
            * w2i ** 2
            * w3r ** 2
            + 2 * rhogeq ** 2 * w3i * w2r ** 2 * w1i * k * vgsol * w2i ** 2 * w3r ** 2
            - w3i ** 3 * vgsol * k * Kdrag ** 2 * w2i * w2r ** 2
            + w3i ** 3 * rhogsol * k ** 2 * cs ** 2 * Kdrag * w2r * w2i ** 2
            + w3i * rhogsol * k ** 2 * cs ** 2 * Kdrag * w2r ** 3 * w3r ** 2
            + w3i ** 3 * rhogsol * k ** 2 * cs ** 2 * Kdrag * w2r ** 3
            + w3i ** 4 * cs ** 2 * k ** 3 * rhogeq ** 2 * vgsol * w2r ** 2
            - rhogeq ** 2 * cs ** 2 * k ** 3 * w3i * w2i * vgsol * w2r ** 2 * w3r ** 2
            + 2 * rhogeq ** 2 * w2i * k * vgsol * w3i * w1r ** 2 * w2r ** 2 * w3r ** 2
            - w3i * vgsol * k * Kdrag ** 2 * w2i * w2r ** 2 * w3r ** 2
            + w3i * Kdrag * rhogsol * w2r ** 4 * w1r * w3r ** 2
            - 2 * w3r * rhogeq * cs ** 2 * k ** 2 * w1i ** 3 * w2r ** 2 * rhogsol * w3i
            + 2 * w3r * rhogeq ** 2 * cs ** 2 * k ** 3 * w2r ** 3 * vgsol * w3i * w1i
            - 2 * w3r * w1r ** 2 * w2i * cs ** 2 * k ** 3 * rhogeq * Kdrag * vdsol * w2r
            - 2 * w3r * w1i ** 3 * w2i * rhogeq * cs ** 2 * k ** 2 * rhogsol * w2r ** 2
            + 2 * w3r * w1r ** 2 * w2i * cs ** 2 * k ** 3 * rhogeq * Kdrag * vgsol * w2r
            + 2
            * w3r
            * w1i ** 2
            * rhogeq
            * cs ** 2
            * k ** 2
            * w2r ** 2
            * w1r ** 2
            * rhogsol
            + w3r * rhogeq * w1i ** 3 * w3i ** 2 * rhogsol * w2i * w2r ** 2
            + w3i * rhogsol * k ** 2 * cs ** 2 * Kdrag * w2r * w2i ** 2 * w3r ** 2
            + 2 * w3r * w2r * w1i ** 3 * rhogeq ** 2 * cs ** 2 * k ** 3 * vgsol * w2i
            + 2 * w3r * w2r * rhogeq ** 2 * cs ** 2 * k ** 3 * w1i ** 3 * vgsol * w3i
            + w3r * w1i ** 4 * rhogeq * cs ** 2 * k ** 2 * w2r ** 2 * rhogsol
            + 2
            * w3r
            * w2r
            * w1r ** 2
            * rhogeq ** 2
            * cs ** 2
            * k ** 3
            * vgsol
            * w2i
            * w1i
            + w3r * w1i ** 2 * rhogeq ** 2 * cs ** 2 * k ** 3 * w2r ** 2 * w1r * vgsol
            + w3r * rhogeq * w2r ** 3 * w1r ** 3 * rhogsol * w3i ** 2
            - 2 * w3r * w2r * rhogeq * cs ** 2 * k ** 3 * w1i ** 2 * Kdrag * vdsol * w3i
            + 2 * w3r * w2r * rhogeq * cs ** 2 * k ** 3 * w1i ** 2 * Kdrag * vgsol * w3i
            - 2 * w3r * w2r * cs ** 2 * k ** 2 * rhogeq * rhogsol * w1r ** 3 * w3i ** 2
            + 2 * w3r * rhogeq * cs ** 2 * k ** 3 * w1i * Kdrag * vgsol * w2r ** 2 * w1r
            - 2 * w3r * rhogeq * cs ** 2 * k ** 3 * w1i * Kdrag * vdsol * w2r ** 2 * w1r
            + 2
            * w3r
            * rhogeq ** 2
            * cs ** 2
            * k ** 3
            * w2r ** 2
            * vgsol
            * w2i
            * w1i
            * w1r
            - 2 * w3r * w3i * cs ** 2 * k ** 3 * rhogeq ** 2 * vgsol * w2i ** 3 * w1r
            + 4 * w3r * rhogeq * rhogsol * k ** 2 * cs ** 2 * w2r ** 3 * w1r * w3i * w1i
            + 2 * w3r * w2r * Kdrag * rhogsol * w1i * w2i ** 2 * w3i ** 2 * w1r
            - 2 * w3r * w1r ** 2 * rhogeq * cs ** 4 * k ** 4 * w2r ** 2 * rhogsol
            + w3r * rhogeq * rhogsol * w2r * w2i ** 2 * w1i ** 2 * w3i ** 2 * w1r
            - w3r * w1r ** 2 * w2r * rhogeq * cs ** 2 * k ** 3 * w1i * Kdrag * vgsol
            + w3r * w1r ** 2 * w2r * rhogeq * cs ** 2 * k ** 3 * w1i * Kdrag * vdsol
            + 2
            * w3r
            * w2r
            * rhogeq ** 2
            * cs ** 2
            * k ** 3
            * w1r ** 2
            * vgsol
            * w3i
            * w1i
            - 2
            * w3r
            * w2r
            * rhogeq
            * w1i ** 2
            * rhogsol
            * k ** 2
            * cs ** 2
            * w3i ** 2
            * w1r
            + w3r * rhogeq * rhogsol * w2r * w2i ** 2 * w1r ** 3 * w3i ** 2
            - 4 * w3r * rhogsol * k ** 4 * cs ** 4 * rhogeq * w2r * w2i * w3i * w1r
            + w3r * w1r ** 3 * rhogeq ** 2 * cs ** 2 * k ** 3 * w2r ** 2 * vgsol
            + w3r * rhogeq ** 2 * w3i ** 2 * k * w1r ** 3 * vgsol * w2r ** 2
            - 4
            * w3r
            * rhogeq ** 2
            * cs ** 2
            * k ** 3
            * w2r ** 2
            * w1r
            * vgsol
            * w3i
            * w1i
            - 2
            * w3r
            * rhogeq
            * cs ** 2
            * k ** 2
            * w2r ** 2
            * w1r ** 2
            * rhogsol
            * w3i
            * w1i
            + 2
            * w3r
            * cs ** 2
            * k ** 3
            * rhogeq ** 2
            * vgsol
            * w2i ** 2
            * w2r
            * w3i
            * w1i
            + 4
            * w3r
            * w2i
            * cs ** 2
            * k ** 2
            * rhogeq
            * rhogsol
            * w2r
            * w1i
            * w3i ** 2
            * w1r
            - 2
            * w3r
            * w1i ** 2
            * rhogeq
            * cs ** 2
            * k ** 2
            * rhogsol
            * w2r
            * w2i ** 2
            * w1r
            - 2 * w3r * w1i ** 2 * w2i * cs ** 2 * k ** 3 * rhogeq * Kdrag * vdsol * w2r
            + 2 * w3r * w1i ** 2 * w2i * cs ** 2 * k ** 3 * rhogeq * Kdrag * vgsol * w2r
            + w3r * rhogeq * w1i ** 2 * w2r ** 3 * rhogsol * w3i ** 2 * w1r
            - 2
            * w3r
            * rhogeq ** 2
            * cs ** 2
            * k ** 3
            * w3i
            * w2i
            * vgsol
            * w2r ** 2
            * w1r
            + w3r ** 3 * rhogeq * w2r ** 3 * w1r ** 3 * rhogsol
            - 2 * w3r ** 3 * rhogeq * w2r ** 2 * w1i * k * Kdrag * vgsol * w1r
            - w3i * rhogsol * k ** 4 * cs ** 4 * rhogeq * w2r * w1i * w3r ** 2
        )

        rhod1r = rhod1r + (
            w3i * vdsol * k * Kdrag ** 2 * w2i * w2r ** 2 * w3r ** 2
            + 2 * rhogeq * w2r ** 2 * w3i ** 3 * w1r ** 2 * k * Kdrag * vgsol
            - 4
            * w3r
            * rhogeq ** 2
            * cs ** 2
            * k ** 3
            * w2r
            * w1i ** 2
            * vgsol
            * w2i
            * w3i
            - w3r * w1r ** 4 * w2r * rhogeq ** 2 * cs ** 2 * k ** 3 * vgsol
            + 4
            * w3r
            * w2i ** 2
            * cs ** 2
            * k ** 2
            * rhogeq
            * rhogsol
            * w2r
            * w1i
            * w1r
            * w3i
            + w3r * w1r ** 3 * w2r * rhogeq * cs ** 4 * k ** 4 * rhogsol
            - w3i ** 3 * vgsol * k * rhogeq * Kdrag * w2r ** 4
            - 2 * w3r * rhogeq * w2r ** 2 * w1i * k * Kdrag * vgsol * w3i ** 2 * w1r
            + w3r * rhogeq ** 2 * w3i ** 2 * k * w1r * vgsol * w2r ** 2 * w1i ** 2
            + 4
            * w3r
            * rhogeq ** 2
            * cs ** 2
            * k ** 3
            * w2r
            * w1r ** 2
            * vgsol
            * w2i
            * w3i
            + 2
            * w3r
            * cs ** 2
            * k ** 3
            * rhogeq ** 2
            * w2r
            * vgsol
            * w2i
            * w1i
            * w3i ** 2
            - 2 * w3r * w2r ** 3 * rhogeq * w1i ** 2 * rhogsol * k ** 2 * cs ** 2 * w1r
            - 2 * w3r * w2r ** 3 * cs ** 2 * k ** 2 * rhogeq * rhogsol * w1r ** 3
            + rhogeq * w2r ** 3 * w1r ** 2 * rhogsol * w3i ** 3 * w1i
            - 2 * w3r * rhogeq * cs ** 2 * k ** 3 * w3i * Kdrag * vdsol * w2r * w1r ** 2
            - 4 * w3r * w2i * rhogeq ** 2 * k * vgsol * w2r ** 2 * w1i * w3i ** 2 * w1r
            - 2 * w3r * w1r ** 3 * rhogeq * cs ** 2 * k ** 2 * rhogsol * w2r * w2i ** 2
            - 2
            * w3r
            * w2i
            * w1i
            * rhogeq ** 2
            * cs ** 2
            * k ** 3
            * w1r
            * vgsol
            * w3i ** 2
            + w3i ** 3 * vdsol * k * Kdrag ** 2 * w2i * w2r ** 2
            + 2 * rhogeq * w2r ** 2 * w3i * w1r ** 2 * k * Kdrag * vgsol * w3r ** 2
            - 2 * w3r * w3i * cs ** 2 * k ** 3 * rhogeq ** 2 * vgsol * w2i * w1r ** 3
            - 2
            * w3r
            * w3i
            * cs ** 2
            * k ** 3
            * rhogeq ** 2
            * vgsol
            * w2i
            * w1i ** 2
            * w1r
            + 4
            * w3r
            * rhogeq ** 2
            * cs ** 2
            * k ** 3
            * w3i
            * w1i
            * vgsol
            * w2i ** 2
            * w1r
            + 2 * w3r * rhogeq * cs ** 2 * k ** 3 * w3i * Kdrag * vgsol * w2r * w1r ** 2
            + 2 * w3r * Kdrag * rhogsol * w2r ** 3 * w1i * w3i ** 2 * w1r
            - 2
            * w3r
            * w1i ** 2
            * w2r
            * rhogeq ** 2
            * cs ** 2
            * k ** 3
            * w1r ** 2
            * vgsol
            + w3r * w1i ** 2 * w2r * rhogeq * cs ** 4 * k ** 4 * w1r * rhogsol
            + w3r * w1i ** 3 * w2r * rhogeq * cs ** 2 * k ** 3 * Kdrag * vdsol
            - w3r * w1i ** 3 * w2r * rhogeq * cs ** 2 * k ** 3 * Kdrag * vgsol
            - w3r * w1i ** 4 * w2r * rhogeq ** 2 * cs ** 2 * k ** 3 * vgsol
            + w3r * rhogeq * w1i * w3i ** 2 * rhogsol * w2i * w2r ** 2 * w1r ** 2
            - 2
            * w3r
            * w1r ** 2
            * w2i
            * rhogeq
            * cs ** 2
            * k ** 2
            * rhogsol
            * w2r ** 2
            * w1i
            + Kdrag * w2i ** 2 * rhogsol * w1i ** 2 * w2r * w3i ** 3
            + Kdrag * w2i ** 2 * rhogsol * w1i ** 2 * w2r * w3i * w3r ** 2
            - Kdrag * w2i ** 2 * rhogsol * w3i * w2r * w1r ** 2 * w3r ** 2
            - Kdrag * w2i ** 2 * rhogsol * w3i ** 3 * w2r * w1r ** 2
            - Kdrag * w2r ** 2 * k * rhogeq * vgsol * w2i * w3i ** 4
            - w3i * vgsol * k * rhogeq * Kdrag * w2r ** 4 * w3r ** 2
            + 2 * w1r ** 3 * rhogeq * cs ** 2 * k ** 2 * rhogsol * w2i ** 2 * w3i ** 2
            + Kdrag * w2r ** 3 * rhogsol * w1i ** 2 * w3i * w3r ** 2
            - w1r ** 3 * w3i ** 3 * rhogeq * rhogsol * w2i ** 3
            - 4 * w3r ** 3 * w2i * rhogeq ** 2 * k * vgsol * w2r ** 2 * w1i * w1r
            - w3i ** 3 * rhogsol * k ** 4 * cs ** 4 * rhogeq * w2r * w1i
            + rhogeq * cs ** 2 * k ** 2 * w1i ** 2 * w2r * rhogsol * w3i ** 4
            - w2r ** 3 * k * Kdrag ** 2 * vgsol * w1r * w3r ** 2
            - 2 * w1i ** 2 * rhogeq * cs ** 2 * k ** 3 * w2i * Kdrag * vgsol * w3i ** 2
            - 2 * rhogeq ** 2 * w1i ** 2 * w2r ** 2 * k * vgsol * w2i * w3i ** 3
            - w2r ** 2 * k * Kdrag ** 2 * vdsol * w3i ** 3 * w1i
            + w2i ** 2 * k * Kdrag ** 2 * vdsol * w2r * w1r * w3i ** 2
            + w1i ** 4 * w2r * rhogeq * rhogsol * k ** 2 * cs ** 2 * w3i ** 2
            + w2r ** 2 * k * Kdrag ** 2 * vgsol * w3i ** 3 * w1i
            + w2i ** 2 * k * Kdrag ** 2 * vdsol * w2r * w1r * w3r ** 2
            - 2 * rhogeq ** 2 * w1i ** 2 * w2r ** 2 * k * vgsol * w2i * w3i * w3r ** 2
            + w2r ** 3 * k * Kdrag ** 2 * vdsol * w1r * w3i ** 2
            - w1r ** 2 * vdsol * Kdrag * k * rhogeq * w2i ** 3 * w3i ** 2
            - w1r ** 2 * vdsol * Kdrag * k * rhogeq * w2i ** 3 * w3r ** 2
            + w2r ** 3 * k * Kdrag ** 2 * vdsol * w1r * w3r ** 2
            - Kdrag * w2r ** 2 * rhogsol * k ** 2 * cs ** 2 * w2i * w1r * w3i ** 2
            - w1r ** 2 * Kdrag * k * rhogeq * vgsol * w1i * w2i ** 2 * w3i ** 2
            - w2i ** 2 * k * Kdrag ** 2 * vgsol * w2r * w1r * w3i ** 2
            - w2i ** 2 * k * Kdrag ** 2 * vgsol * w2r * w1r * w3r ** 2
            - w1r ** 2 * Kdrag * k * rhogeq * vgsol * w1i * w2i ** 2 * w3r ** 2
            - Kdrag * w2r ** 2 * rhogsol * k ** 2 * cs ** 2 * w2i * w1r * w3r ** 2
            + w2i ** 2 * k * Kdrag ** 2 * vgsol * w3i ** 3 * w1i
            + 2 * w3r * w3i * cs ** 4 * k ** 4 * rhogeq * rhogsol * w2i * w1r ** 2
            - w2i ** 2 * k * Kdrag ** 2 * vdsol * w3i ** 3 * w1i
            + w2i ** 2 * k * Kdrag ** 2 * vdsol * w3r ** 3 * w1r
            - 2 * w3r ** 3 * w2r * cs ** 2 * k ** 2 * rhogeq * rhogsol * w1r ** 3
            + 2 * w3r ** 3 * w2r * Kdrag * rhogsol * w1i * w2i ** 2 * w1r
            - w3i ** 4 * rhogeq ** 2 * cs ** 2 * k ** 3 * vgsol * w2i ** 2
            - Kdrag * w2i ** 2 * k * rhogeq * vgsol * w2r * w3i ** 3 * w1r
            + 2 * Kdrag * w2i * vgsol * k * rhogeq * w1r ** 2 * w2r ** 2 * w3i ** 2
            + w3r ** 3 * rhogeq * w2i * rhogsol * w1i * w2r ** 2 * w1r ** 2
            - 2 * w3r ** 3 * w2r * rhogeq * w1i ** 2 * rhogsol * k ** 2 * cs ** 2 * w1r
            + 2 * Kdrag * w2i * vgsol * k * rhogeq * w1r ** 2 * w2r ** 2 * w3r ** 2
            - Kdrag * w2r ** 3 * k * rhogeq * vgsol * w3i * w1r * w3r ** 2
            - 2 * rhogeq ** 2 * w2r ** 2 * w3i ** 4 * k * vgsol * w2i ** 2
            - Kdrag * w2i ** 2 * rhogsol * w1i * w2r * w3i ** 4
            - rhogeq * w1i ** 2 * w2i ** 2 * rhogsol * w2r * w3i ** 4
            - Kdrag * w2r ** 3 * rhogsol * w3i * w1r ** 2 * w3r ** 2
        )

        rhod1r = rhod1r + (
            Kdrag * w2r ** 2 * k * rhogeq * vgsol * w2i * w3i ** 3 * w1i
            - 2 * rhogeq * w1i ** 2 * w2r ** 3 * rhogsol * w3i ** 2 * w3r ** 2
            - w2r ** 2 * k * Kdrag ** 2 * vdsol * w3i * w1i * w3r ** 2
            + Kdrag * w2r ** 2 * rhogsol * w2i * w1r * w3i ** 4
            + w2r ** 2 * k * Kdrag ** 2 * vgsol * w3i * w1i * w3r ** 2
            - w2r ** 3 * k * Kdrag ** 2 * vgsol * w1r * w3i ** 2
            + 4 * w3i ** 3 * cs ** 2 * k ** 2 * rhogeq * rhogsol * w2i * w2r * w1i ** 2
            + Kdrag * k * rhogeq * vgsol * w2r ** 2 * w1i * w3i ** 4
            + 2 * w3r ** 3 * cs ** 2 * k ** 3 * rhogeq ** 2 * w2r * vgsol * w2i * w1i
            + w3r ** 3 * rhogeq * w2i * rhogsol * w1i ** 3 * w2r ** 2
            + 2 * w3r ** 3 * Kdrag * rhogsol * w2r ** 3 * w1i * w1r
            - Kdrag ** 2 * k * vdsol * w2i * w1i * w2r ** 2 * w3r ** 2
            + w3r ** 3 * rhogeq ** 2 * k * w1r ** 3 * vgsol * w2r ** 2
            + w3r * rhogeq * cs ** 2 * k ** 2 * w1r ** 4 * rhogsol * w2i ** 2
            + Kdrag ** 2 * k * vgsol * w2i * w1i * w2r ** 2 * w3i ** 2
            - 2
            * cs ** 2
            * k ** 3
            * rhogeq ** 2
            * vgsol
            * w2r
            * w1r
            * w2i ** 2
            * w3r ** 2
            + w3r * rhogeq * cs ** 2 * k ** 3 * w2i * Kdrag * vdsol * w1r ** 3
            + w3i * cs ** 2 * k ** 3 * rhogeq * Kdrag * vgsol * w2i * w1i * w1r ** 2
            - w3r * rhogeq * cs ** 2 * k ** 3 * w2i * Kdrag * vgsol * w1r ** 3
            + Kdrag ** 2 * k * vgsol * w2i * w1i * w2r ** 2 * w3r ** 2
            + w3r ** 3 * rhogeq ** 2 * k * w1r * vgsol * w2r ** 2 * w1i ** 2
            + rhogeq * w2r ** 3 * w3i ** 3 * w1r * k * Kdrag * vdsol
            + 2 * rhogsol * k ** 4 * cs ** 4 * rhogeq * w2r * w2i * w3i ** 3
            + 2 * rhogsol * k ** 4 * cs ** 4 * rhogeq * w2r * w2i * w3i * w3r ** 2
            - w2i ** 2 * k * Kdrag ** 2 * vgsol * w3r ** 3 * w1r
            - Kdrag * w2r ** 2 * rhogsol * k ** 2 * cs ** 2 * w3i * w1r * w3r ** 2
            + rhogeq ** 2 * w1i * w2r ** 2 * k * vgsol * w2i * w3i ** 4
            - Kdrag * w2r ** 2 * rhogsol * k ** 2 * cs ** 2 * w3i ** 3 * w1r
            - Kdrag * w2r ** 3 * k * rhogeq * vgsol * w3i ** 3 * w1r
            - w3i ** 4 * vgsol * k * rhogeq * Kdrag * w2i ** 3
            + w3i * vdsol * k * Kdrag ** 2 * w2i ** 3 * w3r ** 2
            - w3i * vgsol * k * Kdrag ** 2 * w2i ** 3 * w3r ** 2
            + 2 * w2i ** 3 * w1i * w3r * rhogeq ** 2 * cs ** 2 * k ** 3 * w1r * vgsol
            + w3r ** 4 * rhogeq ** 2 * vgsol * k * w2i ** 3 * w1i
            - w3r * Kdrag * rhogsol * w2i ** 4 * w1i * w3i ** 2
            - Kdrag * rhogsol * k ** 2 * cs ** 2 * w1i * w2r * w2i ** 2 * w3i ** 2
            - w2i ** 2 * k * Kdrag ** 2 * vgsol * w3r * w1r * w3i ** 2
            - Kdrag * rhogsol * k ** 2 * cs ** 2 * w3i ** 3 * w1r * w2i ** 2
            - Kdrag * rhogsol * k ** 2 * cs ** 2 * w1i * w2r * w2i ** 2 * w3r ** 2
            + rhogeq * w2r ** 3 * w3i * w1r * k * Kdrag * vdsol * w3r ** 2
            - w3r ** 4 * vgsol * k * rhogeq * Kdrag * w2i ** 3
            + w2i ** 2 * k * Kdrag ** 2 * vgsol * w3i * w1i * w3r ** 2
            + 2
            * w3r
            * rhogeq
            * cs ** 2
            * k ** 2
            * w1i ** 2
            * rhogsol
            * w2i ** 2
            * w1r ** 2
            - w2i ** 2 * k * Kdrag ** 2 * vdsol * w3i * w1i * w3r ** 2
            + w1i ** 3 * rhogeq ** 2 * cs ** 2 * k ** 3 * w3i * vgsol * w2r ** 2
            + w2i ** 2 * k * Kdrag ** 2 * vdsol * w3r * w1r * w3i ** 2
            - 2
            * w3i ** 2
            * rhogeq ** 2
            * cs ** 2
            * k ** 3
            * vgsol
            * w2i ** 2
            * w3r ** 2
            + 4 * w3r ** 3 * w2i * cs ** 2 * k ** 2 * rhogeq * rhogsol * w2r * w1i * w1r
            + Kdrag * w2i ** 2 * vgsol * k * rhogeq * w1i * w3r ** 4
            + w3r ** 3 * rhogeq * rhogsol * w2r * w2i ** 2 * w1i ** 2 * w1r
            - 2 * Kdrag * w2r ** 2 * k * rhogeq * vgsol * w2i ** 2 * w3i ** 3
            + w2r * rhogeq ** 2 * vgsol * k * w2i ** 2 * w1r * w3i ** 4
            - Kdrag ** 2 * k * vdsol * w2i * w1i * w2r ** 2 * w3i ** 2
            - 2 * w2i ** 3 * w1i ** 3 * w3r * rhogeq * cs ** 2 * k ** 2 * rhogsol
            + w3r ** 3 * vdsol * Kdrag * k * rhogeq * w2i ** 3 * w1r
            + 2 * cs ** 2 * k ** 3 * rhogeq ** 2 * vgsol * w2i ** 3 * w1i * w3i ** 2
            + 2 * w3i ** 2 * Kdrag * rhogsol * w2i ** 3 * w1r * w3r ** 2
            - Kdrag * rhogsol * k ** 2 * cs ** 2 * w3i * w1r * w2i ** 2 * w3r ** 2
            + 2 * Kdrag * w2i ** 2 * vgsol * k * rhogeq * w1i * w3r ** 2 * w3i ** 2
            - w3r * rhogeq * cs ** 4 * k ** 4 * rhogsol * w2i * w1i * w1r ** 2
            - w3i * cs ** 4 * k ** 4 * rhogeq * rhogsol * w2i * w1r ** 3
            + 2 * w3r ** 2 * rhogeq ** 2 * vgsol * k * w2i ** 3 * w1i * w3i ** 2
            - w3i * cs ** 2 * k ** 3 * rhogeq * Kdrag * vdsol * w2i * w1i * w1r ** 2
            + w3i * cs ** 2 * k ** 3 * rhogeq ** 2 * vgsol * w2i * w1r ** 4
            - 2 * Kdrag * w2r ** 2 * k * rhogeq * vgsol * w2i ** 2 * w3i * w3r ** 2
            - cs ** 2 * k ** 3 * rhogeq * vgsol * Kdrag * w2i ** 3 * w3i ** 2
            + 2 * w3r ** 2 * rhogeq * rhogsol * w2i ** 4 * w1r * w3i ** 2
            + 2 * Kdrag * k * rhogeq * vgsol * w2r ** 2 * w1i * w2i ** 2 * w3i ** 2
            + 2 * Kdrag * k * rhogeq * vgsol * w2r ** 2 * w1i * w2i ** 2 * w3r ** 2
            + w3r ** 3 * rhogsol * k ** 2 * cs ** 2 * Kdrag * w2i ** 3
            + w3i * Kdrag * rhogsol * w2i ** 4 * w1r * w3r ** 2
            + cs ** 2 * k ** 3 * rhogeq * vgsol * Kdrag * w2i ** 3 * w3r ** 2
            + w1r ** 2 * w3r ** 3 * rhogeq * rhogsol * w2i ** 3 * w1i
            + 2
            * w3i
            * cs ** 2
            * k ** 3
            * rhogeq ** 2
            * vgsol
            * w2i
            * w1i ** 2
            * w1r ** 2
            - 2 * w3r ** 2 * vgsol * k * rhogeq * Kdrag * w2i ** 3 * w3i ** 2
            + w3r ** 4 * rhogeq ** 2 * cs ** 2 * k ** 3 * vgsol * w2i * w1i
            + w1r ** 2 * vgsol * k * rhogeq ** 2 * w2i ** 3 * w1i * w3r ** 2
            + w1r ** 2 * vgsol * k * rhogeq ** 2 * w2i ** 3 * w1i * w3i ** 2
        )

        rhod1r = rhod1r + (
            w1r ** 2 * w3r * rhogeq * rhogsol * w2i ** 3 * w1i * w3i ** 2
            - 2 * w2i * Kdrag * rhogsol * w2r ** 2 * w1r * w3i ** 3 * w1i
            + rhogeq * rhogsol * w2r * w2i ** 2 * w1r ** 2 * w3i ** 3 * w1i
            + w3r * rhogeq ** 2 * k * vgsol * w2i ** 4 * w1r * w3i ** 2
            - 2 * w3r * rhogeq * cs ** 2 * k ** 3 * w1i * Kdrag * vdsol * w2i ** 2 * w1r
            + 2 * w3r * rhogeq * cs ** 2 * k ** 3 * w1i * Kdrag * vgsol * w2i ** 2 * w1r
            + w3r ** 3 * rhogeq ** 2 * k * vgsol * w2i ** 4 * w1r
            + cs ** 2 * k ** 3 * rhogeq * vdsol * Kdrag * w2i ** 3 * w3i ** 2
            - cs ** 2 * k ** 3 * rhogeq * vdsol * Kdrag * w2i ** 3 * w3r ** 2
            - w3i ** 3 * cs ** 2 * k ** 3 * rhogeq ** 2 * vgsol * w2i * w1i ** 2
            - w3r * vgsol * k * rhogeq * Kdrag * w2i ** 3 * w1r * w3i ** 2
            + w3i * vgsol * k * rhogeq * Kdrag * w2i ** 3 * w1i * w3r ** 2
            - w3i ** 3 * cs ** 2 * k ** 3 * rhogeq ** 2 * vgsol * w2i ** 3
            + w3i ** 3 * cs ** 2 * k ** 3 * rhogeq ** 2 * vgsol * w2i * w1r ** 2
            - w3i * cs ** 2 * k ** 3 * rhogeq ** 2 * vgsol * w2i ** 3 * w3r ** 2
            + w3r * vdsol * Kdrag * k * rhogeq * w2i ** 3 * w1r * w3i ** 2
            - 2 * w2i ** 3 * w1i * w3i ** 3 * Kdrag * rhogsol * w1r
            + w3r * rhogeq * cs ** 2 * k ** 2 * w1i ** 4 * rhogsol * w2i ** 2
            - 2 * w2i ** 3 * w1i * w3r * rhogeq * cs ** 2 * k ** 2 * w1r ** 2 * rhogsol
            - 2 * w2i ** 2 * w1i ** 2 * w3i * cs ** 2 * k ** 3 * rhogeq * Kdrag * vgsol
            + 2 * w2i ** 2 * w1i ** 2 * w3i * cs ** 2 * k ** 3 * rhogeq * Kdrag * vdsol
            + 2 * w3i ** 3 * cs ** 2 * k ** 3 * rhogeq ** 2 * vgsol * w2i ** 2 * w1i
            + 2 * w2i ** 2 * w1i ** 2 * w3r * rhogeq * cs ** 4 * k ** 4 * rhogsol
            - w3i ** 3 * cs ** 2 * k ** 3 * rhogeq * Kdrag * vdsol * w2i * w1i
            + w3r * rhogsol * k ** 2 * cs ** 2 * Kdrag * w2i ** 3 * w3i ** 2
            - w3r ** 3 * vgsol * k * rhogeq * Kdrag * w2i ** 3 * w1r
            + 2
            * w3r ** 2
            * rhogeq ** 2
            * cs ** 2
            * k ** 3
            * vgsol
            * w2i
            * w1i
            * w3i ** 2
            + w3i ** 3 * vgsol * k * rhogeq * Kdrag * w2i ** 3 * w1i
            + 2 * w2i * w1i ** 2 * w3r * w3i * cs ** 4 * k ** 4 * rhogeq * rhogsol
            + w3i ** 3 * rhogeq ** 2 * k * vgsol * w2i ** 4 * w1i
            - 2 * w2i ** 3 * w1i * w3i * w3r ** 2 * Kdrag * rhogsol * w1r
            + w3r ** 3 * rhogeq * w1i ** 2 * w2r ** 3 * rhogsol * w1r
            + w3r ** 3 * rhogeq * rhogsol * w2r * w2i ** 2 * w1r ** 3
            + rhogeq * rhogsol * k ** 2 * cs ** 2 * w2i ** 4 * w1r * w3i ** 2
            - rhogeq * rhogsol * k ** 2 * cs ** 2 * w2i ** 4 * w1r * w3r ** 2
            + w3i * vdsol * Kdrag * k * rhogeq * w2i ** 3 * w1i * w3r ** 2
            - w3r * rhogeq * cs ** 2 * k ** 3 * w2i * Kdrag * vgsol * w1r * w1i ** 2
            + w3i * cs ** 2 * k ** 3 * rhogeq * Kdrag * vgsol * w2i * w1i ** 3
            + w3i ** 3 * vdsol * Kdrag * k * rhogeq * w2i ** 3 * w1i
            + w3r * rhogeq * cs ** 2 * k ** 3 * w2i * Kdrag * vdsol * w1r * w1i ** 2
            - 2 * w3r ** 2 * rhogeq ** 2 * vgsol * k * w2i ** 4 * w3i ** 2
            - w3r * rhogeq * cs ** 4 * k ** 4 * rhogsol * w2i * w1i ** 3
            - w3i * cs ** 4 * k ** 4 * rhogeq * rhogsol * w2i * w1r * w1i ** 2
            + w3i * cs ** 2 * k ** 3 * rhogeq ** 2 * vgsol * w2i * w1i ** 4
            - w3i * cs ** 2 * k ** 3 * rhogeq * Kdrag * vdsol * w2i * w1i ** 3
            - w3i * cs ** 2 * k ** 3 * rhogeq ** 2 * vgsol * w2i * w1i ** 2 * w3r ** 2
            + w3i * cs ** 2 * k ** 3 * rhogeq ** 2 * vgsol * w2i * w1r ** 2 * w3r ** 2
            + 2
            * w3i
            * cs ** 2
            * k ** 3
            * rhogeq ** 2
            * vgsol
            * w2i ** 2
            * w1i
            * w3r ** 2
            + 2 * w3i * rhogsol * k ** 4 * cs ** 4 * rhogeq * w1i * w2i ** 2 * w1r
            + 2 * w3i * rhogeq ** 2 * k * vgsol * w2i ** 3 * w1r ** 2 * w3r ** 2
            - 2 * w3i * rhogeq ** 2 * k * vgsol * w2i ** 3 * w1i ** 2 * w3r ** 2
            - 2
            * w3r
            * rhogeq
            * cs ** 2
            * k ** 2
            * w1r ** 2
            * rhogsol
            * w2i ** 2
            * w3i ** 2
            + w3i * rhogeq ** 2 * k * vgsol * w2i ** 4 * w1i * w3r ** 2
            + w3i ** 4 * rhogeq ** 2 * vgsol * k * w2i ** 3 * w1i
            + 2 * w3i ** 3 * rhogeq ** 2 * k * vgsol * w2i ** 3 * w1r ** 2
            - 2 * w3i ** 3 * rhogeq ** 2 * k * vgsol * w2i ** 3 * w1i ** 2
            - w3i * cs ** 4 * k ** 4 * rhogeq * rhogsol * w2i * w1r * w3r ** 2
            - w3r ** 4 * rhogeq ** 2 * cs ** 2 * k ** 3 * vgsol * w2i ** 2
            - w3r * Kdrag * rhogsol * k ** 2 * cs ** 2 * w1i * w2i ** 2 * w3i ** 2
            + w3r ** 3 * rhogeq * cs ** 4 * k ** 4 * rhogsol * w2i ** 2
            - w3r * rhogeq * rhogsol * w2i ** 4 * w1r ** 2 * w3i ** 2
            + rhogeq * cs ** 2 * k ** 3 * w1i * Kdrag * vdsol * w2i ** 2 * w3r ** 2
            - rhogeq * cs ** 4 * k ** 4 * w1r * rhogsol * w2i ** 2 * w3i ** 2
            - rhogeq * cs ** 4 * k ** 4 * w1r * rhogsol * w2i ** 2 * w3r ** 2
            - w3r ** 3 * rhogeq * cs ** 4 * k ** 4 * rhogsol * w2i * w1i
            + 3 * rhogeq * cs ** 2 * k ** 3 * w1i * Kdrag * vgsol * w2i ** 2 * w3i ** 2
            + 2 * w2i * w1i * rhogeq * cs ** 4 * k ** 4 * w1r * rhogsol * w3i ** 2
            + 2 * w2i * w1i * rhogeq * cs ** 4 * k ** 4 * w1r * rhogsol * w3r ** 2
            - 2 * w2i ** 3 * w1i * rhogeq * cs ** 2 * k ** 2 * w1r * rhogsol * w3i ** 2
            + 2 * w2i ** 3 * w1i * rhogeq * cs ** 2 * k ** 2 * w1r * rhogsol * w3r ** 2
            - rhogeq * cs ** 2 * k ** 3 * w1i * Kdrag * vgsol * w2i ** 2 * w3r ** 2
            + w3i * cs ** 2 * k ** 3 * rhogeq * Kdrag * vdsol * w2i ** 2 * w3r ** 2
            - w3i * cs ** 2 * k ** 3 * rhogeq * Kdrag * vgsol * w2i ** 2 * w3r ** 2
            - w3i ** 3 * cs ** 4 * k ** 4 * rhogeq * rhogsol * w2i * w1r
            - w3r ** 3 * Kdrag * rhogsol * k ** 2 * cs ** 2 * w1i * w2i ** 2
            - w3r * rhogeq * cs ** 4 * k ** 4 * rhogsol * w2i * w1i * w3i ** 2
        )

        rhod1r = rhod1r + (
            w3r * rhogeq * cs ** 4 * k ** 4 * rhogsol * w2i ** 2 * w3i ** 2
            - 2 * w3r ** 2 * rhogeq ** 2 * w1i ** 2 * k * vgsol * w2r ** 2 * w1r ** 2
            + w3r ** 2 * w1r ** 3 * rhogeq ** 2 * w2r ** 3 * k * vgsol
            - 2 * w3r ** 3 * rhogeq * cs ** 2 * k ** 2 * w1r ** 2 * rhogsol * w2i ** 2
            - w3r ** 2 * w3i * rhogeq * rhogsol * w2i * w1r * w2r ** 2 * w1i ** 2
            - w3r ** 2 * w3i * rhogeq * k * Kdrag * vdsol * w2r ** 2 * w1i ** 2
            - w3r * rhogeq * rhogsol * w2i ** 4 * w1i ** 2 * w3i ** 2
            - 4 * w3r ** 2 * rhogeq ** 2 * k * vgsol * w2r * w2i ** 2 * w1r * w3i * w1i
            - 4
            * w3r ** 2
            * w2r
            * rhogeq ** 2
            * cs ** 2
            * k ** 3
            * vgsol
            * w2i
            * w1i
            * w1r
            - w2i ** 2 * k * Kdrag ** 2 * vdsol * w1r ** 2 * w3r ** 2
            + w2i ** 2 * k * Kdrag ** 2 * vgsol * w1r ** 2 * w3i ** 2
            + w2i ** 2 * k * Kdrag ** 2 * vgsol * w1r ** 2 * w3r ** 2
            - 4 * vgsol * k * rhogeq ** 2 * w2i ** 3 * w1i * w3i ** 2 * w3r * w1r
            - 4 * vgsol * k * rhogeq ** 2 * w2i ** 3 * w1i * w3r ** 3 * w1r
            - 2
            * rhogeq ** 2
            * cs ** 2
            * k ** 3
            * w1i ** 2
            * vgsol
            * w2i ** 2
            * w3r ** 2
            + 2 * w2i ** 2 * w1i * Kdrag * rhogsol * k ** 2 * cs ** 2 * w1r * w3r ** 2
            + 2 * w2i ** 2 * w1i * Kdrag * rhogsol * k ** 2 * cs ** 2 * w1r * w3i ** 2
            + w2i ** 2 * w1i ** 2 * Kdrag ** 2 * k * vdsol * w3r ** 2
            - 3 * rhogeq * cs ** 2 * k ** 3 * w1i * Kdrag * vdsol * w2i ** 2 * w3i ** 2
            + 2
            * rhogeq ** 2
            * cs ** 2
            * k ** 3
            * w1r ** 2
            * vgsol
            * w2i ** 2
            * w3r ** 2
            + w3i ** 3 * cs ** 2 * k ** 3 * rhogeq * Kdrag * vdsol * w2i ** 2
            - w2i ** 2 * w1i ** 2 * Kdrag ** 2 * k * vgsol * w3r ** 2
            + w2i ** 2 * w1i ** 2 * Kdrag ** 2 * k * vdsol * w3i ** 2
            - w2i ** 2 * w1i ** 2 * Kdrag ** 2 * k * vgsol * w3i ** 2
            - w3i ** 3 * cs ** 2 * k ** 3 * rhogeq * Kdrag * vgsol * w2i ** 2
            + w3i ** 4 * rhogeq ** 2 * cs ** 2 * k ** 3 * vgsol * w2i * w1i
            + w3i ** 3 * cs ** 2 * k ** 3 * rhogeq * Kdrag * vgsol * w2i * w1i
            - w3r ** 3 * rhogeq * cs ** 2 * k ** 3 * w2i * Kdrag * vgsol * w1r
            + w3r ** 3 * rhogeq * cs ** 2 * k ** 3 * w2i * Kdrag * vdsol * w1r
            - w2i ** 2 * k * Kdrag ** 2 * vdsol * w1r ** 2 * w3i ** 2
            + 2 * Kdrag * w2i ** 2 * k * rhogeq * vgsol * w3i * w1r ** 2 * w3r ** 2
            + 2 * w1r ** 2 * rhogeq * cs ** 2 * k ** 3 * w2i * Kdrag * vgsol * w3r ** 2
            - w1r ** 2 * rhogeq ** 2 * cs ** 2 * k ** 3 * vgsol * w2i * w1i * w3i ** 2
            + w1r ** 2 * rhogeq ** 2 * cs ** 2 * k ** 3 * vgsol * w2i * w1i * w3r ** 2
            + 2 * Kdrag * w2i ** 2 * k * rhogeq * vgsol * w3i ** 3 * w1r ** 2
            - 2 * w1r ** 2 * rhogeq ** 2 * w1i ** 2 * k * vgsol * w2i ** 2 * w3i ** 2
            - 2 * w1r ** 2 * rhogeq ** 2 * w1i ** 2 * k * vgsol * w2i ** 2 * w3r ** 2
            + Kdrag * w2i ** 2 * k * rhogeq * vgsol * w3i ** 4 * w1i
            - w1i ** 2 * vdsol * Kdrag * k * rhogeq * w2i ** 3 * w3i ** 2
            - w1i ** 2 * vdsol * Kdrag * k * rhogeq * w2i ** 3 * w3r ** 2
            + w1i ** 3 * vdsol * Kdrag * k * rhogeq * w2i ** 2 * w3r ** 2
            + w1i ** 3 * vdsol * Kdrag * k * rhogeq * w2i ** 2 * w3i ** 2
            - w1i ** 2 * w3i ** 3 * rhogeq * rhogsol * w2i ** 3 * w1r
            - w1i ** 2 * w3i * rhogeq * rhogsol * w2i ** 3 * w1r * w3r ** 2
            - Kdrag ** 2 * k * vdsol * w2r ** 2 * w1r ** 2 * w3i ** 2
            + 2 * w1i ** 2 * rhogeq * cs ** 2 * k ** 3 * w2i * Kdrag * vdsol * w3i ** 2
            + 2
            * w1i ** 2
            * rhogeq
            * cs ** 2
            * k ** 2
            * w1r
            * rhogsol
            * w2i ** 2
            * w3i ** 2
            - 2 * rhogeq ** 2 * cs ** 2 * k ** 3 * vgsol * w2i * w1i * w3r ** 3 * w1r
            - 2 * rhogeq * cs ** 2 * k ** 2 * w1r * rhogsol * w2i ** 2 * w3i ** 3 * w1i
            + Kdrag ** 2 * k * vgsol * w2r ** 2 * w1r ** 2 * w3i ** 2
            - 2
            * rhogeq
            * cs ** 2
            * k ** 2
            * w1r
            * rhogsol
            * w2i ** 2
            * w3r ** 2
            * w3i
            * w1i
            + w1i ** 3 * vgsol * k * rhogeq ** 2 * w2i ** 3 * w3r ** 2
            + w1i ** 3 * vgsol * k * rhogeq ** 2 * w2i ** 3 * w3i ** 2
            - 2 * rhogeq ** 2 * w1i ** 2 * k * vgsol * w3i ** 2 * w2r ** 2 * w1r ** 2
            + 2 * Kdrag * rhogsol * k ** 2 * cs ** 2 * w1i * w2r ** 2 * w3i ** 2 * w1r
            + 2 * w3r ** 2 * Kdrag * rhogsol * k ** 2 * cs ** 2 * w1i * w2r ** 2 * w1r
            - 2 * w3r ** 2 * rhogeq * cs ** 2 * k ** 2 * w2r ** 2 * w1r ** 3 * rhogsol
            + w1i ** 3 * w3r * rhogeq * rhogsol * w2i ** 3 * w3i ** 2
            + w3r ** 2 * w1r ** 3 * w2r * rhogeq ** 2 * cs ** 2 * k ** 3 * vgsol
            - w1r ** 4 * rhogeq ** 2 * k * vgsol * w2i ** 2 * w3i ** 2
            - w1r ** 4 * rhogeq ** 2 * k * vgsol * w2i ** 2 * w3r ** 2
            - w1i ** 4 * rhogeq ** 2 * k * vgsol * w2i ** 2 * w3r ** 2
            - w1i ** 4 * rhogeq ** 2 * k * vgsol * w2i ** 2 * w3i ** 2
            + w1r ** 2 * vdsol * Kdrag * k * rhogeq * w2i ** 2 * w1i * w3r ** 2
            - w1r ** 3 * w3i * rhogeq * rhogsol * w2i ** 3 * w3r ** 2
            + w1r ** 2 * vdsol * Kdrag * k * rhogeq * w2i ** 2 * w1i * w3i ** 2
            - 2 * w1r ** 2 * rhogeq * cs ** 2 * k ** 3 * w2i * Kdrag * vdsol * w3r ** 2
            - w3r ** 2 * w2i * rhogeq * k * Kdrag * vdsol * w2r ** 2 * w1r ** 2
            - w3r ** 2 * rhogeq ** 2 * w1r ** 4 * k * vgsol * w2r ** 2
            + w3r ** 2 * w1r ** 4 * w2r * cs ** 2 * k ** 2 * rhogeq * rhogsol
            - w1i ** 3 * Kdrag * k * rhogeq * vgsol * w2i ** 2 * w3r ** 2
            - w1i ** 3 * Kdrag * k * rhogeq * vgsol * w2i ** 2 * w3i ** 2
        )

        rhod1r = rhod1r + (
            rhogeq ** 2 * w3i ** 3 * w1i * k * vgsol * w2r ** 2 * w1r ** 2
            + w3r ** 2 * rhogeq * w1i * k * Kdrag * vdsol * w2r ** 2 * w1r ** 2
            + w3r ** 2 * rhogeq * rhogsol * w2r * w2i ** 2 * w1r ** 2 * w3i * w1i
            + w3r ** 2 * w1i ** 2 * rhogeq ** 2 * k * vgsol * w2r * w2i ** 2 * w1r
            - 4 * w3r ** 2 * rhogeq ** 2 * w2r ** 3 * k * w1r * vgsol * w3i * w1i
            - 2
            * w3r ** 2
            * w1r ** 2
            * w2i
            * cs ** 2
            * k ** 2
            * rhogeq
            * rhogsol
            * w2r
            * w1i
            + w3r ** 2 * rhogeq * w1i ** 3 * k * Kdrag * vdsol * w2r ** 2
            + w3r ** 2 * rhogeq * rhogsol * w2r * w2i ** 2 * w1i ** 3 * w3i
            - w3r ** 2 * rhogeq ** 2 * w1i ** 4 * k * vgsol * w2r ** 2
            - w3r ** 2 * Kdrag ** 2 * k * vgsol * w2r ** 2 * w1i ** 2
            + w3r ** 2 * Kdrag ** 2 * k * vdsol * w2r ** 2 * w1i ** 2
            - 2 * w3r ** 2 * w1i ** 3 * w2i * cs ** 2 * k ** 2 * rhogeq * rhogsol * w2r
            + w3r ** 2 * w1i ** 3 * w2i * rhogeq ** 2 * k * vgsol * w2r ** 2
            + w3r ** 2 * rhogeq * w2r ** 3 * w1r ** 2 * rhogsol * w3i * w1i
            + 2
            * w3r ** 2
            * rhogeq
            * rhogsol
            * k ** 2
            * cs ** 2
            * w2r ** 2
            * w1r
            * w3i
            * w1i
            - 2
            * w3r ** 2
            * w2r
            * cs ** 2
            * k ** 2
            * rhogeq
            * rhogsol
            * w1r ** 2
            * w3i
            * w1i
            + w3r ** 2 * w1i ** 2 * rhogeq ** 2 * w2r ** 3 * k * w1r * vgsol
            - 2 * w3r ** 2 * w2r * rhogeq * w1i ** 3 * rhogsol * k ** 2 * cs ** 2 * w3i
            - 2 * w3r ** 2 * w2i * Kdrag * rhogsol * w2r ** 2 * w1r * w3i * w1i
            - 2 * w3r ** 2 * w2r * Kdrag * rhogeq * k * vgsol * w2i ** 2 * w1i * w1r
            - 2 * w3r ** 2 * Kdrag * k * rhogeq * vgsol * w2r ** 3 * w1i * w1r
            - w3r ** 2 * w3i * rhogeq * rhogsol * w2i * w1r ** 3 * w2r ** 2
            - w3r ** 2 * w1i ** 3 * rhogeq * w2r ** 2 * k * Kdrag * vgsol
            + 2
            * w3r ** 2
            * w2r
            * rhogeq ** 2
            * cs ** 2
            * k ** 3
            * w1r
            * vgsol
            * w3i
            * w1i
            - 2 * w3r ** 2 * w2r * rhogeq * cs ** 2 * k ** 3 * w1i * Kdrag * vdsol * w1r
            + 2 * w3r ** 2 * w2r * rhogeq * cs ** 2 * k ** 3 * w1i * Kdrag * vgsol * w1r
            + w3r ** 2 * w1r ** 3 * rhogeq ** 2 * k * vgsol * w2r * w2i ** 2
            - 2 * w3r ** 2 * w2r * rhogeq * cs ** 4 * k ** 4 * w1r ** 2 * rhogsol
            + w1i ** 3 * rhogeq ** 2 * cs ** 2 * k ** 3 * vgsol * w2i * w3r ** 2
            - w1i ** 3 * rhogeq ** 2 * cs ** 2 * k ** 3 * vgsol * w2i * w3i ** 2
            - w3r ** 2 * w1r ** 2 * rhogeq * w2r ** 2 * w1i * k * Kdrag * vgsol
            + rhogeq * w1i ** 3 * w3i ** 2 * k * Kdrag * vdsol * w2r ** 2
            + w3r ** 2 * w1r ** 2 * w2i * rhogeq ** 2 * k * vgsol * w2r ** 2 * w1i
            + w2i ** 2 * w1r ** 3 * w3r ** 3 * rhogeq ** 2 * k * vgsol
            + w3r ** 2 * Kdrag ** 2 * k * vgsol * w2r ** 2 * w1r ** 2
            - w3r ** 2 * Kdrag ** 2 * k * vdsol * w2r ** 2 * w1r ** 2
            + w3r ** 2 * w1i ** 4 * w2r * rhogeq * rhogsol * k ** 2 * cs ** 2
            - rhogeq * w3i ** 3 * k * Kdrag * vdsol * w2r ** 2 * w1r ** 2
            - w2i ** 2 * rhogeq ** 2 * cs ** 2 * k ** 3 * w3i * w1i ** 3 * vgsol
            + w3r ** 2 * rhogeq * w1i ** 3 * w2r ** 3 * rhogsol * w3i
            - w2i ** 2 * w3r * rhogeq ** 2 * cs ** 2 * k ** 3 * w1i ** 2 * vgsol * w1r
            + 2
            * w3r ** 2
            * w2i
            * rhogeq
            * cs ** 2
            * k ** 2
            * rhogsol
            * w2r ** 2
            * w1i
            * w1r
            + w3r ** 2 * w3i * rhogeq ** 2 * k * vgsol * w1i * w2r ** 2 * w1r ** 2
            - w3r ** 2 * w3i * rhogeq * k * Kdrag * vdsol * w2r ** 2 * w1r ** 2
            - w2i ** 2 * w3r * rhogeq ** 2 * cs ** 2 * k ** 3 * w1r ** 3 * vgsol
            - w3r ** 2 * w2i * w1i ** 2 * rhogeq * k * Kdrag * vdsol * w2r ** 2
            - w2i ** 2 * rhogeq ** 2 * cs ** 2 * k ** 3 * w3i * w1i * vgsol * w1r ** 2
            - 2
            * w2i ** 2
            * w3r
            * w3i
            * cs ** 2
            * k ** 2
            * rhogeq
            * rhogsol
            * w1i
            * w1r ** 2
            - 2 * w2i ** 2 * Kdrag * k * rhogeq * vgsol * w1i * w3r ** 3 * w1r
            - 2 * w2i ** 2 * w3r * w3i * cs ** 2 * k ** 2 * rhogeq * rhogsol * w1i ** 3
            - w2i ** 2 * w1r ** 2 * w3i * vdsol * Kdrag * k * rhogeq * w3r ** 2
            - 2
            * w3r ** 2
            * w1i ** 2
            * rhogeq
            * rhogsol
            * k ** 2
            * cs ** 2
            * w2r ** 2
            * w1r
            + w3r ** 2 * w3i * rhogeq ** 2 * k * vgsol * w1i ** 3 * w2r ** 2
            - 2
            * w3r ** 2
            * w2r
            * rhogeq ** 2
            * cs ** 2
            * k ** 3
            * w1r
            * vgsol
            * w2i
            * w3i
            + 2
            * w3r ** 2
            * w1i ** 2
            * w2r
            * cs ** 2
            * k ** 2
            * rhogeq
            * rhogsol
            * w1r ** 2
            + w3r ** 2 * w1i ** 2 * w2r * rhogeq ** 2 * cs ** 2 * k ** 3 * w1r * vgsol
            + w2i ** 2 * w1i ** 3 * rhogeq ** 2 * w3i ** 3 * k * vgsol
            + w2i ** 2 * w1i ** 2 * w3r * rhogeq ** 2 * k * w1r * vgsol * w3i ** 2
            + w2i ** 2 * w1i ** 3 * rhogeq ** 2 * w3i * k * vgsol * w3r ** 2
            + w2i ** 2 * w1i ** 2 * w3r ** 3 * rhogeq ** 2 * k * w1r * vgsol
            - w2i ** 2 * w1r ** 2 * w3i ** 3 * vdsol * Kdrag * k * rhogeq
            - 2 * w2i ** 2 * Kdrag * k * rhogeq * vgsol * w1i * w3i ** 2 * w3r * w1r
            - w2i ** 2 * w1i ** 2 * w3i ** 3 * vdsol * Kdrag * k * rhogeq
            - w2i ** 2 * w1i ** 2 * w3i * vdsol * Kdrag * k * rhogeq * w3r ** 2
            + w2i ** 2 * w1r ** 3 * w3r * rhogeq ** 2 * k * vgsol * w3i ** 2
            + w2i ** 2 * w1r ** 2 * rhogeq ** 2 * w3i ** 3 * w1i * k * vgsol
            + w2i ** 2 * w1r ** 2 * rhogeq ** 2 * w3i * w1i * k * vgsol * w3r ** 2
            - 4
            * w3r
            * w2i
            * cs ** 2
            * k ** 3
            * rhogeq
            * Kdrag
            * vgsol
            * w2r
            * w3i
            * w1i
            + w3r * w1r ** 4 * rhogeq * cs ** 2 * k ** 2 * w2r ** 2 * rhogsol
            + rhogeq * w1i ** 3 * w2r ** 3 * rhogsol * w3i ** 3
        )
        rhod1r = (
            rhod1r
            * rhodeq
            / (
                w1i ** 2
                - 2 * w3i * w1i
                + w3r ** 2
                + w1r ** 2
                + w3i ** 2
                - 2 * w3r * w1r
            )
            / (w3r ** 2 + w3i ** 2)
            / Kdrag
            / (
                w2r ** 2
                + w1r ** 2
                + w2i ** 2
                - 2 * w2i * w1i
                - 2 * w2r * w1r
                + w1i ** 2
            )
            / rhogeq
            / (w2i ** 2 + w2r ** 2)
        )

        rhod1i = -(
            -2 * rhogsol * Kdrag * w1i ** 3 * rhodeq * w3r ** 2 * w2i ** 3
            - 2 * rhodsol * rhogeq * Kdrag * w3i ** 4 * w2i ** 2 * w2r ** 2
            - rhodsol * rhogeq * Kdrag * w2r ** 4 * w3i ** 4
            - rhodsol * rhogeq * Kdrag * w3i ** 4 * w2i ** 4
            - 2 * rhogsol * Kdrag * w1i ** 3 * rhodeq * w3r ** 2 * w2i * w2r ** 2
            - 2 * rhodsol * rhogeq * Kdrag * w2r ** 4 * w3i ** 2 * w3r ** 2
            - 2 * rhogsol * Kdrag * w1i ** 3 * rhodeq * w3r ** 2 * w2r ** 2 * w3i
            - 2 * rhogsol * Kdrag * w1i ** 3 * rhodeq * w3r ** 2 * w2i ** 2 * w3i
            - 2 * rhogsol * Kdrag * w1i ** 3 * rhodeq * w2r ** 2 * w3i ** 3
            - 2 * rhogsol * Kdrag * w1i ** 3 * rhodeq * w2i ** 2 * w3i ** 3
            - 2 * rhogsol * Kdrag * w1i ** 3 * rhodeq * w2i ** 3 * w3i ** 2
            - rhodsol * rhogeq * Kdrag * w3r ** 4 * w2i ** 4
            - rhodsol * rhogeq * Kdrag * w3r ** 4 * w2r ** 4
            - 2 * rhodsol * rhogeq * Kdrag * w3r ** 4 * w2i ** 2 * w2r ** 2
            - 4 * rhodsol * rhogeq * Kdrag * w3r ** 2 * w2i ** 2 * w2r ** 2 * w3i ** 2
            - 2 * rhodsol * rhogeq * Kdrag * w3i ** 2 * w2i ** 4 * w3r ** 2
            - 2 * rhogsol * Kdrag * w1i ** 3 * rhodeq * w2i * w2r ** 2 * w3i ** 2
            + rhogsol * rhogeq * k ** 4 * cs ** 4 * w1i ** 3 * rhodeq * w3r * w2r
            - rhogsol * rhogeq * k ** 4 * cs ** 4 * w1i ** 3 * rhodeq * w2i * w3i
            + w1r
            * vdsol
            * Kdrag
            * rhogeq
            * k ** 3
            * cs ** 2
            * rhodeq
            * w3r ** 2
            * w2i ** 2
            - w1r
            * vdsol
            * Kdrag
            * rhogeq
            * k ** 3
            * cs ** 2
            * rhodeq
            * w2r
            * w3r
            * w3i ** 2
            - w1r
            * vdsol
            * Kdrag
            * rhogeq
            * k ** 3
            * cs ** 2
            * rhodeq
            * w3r
            * w2r
            * w2i ** 2
            - w1r * vdsol * Kdrag * rhogeq * k ** 3 * cs ** 2 * rhodeq * w3r * w2r ** 3
            - w1i * rhogsol * rhogeq * k ** 2 * cs ** 2 * rhodeq * w3i ** 2 * w2i ** 4
            - w1i * rhogsol * rhogeq * k ** 2 * cs ** 2 * rhodeq * w3i ** 4 * w2i ** 2
            + w1i * rhogsol * rhogeq * k ** 2 * cs ** 2 * rhodeq * w3i ** 4 * w2r ** 2
            - vgsol * Kdrag * w1i * k ** 3 * cs ** 2 * rhogeq * rhodeq * w3r * w2i ** 3
            + w1i * rhogsol * rhogeq * k ** 2 * cs ** 2 * rhodeq * w3r ** 2 * w2i ** 4
            - w1i * rhogsol * rhogeq * k ** 2 * cs ** 2 * rhodeq * w2r ** 4 * w3i ** 2
            - 2
            * w1i
            * rhogsol
            * rhogeq
            * k ** 2
            * cs ** 2
            * rhodeq
            * w3i ** 2
            * w2i ** 2
            * w2r ** 2
            - vgsol * Kdrag * w1i * k ** 3 * cs ** 2 * rhogeq * rhodeq * w3r ** 3 * w2i
            - w1i * rhogsol * rhogeq * k ** 2 * cs ** 2 * rhodeq * w3r ** 4 * w2i ** 2
            + w1i * rhogsol * rhogeq * k ** 2 * cs ** 2 * rhodeq * w3r ** 2 * w2r ** 4
            - 2
            * cs ** 2
            * k ** 2
            * rhogsol
            * w1i ** 2
            * rhogeq
            * rhodeq
            * w2r ** 2
            * w3i ** 3
            + cs ** 2 * k ** 2 * rhogsol * w1i ** 2 * rhogeq * rhodeq * w3i ** 4 * w2i
            + 2
            * w1i
            * rhogsol
            * rhogeq
            * k ** 2
            * cs ** 2
            * rhodeq
            * w3r ** 2
            * w2i ** 2
            * w2r ** 2
            - 2
            * w1i
            * rhogsol
            * rhogeq
            * k ** 2
            * cs ** 2
            * rhodeq
            * w3r ** 2
            * w2i ** 2
            * w3i ** 2
            + w1i * rhogsol * rhogeq * k ** 2 * cs ** 2 * rhodeq * w3r ** 4 * w2r ** 2
            + rhogeq * k * Kdrag * vdsol * w1i ** 2 * rhodeq * w3r * w2i ** 2 * w3i ** 2
            + rhogeq * k * Kdrag * vdsol * w1i ** 2 * rhodeq * w3r * w2r ** 2 * w3i ** 2
            + rhogeq * k * Kdrag * vdsol * w1i ** 2 * rhodeq * w2r * w2i ** 2 * w3i ** 2
            + rhogeq * k * Kdrag * vdsol * w1i ** 2 * rhodeq * w2r ** 3 * w3r ** 2
            + rhogeq * k * Kdrag * vdsol * w1i ** 2 * rhodeq * w2r * w2i ** 2 * w3r ** 2
            + rhogeq * k * Kdrag * vdsol * w1i ** 2 * rhodeq * w2r ** 3 * w3i ** 2
            + rhogeq * k * Kdrag * vdsol * w1i ** 2 * rhodeq * w3r ** 3 * w2i ** 2
            + rhogeq * k * Kdrag * vdsol * w1i ** 2 * rhodeq * w3r ** 3 * w2r ** 2
            + 2
            * cs ** 2
            * k ** 2
            * rhogsol
            * w1i ** 2
            * rhogeq
            * rhodeq
            * w2i ** 2
            * w3i ** 3
            + 2
            * cs ** 2
            * k ** 2
            * rhogsol
            * w1i ** 2
            * rhogeq
            * rhodeq
            * w2i
            * w2r ** 2
            * w3i ** 2
            + cs ** 2 * k ** 2 * rhogsol * w1i ** 2 * rhogeq * rhodeq * w3i * w2i ** 4
            + 2
            * cs ** 2
            * k ** 2
            * rhogsol
            * w1i ** 2
            * rhogeq
            * rhodeq
            * w2i ** 3
            * w3i ** 2
            + 2
            * cs ** 2
            * k ** 2
            * rhogsol
            * w1i ** 2
            * rhogeq
            * rhodeq
            * w3i
            * w2r ** 2
            * w2i ** 2
            + cs ** 2 * k ** 2 * rhogsol * w1i ** 2 * rhogeq * rhodeq * w3i * w2r ** 4
            + 2
            * cs ** 2
            * k ** 2
            * rhogsol
            * w1i ** 2
            * rhogeq
            * rhodeq
            * w3i ** 2
            * w2i
            * w3r ** 2
            + 2
            * cs ** 2
            * k ** 2
            * rhogsol
            * w1i ** 2
            * rhogeq
            * rhodeq
            * w3r ** 2
            * w2i ** 2
            * w3i
            - 2
            * cs ** 2
            * k ** 2
            * rhogsol
            * w1i ** 2
            * rhogeq
            * rhodeq
            * w3r ** 2
            * w2r ** 2
            * w3i
            - 2
            * cs ** 2
            * k ** 2
            * rhogsol
            * w1i ** 2
            * rhogeq
            * rhodeq
            * w3r ** 2
            * w2i
            * w2r ** 2
            - 2
            * cs ** 2
            * k ** 2
            * rhogsol
            * w1i ** 2
            * rhogeq
            * rhodeq
            * w3r ** 2
            * w2i ** 3
            - 2
            * w1i
            * w1r ** 2
            * rhogsol
            * rhogeq
            * k ** 2
            * cs ** 2
            * rhodeq
            * w2i ** 3
            * w3i
            - 2
            * w1i
            * w1r ** 2
            * rhogsol
            * rhogeq
            * k ** 2
            * cs ** 2
            * rhodeq
            * w3i ** 2
            * w2i ** 2
            - 2
            * w1i
            * w1r ** 2
            * rhogsol
            * rhogeq
            * k ** 2
            * cs ** 2
            * rhodeq
            * w3i
            * w2i
            * w3r ** 2
            - 2
            * w1i
            * w1r ** 2
            * rhogsol
            * rhogeq
            * k ** 2
            * cs ** 2
            * rhodeq
            * w2i
            * w3i
            * w2r ** 2
            + 2
            * w1i
            * w1r ** 2
            * rhogsol
            * rhogeq
            * k ** 2
            * cs ** 2
            * rhodeq
            * w3r ** 2
            * w2r ** 2
            - 2
            * w1i
            * w1r ** 2
            * rhogsol
            * rhogeq
            * k ** 2
            * cs ** 2
            * rhodeq
            * w3i ** 3
            * w2i
        )

        rhod1i = rhod1i - (
            w1r
            * vdsol
            * Kdrag
            * rhogeq
            * k ** 3
            * cs ** 2
            * rhodeq
            * w2r ** 2
            * w3i ** 2
            + cs ** 2 * k ** 2 * rhogsol * w1i ** 2 * rhogeq * rhodeq * w3r ** 4 * w2i
            + w1r * vdsol * Kdrag * rhogeq * k ** 3 * cs ** 2 * rhodeq * w2i ** 3 * w3i
            + w1r
            * vdsol
            * Kdrag
            * rhogeq
            * k ** 3
            * cs ** 2
            * rhodeq
            * w3i ** 2
            * w2i ** 2
            + w1r * vdsol * Kdrag * rhogeq * k ** 3 * cs ** 2 * rhodeq * w3i ** 3 * w2i
            + w1r
            * vdsol
            * Kdrag
            * rhogeq
            * k ** 3
            * cs ** 2
            * rhodeq
            * w2i
            * w3i
            * w2r ** 2
            - 3
            * w1r
            * vdsol
            * Kdrag
            * rhogeq
            * k ** 3
            * cs ** 2
            * rhodeq
            * w3r ** 2
            * w2r ** 2
            + 4
            * w1r
            * vdsol
            * Kdrag
            * rhogeq
            * k ** 3
            * cs ** 2
            * rhodeq
            * w3r
            * w2r
            * w2i
            * w3i
            - w1r * vdsol * Kdrag * rhogeq * k ** 3 * cs ** 2 * rhodeq * w2r * w3r ** 3
            + w1r
            * vdsol
            * Kdrag
            * rhogeq
            * k ** 3
            * cs ** 2
            * rhodeq
            * w3i
            * w2i
            * w3r ** 2
            + 2 * w1r * w1i * rhogsol * Kdrag * rhodeq * w3r * w2r ** 2 * w2i * w3i ** 2
            + w1r
            * vgsol
            * k
            * w1i ** 2
            * rhogeq ** 2
            * rhodeq
            * w3r ** 2
            * w2r ** 2
            * w3i
            + w1r
            * vgsol
            * k
            * w1i ** 2
            * rhogeq ** 2
            * rhodeq
            * w3r ** 2
            * w2i ** 2
            * w3i
            + w1r ** 2 * rhogsol * rhogeq * rhodeq * w2r ** 4 * w3i ** 3
            - 2
            * rhogsol
            * rhogeq
            * k ** 2
            * cs ** 2
            * w1r
            * rhodeq
            * w2r
            * w3i ** 4
            * w2i
            + 2 * w1r * w1i * rhogsol * Kdrag * rhodeq * w3r ** 3 * w2i * w2r ** 2
            + 2 * w1r * w1i * rhogsol * Kdrag * rhodeq * w3i * w2i ** 2 * w3r ** 2 * w2r
            + 2 * w1r * w1i * rhogsol * Kdrag * rhodeq * w3i * w2r ** 3 * w3r ** 2
            + 2 * w1r * w1i * rhogsol * Kdrag * rhodeq * w3r * w2i ** 3 * w3i ** 2
            - 4
            * vgsol
            * Kdrag
            * w1i
            * k ** 3
            * cs ** 2
            * rhogeq
            * rhodeq
            * w2r
            * w2i
            * w3i ** 2
            - vgsol * Kdrag * w1i * k ** 3 * cs ** 2 * rhogeq * rhodeq * w2r * w3i ** 3
            - 2
            * rhogsol
            * rhogeq
            * k ** 2
            * cs ** 2
            * w1r
            * rhodeq
            * w2r
            * w2i
            * w3r ** 4
            - 4
            * rhogsol
            * rhogeq
            * k ** 2
            * cs ** 2
            * w1r
            * rhodeq
            * w2r
            * w2i
            * w3i ** 2
            * w3r ** 2
            - 2
            * rhogsol
            * rhogeq
            * k ** 2
            * cs ** 2
            * w1r
            * rhodeq
            * w3i
            * w3r
            * w2i ** 4
            - 2
            * rhogsol
            * rhogeq
            * k ** 2
            * cs ** 2
            * w1r
            * rhodeq
            * w2r ** 4
            * w3i
            * w3r
            - 4
            * rhogsol
            * rhogeq
            * k ** 2
            * cs ** 2
            * w1r
            * rhodeq
            * w3r
            * w3i
            * w2r ** 2
            * w2i ** 2
            + 2 * w1r * w1i * rhogsol * Kdrag * rhodeq * w2i ** 3 * w3r ** 3
            - vgsol
            * Kdrag
            * w1i
            * k ** 3
            * cs ** 2
            * rhogeq
            * rhodeq
            * w2r
            * w3i
            * w3r ** 2
            - vgsol
            * Kdrag
            * w1i
            * k ** 3
            * cs ** 2
            * rhogeq
            * rhodeq
            * w3r
            * w2i
            * w3i ** 2
            - vgsol
            * Kdrag
            * w1i
            * k ** 3
            * cs ** 2
            * rhogeq
            * rhodeq
            * w3r
            * w2i
            * w2r ** 2
            - 4
            * vgsol
            * Kdrag
            * w1i
            * k ** 3
            * cs ** 2
            * rhogeq
            * rhodeq
            * w3i
            * w3r
            * w2i ** 2
            - vgsol * Kdrag * w1i * k ** 3 * cs ** 2 * rhogeq * rhodeq * w2r ** 3 * w3i
            - vgsol
            * Kdrag
            * w1i
            * k ** 3
            * cs ** 2
            * rhogeq
            * rhodeq
            * w2r
            * w3i
            * w2i ** 2
            + w1r
            * vgsol
            * k
            * w1i ** 2
            * rhogeq ** 2
            * rhodeq
            * w2i
            * w2r ** 2
            * w3i ** 2
            + 4
            * w1r
            * rhogsol
            * rhogeq
            * k ** 4
            * cs ** 4
            * rhodeq
            * w2r
            * w2i
            * w3r ** 2
            + w1r * rhogsol * rhogeq * k ** 4 * cs ** 4 * rhodeq * w2r * w3i * w3r ** 2
            + w1r * rhogsol * rhogeq * k ** 4 * cs ** 4 * rhodeq * w3r * w2i * w3i ** 2
            + w1r ** 2 * rhogsol * rhogeq * rhodeq * w2r ** 2 * w2i * w3r ** 4
            + w1r * vgsol * k * w1i ** 2 * rhogeq ** 2 * rhodeq * w2r ** 2 * w3i ** 3
            + w1r * vgsol * k * w1i ** 2 * rhogeq ** 2 * rhodeq * w2i ** 2 * w3i ** 3
            + w1r
            * vgsol
            * k
            * w1i ** 2
            * rhogeq ** 2
            * rhodeq
            * w3r ** 2
            * w2i
            * w2r ** 2
            + w1r * vgsol * k * w1i ** 2 * rhogeq ** 2 * rhodeq * w3r ** 2 * w2i ** 3
            + 2 * w1r * w1i * rhogsol * Kdrag * rhodeq * w2r * w2i ** 2 * w3i ** 3
            + 2 * w1r * w1i * rhogsol * Kdrag * rhodeq * w3i ** 3 * w2r ** 3
            - 2
            * w1r
            * cs ** 2
            * k ** 2
            * rhogsol
            * w1i
            * rhogeq
            * rhodeq
            * w2r ** 3
            * w3r ** 2
            - 2
            * w1r
            * cs ** 2
            * k ** 2
            * rhogsol
            * w1i
            * rhogeq
            * rhodeq
            * w2r
            * w2i ** 2
            * w3r ** 2
            + w1r * rhogsol * rhogeq * k ** 4 * cs ** 4 * rhodeq * w2r ** 3 * w3i
            + w1r * rhogsol * rhogeq * k ** 4 * cs ** 4 * rhodeq * w2r * w3i * w2i ** 2
            + w1r * rhogsol * rhogeq * k ** 4 * cs ** 4 * rhodeq * w2r * w3i ** 3
            + vgsol * Kdrag * k * rhogeq * w1r ** 3 * rhodeq * w3r ** 2 * w2i ** 2
            + w1r * vgsol * k * w1i ** 2 * rhogeq ** 2 * rhodeq * w2i ** 3 * w3i ** 2
            + w1r * rhogsol * rhogeq * k ** 4 * cs ** 4 * rhodeq * w3r ** 3 * w2i
            + w1r * rhogsol * rhogeq * k ** 4 * cs ** 4 * rhodeq * w3r * w2i ** 3
            + w1r * rhogsol * rhogeq * k ** 4 * cs ** 4 * rhodeq * w3r * w2i * w2r ** 2
            + vgsol * Kdrag * k * rhogeq * w1r ** 3 * rhodeq * w3r ** 2 * w2r ** 2
            + vgsol * Kdrag * k * rhogeq * w1r ** 3 * rhodeq * w3i ** 2 * w2i ** 2
            + vgsol * Kdrag * k * rhogeq * w1r ** 3 * rhodeq * w2r ** 2 * w3i ** 2
            + 2
            * w1i
            * rhogsol
            * rhogeq
            * k ** 2
            * cs ** 2
            * rhodeq
            * w3r ** 2
            * w2r ** 2
            * w3i ** 2
            - 2
            * w1r
            * cs ** 2
            * k ** 2
            * rhogsol
            * w1i
            * rhogeq
            * rhodeq
            * w3r ** 3
            * w2r ** 2
            + 2
            * w1r
            * cs ** 2
            * k ** 2
            * rhogsol
            * w1i
            * rhogeq
            * rhodeq
            * w3r ** 3
            * w2i ** 2
            + 4
            * w1r
            * rhogsol
            * rhogeq
            * k ** 4
            * cs ** 4
            * rhodeq
            * w3i
            * w3r
            * w2r ** 2
            - 2
            * w1r
            * cs ** 2
            * k ** 2
            * rhogsol
            * w1i
            * rhogeq
            * rhodeq
            * w3r
            * w2r ** 2
            * w3i ** 2
        )

        rhod1i = rhod1i - (
            2
            * w1r
            * cs ** 2
            * k ** 2
            * rhogsol
            * w1i
            * rhogeq
            * rhodeq
            * w3r
            * w2i ** 2
            * w3i ** 2
            + 4
            * w1r
            * cs ** 2
            * k ** 2
            * rhogsol
            * w1i
            * rhogeq
            * rhodeq
            * w2r
            * w2i
            * w3r ** 2
            * w3i
            + 4
            * w1r
            * cs ** 2
            * k ** 2
            * rhogsol
            * w1i
            * rhogeq
            * rhodeq
            * w2i
            * w3i
            * w3r
            * w2r ** 2
            + 4
            * w1r
            * cs ** 2
            * k ** 2
            * rhogsol
            * w1i
            * rhogeq
            * rhodeq
            * w2i ** 3
            * w3i
            * w3r
            - 2 * w1r * vdsol * Kdrag ** 2 * k * w1i * rhodeq * w3r ** 2 * w2r ** 2
            - 2 * w1r * vdsol * Kdrag ** 2 * k * w1i * rhodeq * w3i ** 2 * w2i ** 2
            - 2 * w1r * vdsol * Kdrag ** 2 * k * w1i * rhodeq * w2r ** 2 * w3i ** 2
            - w1r * vgsol * k ** 3 * cs ** 2 * rhogeq ** 2 * rhodeq * w3r ** 4 * w2i
            + 4
            * w1r
            * cs ** 2
            * k ** 2
            * rhogsol
            * w1i
            * rhogeq
            * rhodeq
            * w2r
            * w2i
            * w3i ** 3
            + 2
            * w1r
            * cs ** 2
            * k ** 2
            * rhogsol
            * w1i
            * rhogeq
            * rhodeq
            * w2r
            * w2i ** 2
            * w3i ** 2
            - 2 * w1r * vdsol * Kdrag ** 2 * k * w1i * rhodeq * w3r ** 2 * w2i ** 2
            - 2 * w1r ** 3 * rhogsol * Kdrag * rhodeq * w3r ** 3 * w2r ** 2
            - 2 * w1r ** 3 * rhogsol * Kdrag * rhodeq * w2r ** 3 * w3i ** 2
            + w1r * cs ** 2 * k ** 2 * rhogsol * Kdrag * rhodeq * w3r ** 3 * w2i ** 2
            - w1i ** 4 * Kdrag * rhogeq * rhodsol * w2r ** 2 * w3i ** 2
            - 2 * w1r ** 3 * rhogsol * Kdrag * rhodeq * w3r ** 3 * w2i ** 2
            - 2 * w1r ** 3 * rhogsol * Kdrag * rhodeq * w2r * w2i ** 2 * w3r ** 2
            + w1r ** 2 * rhogsol * Kdrag * rhodeq * w2i * w2r ** 2 * w3i ** 3
            + w1r ** 2 * rhogsol * Kdrag * rhodeq * w2i ** 3 * w3i ** 3
            + w1r ** 2 * rhogsol * Kdrag * rhodeq * w3i ** 4 * w2r ** 2
            - w1i ** 4 * Kdrag * rhogeq * rhodsol * w3r ** 2 * w2i ** 2
            - w1i ** 4 * Kdrag * rhogeq * rhodsol * w3r ** 2 * w2r ** 2
            - w1i ** 4 * Kdrag * rhogeq * rhodsol * w3i ** 2 * w2i ** 2
            + 2 * w1r ** 2 * rhogsol * Kdrag * rhodeq * w3r ** 2 * w2i ** 2 * w3i ** 2
            + 2 * w1r ** 2 * rhogsol * Kdrag * rhodeq * w3r ** 2 * w2r ** 2 * w3i ** 2
            + 3 * w1r ** 2 * rhogsol * Kdrag * rhodeq * w3r * w2r ** 3 * w3i ** 2
            + w1r ** 2 * rhogsol * Kdrag * rhodeq * w2r ** 4 * w3i ** 2
            + w1r ** 2 * rhogsol * Kdrag * rhodeq * w3r ** 2 * w2r ** 4
            + w1r ** 2 * rhogsol * Kdrag * rhodeq * w2i ** 3 * w3i * w3r ** 2
            + w1r ** 2 * rhogsol * Kdrag * rhodeq * w3r ** 2 * w2i ** 4
            + w1r ** 2 * rhogsol * Kdrag * rhodeq * w2i * w3i * w2r ** 2 * w3r ** 2
            + 3 * w1r ** 2 * rhogsol * Kdrag * rhodeq * w2r * w3r ** 3 * w2i ** 2
            + 2 * w1r ** 2 * rhogsol * Kdrag * rhodeq * w3r ** 2 * w2i ** 2 * w2r ** 2
            + 3 * w1r ** 2 * rhogsol * Kdrag * rhodeq * w2r * w3r * w2i ** 2 * w3i ** 2
            + 2 * w1r ** 2 * rhogsol * Kdrag * rhodeq * w3i ** 2 * w2i ** 2 * w2r ** 2
            + w1r ** 2 * rhogsol * Kdrag * rhodeq * w3i ** 2 * w2i ** 4
            + w1r ** 2 * rhogsol * Kdrag * rhodeq * w3i ** 4 * w2i ** 2
            + w1r ** 2 * rhogeq * k * Kdrag * vdsol * rhodeq * w3r * w2i ** 2 * w3i ** 2
            + w1r ** 2 * rhogeq * k * Kdrag * vdsol * rhodeq * w3r * w2r ** 2 * w3i ** 2
            + w1r ** 2 * rhogeq * k * Kdrag * vdsol * rhodeq * w2r ** 3 * w3i ** 2
            + w1r ** 2 * rhogeq * k * Kdrag * vdsol * rhodeq * w2r * w2i ** 2 * w3r ** 2
            + w1r ** 2 * rhogeq * k * Kdrag * vdsol * rhodeq * w2r * w2i ** 2 * w3i ** 2
            + w1r ** 2 * rhogsol * Kdrag * rhodeq * w3r ** 4 * w2i ** 2
            + w1r ** 2 * rhogsol * Kdrag * rhodeq * w3r ** 4 * w2r ** 2
            + 3 * w1r ** 2 * rhogsol * Kdrag * rhodeq * w3r ** 3 * w2r ** 3
            - 2
            * rhogsol
            * rhogeq
            * k ** 2
            * cs ** 2
            * w1i ** 3
            * rhodeq
            * w3i
            * w2i
            * w3r ** 2
            - 2
            * rhogsol
            * rhogeq
            * k ** 2
            * cs ** 2
            * w1i ** 3
            * rhodeq
            * w3i ** 3
            * w2i
            - 2
            * rhogsol
            * rhogeq
            * k ** 2
            * cs ** 2
            * w1i ** 3
            * rhodeq
            * w2i
            * w3i
            * w2r ** 2
            - 2
            * rhogsol
            * rhogeq
            * k ** 2
            * cs ** 2
            * w1i ** 3
            * rhodeq
            * w2i ** 3
            * w3i
            + 2
            * rhogsol
            * rhogeq
            * k ** 2
            * cs ** 2
            * w1i ** 3
            * rhodeq
            * w3r ** 2
            * w2r ** 2
            - 2
            * rhogsol
            * rhogeq
            * k ** 2
            * cs ** 2
            * w1i ** 3
            * rhodeq
            * w3i ** 2
            * w2i ** 2
            + w1r ** 2 * rhogeq * k * Kdrag * vdsol * rhodeq * w3r ** 3 * w2i ** 2
            + w1r ** 2 * rhogeq * k * Kdrag * vdsol * rhodeq * w3r ** 3 * w2r ** 2
            + w1r ** 2 * rhogeq * k * Kdrag * vdsol * rhodeq * w2r ** 3 * w3r ** 2
            + rhogsol * rhogeq * w1i ** 3 * rhodeq * w2r * w3r * w2i ** 2 * w3i ** 2
            - rhogsol * rhogeq * w1i ** 3 * rhodeq * w2i * w2r ** 2 * w3i ** 3
            - rhogsol * rhogeq * w1i ** 3 * rhodeq * w2i ** 3 * w3i ** 3
            + w1i ** 4 * rhogsol * Kdrag * rhodeq * w3r ** 2 * w2i ** 2
            + rhogsol * rhogeq * w1i ** 3 * rhodeq * w2r * w3r ** 3 * w2i ** 2
            - rhogsol * rhogeq * w1i ** 3 * rhodeq * w2i ** 3 * w3i * w3r ** 2
            - rhogsol * rhogeq * w1i ** 3 * rhodeq * w2i * w3i * w2r ** 2 * w3r ** 2
            + rhogsol * rhogeq * w1i ** 3 * rhodeq * w3r ** 3 * w2r ** 3
            + rhogsol * rhogeq * w1i ** 3 * rhodeq * w3r * w2r ** 3 * w3i ** 2
            + w1i ** 4 * rhogsol * Kdrag * rhodeq * w3r ** 2 * w2r ** 2
            + w1i ** 4 * rhogsol * Kdrag * rhodeq * w3i ** 2 * w2i ** 2
            + w1i ** 4 * rhogsol * Kdrag * rhodeq * w2r ** 2 * w3i ** 2
            + w1r * Kdrag * rhogeq * k * vgsol * w1i ** 2 * rhodeq * w3r ** 2 * w2i ** 2
            + w1r * Kdrag * rhogeq * k * vgsol * w1i ** 2 * rhodeq * w3r ** 2 * w2r ** 2
            + w1r * Kdrag * rhogeq * k * vgsol * w1i ** 2 * rhodeq * w3i ** 2 * w2i ** 2
            + w1r * Kdrag * rhogeq * k * vgsol * w1i ** 2 * rhodeq * w2r ** 2 * w3i ** 2
            + w1i * vdsol * k * Kdrag ** 2 * rhodeq * w3r ** 3 * w2r ** 2
            + w1i * vdsol * k * Kdrag ** 2 * rhodeq * w2r ** 3 * w3r ** 2
            + w1i * vdsol * k * Kdrag ** 2 * rhodeq * w2r * w2i ** 2 * w3r ** 2
            + w1i * vdsol * k * Kdrag ** 2 * rhodeq * w3r * w2i ** 2 * w3i ** 2
            + w1i * vdsol * k * Kdrag ** 2 * rhodeq * w3r * w2r ** 2 * w3i ** 2
            + w1i * vdsol * k * Kdrag ** 2 * rhodeq * w2r ** 3 * w3i ** 2
        )

        rhod1i = rhod1i - (
            w1i * vdsol * k * Kdrag ** 2 * rhodeq * w2r * w2i ** 2 * w3i ** 2
            + Kdrag
            * cs ** 2
            * k ** 2
            * rhogsol
            * w1i ** 2
            * rhodeq
            * w2r ** 2
            * w3i ** 2
            - rhogsol * w1r ** 3 * rhogeq * rhodeq * w2r * w2i ** 2 * w3i ** 3
            - rhogsol * w1r ** 3 * rhogeq * rhodeq * w3i ** 3 * w2r ** 3
            + w1i * vdsol * k * Kdrag ** 2 * rhodeq * w3r ** 3 * w2i ** 2
            - rhogsol * w1r ** 3 * rhogeq * rhodeq * w3r * w2r ** 2 * w2i * w3i ** 2
            + Kdrag
            * cs ** 2
            * k ** 2
            * rhogsol
            * w1i ** 2
            * rhodeq
            * w3r ** 2
            * w2i ** 2
            + Kdrag
            * cs ** 2
            * k ** 2
            * rhogsol
            * w1i ** 2
            * rhodeq
            * w3r ** 2
            * w2r ** 2
            + Kdrag
            * cs ** 2
            * k ** 2
            * rhogsol
            * w1i ** 2
            * rhodeq
            * w3i ** 2
            * w2i ** 2
            - w1i ** 2 * Kdrag * rhogeq * rhodsol * w3i ** 4 * w2i ** 2
            - w1i ** 2 * Kdrag * rhogeq * rhodsol * w3i ** 4 * w2r ** 2
            - 4 * w1i ** 2 * Kdrag * rhogeq * rhodsol * w2i ** 3 * w3i ** 3
            - rhogsol * w1r ** 3 * rhogeq * rhodeq * w3r ** 3 * w2i * w2r ** 2
            - 2 * w1i ** 2 * Kdrag * rhogeq * rhodsol * w3r ** 2 * w2r ** 2 * w3i ** 2
            - w1i ** 2 * Kdrag * rhogeq * rhodsol * w2r ** 4 * w3i ** 2
            - 2 * w1i ** 2 * Kdrag * rhogeq * rhodsol * w3i ** 2 * w2i ** 2 * w2r ** 2
            - 4 * w1i ** 2 * Kdrag * rhogeq * rhodsol * w2i * w2r ** 2 * w3i ** 3
            - 4 * w1i ** 2 * Kdrag * rhogeq * rhodsol * w2i * w3i * w2r ** 2 * w3r ** 2
            - w1i ** 2 * Kdrag * rhogeq * rhodsol * w3i ** 2 * w2i ** 4
            - rhogsol * w1r ** 3 * rhogeq * rhodeq * w2i ** 3 * w3r ** 3
            - rhogsol * w1r ** 3 * rhogeq * rhodeq * w3i * w2i ** 2 * w3r ** 2 * w2r
            - rhogsol * w1r ** 3 * rhogeq * rhodeq * w3i * w2r ** 3 * w3r ** 2
            - rhogsol * w1r ** 3 * rhogeq * rhodeq * w3r * w2i ** 3 * w3i ** 2
            - 2 * w1i ** 2 * Kdrag * rhogeq * rhodsol * w3r ** 2 * w2i ** 2 * w2r ** 2
            - 4 * w1i ** 2 * Kdrag * rhogeq * rhodsol * w2i ** 3 * w3i * w3r ** 2
            - w1i ** 2 * Kdrag * rhogeq * rhodsol * w3r ** 2 * w2i ** 4
            - 2 * w1i ** 2 * Kdrag * rhogeq * rhodsol * w3r ** 2 * w2i ** 2 * w3i ** 2
            - w1i ** 2 * Kdrag * rhogeq * rhodsol * w3r ** 4 * w2r ** 2
            - w1i ** 2 * Kdrag * rhogeq * rhodsol * w3r ** 2 * w2r ** 4
            + 2 * w1i * Kdrag * rhogeq * rhodsol * w2r ** 4 * w3r ** 2 * w3i
            + 4 * w1i * Kdrag * rhogeq * rhodsol * w2r ** 2 * w3i * w3r ** 2 * w2i ** 2
            + 2 * w1i * Kdrag * rhogeq * rhodsol * w2r ** 4 * w3i ** 3
            + 2 * w1i * Kdrag * rhogeq * rhodsol * w2i ** 4 * w3i ** 3
            + 2 * w1i * Kdrag * rhogeq * rhodsol * w2r ** 2 * w2i * w3r ** 4
            + 4 * w1i * Kdrag * rhogeq * rhodsol * w3i ** 2 * w2r ** 2 * w2i * w3r ** 2
            + 4 * w1i * Kdrag * rhogeq * rhodsol * w2i ** 3 * w3i ** 2 * w3r ** 2
            - 2 * w1i * rhogsol * rhogeq * rhodeq * w3i ** 4 * w2i ** 2 * w2r ** 2
            - w1i * rhogsol * rhogeq * rhodeq * w2r ** 4 * w3i ** 4
            - w1i * rhogsol * rhogeq * rhodeq * w3i ** 4 * w2i ** 4
            + 2 * w1i * Kdrag * rhogeq * rhodsol * w2i ** 3 * w3r ** 4
            - 2 * w1i * rhogsol * rhogeq * rhodeq * w2r ** 4 * w3i ** 2 * w3r ** 2
            + 2 * w1i * Kdrag * rhogeq * rhodsol * w2i ** 4 * w3r ** 2 * w3i
            + 2 * w1i * Kdrag * rhogeq * rhodsol * w2i * w2r ** 2 * w3i ** 4
            + 2 * w1i * Kdrag * rhogeq * rhodsol * w3i ** 4 * w2i ** 3
            + 4 * w1i * Kdrag * rhogeq * rhodsol * w2r ** 2 * w2i ** 2 * w3i ** 3
            - w1i ** 2 * Kdrag * rhogeq * rhodsol * w3r ** 4 * w2i ** 2
            + 2 * w1r ** 2 * rhogsol * rhogeq * rhodeq * w2r ** 2 * w2i ** 2 * w3i ** 3
            - w1i * rhogsol * rhogeq * rhodeq * w3r ** 4 * w2i ** 4
            + w1r ** 2 * rhogsol * rhogeq * rhodeq * w2i ** 4 * w3r ** 2 * w3i
            + w1r ** 2 * rhogsol * rhogeq * rhodeq * w2r ** 4 * w3r ** 2 * w3i
            + w1r ** 2 * rhogsol * rhogeq * rhodeq * w2i * w2r ** 2 * w3i ** 4
            + w1r ** 2 * rhogsol * rhogeq * rhodeq * w2i ** 4 * w3i ** 3
            + 2
            * w1r ** 2
            * rhogsol
            * rhogeq
            * rhodeq
            * w3i ** 2
            * w2r ** 2
            * w2i
            * w3r ** 2
            + 2
            * w1r ** 2
            * rhogsol
            * rhogeq
            * rhodeq
            * w2r ** 2
            * w3i
            * w3r ** 2
            * w2i ** 2
            - 2
            * w1r
            * vgsol
            * k ** 3
            * cs ** 2
            * rhogeq ** 2
            * rhodeq
            * w3i
            * w2r ** 2
            * w2i ** 2
            + 2 * w1r ** 2 * rhogsol * rhogeq * rhodeq * w2i ** 3 * w3i ** 2 * w3r ** 2
            + w1r ** 2 * rhogsol * rhogeq * rhodeq * w3i ** 4 * w2i ** 3
            - w1i * rhogsol * rhogeq * rhodeq * w3r ** 4 * w2r ** 4
            - 2 * w1i * rhogsol * rhogeq * rhodeq * w3r ** 4 * w2i ** 2 * w2r ** 2
            - 4
            * w1i
            * rhogsol
            * rhogeq
            * rhodeq
            * w3r ** 2
            * w2i ** 2
            * w2r ** 2
            * w3i ** 2
            - 2 * w1i * rhogsol * rhogeq * rhodeq * w3i ** 2 * w2i ** 4 * w3r ** 2
            - w1r * vgsol * k ** 3 * cs ** 2 * rhogeq ** 2 * rhodeq * w3i * w2i ** 4
            - 2
            * w1r
            * vgsol
            * k ** 3
            * cs ** 2
            * rhogeq ** 2
            * rhodeq
            * w2i ** 3
            * w3i ** 2
            - 2
            * w1r
            * vgsol
            * k ** 3
            * cs ** 2
            * rhogeq ** 2
            * rhodeq
            * w2i ** 2
            * w3i ** 3
            - 2
            * w1r
            * vgsol
            * k ** 3
            * cs ** 2
            * rhogeq ** 2
            * rhodeq
            * w2r
            * w3r
            * w2i
            * w3i ** 2
            - w1r * vgsol * k ** 3 * cs ** 2 * rhogeq ** 2 * rhodeq * w3i ** 4 * w2i
            - w1r * vgsol * k ** 3 * cs ** 2 * rhogeq ** 2 * rhodeq * w3i * w2r ** 4
            - 2
            * w1r
            * vgsol
            * k ** 3
            * cs ** 2
            * rhogeq ** 2
            * rhodeq
            * w3i
            * w3r
            * w2i ** 2
            * w2r
            - 2
            * w1r
            * vgsol
            * k ** 3
            * cs ** 2
            * rhogeq ** 2
            * rhodeq
            * w3i
            * w3r
            * w2r ** 3
            - 2
            * w1r
            * vgsol
            * k ** 3
            * cs ** 2
            * rhogeq ** 2
            * rhodeq
            * w2i
            * w2r ** 2
            * w3i ** 2
            - 2
            * w1r
            * vgsol
            * k ** 3
            * cs ** 2
            * rhogeq ** 2
            * rhodeq
            * w3i ** 2
            * w2i
            * w3r ** 2
            - 2
            * w1r
            * vgsol
            * k ** 3
            * cs ** 2
            * rhogeq ** 2
            * rhodeq
            * w3r ** 2
            * w2i ** 2
            * w3i
            + w1r ** 2 * rhogsol * rhogeq * rhodeq * w2i ** 3 * w3r ** 4
            - 2
            * w1r
            * vgsol
            * k ** 3
            * cs ** 2
            * rhogeq ** 2
            * rhodeq
            * w2r
            * w3r ** 3
            * w2i
            - vgsol * Kdrag * rhogeq * k ** 3 * cs ** 2 * w1i ** 3 * rhodeq * w2r * w3i
            - w1r * rhogsol * rhogeq * w1i ** 2 * rhodeq * w3i ** 3 * w2r ** 3
            + rhogsol * w1r * k ** 4 * cs ** 4 * rhogeq * w1i ** 2 * rhodeq * w2i * w3r
            - w1r * rhogsol * rhogeq * w1i ** 2 * rhodeq * w3i * w2r ** 3 * w3r ** 2
            - w1r * rhogsol * rhogeq * w1i ** 2 * rhodeq * w3r * w2i ** 3 * w3i ** 2
            - w1r
            * rhogsol
            * rhogeq
            * w1i ** 2
            * rhodeq
            * w3r
            * w2r ** 2
            * w2i
            * w3i ** 2
            - w1r * rhogsol * rhogeq * w1i ** 2 * rhodeq * w3r ** 3 * w2i * w2r ** 2
            - w1r * rhogsol * rhogeq * w1i ** 2 * rhodeq * w2i ** 3 * w3r ** 3
            + rhogsol * rhogeq * k ** 2 * cs ** 2 * w1i ** 4 * rhodeq * w2r ** 2 * w3i
            - w1r
            * rhogsol
            * rhogeq
            * w1i ** 2
            * rhodeq
            * w3i
            * w2i ** 2
            * w3r ** 2
            * w2r
            - w1r * rhogsol * rhogeq * w1i ** 2 * rhodeq * w2r * w2i ** 2 * w3i ** 3
        )

        rhod1i = rhod1i - (
            rhogsol * w1r * k ** 4 * cs ** 4 * rhogeq * w1i ** 2 * rhodeq * w2r * w3i
            - vgsol * Kdrag * rhogeq * k ** 3 * cs ** 2 * w1i ** 3 * rhodeq * w2i * w3r
            - w1i * rhogsol * Kdrag * rhodeq * w2i * w2r ** 2 * w3i ** 4
            - w1i * rhogsol * Kdrag * rhodeq * w2i ** 4 * w3i ** 3
            - w1i * rhogsol * Kdrag * rhodeq * w3i ** 4 * w2i ** 3
            + rhogsol * rhogeq * k ** 2 * cs ** 2 * w1i ** 4 * rhodeq * w2i * w3r ** 2
            - 2 * w1i * rhogsol * Kdrag * rhodeq * w2i ** 3 * w3i ** 2 * w3r ** 2
            - w1i * rhogsol * Kdrag * rhodeq * w2i ** 4 * w3r ** 2 * w3i
            - w1i * rhogsol * Kdrag * rhodeq * w2r ** 4 * w3r ** 2 * w3i
            - w1i * rhogsol * Kdrag * rhodeq * w2r ** 4 * w3i ** 3
            - 2 * w1i * rhogsol * Kdrag * rhodeq * w3i ** 2 * w2r ** 2 * w2i * w3r ** 2
            - 2 * w1i * rhogsol * Kdrag * rhodeq * w2r ** 2 * w3i * w3r ** 2 * w2i ** 2
            - 2 * w1i * rhogsol * Kdrag * rhodeq * w2r ** 2 * w2i ** 2 * w3i ** 3
            + rhogsol * rhogeq * k ** 2 * cs ** 2 * w1i ** 4 * rhodeq * w2i ** 2 * w3i
            + rhogsol * rhogeq * k ** 2 * cs ** 2 * w1i ** 4 * rhodeq * w2i * w3i ** 2
            + rhogsol * rhogeq * k ** 4 * cs ** 4 * w1r ** 3 * rhodeq * w2i * w3r
            + rhogsol * rhogeq * k ** 4 * cs ** 4 * w1r ** 3 * rhodeq * w2r * w3i
            + rhogsol * Kdrag * w1r ** 4 * rhodeq * w3r ** 2 * w2i ** 2
            + rhogsol * Kdrag * w1r ** 4 * rhodeq * w3r ** 2 * w2r ** 2
            + rhogsol * Kdrag * w1r ** 4 * rhodeq * w3i ** 2 * w2i ** 2
            + rhogsol * Kdrag * w1r ** 4 * rhodeq * w2r ** 2 * w3i ** 2
            - w1i * rhogsol * Kdrag * rhodeq * w2i ** 3 * w3r ** 4
            - w1i * rhogsol * Kdrag * rhodeq * w2r ** 2 * w2i * w3r ** 4
            - vgsol * k ** 3 * cs ** 2 * rhogeq ** 2 * w1r ** 4 * rhodeq * w2i * w3r
            - vgsol * k ** 3 * cs ** 2 * rhogeq ** 2 * w1r ** 4 * rhodeq * w2r * w3i
            + 2 * w1r ** 3 * Kdrag * rhogeq * rhodsol * w3r * w2i ** 2 * w3i ** 2
            + 2 * w1r ** 3 * Kdrag * rhogeq * rhodsol * w3r * w2r ** 2 * w3i ** 2
            + 2 * w1r ** 3 * Kdrag * rhogeq * rhodsol * w2r ** 3 * w3i ** 2
            + 2 * w1r ** 3 * Kdrag * rhogeq * rhodsol * w2r * w2i ** 2 * w3i ** 2
            + 2 * w1r ** 3 * Kdrag * rhogeq * rhodsol * w3r ** 3 * w2i ** 2
            + 2 * w1r ** 3 * Kdrag * rhogeq * rhodsol * w3r ** 3 * w2r ** 2
            + 2 * w1r ** 3 * Kdrag * rhogeq * rhodsol * w2r ** 3 * w3r ** 2
            + 2 * w1r ** 3 * Kdrag * rhogeq * rhodsol * w2r * w2i ** 2 * w3r ** 2
            + 2 * w1r * w1i ** 2 * Kdrag * rhogeq * rhodsol * w3r ** 3 * w2r ** 2
            + 2 * w1r * w1i ** 2 * Kdrag * rhogeq * rhodsol * w2r ** 3 * w3r ** 2
            + 2 * w1r * w1i ** 2 * Kdrag * rhogeq * rhodsol * w2r * w2i ** 2 * w3r ** 2
            + 2 * w1i ** 3 * Kdrag * rhogeq * rhodsol * w2i ** 2 * w3i ** 3
            + 2 * w1i ** 3 * Kdrag * rhogeq * rhodsol * w2i ** 3 * w3i ** 2
            + 2 * w1r * w1i ** 2 * Kdrag * rhogeq * rhodsol * w3r ** 3 * w2i ** 2
            + 2 * w1i ** 3 * Kdrag * rhogeq * rhodsol * w3r ** 2 * w2i ** 3
            + 2 * w1i ** 3 * Kdrag * rhogeq * rhodsol * w3r ** 2 * w2r ** 2 * w3i
            + 2 * w1i ** 3 * Kdrag * rhogeq * rhodsol * w3r ** 2 * w2i ** 2 * w3i
            + 2 * w1i ** 3 * Kdrag * rhogeq * rhodsol * w2i * w2r ** 2 * w3i ** 2
            + 2 * w1i ** 3 * Kdrag * rhogeq * rhodsol * w3r ** 2 * w2i * w2r ** 2
            + 2 * w1i ** 3 * Kdrag * rhogeq * rhodsol * w2r ** 2 * w3i ** 3
            + 2 * w1r * w1i ** 2 * Kdrag * rhogeq * rhodsol * w3r * w2i ** 2 * w3i ** 2
            + 2 * w1r * w1i ** 2 * Kdrag * rhogeq * rhodsol * w3r * w2r ** 2 * w3i ** 2
            + 2 * w1r * w1i ** 2 * Kdrag * rhogeq * rhodsol * w2r ** 3 * w3i ** 2
            + 2 * w1r * w1i ** 2 * Kdrag * rhogeq * rhodsol * w2r * w2i ** 2 * w3i ** 2
            + 2 * w1i * w1r ** 2 * Kdrag * rhogeq * rhodsol * w2i ** 2 * w3i ** 3
            + 2 * w1i * w1r ** 2 * Kdrag * rhogeq * rhodsol * w2i * w2r ** 2 * w3i ** 2
            + 2 * w1i * w1r ** 2 * Kdrag * rhogeq * rhodsol * w2i ** 3 * w3i ** 2
            + 2 * w1i * w1r ** 2 * Kdrag * rhogeq * rhodsol * w2r ** 2 * w3i ** 3
            - 2 * w1r ** 2 * w1i ** 2 * Kdrag * rhogeq * rhodsol * w3r ** 2 * w2i ** 2
            - 2 * w1r ** 2 * w1i ** 2 * Kdrag * rhogeq * rhodsol * w3r ** 2 * w2r ** 2
            - 2 * w1r ** 2 * w1i ** 2 * Kdrag * rhogeq * rhodsol * w3i ** 2 * w2i ** 2
            - 2 * w1r ** 2 * w1i ** 2 * Kdrag * rhogeq * rhodsol * w2r ** 2 * w3i ** 2
            + 2 * w1i * w1r ** 2 * Kdrag * rhogeq * rhodsol * w3r ** 2 * w2i * w2r ** 2
            + 2 * w1i * w1r ** 2 * Kdrag * rhogeq * rhodsol * w3r ** 2 * w2i ** 3
            + 2 * w1i * w1r ** 2 * Kdrag * rhogeq * rhodsol * w3r ** 2 * w2r ** 2 * w3i
            + 2 * w1i * w1r ** 2 * Kdrag * rhogeq * rhodsol * w3r ** 2 * w2i ** 2 * w3i
            - 2 * w1r ** 2 * Kdrag * rhogeq * rhodsol * w3r ** 2 * w2i ** 2 * w3i ** 2
            - 2 * w1r ** 2 * Kdrag * rhogeq * rhodsol * w3r ** 2 * w2r ** 2 * w3i ** 2
            - 4 * w1r ** 2 * Kdrag * rhogeq * rhodsol * w3r * w2r ** 3 * w3i ** 2
            - w1r ** 2 * Kdrag * rhogeq * rhodsol * w2r ** 4 * w3i ** 2
            - 2 * w1r ** 2 * Kdrag * rhogeq * rhodsol * w3i ** 2 * w2i ** 2 * w2r ** 2
            - w1r ** 2 * Kdrag * rhogeq * rhodsol * w3i ** 2 * w2i ** 4
            - w1r ** 2 * Kdrag * rhogeq * rhodsol * w3i ** 4 * w2i ** 2
            - w1r ** 2 * Kdrag * rhogeq * rhodsol * w3i ** 4 * w2r ** 2
            - w1r ** 2 * Kdrag * rhogeq * rhodsol * w3r ** 2 * w2r ** 4
            - 2 * w1r ** 2 * Kdrag * rhogeq * rhodsol * w3r ** 2 * w2i ** 2 * w2r ** 2
            - w1r ** 2 * Kdrag * rhogeq * rhodsol * w3r ** 2 * w2i ** 4
            - 4 * w1r ** 2 * Kdrag * rhogeq * rhodsol * w2r * w3r * w2i ** 2 * w3i ** 2
            - w1r ** 2 * Kdrag * rhogeq * rhodsol * w3r ** 4 * w2i ** 2
            - w1r ** 2 * Kdrag * rhogeq * rhodsol * w3r ** 4 * w2r ** 2
            - 4 * w1r ** 2 * Kdrag * rhogeq * rhodsol * w2r * w3r ** 3 * w2i ** 2
            - 4 * w1r ** 2 * Kdrag * rhogeq * rhodsol * w3r ** 3 * w2r ** 3
            + 2 * w1r * Kdrag * rhogeq * rhodsol * w2r * w3i ** 4 * w2i ** 2
        )

        rhod1i = rhod1i - (
            4 * w1r * Kdrag * rhogeq * rhodsol * w3r ** 3 * w2r ** 2 * w2i ** 2
            + 4 * w1r * Kdrag * rhogeq * rhodsol * w2r ** 3 * w3i ** 2 * w3r ** 2
            + 4 * w1r * Kdrag * rhogeq * rhodsol * w2r * w3i ** 2 * w2i ** 2 * w3r ** 2
            + 2 * w1r * Kdrag * rhogeq * rhodsol * w2r * w2i ** 2 * w3r ** 4
            + 2 * w1r * Kdrag * rhogeq * rhodsol * w3r ** 3 * w2i ** 4
            - 4 * w1i * w1r * Kdrag * rhogeq * rhodsol * w3r * w2r ** 2 * w2i * w3i ** 2
            - 4 * w1i * w1r * Kdrag * rhogeq * rhodsol * w2r * w2i ** 2 * w3i ** 3
            - 4 * w1i * w1r * Kdrag * rhogeq * rhodsol * w3i ** 3 * w2r ** 3
            - 4 * w1i * w1r * Kdrag * rhogeq * rhodsol * w3r * w2i ** 3 * w3i ** 2
            + 2 * w1r * Kdrag * rhogeq * rhodsol * w2r ** 3 * w3r ** 4
            + 2 * w1r * Kdrag * rhogeq * rhodsol * w3r ** 3 * w2r ** 4
            + 2 * w1r * Kdrag * rhogeq * rhodsol * w3r * w2r ** 4 * w3i ** 2
            + 2 * w1r * Kdrag * rhogeq * rhodsol * w3r * w2i ** 4 * w3i ** 2
            + 4 * w1r * Kdrag * rhogeq * rhodsol * w3r * w2r ** 2 * w3i ** 2 * w2i ** 2
            + 2 * w1r * Kdrag * rhogeq * rhodsol * w2r ** 3 * w3i ** 4
            - 4 * w1i * w1r * Kdrag * rhogeq * rhodsol * w2i ** 3 * w3r ** 3
            - 4 * w1i * w1r * Kdrag * rhogeq * rhodsol * w3i * w2i ** 2 * w3r ** 2 * w2r
            - 4 * w1i * w1r * Kdrag * rhogeq * rhodsol * w3i * w2r ** 3 * w3r ** 2
            - 4 * w1i * w1r * Kdrag * rhogeq * rhodsol * w3r ** 3 * w2i * w2r ** 2
            - vdsol * Kdrag * k * rhogeq * w1r ** 3 * rhodeq * w3i ** 2 * w2i ** 2
            - vdsol * Kdrag * k * rhogeq * w1r ** 3 * rhodeq * w2r ** 2 * w3i ** 2
            + vdsol * Kdrag * k ** 3 * cs ** 2 * rhogeq * w1r ** 3 * rhodeq * w2i * w3i
            - vdsol * Kdrag * k * rhogeq * w1r ** 3 * rhodeq * w3r ** 2 * w2i ** 2
            - vdsol * Kdrag * k ** 3 * cs ** 2 * rhogeq * w1r ** 3 * rhodeq * w3r * w2r
            - vdsol * Kdrag * k * rhogeq * w1r ** 3 * rhodeq * w3r ** 2 * w2r ** 2
            - cs ** 2
            * k ** 2
            * rhogsol
            * Kdrag
            * w1r ** 2
            * rhodeq
            * w3r ** 2
            * w2i ** 2
            - cs ** 2
            * k ** 2
            * rhogsol
            * Kdrag
            * w1r ** 2
            * rhodeq
            * w3r ** 2
            * w2r ** 2
            - cs ** 2
            * k ** 2
            * rhogsol
            * Kdrag
            * w1r ** 2
            * rhodeq
            * w3i ** 2
            * w2i ** 2
            - cs ** 2
            * k ** 2
            * rhogsol
            * Kdrag
            * w1r ** 2
            * rhodeq
            * w2r ** 2
            * w3i ** 2
            - 2
            * rhogsol
            * rhogeq
            * k ** 2
            * cs ** 2
            * w1r ** 3
            * rhodeq
            * w3r
            * w2i
            * w3i ** 2
            - 2
            * rhogsol
            * rhogeq
            * k ** 2
            * cs ** 2
            * w1r ** 3
            * rhodeq
            * w3i
            * w3r
            * w2i ** 2
            - 2
            * rhogsol
            * rhogeq
            * k ** 2
            * cs ** 2
            * w1r ** 3
            * rhodeq
            * w2r ** 3
            * w3i
            - 2
            * rhogsol
            * rhogeq
            * k ** 2
            * cs ** 2
            * w1r ** 3
            * rhodeq
            * w3r ** 3
            * w2i
            + 2
            * vgsol
            * Kdrag
            * rhogeq
            * k ** 3
            * cs ** 2
            * w1r ** 2
            * rhodeq
            * w2r
            * w2i
            * w3i
            - 2
            * rhogsol
            * rhogeq
            * k ** 2
            * cs ** 2
            * w1r ** 3
            * rhodeq
            * w2r
            * w2i
            * w3r ** 2
            - 2
            * rhogsol
            * rhogeq
            * k ** 2
            * cs ** 2
            * w1r ** 3
            * rhodeq
            * w3i
            * w3r
            * w2r ** 2
            - 2
            * rhogsol
            * rhogeq
            * k ** 2
            * cs ** 2
            * w1r ** 3
            * rhodeq
            * w2r
            * w3i
            * w2i ** 2
            - 2
            * rhogsol
            * rhogeq
            * k ** 2
            * cs ** 2
            * w1r ** 3
            * rhodeq
            * w2r
            * w2i
            * w3i ** 2
            - 2
            * vdsol
            * Kdrag
            * rhogeq
            * k ** 3
            * cs ** 2
            * w1r ** 2
            * rhodeq
            * w3r
            * w2i
            * w3i
            - 2
            * vdsol
            * Kdrag
            * rhogeq
            * k ** 3
            * cs ** 2
            * w1r ** 2
            * rhodeq
            * w2r
            * w2i
            * w3i
            - 2
            * vgsol
            * Kdrag
            * rhogeq
            * k ** 3
            * cs ** 2
            * w1r ** 2
            * rhodeq
            * w2r
            * w3r ** 2
            - 2
            * vgsol
            * Kdrag
            * rhogeq
            * k ** 3
            * cs ** 2
            * w1r ** 2
            * rhodeq
            * w2r ** 2
            * w3r
            + 2
            * vgsol
            * Kdrag
            * rhogeq
            * k ** 3
            * cs ** 2
            * w1r ** 2
            * rhodeq
            * w3r
            * w2i
            * w3i
            + 2
            * vdsol
            * Kdrag
            * rhogeq
            * k ** 3
            * cs ** 2
            * w1r ** 2
            * rhodeq
            * w2r
            * w3r ** 2
            + 2
            * vdsol
            * Kdrag
            * rhogeq
            * k ** 3
            * cs ** 2
            * w1r ** 2
            * rhodeq
            * w2r ** 2
            * w3r
            - 2
            * rhogsol
            * k ** 4
            * cs ** 4
            * rhogeq
            * w1r ** 2
            * rhodeq
            * w3r
            * w2r
            * w2i
            - 2
            * rhogsol
            * k ** 4
            * cs ** 4
            * rhogeq
            * w1r ** 2
            * rhodeq
            * w3r
            * w2r
            * w3i
            - 2
            * rhogsol
            * k ** 4
            * cs ** 4
            * rhogeq
            * w1r ** 2
            * rhodeq
            * w2r ** 2
            * w3i
            - vgsol
            * k ** 3
            * cs ** 2
            * rhogeq ** 2
            * w1r ** 3
            * rhodeq
            * w2i
            * w3i ** 2
            - 2
            * rhogsol
            * k ** 4
            * cs ** 4
            * rhogeq
            * w1r ** 2
            * rhodeq
            * w2i
            * w3r ** 2
            + 2
            * vgsol
            * k ** 3
            * cs ** 2
            * rhogeq ** 2
            * w1r ** 3
            * rhodeq
            * w3r
            * w2r
            * w3i
            + 2
            * vgsol
            * k ** 3
            * cs ** 2
            * rhogeq ** 2
            * w1r ** 3
            * rhodeq
            * w3r
            * w2r
            * w2i
            + vgsol
            * k ** 3
            * cs ** 2
            * rhogeq ** 2
            * w1r ** 3
            * rhodeq
            * w2r ** 2
            * w3i
            + w1i
            * vdsol
            * Kdrag
            * rhogeq
            * k ** 3
            * cs ** 2
            * w1r ** 2
            * rhodeq
            * w2r
            * w3i
        )

        rhod1i = rhod1i - (
            w1i
            * vdsol
            * Kdrag
            * rhogeq
            * k ** 3
            * cs ** 2
            * w1r ** 2
            * rhodeq
            * w2i
            * w3r
            + vgsol
            * k ** 3
            * cs ** 2
            * rhogeq ** 2
            * w1r ** 3
            * rhodeq
            * w2i
            * w3r ** 2
            - vgsol
            * k ** 3
            * cs ** 2
            * rhogeq ** 2
            * w1r ** 3
            * rhodeq
            * w2i ** 2
            * w3i
            - w1i
            * vgsol
            * Kdrag
            * rhogeq
            * k ** 3
            * cs ** 2
            * w1r ** 2
            * rhodeq
            * w2i
            * w3r
            - w1i
            * vgsol
            * Kdrag
            * rhogeq
            * k ** 3
            * cs ** 2
            * w1r ** 2
            * rhodeq
            * w2r
            * w3i
            - w1r * rhogsol * Kdrag * rhodeq * w2r ** 3 * w3i ** 4
            - w1r * rhogsol * Kdrag * rhodeq * w2r * w3i ** 4 * w2i ** 2
            - w1r * rhogsol * Kdrag * rhodeq * w3r * w2r ** 4 * w3i ** 2
            - w1r * rhogsol * Kdrag * rhodeq * w3r * w2i ** 4 * w3i ** 2
            - 2 * w1r * rhogsol * Kdrag * rhodeq * w3r * w2r ** 2 * w3i ** 2 * w2i ** 2
            - 2 * w1r * rhogsol * Kdrag * rhodeq * w3r ** 3 * w2r ** 2 * w2i ** 2
            - 2 * w1r * rhogsol * Kdrag * rhodeq * w2r ** 3 * w3i ** 2 * w3r ** 2
            - 2 * w1r * rhogsol * Kdrag * rhodeq * w2r * w3i ** 2 * w2i ** 2 * w3r ** 2
            - w1r * rhogsol * Kdrag * rhodeq * w2r ** 3 * w3r ** 4
            - w1r * rhogsol * Kdrag * rhodeq * w2r * w2i ** 2 * w3r ** 4
            - w1r * rhogsol * Kdrag * rhodeq * w3r ** 3 * w2i ** 4
            - w1r * rhogsol * Kdrag * rhodeq * w3r ** 3 * w2r ** 4
            - 2 * w1r * w1i ** 2 * rhogsol * Kdrag * rhodeq * w2r * w2i ** 2 * w3r ** 2
            - 2 * w1r * w1i ** 2 * rhogsol * Kdrag * rhodeq * w3r * w2i ** 2 * w3i ** 2
            - 2 * w1r * w1i ** 2 * rhogsol * Kdrag * rhodeq * w2r ** 3 * w3i ** 2
            - 2 * w1r * w1i ** 2 * rhogsol * Kdrag * rhodeq * w3r ** 3 * w2i ** 2
            - 2 * w1r * w1i ** 2 * rhogsol * Kdrag * rhodeq * w3r ** 3 * w2r ** 2
            + w1r
            * cs ** 2
            * k ** 2
            * rhogsol
            * Kdrag
            * rhodeq
            * w3r
            * w2r ** 2
            * w3i ** 2
            + w1r * cs ** 2 * k ** 2 * rhogsol * Kdrag * rhodeq * w2r ** 3 * w3i ** 2
            + w1r
            * cs ** 2
            * k ** 2
            * rhogsol
            * Kdrag
            * rhodeq
            * w2r
            * w2i ** 2
            * w3i ** 2
            + w1r
            * cs ** 2
            * k ** 2
            * rhogsol
            * Kdrag
            * rhodeq
            * w3r
            * w2i ** 2
            * w3i ** 2
            - 2 * w1r * w1i ** 2 * rhogsol * Kdrag * rhodeq * w2r ** 3 * w3r ** 2
            - 2 * w1r * w1i ** 2 * rhogsol * Kdrag * rhodeq * w3r * w2r ** 2 * w3i ** 2
            - 2 * w1r * w1i ** 2 * rhogsol * Kdrag * rhodeq * w2r * w2i ** 2 * w3i ** 2
            + w1r * cs ** 2 * k ** 2 * rhogsol * Kdrag * rhodeq * w3r ** 3 * w2r ** 2
            + w1r * cs ** 2 * k ** 2 * rhogsol * Kdrag * rhodeq * w2r ** 3 * w3r ** 2
            + w1r
            * cs ** 2
            * k ** 2
            * rhogsol
            * Kdrag
            * rhodeq
            * w2r
            * w2i ** 2
            * w3r ** 2
            - 2 * w1r ** 3 * rhogsol * Kdrag * rhodeq * w2r ** 3 * w3r ** 2
            - 2 * w1r ** 3 * rhogsol * Kdrag * rhodeq * w3r * w2i ** 2 * w3i ** 2
            - 2 * w1r ** 3 * rhogsol * Kdrag * rhodeq * w3r * w2r ** 2 * w3i ** 2
            - 2 * w1r ** 3 * rhogsol * Kdrag * rhodeq * w2r * w2i ** 2 * w3i ** 2
            - 2 * w1i ** 2 * vgsol * rhogeq * k * Kdrag * rhodeq * w3r ** 3 * w2i ** 2
            - 2 * w1i ** 2 * vgsol * rhogeq * k * Kdrag * rhodeq * w3r ** 3 * w2r ** 2
            - 2 * w1i ** 2 * vgsol * rhogeq * k * Kdrag * rhodeq * w2r ** 3 * w3r ** 2
            - 2
            * w1i ** 2
            * vgsol
            * rhogeq
            * k
            * Kdrag
            * rhodeq
            * w2r
            * w2i ** 2
            * w3r ** 2
            - 2
            * w1i ** 2
            * vgsol
            * rhogeq
            * k
            * Kdrag
            * rhodeq
            * w3r
            * w2i ** 2
            * w3i ** 2
            - 2
            * w1i ** 2
            * vgsol
            * rhogeq
            * k
            * Kdrag
            * rhodeq
            * w3r
            * w2r ** 2
            * w3i ** 2
            - 2 * w1i ** 2 * vgsol * rhogeq * k * Kdrag * rhodeq * w2r ** 3 * w3i ** 2
            - 2
            * w1i ** 2
            * vgsol
            * rhogeq
            * k
            * Kdrag
            * rhodeq
            * w2r
            * w2i ** 2
            * w3i ** 2
            + w1i * k * vgsol * rhogeq ** 2 * rhodeq * w2r * w3i ** 4 * w2i ** 2
            + w1i * k * vgsol * rhogeq ** 2 * rhodeq * w2r ** 3 * w3i ** 4
            + w1r
            * vgsol
            * Kdrag
            * k ** 3
            * cs ** 2
            * rhogeq
            * w1i ** 2
            * rhodeq
            * w3r
            * w2r
            - w1r
            * vgsol
            * Kdrag
            * k ** 3
            * cs ** 2
            * rhogeq
            * w1i ** 2
            * rhodeq
            * w2i
            * w3i
            - vgsol * k ** 3 * cs ** 2 * rhogeq ** 2 * w1i ** 4 * rhodeq * w2i * w3r
            - vgsol * k ** 3 * cs ** 2 * rhogeq ** 2 * w1i ** 4 * rhodeq * w2r * w3i
            + w1i * k * vgsol * rhogeq ** 2 * rhodeq * w3r * w2r ** 4 * w3i ** 2
            + w1i * k * vgsol * rhogeq ** 2 * rhodeq * w3r * w2i ** 4 * w3i ** 2
            + 2
            * w1i
            * k
            * vgsol
            * rhogeq ** 2
            * rhodeq
            * w3r
            * w2r ** 2
            * w3i ** 2
            * w2i ** 2
            + w1i * k * vgsol * rhogeq ** 2 * rhodeq * w2r ** 3 * w3r ** 4
            + w1i * k * vgsol * rhogeq ** 2 * rhodeq * w2r * w2i ** 2 * w3r ** 4
            + w1i * k * vgsol * rhogeq ** 2 * rhodeq * w3r ** 3 * w2i ** 4
            - vdsol * Kdrag * k * w1r * rhogeq * w1i ** 2 * rhodeq * w3i ** 2 * w2i ** 2
            - vdsol * Kdrag * k * w1r * rhogeq * w1i ** 2 * rhodeq * w2r ** 2 * w3i ** 2
            - vdsol * Kdrag * k * w1r * rhogeq * w1i ** 2 * rhodeq * w3r ** 2 * w2r ** 2
            + w1i * k * vgsol * rhogeq ** 2 * rhodeq * w3r ** 3 * w2r ** 4
            + 2
            * w1i
            * k
            * vgsol
            * rhogeq ** 2
            * rhodeq
            * w3r ** 3
            * w2r ** 2
            * w2i ** 2
            + 2
            * w1i
            * k
            * vgsol
            * rhogeq ** 2
            * rhodeq
            * w2r ** 3
            * w3i ** 2
            * w3r ** 2
            + 2
            * w1i
            * k
            * vgsol
            * rhogeq ** 2
            * rhodeq
            * w2r
            * w3i ** 2
            * w2i ** 2
            * w3r ** 2
            - 2 * w1i * w1r ** 2 * rhogsol * Kdrag * rhodeq * w2i ** 2 * w3i ** 3
            - 2 * w1i * w1r ** 2 * rhogsol * Kdrag * rhodeq * w2i * w2r ** 2 * w3i ** 2
            - 2 * w1i * w1r ** 2 * rhogsol * Kdrag * rhodeq * w2i ** 3 * w3i ** 2
            - 2 * w1i * w1r ** 2 * rhogsol * Kdrag * rhodeq * w2r ** 2 * w3i ** 3
            - 2
            * vgsol
            * w1r ** 2
            * k ** 3
            * cs ** 2
            * rhogeq ** 2
            * w1i ** 2
            * rhodeq
            * w2i
            * w3r
            - 2
            * vgsol
            * w1r ** 2
            * k ** 3
            * cs ** 2
            * rhogeq ** 2
            * w1i ** 2
            * rhodeq
            * w2r
            * w3i
            - vdsol * Kdrag * k * w1r * rhogeq * w1i ** 2 * rhodeq * w3r ** 2 * w2i ** 2
            - 2 * w1i * w1r ** 2 * rhogsol * Kdrag * rhodeq * w3r ** 2 * w2i * w2r ** 2
            - 2 * w1i * w1r ** 2 * rhogsol * Kdrag * rhodeq * w3r ** 2 * w2i ** 3
            - 2 * w1i * w1r ** 2 * rhogsol * Kdrag * rhodeq * w3r ** 2 * w2r ** 2 * w3i
            - 2 * w1i * w1r ** 2 * rhogsol * Kdrag * rhodeq * w3r ** 2 * w2i ** 2 * w3i
            - vgsol * k * w1r * rhogeq ** 2 * rhodeq * w3i ** 4 * w2i ** 3
            - 2
            * vgsol
            * k
            * w1r
            * rhogeq ** 2
            * rhodeq
            * w2r ** 2
            * w2i ** 2
            * w3i ** 3
            - 2
            * vgsol
            * k
            * w1r
            * rhogeq ** 2
            * rhodeq
            * w2r ** 2
            * w3i
            * w3r ** 2
            * w2i ** 2
            - vgsol * k * w1r * rhogeq ** 2 * rhodeq * w2r ** 4 * w3i ** 3
            - vgsol * k * w1r * rhogeq ** 2 * rhodeq * w2i ** 4 * w3i ** 3
            - vgsol * k * w1r * rhogeq ** 2 * rhodeq * w2i ** 4 * w3r ** 2 * w3i
            - vgsol * k * w1r * rhogeq ** 2 * rhodeq * w2r ** 4 * w3r ** 2 * w3i
            - vgsol * k * w1r * rhogeq ** 2 * rhodeq * w2i * w2r ** 2 * w3i ** 4
            - vgsol * k * w1r * rhogeq ** 2 * rhodeq * w2r ** 2 * w2i * w3r ** 4
            - 2
            * vgsol
            * k
            * w1r
            * rhogeq ** 2
            * rhodeq
            * w3i ** 2
            * w2r ** 2
            * w2i
            * w3r ** 2
            - 2
            * vgsol
            * k
            * w1r
            * rhogeq ** 2
            * rhodeq
            * w2i ** 3
            * w3i ** 2
            * w3r ** 2
            + rhogsol * w1i * k ** 4 * cs ** 4 * rhogeq * w1r ** 2 * rhodeq * w3r * w2r
            - rhogsol * w1i * k ** 4 * cs ** 4 * rhogeq * w1r ** 2 * rhodeq * w2i * w3i
        )

        rhod1i = rhod1i - (
            2 * w1i ** 2 * rhogsol * Kdrag * rhodeq * w3i ** 2 * w2i ** 2 * w2r ** 2
            - vgsol * k * w1r * rhogeq ** 2 * rhodeq * w2i ** 3 * w3r ** 4
            + 2 * w1i ** 2 * rhogsol * Kdrag * rhodeq * w3r ** 2 * w2r ** 2 * w3i ** 2
            + w1i ** 2 * rhogsol * Kdrag * rhodeq * w3r ** 2 * w2r ** 4
            + 2 * w1i ** 2 * rhogsol * Kdrag * rhodeq * w3r ** 2 * w2i ** 2 * w2r ** 2
            + w1i ** 2 * rhogsol * Kdrag * rhodeq * w3r * w2r ** 3 * w3i ** 2
            + w1i ** 2 * rhogsol * Kdrag * rhodeq * w2r * w3r * w2i ** 2 * w3i ** 2
            + w1i ** 2 * rhogsol * Kdrag * rhodeq * w3r ** 3 * w2r ** 3
            + 3 * w1i ** 2 * rhogsol * Kdrag * rhodeq * w2i ** 3 * w3i * w3r ** 2
            + w1i ** 2 * rhogsol * Kdrag * rhodeq * w3r ** 2 * w2i ** 4
            + 3 * w1i ** 2 * rhogsol * Kdrag * rhodeq * w2i * w3i * w2r ** 2 * w3r ** 2
            + w1i ** 2 * rhogsol * Kdrag * rhodeq * w3r ** 4 * w2r ** 2
            + 2 * w1i ** 2 * rhogsol * Kdrag * rhodeq * w3r ** 2 * w2i ** 2 * w3i ** 2
            + w1i ** 2 * rhogsol * Kdrag * rhodeq * w2r ** 4 * w3i ** 2
            + w1i ** 2 * rhogsol * Kdrag * rhodeq * w3i ** 2 * w2i ** 4
            + w1i ** 2 * rhogsol * Kdrag * rhodeq * w3i ** 4 * w2i ** 2
            + w1i ** 2 * rhogsol * Kdrag * rhodeq * w3i ** 4 * w2r ** 2
            + 3 * w1i ** 2 * rhogsol * Kdrag * rhodeq * w2i * w2r ** 2 * w3i ** 3
            + 3 * w1i ** 2 * rhogsol * Kdrag * rhodeq * w2i ** 3 * w3i ** 3
            + 2
            * w1i
            * vgsol
            * w1r
            * k ** 3
            * cs ** 2
            * rhogeq ** 2
            * rhodeq
            * w2i ** 3
            * w3i
            + 2
            * w1i
            * vgsol
            * w1r
            * k ** 3
            * cs ** 2
            * rhogeq ** 2
            * rhodeq
            * w2i
            * w3i
            * w2r ** 2
            + 4
            * w1i
            * vgsol
            * w1r
            * k ** 3
            * cs ** 2
            * rhogeq ** 2
            * rhodeq
            * w3i ** 2
            * w2i ** 2
            - 8
            * w1i
            * vgsol
            * w1r
            * k ** 3
            * cs ** 2
            * rhogeq ** 2
            * rhodeq
            * w3r
            * w2r
            * w2i
            * w3i
            + w1i ** 2 * rhogsol * Kdrag * rhodeq * w2r * w3r ** 3 * w2i ** 2
            + 2
            * w1i
            * vgsol
            * w1r
            * k ** 3
            * cs ** 2
            * rhogeq ** 2
            * rhodeq
            * w2r
            * w3r ** 3
            + 2
            * w1i
            * vgsol
            * w1r
            * k ** 3
            * cs ** 2
            * rhogeq ** 2
            * rhodeq
            * w3i
            * w2i
            * w3r ** 2
            + 4
            * w1i
            * vgsol
            * w1r
            * k ** 3
            * cs ** 2
            * rhogeq ** 2
            * rhodeq
            * w3r ** 2
            * w2r ** 2
            - w1r * vgsol * k * Kdrag ** 2 * rhodeq * w2i ** 2 * w3i ** 3
            - w1r * vgsol * k * Kdrag ** 2 * rhodeq * w2i ** 3 * w3i ** 2
            - w1r * vgsol * k * Kdrag ** 2 * rhodeq * w2r ** 2 * w3i ** 3
            + 2
            * w1i
            * vgsol
            * w1r
            * k ** 3
            * cs ** 2
            * rhogeq ** 2
            * rhodeq
            * w2r
            * w3r
            * w3i ** 2
            + 2
            * w1i
            * vgsol
            * w1r
            * k ** 3
            * cs ** 2
            * rhogeq ** 2
            * rhodeq
            * w3r
            * w2r
            * w2i ** 2
            + 2
            * w1i
            * vgsol
            * w1r
            * k ** 3
            * cs ** 2
            * rhogeq ** 2
            * rhodeq
            * w3r
            * w2r ** 3
            + 2
            * w1i
            * vgsol
            * w1r
            * k ** 3
            * cs ** 2
            * rhogeq ** 2
            * rhodeq
            * w3i ** 3
            * w2i
            - w1i * rhogsol * w1r ** 2 * rhogeq * rhodeq * w2i ** 3 * w3i ** 3
            + w1i ** 2 * rhogsol * Kdrag * rhodeq * w3r ** 4 * w2i ** 2
            - w1i
            * rhogsol
            * w1r ** 2
            * rhogeq
            * rhodeq
            * w2i
            * w3i
            * w2r ** 2
            * w3r ** 2
            + w1i * rhogsol * w1r ** 2 * rhogeq * rhodeq * w3r * w2r ** 3 * w3i ** 2
            + w1i
            * rhogsol
            * w1r ** 2
            * rhogeq
            * rhodeq
            * w2r
            * w3r
            * w2i ** 2
            * w3i ** 2
            - w1i * rhogsol * w1r ** 2 * rhogeq * rhodeq * w2i ** 3 * w3i * w3r ** 2
            - w1i * rhogsol * w1r ** 2 * rhogeq * rhodeq * w2i * w2r ** 2 * w3i ** 3
            - w1r * vgsol * k * Kdrag ** 2 * rhodeq * w3r ** 2 * w2i * w2r ** 2
            - w1r * vgsol * k * Kdrag ** 2 * rhodeq * w3r ** 2 * w2i ** 3
            - w1r * vgsol * k * Kdrag ** 2 * rhodeq * w3r ** 2 * w2r ** 2 * w3i
            - w1r * vgsol * k * Kdrag ** 2 * rhodeq * w3r ** 2 * w2i ** 2 * w3i
            - w1r * vgsol * k * Kdrag ** 2 * rhodeq * w2i * w2r ** 2 * w3i ** 2
            + w1i * rhogsol * w1r ** 2 * rhogeq * rhodeq * w3r ** 3 * w2r ** 3
            + w1i * rhogsol * w1r ** 2 * rhogeq * rhodeq * w2r * w3r ** 3 * w2i ** 2
            + vgsol * Kdrag ** 2 * k * rhodeq * w3r * w2r ** 2 * w2i * w3i ** 2
            + vgsol * Kdrag ** 2 * k * rhodeq * w2r * w2i ** 2 * w3i ** 3
            + vgsol * Kdrag ** 2 * k * rhodeq * w3i ** 3 * w2r ** 3
            + vgsol * Kdrag ** 2 * k * rhodeq * w3r ** 3 * w2i * w2r ** 2
            + vgsol * Kdrag ** 2 * k * rhodeq * w3i * w2r ** 3 * w3r ** 2
            + vgsol * Kdrag ** 2 * k * rhodeq * w3r * w2i ** 3 * w3i ** 2
            - 2
            * w1r
            * rhogsol
            * k ** 2
            * cs ** 2
            * rhogeq
            * w1i ** 2
            * rhodeq
            * w3i
            * w3r
            * w2i ** 2
            - 2
            * w1r
            * rhogsol
            * k ** 2
            * cs ** 2
            * rhogeq
            * w1i ** 2
            * rhodeq
            * w3i
            * w3r
            * w2r ** 2
            + vgsol * Kdrag ** 2 * k * rhodeq * w2i ** 3 * w3r ** 3
            - 2
            * w1r
            * rhogsol
            * k ** 2
            * cs ** 2
            * rhogeq
            * w1i ** 2
            * rhodeq
            * w2r
            * w2i
            * w3r ** 2
            + vgsol * Kdrag ** 2 * k * rhodeq * w3i * w2i ** 2 * w3r ** 2 * w2r
            - 2
            * w1r
            * rhogsol
            * k ** 2
            * cs ** 2
            * rhogeq
            * w1i ** 2
            * rhodeq
            * w3r ** 3
            * w2i
            - 2
            * w1r
            * rhogsol
            * k ** 2
            * cs ** 2
            * rhogeq
            * w1i ** 2
            * rhodeq
            * w3r
            * w2i
            * w3i ** 2
            - 2
            * w1r
            * rhogsol
            * k ** 2
            * cs ** 2
            * rhogeq
            * w1i ** 2
            * rhodeq
            * w2r ** 3
            * w3i
            - 2
            * w1r
            * rhogsol
            * k ** 2
            * cs ** 2
            * rhogeq
            * w1i ** 2
            * rhodeq
            * w2r
            * w3i
            * w2i ** 2
            - 2
            * w1r
            * rhogsol
            * k ** 2
            * cs ** 2
            * rhogeq
            * w1i ** 2
            * rhodeq
            * w2r
            * w2i
            * w3i ** 2
            + 2 * w1i * w1r * vgsol * k * Kdrag ** 2 * rhodeq * w3r ** 2 * w2r ** 2
            + 2 * w1i * w1r * vgsol * k * Kdrag ** 2 * rhodeq * w3i ** 2 * w2i ** 2
            + 2 * w1i * w1r * vgsol * k * Kdrag ** 2 * rhodeq * w3r ** 2 * w2i ** 2
            + 2 * w1i * w1r * vgsol * k * Kdrag ** 2 * rhodeq * w2r ** 2 * w3i ** 2
            + vgsol
            * k ** 3
            * cs ** 2
            * rhogeq ** 2
            * w1i ** 2
            * rhodeq
            * w2r
            * w3i ** 3
            - 2
            * vgsol
            * k ** 3
            * cs ** 2
            * rhogeq ** 2
            * w1i ** 2
            * rhodeq
            * w2r
            * w2i
            * w3i ** 2
        )

        rhod1i = rhod1i - (
            2
            * vgsol
            * k ** 3
            * cs ** 2
            * rhogeq ** 2
            * w1i ** 2
            * rhodeq
            * w3i
            * w3r
            * w2r ** 2
            - vgsol
            * k ** 3
            * cs ** 2
            * rhogeq ** 2
            * w1i ** 2
            * rhodeq
            * w2r ** 3
            * w3i
            - vgsol
            * k ** 3
            * cs ** 2
            * rhogeq ** 2
            * w1i ** 2
            * rhodeq
            * w2r
            * w3i
            * w2i ** 2
            + vgsol
            * k ** 3
            * cs ** 2
            * rhogeq ** 2
            * w1i ** 2
            * rhodeq
            * w3r
            * w2i ** 3
            - 2
            * vgsol
            * k ** 3
            * cs ** 2
            * rhogeq ** 2
            * w1i ** 2
            * rhodeq
            * w3i
            * w3r
            * w2i ** 2
            - vgsol
            * k ** 3
            * cs ** 2
            * rhogeq ** 2
            * w1i ** 2
            * rhodeq
            * w3r ** 3
            * w2i
            + vgsol
            * k ** 3
            * cs ** 2
            * rhogeq ** 2
            * w1i ** 2
            * rhodeq
            * w3r
            * w2i
            * w2r ** 2
            - vgsol
            * k ** 3
            * cs ** 2
            * rhogeq ** 2
            * w1i ** 2
            * rhodeq
            * w3r
            * w2i
            * w3i ** 2
            - w1i
            * rhogeq
            * k
            * Kdrag
            * vdsol
            * rhodeq
            * w3r
            * w2r ** 2
            * w2i
            * w3i ** 2
            - w1i * rhogeq * k * Kdrag * vdsol * rhodeq * w2r * w2i ** 2 * w3i ** 3
            + vgsol
            * k ** 3
            * cs ** 2
            * rhogeq ** 2
            * w1i ** 2
            * rhodeq
            * w2r
            * w3i
            * w3r ** 2
            - w1i * rhogeq * k * Kdrag * vdsol * rhodeq * w3r * w2i ** 3 * w3i ** 2
            - w1i * rhogeq * k * Kdrag * vdsol * rhodeq * w3i ** 3 * w2r ** 3
            + 2
            * vgsol
            * k ** 3
            * cs ** 2
            * rhogeq ** 2
            * w1i ** 2
            * rhodeq
            * w2r
            * w2i
            * w3r ** 2
            - w1i * rhogeq * k * Kdrag * vdsol * rhodeq * w3r ** 3 * w2i * w2r ** 2
            - w1i * rhogeq * k * Kdrag * vdsol * rhodeq * w2i ** 3 * w3r ** 3
            - w1i
            * rhogeq
            * k
            * Kdrag
            * vdsol
            * rhodeq
            * w3i
            * w2i ** 2
            * w3r ** 2
            * w2r
            - w1i * rhogeq * k * Kdrag * vdsol * rhodeq * w3i * w2r ** 3 * w3r ** 2
            + vdsol * Kdrag * k ** 3 * cs ** 2 * rhogeq * w1i ** 3 * rhodeq * w2i * w3r
            + vdsol * Kdrag * k ** 3 * cs ** 2 * rhogeq * w1i ** 3 * rhodeq * w2r * w3i
            + 2
            * w1r
            * vgsol
            * rhogeq ** 2
            * k ** 3
            * cs ** 2
            * w1i ** 2
            * rhodeq
            * w3r
            * w2r
            * w2i
            - w1r
            * vgsol
            * rhogeq ** 2
            * k ** 3
            * cs ** 2
            * w1i ** 2
            * rhodeq
            * w2i ** 2
            * w3i
            - w1r
            * vgsol
            * rhogeq ** 2
            * k ** 3
            * cs ** 2
            * w1i ** 2
            * rhodeq
            * w2i
            * w3i ** 2
            + w1r
            * vgsol
            * rhogeq ** 2
            * k ** 3
            * cs ** 2
            * w1i ** 2
            * rhodeq
            * w2i
            * w3r ** 2
            + 2
            * w1r
            * vgsol
            * rhogeq ** 2
            * k ** 3
            * cs ** 2
            * w1i ** 2
            * rhodeq
            * w3r
            * w2r
            * w3i
            + w1r
            * vgsol
            * rhogeq ** 2
            * k ** 3
            * cs ** 2
            * w1i ** 2
            * rhodeq
            * w2r ** 2
            * w3i
            - vgsol
            * k ** 3
            * cs ** 2
            * rhogeq ** 2
            * w1i ** 3
            * rhodeq
            * w2r
            * w3r ** 2
            - vgsol
            * k ** 3
            * cs ** 2
            * rhogeq ** 2
            * w1i ** 3
            * rhodeq
            * w2r ** 2
            * w3r
            - w1i * cs ** 2 * k ** 2 * rhogsol * Kdrag * rhodeq * w2i ** 3 * w3i ** 2
            + vgsol
            * k ** 3
            * cs ** 2
            * rhogeq ** 2
            * w1i ** 3
            * rhodeq
            * w2i ** 2
            * w3r
            + 2
            * vgsol
            * k ** 3
            * cs ** 2
            * rhogeq ** 2
            * w1i ** 3
            * rhodeq
            * w2r
            * w2i
            * w3i
            + vdsol * Kdrag * k ** 3 * cs ** 2 * rhogeq * rhodeq * w3r ** 3 * w2r ** 2
            - vdsol * Kdrag * k ** 3 * cs ** 2 * rhogeq * rhodeq * w3r ** 3 * w2i ** 2
            + vdsol * Kdrag * k ** 3 * cs ** 2 * rhogeq * rhodeq * w2r ** 3 * w3r ** 2
            - w1i
            * cs ** 2
            * k ** 2
            * rhogsol
            * Kdrag
            * rhodeq
            * w3r ** 2
            * w2i ** 2
            * w3i
            - w1i * cs ** 2 * k ** 2 * rhogsol * Kdrag * rhodeq * w2r ** 2 * w3i ** 3
            - w1i * cs ** 2 * k ** 2 * rhogsol * Kdrag * rhodeq * w2i ** 2 * w3i ** 3
            - w1i
            * cs ** 2
            * k ** 2
            * rhogsol
            * Kdrag
            * rhodeq
            * w2i
            * w2r ** 2
            * w3i ** 2
            + 4
            * w1i
            * vdsol
            * Kdrag
            * k ** 3
            * cs ** 2
            * rhogeq
            * rhodeq
            * w2r
            * w2i
            * w3i ** 2
            - w1i
            * cs ** 2
            * k ** 2
            * rhogsol
            * Kdrag
            * rhodeq
            * w3r ** 2
            * w2i
            * w2r ** 2
            - w1i
            * cs ** 2
            * k ** 2
            * rhogsol
            * Kdrag
            * rhodeq
            * w3r ** 2
            * w2r ** 2
            * w3i
            + w1i
            * vdsol
            * Kdrag
            * k ** 3
            * cs ** 2
            * rhogeq
            * rhodeq
            * w3r
            * w2i
            * w2r ** 2
            + 4
            * w1i
            * vdsol
            * Kdrag
            * k ** 3
            * cs ** 2
            * rhogeq
            * rhodeq
            * w3i
            * w3r
            * w2i ** 2
            + w1i
            * vdsol
            * Kdrag
            * k ** 3
            * cs ** 2
            * rhogeq
            * rhodeq
            * w2r
            * w3i
            * w3r ** 2
            + w1i
            * vdsol
            * Kdrag
            * k ** 3
            * cs ** 2
            * rhogeq
            * rhodeq
            * w3r
            * w2i
            * w3i ** 2
            - w1i * cs ** 2 * k ** 2 * rhogsol * Kdrag * rhodeq * w3r ** 2 * w2i ** 3
            + vgsol
            * k ** 3
            * cs ** 2
            * rhogeq ** 2
            * rhodeq
            * w2r
            * w2i ** 2
            * w3i ** 3
            + 2 * vgsol * k ** 3 * cs ** 2 * rhogeq ** 2 * rhodeq * w2r * w3i ** 4 * w2i
            + w1i * vdsol * Kdrag * k ** 3 * cs ** 2 * rhogeq * rhodeq * w3r ** 3 * w2i
            + w1i * vdsol * Kdrag * k ** 3 * cs ** 2 * rhogeq * rhodeq * w3r * w2i ** 3
            + w1i * vdsol * Kdrag * k ** 3 * cs ** 2 * rhogeq * rhodeq * w2r ** 3 * w3i
            + w1i
            * vdsol
            * Kdrag
            * k ** 3
            * cs ** 2
            * rhogeq
            * rhodeq
            * w2r
            * w3i
            * w2i ** 2
            + w1i * vdsol * Kdrag * k ** 3 * cs ** 2 * rhogeq * rhodeq * w2r * w3i ** 3
            + vgsol
            * k ** 3
            * cs ** 2
            * rhogeq ** 2
            * rhodeq
            * w3r
            * w2i ** 3
            * w3i ** 2
            + vgsol
            * k ** 3
            * cs ** 2
            * rhogeq ** 2
            * rhodeq
            * w3r
            * w2r ** 2
            * w2i
            * w3i ** 2
            + 2 * vgsol * k ** 3 * cs ** 2 * rhogeq ** 2 * rhodeq * w3i * w3r * w2i ** 4
            + vgsol
            * k ** 3
            * cs ** 2
            * rhogeq ** 2
            * rhodeq
            * w3r ** 3
            * w2i
            * w2r ** 2
            + vgsol
            * k ** 3
            * cs ** 2
            * rhogeq ** 2
            * rhodeq
            * w3i
            * w2r ** 3
            * w3r ** 2
            + 4
            * vgsol
            * k ** 3
            * cs ** 2
            * rhogeq ** 2
            * rhodeq
            * w2r
            * w2i
            * w3i ** 2
            * w3r ** 2
            + 2 * vgsol * k ** 3 * cs ** 2 * rhogeq ** 2 * rhodeq * w2r * w2i * w3r ** 4
            + vgsol
            * Kdrag
            * rhogeq
            * k ** 3
            * cs ** 2
            * rhodeq
            * w2r
            * w2i ** 2
            * w3i ** 2
            + vgsol * k ** 3 * cs ** 2 * rhogeq ** 2 * rhodeq * w2i ** 3 * w3r ** 3
            + vgsol
            * k ** 3
            * cs ** 2
            * rhogeq ** 2
            * rhodeq
            * w3i
            * w2i ** 2
            * w3r ** 2
            * w2r
            + 4
            * vgsol
            * k ** 3
            * cs ** 2
            * rhogeq ** 2
            * rhodeq
            * w3r
            * w3i
            * w2r ** 2
            * w2i ** 2
            + 2 * vgsol * k ** 3 * cs ** 2 * rhogeq ** 2 * rhodeq * w2r ** 4 * w3i * w3r
            + vgsol * k ** 3 * cs ** 2 * rhogeq ** 2 * rhodeq * w3i ** 3 * w2r ** 3
            + 2
            * vgsol
            * Kdrag
            * rhogeq
            * k ** 3
            * cs ** 2
            * rhodeq
            * w2r
            * w2i
            * w3r ** 2
            * w3i
            - vgsol
            * Kdrag
            * rhogeq
            * k ** 3
            * cs ** 2
            * rhodeq
            * w3r
            * w2r ** 2
            * w3i ** 2
            - vgsol
            * Kdrag
            * rhogeq
            * k ** 3
            * cs ** 2
            * rhodeq
            * w2r
            * w2i ** 2
            * w3r ** 2
            + vgsol
            * Kdrag
            * rhogeq
            * k ** 3
            * cs ** 2
            * rhodeq
            * w3r
            * w2i ** 2
            * w3i ** 2
        )

        rhod1i = rhod1i - (
            2
            * vgsol
            * Kdrag
            * rhogeq
            * k ** 3
            * cs ** 2
            * rhodeq
            * w2i
            * w3i
            * w3r
            * w2r ** 2
            + 2
            * vgsol
            * Kdrag
            * rhogeq
            * k ** 3
            * cs ** 2
            * rhodeq
            * w2i ** 3
            * w3i
            * w3r
            + vgsol * Kdrag * rhogeq * k ** 3 * cs ** 2 * rhodeq * w2r ** 3 * w3i ** 2
            + 2
            * vgsol
            * Kdrag
            * rhogeq
            * k ** 3
            * cs ** 2
            * rhodeq
            * w2r
            * w2i
            * w3i ** 3
            - vdsol * Kdrag ** 2 * k * rhodeq * w3r * w2r ** 2 * w2i * w3i ** 2
            - vdsol * Kdrag ** 2 * k * rhodeq * w2r * w2i ** 2 * w3i ** 3
            - vdsol * Kdrag ** 2 * k * rhodeq * w3r ** 3 * w2i * w2r ** 2
            - vdsol * Kdrag ** 2 * k * rhodeq * w3i * w2i ** 2 * w3r ** 2 * w2r
            - vdsol * Kdrag ** 2 * k * rhodeq * w3i * w2r ** 3 * w3r ** 2
            - vdsol * Kdrag ** 2 * k * rhodeq * w3r * w2i ** 3 * w3i ** 2
            - vgsol * Kdrag * rhogeq * k ** 3 * cs ** 2 * rhodeq * w3r ** 3 * w2r ** 2
            + vgsol * Kdrag * rhogeq * k ** 3 * cs ** 2 * rhodeq * w3r ** 3 * w2i ** 2
            - vgsol * Kdrag * rhogeq * k ** 3 * cs ** 2 * rhodeq * w2r ** 3 * w3r ** 2
            - w1i * w1r ** 2 * vgsol * k * rhogeq ** 2 * rhodeq * w3r ** 3 * w2r ** 2
            - w1i * w1r ** 2 * vgsol * k * rhogeq ** 2 * rhodeq * w2r ** 3 * w3r ** 2
            - w1i
            * w1r ** 2
            * vgsol
            * k
            * rhogeq ** 2
            * rhodeq
            * w2r
            * w2i ** 2
            * w3r ** 2
            + 2
            * w1i
            * w1r
            * vgsol
            * Kdrag
            * rhogeq
            * k ** 3
            * cs ** 2
            * rhodeq
            * w2i
            * w3i ** 2
            + 2
            * w1i
            * w1r
            * vgsol
            * Kdrag
            * rhogeq
            * k ** 3
            * cs ** 2
            * rhodeq
            * w2r ** 2
            * w3i
            + 2
            * w1i
            * w1r
            * vgsol
            * Kdrag
            * rhogeq
            * k ** 3
            * cs ** 2
            * rhodeq
            * w2i ** 2
            * w3i
            - w1i
            * w1r ** 2
            * vgsol
            * k
            * rhogeq ** 2
            * rhodeq
            * w3r
            * w2i ** 2
            * w3i ** 2
            - w1i
            * w1r ** 2
            * vgsol
            * k
            * rhogeq ** 2
            * rhodeq
            * w3r
            * w2r ** 2
            * w3i ** 2
            - w1i * w1r ** 2 * vgsol * k * rhogeq ** 2 * rhodeq * w2r ** 3 * w3i ** 2
            - w1i
            * w1r ** 2
            * vgsol
            * k
            * rhogeq ** 2
            * rhodeq
            * w2r
            * w2i ** 2
            * w3i ** 2
            - w1r
            * vgsol
            * Kdrag
            * k ** 3
            * cs ** 2
            * rhogeq
            * rhodeq
            * w2i
            * w3i
            * w2r ** 2
            - w1r * vgsol * Kdrag * k ** 3 * cs ** 2 * rhogeq * rhodeq * w2i ** 3 * w3i
            - w1r
            * vgsol
            * Kdrag
            * k ** 3
            * cs ** 2
            * rhogeq
            * rhodeq
            * w3i ** 2
            * w2i ** 2
            - w1r
            * vgsol
            * Kdrag
            * k ** 3
            * cs ** 2
            * rhogeq
            * rhodeq
            * w2r ** 2
            * w3i ** 2
            + 2
            * w1i
            * w1r
            * vgsol
            * Kdrag
            * rhogeq
            * k ** 3
            * cs ** 2
            * rhodeq
            * w2i
            * w3r ** 2
            - 4
            * w1r
            * vgsol
            * Kdrag
            * k ** 3
            * cs ** 2
            * rhogeq
            * rhodeq
            * w3r
            * w2r
            * w2i
            * w3i
            - w1r * vgsol * Kdrag * k ** 3 * cs ** 2 * rhogeq * rhodeq * w3i ** 3 * w2i
            - w1i * w1r ** 2 * vgsol * k * rhogeq ** 2 * rhodeq * w3r ** 3 * w2i ** 2
            - w1r
            * vgsol
            * Kdrag
            * k ** 3
            * cs ** 2
            * rhogeq
            * rhodeq
            * w3i
            * w2i
            * w3r ** 2
            - w1r
            * vgsol
            * Kdrag
            * k ** 3
            * cs ** 2
            * rhogeq
            * rhodeq
            * w3r ** 2
            * w2i ** 2
            + 3
            * w1r
            * vgsol
            * Kdrag
            * k ** 3
            * cs ** 2
            * rhogeq
            * rhodeq
            * w3r ** 2
            * w2r ** 2
            - vgsol * k * rhogeq ** 2 * w1i ** 3 * rhodeq * w2r * w2i ** 2 * w3i ** 2
            + w1r * vgsol * Kdrag * k ** 3 * cs ** 2 * rhogeq * rhodeq * w2r * w3r ** 3
            - vgsol * k * rhogeq ** 2 * w1i ** 3 * rhodeq * w2r ** 3 * w3r ** 2
            - vgsol * k * rhogeq ** 2 * w1i ** 3 * rhodeq * w2r * w2i ** 2 * w3r ** 2
            - vgsol * k * rhogeq ** 2 * w1i ** 3 * rhodeq * w3r * w2i ** 2 * w3i ** 2
            - vgsol * k * rhogeq ** 2 * w1i ** 3 * rhodeq * w3r * w2r ** 2 * w3i ** 2
            - vdsol * Kdrag ** 2 * k * rhodeq * w2i ** 3 * w3r ** 3
            - vgsol * k * rhogeq ** 2 * w1i ** 3 * rhodeq * w3r ** 3 * w2r ** 2
            - vgsol * k * rhogeq ** 2 * w1i ** 3 * rhodeq * w2r ** 3 * w3i ** 2
            + w1r
            * vgsol
            * Kdrag
            * k ** 3
            * cs ** 2
            * rhogeq
            * rhodeq
            * w2r
            * w3r
            * w3i ** 2
            + w1r
            * vgsol
            * Kdrag
            * k ** 3
            * cs ** 2
            * rhogeq
            * rhodeq
            * w3r
            * w2r
            * w2i ** 2
            + w1r * vgsol * Kdrag * k ** 3 * cs ** 2 * rhogeq * rhodeq * w3r * w2r ** 3
            - vgsol * k * rhogeq ** 2 * w1i ** 3 * rhodeq * w3r ** 3 * w2i ** 2
            - vdsol * Kdrag ** 2 * k * rhodeq * w3i ** 3 * w2r ** 3
            + w1r * vdsol * k * Kdrag ** 2 * rhodeq * w2i ** 2 * w3i ** 3
            + w1r * vdsol * k * Kdrag ** 2 * rhodeq * w2i * w2r ** 2 * w3i ** 2
            + w1r * vdsol * k * Kdrag ** 2 * rhodeq * w2i ** 3 * w3i ** 2
            + w1r * vdsol * k * Kdrag ** 2 * rhodeq * w3r ** 2 * w2r ** 2 * w3i
            + w1r * vdsol * k * Kdrag ** 2 * rhodeq * w3r ** 2 * w2i ** 2 * w3i
            + w1r * vdsol * k * Kdrag ** 2 * rhodeq * w2r ** 2 * w3i ** 3
            + w1r * vdsol * k * Kdrag ** 2 * rhodeq * w3r ** 2 * w2i * w2r ** 2
            + w1r * vdsol * k * Kdrag ** 2 * rhodeq * w3r ** 2 * w2i ** 3
            + cs ** 2 * k ** 2 * rhogsol * Kdrag * rhodeq * w2i ** 3 * w3i * w3r ** 2
            - cs ** 2 * k ** 2 * rhogsol * Kdrag * rhodeq * w3r * w2r ** 3 * w3i ** 2
            - cs ** 2
            * k ** 2
            * rhogsol
            * Kdrag
            * rhodeq
            * w2r
            * w3r
            * w2i ** 2
            * w3i ** 2
            - cs ** 2 * k ** 2 * rhogsol * Kdrag * rhodeq * w3r ** 3 * w2r ** 3
            + cs ** 2
            * k ** 2
            * rhogsol
            * Kdrag
            * rhodeq
            * w2i
            * w3i
            * w2r ** 2
            * w3r ** 2
            - w1r ** 4 * Kdrag * rhogeq * rhodsol * w2r ** 2 * w3i ** 2
            + vgsol * Kdrag * k ** 3 * cs ** 2 * rhogeq * w1r ** 3 * rhodeq * w3r * w2r
            - w1r ** 4 * Kdrag * rhogeq * rhodsol * w3i ** 2 * w2i ** 2
            - vgsol * Kdrag * k ** 3 * cs ** 2 * rhogeq * w1r ** 3 * rhodeq * w2i * w3i
            - cs ** 2 * k ** 2 * rhogsol * Kdrag * rhodeq * w2r * w3r ** 3 * w2i ** 2
            + cs ** 2 * k ** 2 * rhogsol * Kdrag * rhodeq * w2i * w2r ** 2 * w3i ** 3
            + cs ** 2 * k ** 2 * rhogsol * Kdrag * rhodeq * w2i ** 3 * w3i ** 3
            + 2 * w1i ** 2 * rhogsol * rhogeq * rhodeq * w2i ** 3 * w3i ** 2 * w3r ** 2
            + w1i ** 2 * rhogsol * rhogeq * rhodeq * w2i ** 4 * w3r ** 2 * w3i
            + w1i ** 2 * rhogsol * rhogeq * rhodeq * w2r ** 4 * w3r ** 2 * w3i
            + w1i ** 2 * rhogsol * rhogeq * rhodeq * w2r ** 4 * w3i ** 3
            + 2
            * w1i ** 2
            * rhogsol
            * rhogeq
            * rhodeq
            * w3i ** 2
            * w2r ** 2
            * w2i
            * w3r ** 2
        )

        rhod1i = rhod1i - (
            2
            * w1i ** 2
            * rhogsol
            * rhogeq
            * rhodeq
            * w2r ** 2
            * w3i
            * w3r ** 2
            * w2i ** 2
            + w1i ** 2 * rhogsol * rhogeq * rhodeq * w2i * w2r ** 2 * w3i ** 4
            + w1i ** 2 * rhogsol * rhogeq * rhodeq * w2i ** 4 * w3i ** 3
            + w1i ** 2 * rhogsol * rhogeq * rhodeq * w3i ** 4 * w2i ** 3
            + 2 * w1i ** 2 * rhogsol * rhogeq * rhodeq * w2r ** 2 * w2i ** 2 * w3i ** 3
            - w1r ** 4 * Kdrag * rhogeq * rhodsol * w3r ** 2 * w2i ** 2
            - w1r ** 4 * Kdrag * rhogeq * rhodsol * w3r ** 2 * w2r ** 2
            + w1r ** 3 * vgsol * k * rhogeq ** 2 * rhodeq * w2i ** 2 * w3i ** 3
            + w1r ** 3 * vgsol * k * rhogeq ** 2 * rhodeq * w2i * w2r ** 2 * w3i ** 2
            + w1r ** 3 * vgsol * k * rhogeq ** 2 * rhodeq * w2i ** 3 * w3i ** 2
            + w1r ** 3 * vgsol * k * rhogeq ** 2 * rhodeq * w2r ** 2 * w3i ** 3
            + w1i ** 2 * rhogsol * rhogeq * rhodeq * w2i ** 3 * w3r ** 4
            + w1i ** 2 * rhogsol * rhogeq * rhodeq * w2r ** 2 * w2i * w3r ** 4
            + w1r ** 3 * vgsol * k * rhogeq ** 2 * rhodeq * w3r ** 2 * w2i * w2r ** 2
            + w1r ** 3 * vgsol * k * rhogeq ** 2 * rhodeq * w3r ** 2 * w2r ** 2 * w3i
            + w1r ** 3 * vgsol * k * rhogeq ** 2 * rhodeq * w3r ** 2 * w2i ** 2 * w3i
            + w1r
            * vdsol
            * Kdrag
            * rhogeq
            * k ** 3
            * cs ** 2
            * w1i ** 2
            * rhodeq
            * w2i
            * w3i
            + 2
            * w1r ** 2
            * rhogsol
            * rhogeq
            * k ** 2
            * cs ** 2
            * w1i ** 2
            * rhodeq
            * w2i
            * w3r ** 2
            + 2
            * w1r ** 2
            * rhogsol
            * rhogeq
            * k ** 2
            * cs ** 2
            * w1i ** 2
            * rhodeq
            * w2i ** 2
            * w3i
            - w1r
            * vdsol
            * Kdrag
            * rhogeq
            * k ** 3
            * cs ** 2
            * w1i ** 2
            * rhodeq
            * w3r
            * w2r
            - 2
            * rhogsol
            * rhogeq
            * k ** 4
            * cs ** 4
            * w1i ** 2
            * rhodeq
            * w3r
            * w2r
            * w3i
            + 2
            * rhogsol
            * rhogeq
            * k ** 4
            * cs ** 4
            * w1i ** 2
            * rhodeq
            * w2i ** 2
            * w3i
            + 2
            * rhogsol
            * rhogeq
            * k ** 4
            * cs ** 4
            * w1i ** 2
            * rhodeq
            * w2i
            * w3i ** 2
            + 2 * w1i * w1r * Kdrag * rhogeq * k * vgsol * rhodeq * w2i ** 2 * w3i ** 3
            + 2
            * w1i
            * w1r
            * Kdrag
            * rhogeq
            * k
            * vgsol
            * rhodeq
            * w2i
            * w2r ** 2
            * w3i ** 2
            + 2 * w1i * w1r * Kdrag * rhogeq * k * vgsol * rhodeq * w2i ** 3 * w3i ** 2
            + 2
            * w1i
            * w1r
            * Kdrag
            * rhogeq
            * k
            * vgsol
            * rhodeq
            * w3r ** 2
            * w2i ** 2
            * w3i
            - 2
            * rhogsol
            * rhogeq
            * k ** 4
            * cs ** 4
            * w1i ** 2
            * rhodeq
            * w3r
            * w2r
            * w2i
            + 2
            * w1r ** 2
            * rhogsol
            * rhogeq
            * k ** 2
            * cs ** 2
            * w1i ** 2
            * rhodeq
            * w2i
            * w3i ** 2
            + 2
            * w1r ** 2
            * rhogsol
            * rhogeq
            * k ** 2
            * cs ** 2
            * w1i ** 2
            * rhodeq
            * w2r ** 2
            * w3i
            + 2 * w1i * w1r * Kdrag * rhogeq * k * vgsol * rhodeq * w3r ** 2 * w2i ** 3
            + 2
            * w1i
            * w1r
            * Kdrag
            * rhogeq
            * k
            * vgsol
            * rhodeq
            * w3r ** 2
            * w2r ** 2
            * w3i
            + 2 * w1i * w1r * Kdrag * rhogeq * k * vgsol * rhodeq * w2r ** 2 * w3i ** 3
            + 2
            * Kdrag
            * vgsol
            * rhogeq
            * k ** 3
            * cs ** 2
            * w1i ** 2
            * rhodeq
            * w2r
            * w2i
            * w3i
            + 2
            * Kdrag
            * vgsol
            * rhogeq
            * k ** 3
            * cs ** 2
            * w1i ** 2
            * rhodeq
            * w2r
            * w3i ** 2
            - 2
            * Kdrag
            * vdsol
            * rhogeq
            * k ** 3
            * cs ** 2
            * w1i ** 2
            * rhodeq
            * w2r
            * w3i ** 2
            + 2
            * Kdrag
            * vgsol
            * rhogeq
            * k ** 3
            * cs ** 2
            * w1i ** 2
            * rhodeq
            * w2i ** 2
            * w3r
            - 2
            * Kdrag
            * vdsol
            * rhogeq
            * k ** 3
            * cs ** 2
            * w1i ** 2
            * rhodeq
            * w2i ** 2
            * w3r
            - 2
            * Kdrag
            * vdsol
            * rhogeq
            * k ** 3
            * cs ** 2
            * w1i ** 2
            * rhodeq
            * w3r
            * w2i
            * w3i
            + w1r ** 3 * vgsol * k * rhogeq ** 2 * rhodeq * w3r ** 2 * w2i ** 3
            - 2
            * Kdrag
            * vdsol
            * rhogeq
            * k ** 3
            * cs ** 2
            * w1i ** 2
            * rhodeq
            * w2r
            * w2i
            * w3i
            + 2
            * Kdrag
            * vgsol
            * rhogeq
            * k ** 3
            * cs ** 2
            * w1i ** 2
            * rhodeq
            * w3r
            * w2i
            * w3i
            + 2
            * w1i
            * w1r
            * Kdrag
            * rhogeq
            * k
            * vgsol
            * rhodeq
            * w3r ** 2
            * w2i
            * w2r ** 2
            - w1i * vgsol * Kdrag ** 2 * k * rhodeq * w3r ** 3 * w2i ** 2
            - w1i * vgsol * Kdrag ** 2 * k * rhodeq * w3r ** 3 * w2r ** 2
            - w1i * vgsol * Kdrag ** 2 * k * rhodeq * w2r ** 3 * w3r ** 2
            - w1i * vgsol * Kdrag ** 2 * k * rhodeq * w2r * w2i ** 2 * w3r ** 2
            - w1i * vgsol * Kdrag ** 2 * k * rhodeq * w2r * w2i ** 2 * w3i ** 2
            - 2
            * w1i
            * vgsol
            * k ** 3
            * cs ** 2
            * rhogeq ** 2
            * rhodeq
            * w3r ** 3
            * w2r ** 2
            - 2
            * w1i
            * vgsol
            * k ** 3
            * cs ** 2
            * rhogeq ** 2
            * rhodeq
            * w2r
            * w2i ** 2
            * w3r ** 2
            - 2
            * w1i
            * vgsol
            * k ** 3
            * cs ** 2
            * rhogeq ** 2
            * rhodeq
            * w2r
            * w2i
            * w3r ** 2
            * w3i
            - 2
            * w1i
            * w1r
            * rhogsol
            * rhogeq
            * k ** 4
            * cs ** 4
            * rhodeq
            * w2r
            * w3r ** 2
            - 2
            * w1i
            * w1r
            * rhogsol
            * rhogeq
            * k ** 4
            * cs ** 4
            * rhodeq
            * w2i ** 2
            * w3r
            - 2
            * w1i
            * w1r
            * rhogsol
            * rhogeq
            * k ** 4
            * cs ** 4
            * rhodeq
            * w2r ** 2
            * w3r
            - 2
            * w1i
            * w1r
            * rhogsol
            * rhogeq
            * k ** 4
            * cs ** 4
            * rhodeq
            * w2r
            * w3i ** 2
            - w1i * rhogsol * k ** 4 * cs ** 4 * rhogeq * rhodeq * w2i ** 3 * w3i
            - 3
            * w1i
            * rhogsol
            * k ** 4
            * cs ** 4
            * rhogeq
            * rhodeq
            * w3i ** 2
            * w2i ** 2
        )

        rhod1i = rhod1i - (
            w1i * rhogsol * k ** 4 * cs ** 4 * rhogeq * rhodeq * w2r ** 2 * w3i ** 2
            + 2 * w1r ** 2 * rhogsol * Kdrag * w1i ** 2 * rhodeq * w3r ** 2 * w2r ** 2
            + 2 * w1r ** 2 * rhogsol * Kdrag * w1i ** 2 * rhodeq * w3i ** 2 * w2i ** 2
            + 2 * w1r ** 2 * rhogsol * Kdrag * w1i ** 2 * rhodeq * w2r ** 2 * w3i ** 2
            - w1i * rhogsol * k ** 4 * cs ** 4 * rhogeq * rhodeq * w3i * w2i * w3r ** 2
            + w1i * rhogsol * k ** 4 * cs ** 4 * rhogeq * rhodeq * w3r ** 2 * w2r ** 2
            + w1i * rhogsol * k ** 4 * cs ** 4 * rhogeq * rhodeq * w2r * w3r * w3i ** 2
            + w1i * rhogsol * k ** 4 * cs ** 4 * rhogeq * rhodeq * w3r ** 2 * w2i ** 2
            + 4
            * w1i
            * rhogsol
            * k ** 4
            * cs ** 4
            * rhogeq
            * rhodeq
            * w3r
            * w2r
            * w2i
            * w3i
            + w1i * rhogsol * k ** 4 * cs ** 4 * rhogeq * rhodeq * w3r * w2r ** 3
            + w1i * rhogsol * k ** 4 * cs ** 4 * rhogeq * rhodeq * w3r * w2r * w2i ** 2
            - w1i * rhogsol * k ** 4 * cs ** 4 * rhogeq * rhodeq * w3i ** 3 * w2i
            - w1i * rhogsol * k ** 4 * cs ** 4 * rhogeq * rhodeq * w2i * w3i * w2r ** 2
            - 2
            * w1i
            * w1r
            * vdsol
            * Kdrag
            * rhogeq
            * k ** 3
            * cs ** 2
            * rhodeq
            * w2i
            * w3r ** 2
            - 2
            * w1i
            * w1r
            * vdsol
            * Kdrag
            * rhogeq
            * k ** 3
            * cs ** 2
            * rhodeq
            * w2i ** 2
            * w3i
            - 2
            * w1i
            * w1r
            * vdsol
            * Kdrag
            * rhogeq
            * k ** 3
            * cs ** 2
            * rhodeq
            * w2i
            * w3i ** 2
            - 2
            * w1i
            * w1r
            * vdsol
            * Kdrag
            * rhogeq
            * k ** 3
            * cs ** 2
            * rhodeq
            * w2r ** 2
            * w3i
            + 2 * w1r ** 2 * rhogsol * Kdrag * w1i ** 2 * rhodeq * w3r ** 2 * w2i ** 2
            + rhogsol * k ** 4 * cs ** 4 * rhogeq * rhodeq * w3r ** 2 * w2i ** 2 * w3i
            - 2 * rhogsol * k ** 4 * cs ** 4 * rhogeq * rhodeq * w3i * w3r * w2r ** 3
            - 2
            * rhogsol
            * k ** 4
            * cs ** 4
            * rhogeq
            * rhodeq
            * w2r
            * w3r
            * w2i
            * w3i ** 2
            + rhogsol * k ** 4 * cs ** 4 * rhogeq * rhodeq * w2i ** 3 * w3i ** 2
            + rhogsol * k ** 4 * cs ** 4 * rhogeq * rhodeq * w2i * w2r ** 2 * w3i ** 2
            - rhogsol * k ** 4 * cs ** 4 * rhogeq * rhodeq * w2r ** 2 * w3i ** 3
            + w1i * rhogsol * k ** 4 * cs ** 4 * rhogeq * rhodeq * w2r * w3r ** 3
            + w1r ** 2 * rhogsol * rhogeq * k ** 2 * cs ** 2 * rhodeq * w3i ** 4 * w2i
            - 2 * rhogsol * k ** 4 * cs ** 4 * rhogeq * rhodeq * w2r * w3r ** 3 * w2i
            - rhogsol * k ** 4 * cs ** 4 * rhogeq * rhodeq * w3r ** 2 * w2i * w2r ** 2
            - rhogsol * k ** 4 * cs ** 4 * rhogeq * rhodeq * w3r ** 2 * w2r ** 2 * w3i
            - 2
            * rhogsol
            * k ** 4
            * cs ** 4
            * rhogeq
            * rhodeq
            * w3i
            * w3r
            * w2i ** 2
            * w2r
            + w1r ** 2 * rhogsol * rhogeq * k ** 2 * cs ** 2 * rhodeq * w3i * w2r ** 4
            + w1r ** 2 * rhogsol * rhogeq * k ** 2 * cs ** 2 * rhodeq * w3i * w2i ** 4
            + 2
            * w1r ** 2
            * rhogsol
            * rhogeq
            * k ** 2
            * cs ** 2
            * rhodeq
            * w3i
            * w2r ** 2
            * w2i ** 2
            + w1r ** 2 * rhogsol * rhogeq * k ** 2 * cs ** 2 * rhodeq * w3r ** 4 * w2i
            + 4
            * w1r ** 2
            * rhogsol
            * rhogeq
            * k ** 2
            * cs ** 2
            * rhodeq
            * w2r
            * w3r ** 3
            * w2i
            + 2
            * w1r ** 2
            * rhogsol
            * rhogeq
            * k ** 2
            * cs ** 2
            * rhodeq
            * w3i ** 2
            * w2i
            * w3r ** 2
            + vdsol * Kdrag * k * w1r * rhogeq * rhodeq * w2i * w2r ** 2 * w3i ** 3
            + rhogsol * k ** 4 * cs ** 4 * rhogeq * rhodeq * w2i ** 2 * w3i ** 3
            - vdsol
            * Kdrag
            * k
            * w1r
            * rhogeq
            * rhodeq
            * w2r
            * w3r
            * w2i ** 2
            * w3i ** 2
            + 4
            * w1r ** 2
            * rhogsol
            * rhogeq
            * k ** 2
            * cs ** 2
            * rhodeq
            * w3i
            * w3r
            * w2i ** 2
            * w2r
            + 4
            * w1r ** 2
            * rhogsol
            * rhogeq
            * k ** 2
            * cs ** 2
            * rhodeq
            * w3i
            * w3r
            * w2r ** 3
            + 4
            * w1r ** 2
            * rhogsol
            * rhogeq
            * k ** 2
            * cs ** 2
            * rhodeq
            * w2r
            * w3r
            * w2i
            * w3i ** 2
            - rhogsol * k ** 4 * cs ** 4 * rhogeq * rhodeq * w3r ** 2 * w2i ** 3
            - vdsol * Kdrag * k * w1r * rhogeq * rhodeq * w3r ** 3 * w2r ** 3
            + w1i
            * vgsol
            * w1r ** 2
            * k ** 3
            * cs ** 2
            * rhogeq ** 2
            * rhodeq
            * w2r
            * w3i ** 2
            - vdsol * Kdrag * k * w1r * rhogeq * rhodeq * w2r * w3r ** 3 * w2i ** 2
            + vdsol * Kdrag * k * w1r * rhogeq * rhodeq * w2i ** 3 * w3i * w3r ** 2
            + vdsol
            * Kdrag
            * k
            * w1r
            * rhogeq
            * rhodeq
            * w2i
            * w3i
            * w2r ** 2
            * w3r ** 2
            - vdsol * Kdrag * k * w1r * rhogeq * rhodeq * w3r * w2r ** 3 * w3i ** 2
            + vdsol * Kdrag * k * w1r * rhogeq * rhodeq * w2i ** 3 * w3i ** 3
            + Kdrag * rhogeq * k * vgsol * rhodeq * w2r ** 3 * w3i ** 4
            + Kdrag * rhogeq * k * vgsol * rhodeq * w2r * w3i ** 4 * w2i ** 2
            + 2
            * Kdrag
            * rhogeq
            * k
            * vgsol
            * rhodeq
            * w3r
            * w2r ** 2
            * w3i ** 2
            * w2i ** 2
            - w1i
            * vgsol
            * w1r ** 2
            * k ** 3
            * cs ** 2
            * rhogeq ** 2
            * rhodeq
            * w2r
            * w3r ** 2
            - w1i
            * vgsol
            * w1r ** 2
            * k ** 3
            * cs ** 2
            * rhogeq ** 2
            * rhodeq
            * w2r ** 2
            * w3r
            + w1i
            * vgsol
            * w1r ** 2
            * k ** 3
            * cs ** 2
            * rhogeq ** 2
            * rhodeq
            * w2i ** 2
            * w3r
            + 2
            * w1i
            * vgsol
            * w1r ** 2
            * k ** 3
            * cs ** 2
            * rhogeq ** 2
            * rhodeq
            * w3r
            * w2i
            * w3i
            + 2
            * w1i
            * vgsol
            * w1r ** 2
            * k ** 3
            * cs ** 2
            * rhogeq ** 2
            * rhodeq
            * w2r
            * w2i
            * w3i
            + 2
            * Kdrag
            * rhogeq
            * k
            * vgsol
            * rhodeq
            * w2r
            * w3i ** 2
            * w2i ** 2
            * w3r ** 2
            + Kdrag * rhogeq * k * vgsol * rhodeq * w3r * w2r ** 4 * w3i ** 2
            + Kdrag * rhogeq * k * vgsol * rhodeq * w3r * w2i ** 4 * w3i ** 2
            + Kdrag * rhogeq * k * vgsol * rhodeq * w3r ** 3 * w2i ** 4
            + Kdrag * rhogeq * k * vgsol * rhodeq * w3r ** 3 * w2r ** 4
            + 2 * Kdrag * rhogeq * k * vgsol * rhodeq * w2r ** 3 * w3i ** 2 * w3r ** 2
            + Kdrag * rhogeq * k * vgsol * rhodeq * w2r * w2i ** 2 * w3r ** 4
            + 2 * Kdrag * rhogeq * k * vgsol * rhodeq * w3r ** 3 * w2r ** 2 * w2i ** 2
            - vgsol
            * w1r ** 2
            * k ** 3
            * cs ** 2
            * rhogeq ** 2
            * rhodeq
            * w2r
            * w3i ** 3
            + vgsol
            * w1r ** 2
            * k ** 3
            * cs ** 2
            * rhogeq ** 2
            * rhodeq
            * w2r
            * w3i
            * w2i ** 2
            + 2
            * vgsol
            * w1r ** 2
            * k ** 3
            * cs ** 2
            * rhogeq ** 2
            * rhodeq
            * w2r
            * w2i
            * w3i ** 2
            - 2
            * vgsol
            * w1r ** 2
            * k ** 3
            * cs ** 2
            * rhogeq ** 2
            * rhodeq
            * w3i
            * w3r
            * w2r ** 2
            + vgsol
            * w1r ** 2
            * k ** 3
            * cs ** 2
            * rhogeq ** 2
            * rhodeq
            * w2r ** 3
            * w3i
            - vgsol
            * w1r ** 2
            * k ** 3
            * cs ** 2
            * rhogeq ** 2
            * rhodeq
            * w3r
            * w2i ** 3
        )

        rhod1i = rhod1i - (
            2
            * vgsol
            * w1r ** 2
            * k ** 3
            * cs ** 2
            * rhogeq ** 2
            * rhodeq
            * w3i
            * w3r
            * w2i ** 2
            + 2
            * w1r
            * cs ** 2
            * k ** 2
            * rhogsol
            * w1i
            * rhogeq
            * rhodeq
            * w2r ** 3
            * w3i ** 2
            - vgsol
            * w1r ** 2
            * k ** 3
            * cs ** 2
            * rhogeq ** 2
            * rhodeq
            * w2r
            * w3i
            * w3r ** 2
            - vgsol
            * w1r ** 2
            * k ** 3
            * cs ** 2
            * rhogeq ** 2
            * rhodeq
            * w3r
            * w2i
            * w2r ** 2
            + vgsol
            * w1r ** 2
            * k ** 3
            * cs ** 2
            * rhogeq ** 2
            * rhodeq
            * w3r ** 3
            * w2i
            - 2
            * vgsol
            * w1r ** 2
            * k ** 3
            * cs ** 2
            * rhogeq ** 2
            * rhodeq
            * w2r
            * w2i
            * w3r ** 2
            - vdsol
            * Kdrag
            * k ** 3
            * cs ** 2
            * rhogeq
            * rhodeq
            * w2r
            * w2i ** 2
            * w3i ** 2
            + vdsol
            * Kdrag
            * k ** 3
            * cs ** 2
            * rhogeq
            * rhodeq
            * w3r
            * w2r ** 2
            * w3i ** 2
            - vdsol
            * Kdrag
            * k ** 3
            * cs ** 2
            * rhogeq
            * rhodeq
            * w3r
            * w2i ** 2
            * w3i ** 2
            - 2
            * vdsol
            * Kdrag
            * k ** 3
            * cs ** 2
            * rhogeq
            * rhodeq
            * w2i
            * w3i
            * w3r
            * w2r ** 2
            - 2
            * vdsol
            * Kdrag
            * k ** 3
            * cs ** 2
            * rhogeq
            * rhodeq
            * w2r
            * w2i
            * w3r ** 2
            * w3i
            - 2
            * vdsol
            * Kdrag
            * k ** 3
            * cs ** 2
            * rhogeq
            * rhodeq
            * w2i ** 3
            * w3i
            * w3r
            - vdsol * Kdrag * k ** 3 * cs ** 2 * rhogeq * rhodeq * w2r ** 3 * w3i ** 2
            - 2
            * vdsol
            * Kdrag
            * k ** 3
            * cs ** 2
            * rhogeq
            * rhodeq
            * w2r
            * w2i
            * w3i ** 3
            + vdsol
            * Kdrag
            * k ** 3
            * cs ** 2
            * rhogeq
            * rhodeq
            * w2r
            * w2i ** 2
            * w3r ** 2
            + 2
            * vgsol
            * k ** 3
            * cs ** 2
            * rhogeq ** 2
            * w1i ** 3
            * rhodeq
            * w3r
            * w2i
            * w3i
            + vgsol
            * k ** 3
            * cs ** 2
            * rhogeq ** 2
            * w1i ** 3
            * rhodeq
            * w2r
            * w3i ** 2
            + w1i * Kdrag * rhogeq * k * vgsol * rhodeq * w3i ** 3 * w2r ** 3
            + w1i
            * Kdrag
            * rhogeq
            * k
            * vgsol
            * rhodeq
            * w3i
            * w2i ** 2
            * w3r ** 2
            * w2r
            + w1i * Kdrag * rhogeq * k * vgsol * rhodeq * w3r * w2i ** 3 * w3i ** 2
            + w1i
            * Kdrag
            * rhogeq
            * k
            * vgsol
            * rhodeq
            * w3r
            * w2r ** 2
            * w2i
            * w3i ** 2
            + w1i * Kdrag * rhogeq * k * vgsol * rhodeq * w2r * w2i ** 2 * w3i ** 3
            - 2
            * w1r
            * Kdrag
            * rhogeq
            * k
            * vgsol
            * rhodeq
            * w3i ** 2
            * w2i ** 2
            * w2r ** 2
            + w1i * Kdrag * rhogeq * k * vgsol * rhodeq * w2i ** 3 * w3r ** 3
            + w1i * Kdrag * rhogeq * k * vgsol * rhodeq * w3r ** 3 * w2i * w2r ** 2
            + w1i * Kdrag * rhogeq * k * vgsol * rhodeq * w3i * w2r ** 3 * w3r ** 2
            - w1r * Kdrag * rhogeq * k * vgsol * rhodeq * w3i ** 2 * w2i ** 4
            - w1r * Kdrag * rhogeq * k * vgsol * rhodeq * w3i ** 4 * w2i ** 2
            - 3
            * w1r
            * Kdrag
            * rhogeq
            * k
            * vgsol
            * rhodeq
            * w2i
            * w3i
            * w2r ** 2
            * w3r ** 2
            - 2
            * w1r
            * Kdrag
            * rhogeq
            * k
            * vgsol
            * rhodeq
            * w3r ** 2
            * w2i ** 2
            * w2r ** 2
            - w1r * Kdrag * rhogeq * k * vgsol * rhodeq * w2r ** 4 * w3i ** 2
            - w1r * Kdrag * rhogeq * k * vgsol * rhodeq * w3i ** 4 * w2r ** 2
            - 3 * w1r * Kdrag * rhogeq * k * vgsol * rhodeq * w2i * w2r ** 2 * w3i ** 3
            - 3 * w1r * Kdrag * rhogeq * k * vgsol * rhodeq * w2i ** 3 * w3i ** 3
            + vgsol
            * w1r ** 2
            * k ** 3
            * cs ** 2
            * rhogeq ** 2
            * rhodeq
            * w3r
            * w2i
            * w3i ** 2
            + Kdrag * rhogeq * k * vgsol * rhodeq * w2r ** 3 * w3r ** 4
            - 2
            * w1r
            * Kdrag
            * rhogeq
            * k
            * vgsol
            * rhodeq
            * w3r ** 2
            * w2i ** 2
            * w3i ** 2
            - 2
            * w1r
            * Kdrag
            * rhogeq
            * k
            * vgsol
            * rhodeq
            * w3r ** 2
            * w2r ** 2
            * w3i ** 2
            - w1r
            * Kdrag
            * rhogeq
            * k
            * vgsol
            * rhodeq
            * w2r
            * w3r
            * w2i ** 2
            * w3i ** 2
            - 3 * w1r * Kdrag * rhogeq * k * vgsol * rhodeq * w2i ** 3 * w3i * w3r ** 2
            - w1r * Kdrag * rhogeq * k * vgsol * rhodeq * w3r ** 2 * w2i ** 4
            - w1r * Kdrag * rhogeq * k * vgsol * rhodeq * w3r ** 2 * w2r ** 4
            - w1r * Kdrag * rhogeq * k * vgsol * rhodeq * w3r ** 4 * w2r ** 2
            - w1r * Kdrag * rhogeq * k * vgsol * rhodeq * w3r ** 3 * w2r ** 3
            - w1r * Kdrag * rhogeq * k * vgsol * rhodeq * w2r * w3r ** 3 * w2i ** 2
            + rhogsol * rhogeq * k ** 2 * cs ** 2 * w1r ** 4 * rhodeq * w2i * w3r ** 2
            + rhogsol * rhogeq * k ** 2 * cs ** 2 * w1r ** 4 * rhodeq * w2i ** 2 * w3i
            + rhogsol * rhogeq * k ** 2 * cs ** 2 * w1r ** 4 * rhodeq * w2i * w3i ** 2
            + rhogsol * rhogeq * k ** 2 * cs ** 2 * w1r ** 4 * rhodeq * w2r ** 2 * w3i
            - w1r * Kdrag * rhogeq * k * vgsol * rhodeq * w3r ** 4 * w2i ** 2
            - w1i * vgsol * k ** 3 * cs ** 2 * rhogeq ** 2 * rhodeq * w2r ** 4 * w3r
            - w1i * vgsol * k ** 3 * cs ** 2 * rhogeq ** 2 * rhodeq * w3r * w2i ** 4
            - 2
            * w1i
            * vgsol
            * k ** 3
            * cs ** 2
            * rhogeq ** 2
            * rhodeq
            * w2i ** 3
            * w3i
            * w3r
            - 2
            * w1i
            * vgsol
            * k ** 3
            * cs ** 2
            * rhogeq ** 2
            * rhodeq
            * w2i ** 2
            * w3r
            * w2r ** 2
            - 2
            * w1i
            * vgsol
            * k ** 3
            * cs ** 2
            * rhogeq ** 2
            * rhodeq
            * w2i
            * w3i
            * w3r
            * w2r ** 2
            - 2
            * w1i
            * vgsol
            * k ** 3
            * cs ** 2
            * rhogeq ** 2
            * rhodeq
            * w3r
            * w2r ** 2
            * w3i ** 2
            - w1i * vgsol * k ** 3 * cs ** 2 * rhogeq ** 2 * rhodeq * w2r * w3i ** 4
            - 2
            * w1i
            * vgsol
            * k ** 3
            * cs ** 2
            * rhogeq ** 2
            * rhodeq
            * w2r
            * w2i
            * w3i ** 3
            - 2
            * w1i
            * vgsol
            * k ** 3
            * cs ** 2
            * rhogeq ** 2
            * rhodeq
            * w2r
            * w3i ** 2
            * w3r ** 2
            - 2
            * w1i
            * vgsol
            * k ** 3
            * cs ** 2
            * rhogeq ** 2
            * rhodeq
            * w2r ** 3
            * w3r ** 2
            - w1i * vgsol * k ** 3 * cs ** 2 * rhogeq ** 2 * rhodeq * w2r * w3r ** 4
            - w1i * vgsol * Kdrag ** 2 * k * rhodeq * w3r * w2i ** 2 * w3i ** 2
            - w1i * vgsol * Kdrag ** 2 * k * rhodeq * w3r * w2r ** 2 * w3i ** 2
            - w1i * vgsol * Kdrag ** 2 * k * rhodeq * w2r ** 3 * w3i ** 2
            - w1r * Kdrag * rhogeq * k * vgsol * rhodeq * w3r * w2r ** 3 * w3i ** 2
        )
        rhod1i = (
            rhod1i
            / (
                w1i ** 2
                - 2 * w3i * w1i
                + w3r ** 2
                + w1r ** 2
                + w3i ** 2
                - 2 * w3r * w1r
            )
            / (w3r ** 2 + w3i ** 2)
            / Kdrag
            / (
                w2r ** 2
                + w1r ** 2
                + w2i ** 2
                - 2 * w2i * w1i
                - 2 * w2r * w1r
                + w1i ** 2
            )
            / rhogeq
            / (w2i ** 2 + w2r ** 2)
        )

    # ------------------------------------------------------------------
    # FINALIZE RETURN ARRAYS
    # ------------------------------------------------------------------

    vgas = vdust = rhogas = rhodust = np.zeros_like(xposition)

    for i in range(xposition.shape[0]):

        xk = 2.0 * np.pi / wavelength * (xposition[i] - x0)
        arg1 = xk - w1r * time
        arg2 = xk - w2r * time
        arg3 = xk - w3r * time

        vgas[i] = (
            vgeq
            + vg1r * np.exp(w1i * time) * np.cos(arg1)
            - vg1i * np.exp(w1i * time) * np.sin(arg1)
            + vg2r * np.exp(w2i * time) * np.cos(arg2)
            - vg2i * np.exp(w2i * time) * np.sin(arg2)
            + vg3r * np.exp(w3i * time) * np.cos(arg3)
            - vg3i * np.exp(w3i * time) * np.sin(arg3)
        )

        vdust[i] = (
            vdeq
            + vd1r * np.exp(w1i * time) * np.cos(arg1)
            - vd1i * np.exp(w1i * time) * np.sin(arg1)
            + vd2r * np.exp(w2i * time) * np.cos(arg2)
            - vd2i * np.exp(w2i * time) * np.sin(arg2)
            + vd3r * np.exp(w3i * time) * np.cos(arg3)
            - vd3i * np.exp(w3i * time) * np.sin(arg3)
        )

        rhogas[i] = (
            rhogeq
            + rhog1r * np.exp(w1i * time) * np.cos(arg1)
            - rhog1i * np.exp(w1i * time) * np.sin(arg1)
            + rhog2r * np.exp(w2i * time) * np.cos(arg2)
            - rhog2i * np.exp(w2i * time) * np.sin(arg2)
            + rhog3r * np.exp(w3i * time) * np.cos(arg3)
            - rhog3i * np.exp(w3i * time) * np.sin(arg3)
        )

        rhodust[i] = (
            rhodeq
            + rhod1r * np.exp(w1i * time) * np.cos(arg1)
            - rhod1i * np.exp(w1i * time) * np.sin(arg1)
            + rhod2r * np.exp(w2i * time) * np.cos(arg2)
            - rhod2i * np.exp(w2i * time) * np.sin(arg2)
            + rhod3r * np.exp(w3i * time) * np.cos(arg3)
            - rhod3i * np.exp(w3i * time) * np.sin(arg3)
        )

    return rhogas, rhodust, vgas, vdust
