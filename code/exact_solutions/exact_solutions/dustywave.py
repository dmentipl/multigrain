"""
DUSTYWAVE exact solution

See the following references:
- Laibe and Price (2011) MNRAS, 418, 1491
- Laibe and Price (2014) MNRAS, 444, 1940
- Splash source code src/exact_dustywave.f90

Daniel Mentiplay, 2019.
"""

from ._dustywave import exact_dustywave


def rho_gas(time, ampl, cs, Kdrag, wavelength, x0, rhog0, rhod0, xposition):
    """
    Exact solution for gas density in DUSTYWAVE problem.

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
    (N,) ndarray
        Gas density.

    References
    ----------
    See Laibe and Price (2011) MNRAS, 418, 1491, and see in Splash
    source code src/exact_dustywave.f90.
    """
    return exact_dustywave(
        time, ampl, cs, Kdrag, wavelength, x0, rhog0, rhod0, xposition
    )[0]


def rho_dust(time, ampl, cs, Kdrag, wavelength, x0, rhog0, rhod0, xposition):
    """
    Exact solution for dusty density in DUSTYWAVE problem.

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
    (N,) ndarray
        Dust density.

    References
    ----------
    See Laibe and Price (2011) MNRAS, 418, 1491, and see in Splash
    source code src/exact_dustywave.f90.
    """
    return exact_dustywave(
        time, ampl, cs, Kdrag, wavelength, x0, rhog0, rhod0, xposition
    )[1]


def vx_gas(time, ampl, cs, Kdrag, wavelength, x0, rhog0, rhod0, xposition):
    """
    Exact solution for gas velocity in DUSTYWAVE problem.

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
    (N,) ndarray
        Gas velocity.

    References
    ----------
    See Laibe and Price (2011) MNRAS, 418, 1491, and see in Splash
    source code src/exact_dustywave.f90.
    """
    return exact_dustywave(
        time, ampl, cs, Kdrag, wavelength, x0, rhog0, rhod0, xposition
    )[2]


def vx_dust(time, ampl, cs, Kdrag, wavelength, x0, rhog0, rhod0, xposition):
    """
    Exact solution for dust velocity in DUSTYWAVE problem.

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
    (N,) ndarray
        Dust velocity.

    References
    ----------
    See Laibe and Price (2011) MNRAS, 418, 1491, and see in Splash
    source code src/exact_dustywave.f90.
    """
    return exact_dustywave(
        time, ampl, cs, Kdrag, wavelength, x0, rhog0, rhod0, xposition
    )[3]
