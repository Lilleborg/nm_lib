import numpy as np

from nm_lib import nm_lib as nm
from nm_lib import utils as utils


def get_uu_t0(xx: np.ndarray) -> np.ndarray:
    nom = np.cos(6 * np.pi * xx / 5) ** 2
    den = np.cosh(5 * xx**2)
    return nom / den


def get_u_exact(xx: np.ndarray, t: float, a: float) -> np.ndarray:
    """
    Calculates the exact solution of the advective burgers equation with
    constant advection speed at time t, with periodic boundaries in x.

    Arguments:
        xx {np.ndarray} -- discretized interval of x-values
        t {float} -- time coordinate to evaluate the solution for
        a {float} -- advection speed (could also be an array)

    Returns:
        np.ndarray -- the exact solution at time t
    """
    xmin = xx[0]  # pick out min x value to shift the x-array to start at x_0 = 0
    xmax = xx[-1]  # pick out max x value to calculate the length of the interval
    x_start_0 = xx - xmin  # new x-array with the interval shifted to so the first element is zero
    # y = x - at, but here I use the new shifted x-array so I can use the modulo operator for periodic boundaries
    # after calculating, shift the y-array back to match with the original interval:
    y = (x_start_0 - a * t) % (xmax - xmin) + xmin
    return get_uu_t0(y)


def test_ex_2b():
    maxabserr_testing = np.array([0.48993726, 0.35731628, 0.22902555, 0.13157007, 0.07040167, 0.03637074])
    check_time = 52
    nr_increases = 5  # number of times increasing space resolution
    b2s = 6  # start number of intervals in base 2
    nints = np.logspace(b2s, b2s + nr_increases, nr_increases + 1, base=2, dtype=int)
    x0 = -2.6
    xf = 2.6
    a = -1
    maxabserr = np.zeros(len(nints))
    for i, nint in enumerate(nints):
        xx, dx = utils.get_xx(nint, x0, xf)
        nt = int((check_time + 10) / dx[1])  # Set nr times steps high enough to reach the desired check_time
        uu_t0 = get_uu_t0(xx)
        tt, uunt = nm.evolv_adv_burgers(
            xx,
            uu_t0,
            nt=nt,
            a=a,
            ddx=lambda x, u: nm.deriv_dnw(x, u, method="roll"),
            bnd_limits=[0, 1],
        )
        id_time_check = np.argmin(np.abs(tt - check_time))  # Get index of time closes to check_time
        u_exact_check = get_u_exact(xx, tt[id_time_check], a)  # Exact value at check time
        maxabserr[i] = np.max(np.abs(uunt[id_time_check] - u_exact_check))  # Max error evaluated at check time
    close_check = np.isclose(maxabserr, maxabserr_testing)
    maxabserr < maxabserr_testing
    msg = (
        f"Max absolute error at t={check_time:d} has changed for nints={nints[~close_check]}."
        f"{maxabserr[~close_check]} -> {maxabserr_testing[~close_check]}"
    )
    # if np.any(smaller_check):
    #     msg += f"It is now smaller than it used to be for nints={nints[smaller_check]}. Maybe you did a great job?"
    # if np.any(~smaller_check):
    #     msg += f"It is now bigger than it used to be for nints={nints[~smaller_check]}. Buhu!"
    assert np.all(close_check), msg
