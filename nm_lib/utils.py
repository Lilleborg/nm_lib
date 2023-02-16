from typing import Callable

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation


def get_xx(nint: int, x0: float, xf: float) -> tuple[np.ndarray, np.ndarray]:
    """
    Generate an array of x values from x0 to xf with nint number of intervals
    between points.

    Arguments:
        nint {int} -- number of intervals between grid points

    Keyword Arguments:
        xf {float} -- last value in output array (default: {10.0})
        x0 {float} -- first value in output array (default: {-4.0})

    Returns:
        tuple[np.ndarray, float] -- (the resulting x array, spacing between points x[1]-x[0])
    """
    x = np.arange(nint + 1) / nint * (xf - x0) + x0
    return x, np.roll(x, -1) - x


def order_conv(hh: np.ndarray, hh2: np.ndarray, hh4: np.ndarray, **kwargs) -> np.ndarray:
    """
    Computes the order of convergence of a derivative function.

    Parameters
    ----------
    hh : `array`
        Function that depends on xx.
    hh2 : `array`
        Function that depends on xx but with twice number of grid points than hh.
    hh4 : `array`
        Function that depends on xx but with twice number of grid points than hh2.
    Returns
    -------
    `array`
        The order of convergence.
    """
    return np.ma.log2((hh4[::4] - hh2[::2]) / (hh2[::2] - hh))


def animate_u(
    tt: np.ndarray,
    uunt: np.ndarray,
    xx: np.ndarray,
    a: float = None,
    exact: Callable[[np.ndarray, float], np.ndarray] = None,
    initial: np.ndarray = None,
) -> FuncAnimation:
    """
    Animates a function u(x,t) in the interval xx over time tt.

    Arguments:
        uunt {np.ndarray} -- [len(tt),len(xx)] array with the solution of u(x,t)
        xx {np.ndarray} -- spacial coordinate
        tt {np.ndarray} -- time array
        a {float or None} -- advection speed, if not constant pass in None

    Keyword Arguments:
        exact {callable} -- if given, a callable returning the 1D array with exact solution, signature (x,t) (default: {None})
        initial {np.ndarray or None} -- if given, the initial state of uunt (default: {None})
    """
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(10, 5))
    title_str_app = f" a={a:.2f}" if a is not None else ""
    print(f"Animation with nint={len(xx)-1:d}" + title_str_app + f", nframes={len(tt):d}:")

    def init():
        ax.plot(xx, uunt[0, :])

    def animate(i):
        ax.clear()
        ax.plot(xx, uunt[i, :], label="uunt")
        if initial is None:
            ax.plot(xx, uunt[0, :], ls=":", label="init")
        else:
            ax.plot(xx, initial, ls=":", label="init")
        if exact is not None:
            ax.plot(xx, exact(xx, tt[i], a), ls="--", label="exact")
        ax.set_title(f"t={tt[i]:.2f}, nint={len(xx)-1:d}" + title_str_app)
        ax.legend(loc=1)

    return FuncAnimation(fig, animate, interval=200, frames=len(tt), init_func=init)


def animate_us(tt: np.ndarray, uunt_dict: dict[str, np.ndarray], xx: np.ndarray) -> FuncAnimation:
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(10, 5))
    print(f"Animation with nint={len(xx)-1:d}, nframes={len(tt):d}:")

    def init():
        for uunt in uunt_dict.values():
            ax.plot(xx, uunt[0, :])

    def animate(i):
        ax.clear()
        for name, uunt in uunt_dict.items():
            ax.plot(xx, uunt[i, :], label=name)
        ax.set_title(f"t={tt[i]:.2f}, nint={len(xx)-1:d}")
        ax.legend(loc=1)

    return FuncAnimation(fig, animate, interval=200, frames=len(tt), init_func=init)


def instability_maxabs(tt: np.ndarray, uunt: np.ndarray, crit_value: float) -> tuple[int, float]:
    """
    Finds the first index in temporal direction where the max absolute value of
    the spacial part of uunt is larger than the critical value.

    Arguments:
        uunt {np.ndarray} -- [len(tt),len(xx)] array with the solution of u(x,t)
        tt {np.ndarray} -- time coordinates
        crit_value {float} -- the critical value used in the max absolute value check

    Returns:
        tuple[int, float] --    1) the index of time where the check fails
                                2) the specific value of time where the check fails
    """
    # Find first index of time where max abs in uunt > crit_value:
    id_crit = np.argmax(np.max(np.abs(uunt), axis=1) > crit_value)
    if id_crit == 0:
        crit_time = None
        print(f"Instability not found for t<{tt[-1]:.2f}")
    else:
        crit_time = tt[id_crit]
        print(f"Instability after {id_crit:d} timesteps, t={crit_time:.2f}")
    return id_crit, crit_time
