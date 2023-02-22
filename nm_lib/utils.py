from typing import Tuple, Union, Callable

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation
from scipy.interpolate import interp1d


def get_xx(nint: int, x0: float, xf: float) -> Tuple[np.ndarray, np.ndarray]:
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


def get_periodic_value(xmin: float, xmax: float, value: Union[float, np.ndarray]) -> np.ndarray:
    """
    Returns values within and interval with periodic boundaries. Supports both
    scalar and array values, but will always return an array.

    Arguments:
        xmin {float} -- min value of interval
        xmax {float} -- max value of interval
        value {Union[float, np.ndarray]} -- value(s) to be evaluated periodicly. Any value outside
        [xmin, xmax] is transformed to lay inside the interval using periodic boundaries.

    Returns:
        np.ndarray -- the returning values on periodic interval
    """
    if isinstance(value, (float, int)):
        value = np.array([value])
    interval = xmax - xmin
    value_zeroed = value - xmin
    value_periodic = value_zeroed % interval + xmin
    # If any values in value exactly equal xmax, it will be sat to xmin, which is fine but confusing. Adjust it back:
    value_periodic[np.nonzero(value == xmax)] = xmax
    return value_periodic


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

    return FuncAnimation(fig, animate, interval=3000 / len(tt), frames=len(tt), init_func=init)


def animate_us(
    sol_dict: dict[str, Tuple[np.ndarray, np.ndarray]],
    tt_master: np.ndarray,
    xx: np.ndarray,
    initial: np.ndarray = None,
) -> FuncAnimation:
    """
    Animates a series of functions u(x,t) in the interval xx over time
    tt_master. The individual functions are interpolated across time in order
    to animate the different solutions at the same time values.

    Arguments:
        sol_dict {dict[str, Tuple[np.ndarray, np.ndarray]]} -- dictionary with the different solutions
        where the key is the name, the value is the solutions time array and the array u(x,t)
        tt_master {np.ndarray} -- array with the time to be used for animation
        xx {np.ndarray} -- the spacial coordinate

    Keyword Arguments:
        initial {np.ndarray} -- if given, an array to be plotted as the initial state (default: {None})

    Returns:
        FuncAnimation -- the resulting object from using matplotlib.animation.FuncAnimation
    """
    print(f"Animation with nint={len(xx)-1:d}, nframes={len(tt_master)}:")
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(10, 5))
    # Set up interpolation for each solution in order to plot at the same time values
    uu_of_t_dict = {}
    for name, (tt, uunt) in sol_dict.items():
        uu_of_t_dict[name] = interp1d(tt, uunt, axis=0)

    def init():
        if initial is not None:
            ax.plot(xx, initial, label="init", ls=":")
        else:
            ax.plot([], [])

    def animate(t):
        ax.clear()
        for name, uu_of_t in uu_of_t_dict.items():
            ax.plot(xx, uu_of_t(t), label=name)
        ax.set_title(f"t={t:.2f}, nint={len(xx)-1:d}")
        ax.legend(loc=1)
        # ax.set_ylim(-0.1 * np.max(uunt), np.max(uunt) * 1.1)
        ax.set_xlabel("x")
        ax.set_ylabel("u(x,t)")

    return FuncAnimation(fig, animate, interval=3000 / len(tt_master), frames=tt_master, init_func=init)


def instability_maxabs(tt: np.ndarray, uunt: np.ndarray, crit_value: float) -> Tuple[int, float]:
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


def plot_snapshot(
    sol_dict: dict[str, Tuple[np.ndarray, np.ndarray]],
    xx: np.ndarray,
    time_stamps: np.ndarray,
    initial: np.ndarray = None,
) -> Tuple[plt.Figure, plt.Axes]:
    """
    Plot a series of functions u(x,t) in the interval xx at specific time
    stamps.

    Arguments:
        sol_dict {dict[str, Tuple[np.ndarray, np.ndarray]]} -- dictionary with the different solutions
        where the key is the name, the value is the solutions time array and the array u(x,t)
        xx {np.ndarray} -- the spacial coordinate
        time_stamps {np.ndarray} -- the specific times to plot each solutions. If no more than 4 are
        given each curve will have a unique line style

    Keyword Arguments:
        initial {np.ndarray} -- if given, an array to be plotted as the initial state (default: {None})

    Returns:
        Tuple[plt.Figure, plt.Axes] -- the resulting figure and axis object
    """
    linestyles = ["-", "--", ":", "-."]
    if len(time_stamps) > len(linestyles):
        print(f"More time stamps than line styles, duplicated lines expected...")
    fig, ax = plt.subplots()
    if initial is not None:
        ax.plot(xx, initial, alpha=0.5, color="k", ls=":", label="init")
    for i, (name, (tt, uunt)) in enumerate(sol_dict.items()):
        for j, t in enumerate(time_stamps):
            uu_of_t = interp1d(tt, uunt, axis=0)
            ax.plot(xx, uu_of_t(t), label=name + f", t_i={t:.1f}", color=f"C{i}", ls=linestyles[j])
    ax.legend(bbox_to_anchor=(0.0, 1.02, 1.0, 0.102), loc="lower left", mode="expand", ncol=len(sol_dict))
    ax.set_xlabel("x")
    ax.set_ylabel("u(x,t_i)")

    return fig, ax
