
import numpy as np
from fractions import Fraction
import figurefirst as fifi
import fly_plot_lib_plot as fpl


def plot_trajectory(xpos, ypos, phi, color, ax=None, size_radius=None, nskip=0,
                    colormap='bone_r', colornorm=None, edgecolor='none', reverse=False):
    if color is None:
        color = phi

    color = np.array(color)

    # Set size radius
    xymean = np.mean(np.abs(np.hstack((xpos, ypos))))
    if size_radius is None:  # auto set
        xymean = 0.21*xymean
        if xymean < 0.0001:
            sz = np.array(0.01)
        else:
            sz = np.hstack((xymean, 1))
        size_radius = sz[sz > 0][0]
    else:
        if isinstance(size_radius, list):  # scale defualt by scalar in list
            xymean = size_radius[0] * xymean
            sz = np.hstack((xymean, 1))
            size_radius = sz[sz > 0][0]
        else:  # use directly
            size_radius = size_radius

    if colornorm is None:
        colornorm = [np.min(color), np.max(color)]

    if reverse:
        xpos = np.flip(xpos, axis=0)
        ypos = np.flip(ypos, axis=0)
        phi = np.flip(phi, axis=0)
        color = np.flip(color, axis=0)

    fpl.colorline_with_heading(ax, np.flip(xpos), np.flip(ypos), np.flip(color, axis=0), np.flip(phi),
                               nskip=nskip,
                               size_radius=size_radius,
                               deg=False,
                               colormap=colormap,
                               center_point_size=0.0001,
                               colornorm=colornorm,
                               show_centers=False,
                               size_angle=20,
                               alpha=1,
                               edgecolor=edgecolor)

    ax.set_aspect('equal')
    xrange = xpos.max() - xpos.min()
    xrange = np.max([xrange, 0.1])
    yrange = ypos.max() - ypos.min()
    yrange = np.max([yrange, 0.1])

    if yrange < (size_radius/2):
        yrange = 10

    if xrange < (size_radius/2):
        xrange = 10

    ax.set_xlim(xpos.min()-0.2*xrange, xpos.max()+0.2*xrange)
    ax.set_ylim(ypos.min()-0.2*yrange, ypos.max()+0.2*yrange)

    # fifi.mpl_functions.adjust_spines(ax, [])


def pi_yaxis(ax, tickpispace=0.5, lim=None, real_lim=None):
    if lim is None:
        ax.set_ylim(-1 * np.pi, 1 * np.pi)
    else:
        ax.set_ylim(lim)

    lim = ax.get_ylim()
    ticks = np.arange(lim[0], lim[1] + 0.01, tickpispace * np.pi)
    tickpi = np.round(ticks / np.pi, 3)
    y0 = abs(tickpi) < np.finfo(float).eps  # find 0 entry, if present

    tickslabels = tickpi.tolist()
    for y in range(len(tickslabels)):
        tickslabels[y] = ('$' + str(Fraction(tickslabels[y])) + '\pi $')

    tickslabels = np.asarray(tickslabels, dtype=object)
    tickslabels[y0] = '0'  # replace 0 entry with 0 (instead of 0*pi)
    ax.set_yticks(ticks)
    ax.set_yticklabels(tickslabels)

    if real_lim is None:
        real_lim = np.zeros(2)
        real_lim[0] = lim[0] - 0.4
        real_lim[1] = lim[1] + 0.4

    ax.set_ylim(real_lim)

    if lim is None:
        ax.set_ylim(-1 * np.pi, 1 * np.pi)
    else:
        ax.set_ylim(lim)


def pi_xaxis(ax, tickpispace=0.5, lim=None):
    if lim is None:
        ax.set_xlim(-1 * np.pi, 1 * np.pi)
    else:
        ax.set_xlim(lim)

    lim = ax.get_xlim()
    ticks = np.arange(lim[0], lim[1] + 0.01, tickpispace * np.pi)
    tickpi = ticks / np.pi
    x0 = abs(tickpi) < np.finfo(float).eps  # find 0 entry, if present

    tickslabels = tickpi.tolist()
    for x in range(len(tickslabels)):
        tickslabels[x] = ('$' + str(Fraction(tickslabels[x])) + '\pi$')

    tickslabels = np.asarray(tickslabels, dtype=object)
    tickslabels[x0] = '0'  # replace 0 entry with 0 (instead of 0*pi)
    ax.set_xticks(ticks)
    ax.set_xticklabels(tickslabels)


def circplot(t, phi, jump=np.pi):
    """ Stitches t and phi to make unwrapped circular plot. """

    t = np.squeeze(t)
    phi = np.squeeze(phi)

    difference = np.abs(np.diff(phi, prepend=phi[0]))
    ind = np.squeeze(np.array(np.where(difference > jump)))

    phi_stiched = np.copy(phi)
    t_stiched = np.copy(t)
    for i in range(phi.size):
        if np.isin(i, ind):
            phi_stiched = np.concatenate((phi_stiched[0:i], [np.nan], phi_stiched[i+1:None]))
            t_stiched = np.concatenate((t_stiched[0:i], [np.nan], t_stiched[i+1:None]))

    return t_stiched, phi_stiched


def wrapTo2Pi(rad):
    rad = rad % (2 * np.pi)
    return rad


def wrapToPi(rad):
    q = (rad < -np.pi) | (np.pi < rad)
    rad[q] = ((rad[q] + np.pi) % (2 * np.pi)) - np.pi
    return rad
