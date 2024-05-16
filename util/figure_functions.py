import numpy as np
import scipy
import pandas as pd
from fractions import Fraction
import matplotlib.pyplot as plt
from matplotlib import cm, colors
from matplotlib.colors import ListedColormap, LinearSegmentedColormap
from matplotlib.patches import FancyArrowPatch
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

import figurefirst as fifi
import fly_plot_lib_plot as fpl
import utils

class FlyWindVectors:
    def __init__(self, phi=np.pi/4, g=0.4, psi=np.pi/10, w=0.3, zeta=-np.pi/4):
        """ Calculate air velocity vector from fly heading angle,
            ground velocity vector, and ambient wind velocity vector.
        """

        # Inputs
        self.phi = phi  # heading [rad]
        self.g = g  # ground velocity magnitude
        self.psi = psi  # ground velocity direction [rad]
        self.w = w  # ambient wind velocity magnitude
        self.zeta = zeta  # ambient wind velocity direction [rad]

        # Main variables
        self.phi_x = 0.0
        self.phi_y = 0.0

        self.v_para = 0.0
        self.v_perp = 0.0

        self.g_x = 0.0
        self.g_y = 0.0
        self.psi_global = 0.0

        self.a_para = 0.0
        self.a_perp = 0.0
        self.a = 0.0
        self.gamma = 0.0

        self.a_x = 0.0
        self.a_y = 0.0
        self.gamma_global = 0.0
        self.gamma_check = 0.0

        self.w_x = 0.0
        self.w_y = 0.0

        # Figure & axis
        self.fig = None
        self.ax = None

        # Run
        self.run()

    def run(self):
        """ Run main computations for fly-wind vector plot.
        """

        # Ground velocity in fly frame
        self.v_para = self.g * np.cos(self.psi)  # parallel ground velocity in fly frame
        self.v_perp = self.g * np.sin(self.psi)  # perpendicular ground velocity in fly frame

        # Air velocity in fly frame
        self.a_para = self.v_para - self.w * np.cos(self.phi - self.zeta)  # parallel air velocity in fly frame
        self.a_perp = self.v_perp + self.w * np.sin(self.phi - self.zeta)  # perpendicular air velocity in fly frame
        # self.a_para = self.v_para - self.w * np.cos(self.zeta - self.phi)
        # self.a_perp = self.v_perp - self.w * np.sin(self.zeta - self.phi)
        self.a = np.sqrt(self.a_para ** 2 + self.a_perp ** 2)  # air velocity magnitude
        self.gamma = np.arctan2(self.a_perp, self.a_para)  # air velocity direction [rad]
        # a_v = self.g * np.exp(self.psi * 1j) - self.w * np.exp((self.zeta - self.phi) * 1j)
        # self.a = np.abs(a_v)
        # self.gamma = np.angle(a_v)

        # Vector for heading, make same length as ground speed
        self.phi_x = self.g * np.cos(self.phi)  # heading x
        self.phi_y = self.g * np.sin(self.phi)  # heading y

        # Ground velocity in global frame
        self.psi_global = self.phi + self.psi  # direction of travel in global frame
        self.g_x = self.g * np.cos(self.psi_global)  # x-velocity in global frame
        self.g_y = self.g * np.sin(self.psi_global)  # y-velocity in global frame

        # Ambient wind velocity in global frame
        self.w_x = self.w * np.cos(self.zeta)  # ambient wind x in global frame
        self.w_y = self.w * np.sin(self.zeta)  # ambient wind y in global frame

        # Air velocity in global frame
        self.a_x = self.g_x - self.w_x  # x air velocity in global frame
        self.a_y = self.g_y - self.w_y  # y air velocity in global frame
        self.gamma_global = np.arctan2(self.a_y, self.a_x)  # air velocity direction in global frame
        self.gamma_check = self.gamma_global - self.phi  # air velocity direction in fly frame, should match gamma

    def compute_new_w(self, a_new=None):
        """ Compute the new ambient vector for a change in air speed
            while keeping ground velocity & air velocity direction the same.
        """

        # Set new air speed
        self.a = a_new

        # Compute new ambient wind
        w_v = self.g * np.exp(self.psi * 1j) - a_new * np.exp(self.gamma * 1j)
        self.w = np.abs(w_v)
        self.zeta = np.angle(w_v) + self.phi

        # Re-run
        self.run()

    def plot(self, ax=None, fly_origin=(0, 0), axis_size=None, axis_neg=True, show_arrow=True, fig_size=6,
             phi_color=(128/255, 128/255, 128/255),
             g_color=(32/255, 0/255, 255/255),
             a_color=(240/255, 118/255, 0/255),
             w_color=(47/255, 166/255, 0/255),
             lw=1.5, alpha=1.0):

        """ Plot fly wind vectors.
        """

        fly_origin = np.array(fly_origin)

        if axis_size is None:
            axis_size = 1.05*np.max(np.array([self.w, self.g, self.a]))

        if ax is None:
            fig, ax = plt.subplots(1, 1, figsize=(fig_size, fig_size), dpi=100)
            self.fig = fig
            self.ax = ax

        # Plot axes
        ax.plot([axis_size, -axis_neg*axis_size], [0, 0], '--', linewidth=1.0, color='gray')
        ax.plot([0.0, 0.0], [-axis_neg*axis_size, axis_size], '--', linewidth=1.0, color='gray')

        # Plot fly-wind vectors
        ax.plot([fly_origin[0], fly_origin[0] + self.phi_x], [fly_origin[1], fly_origin[1] + self.phi_y], '-',
                linewidth=lw, color=phi_color, alpha=alpha, label='$\phi$')

        ax.plot([fly_origin[0], fly_origin[0] + self.g_x], [fly_origin[1], fly_origin[1] + self.g_y], '-',
                linewidth=lw, color=g_color, alpha=alpha, label=r'$\bar{g}$')

        ax.plot([fly_origin[0], fly_origin[0] + self.a_x], [fly_origin[1], fly_origin[1] + self.a_y], '-',
                linewidth=lw, color=a_color, alpha=alpha, label=r'$\bar{a}$')

        ax.plot([fly_origin[0] + self.a_x, fly_origin[0] + self.w_x + self.a_x],
                [fly_origin[1] + self.a_y, fly_origin[1] + self.w_y + self.a_y], '-',
                linewidth=lw, color=w_color, alpha=alpha, label=r'$\bar{w}$')

        # ax.plot([fly_origin[0], self.w_x], [fly_origin[1], self.w_y], '-',
        #         linewidth=lw, color='limegreen', alpha=alpha)

        ax.legend()

        # Plot arrows
        if show_arrow:
            mut = 10

            arrow_phi = FancyArrowPatch(posA=fly_origin,
                                        posB=fly_origin + (self.phi_x, self.phi_y),
                                        mutation_scale=mut, color=phi_color)

            arrow_g = FancyArrowPatch(posA=fly_origin,
                                      posB=fly_origin + (self.g_x, self.g_y),
                                      mutation_scale=mut, color=g_color)

            arrow_a = FancyArrowPatch(posA=fly_origin,
                                      posB=fly_origin + (self.a_x, self.a_y),
                                      mutation_scale=mut, color=a_color)

            arrow_w = FancyArrowPatch(posA=fly_origin + (self.a_x, self.a_y),
                                      posB=fly_origin + (self.w_x + self.a_x, self.w_y + self.a_y),
                                      mutation_scale=mut, color=w_color)

            ax.add_patch(arrow_phi)
            ax.add_patch(arrow_g)
            ax.add_patch(arrow_w)
            ax.add_patch(arrow_a)

        # Set axis properties
        ax.set_aspect(1)
        ax.autoscale()

        ax.set_xlim(-axis_size, axis_size)
        ax.set_ylim(-axis_size, axis_size)

        fifi.mpl_functions.adjust_spines(ax, [])


class LatexStates:
    """Holds LaTex format corresponding to set symbolic variables.
    """
    def __init__(self):
        self.dict = {'v_para': r'$v_{\parallel}$',
                     'v_perp': r'$v_{\perp}$',
                     'phi': r'$\phi$',
                     'phidot': r'$\dot{\phi}$',
                     'w': r'$w$',
                     'zeta': r'$\zeta$',
                     'I': r'$I$',
                     'm': r'$m$',
                     'C_para': r'$C_{\parallel}$',
                     'C_perp': r'$C_{\perp}$',
                     'C_phi': r'$C_{\phi}$',
                     'km1': r'$k_{m_1}$',
                     'km2': r'$k_{m_2}$',
                     'km3': r'$k_{m_3}$',
                     'km4': r'$k_{m_4}$',
                     'd': r'$d$',
                     'psi': r'$\psi$',
                     'gamma': r'$\gamma$',
                     'alpha': r'$\alpha$',
                     'of': r'$\frac{g}{d}$',
                     'gdot': r'$\dot{g}$',}

    def convert_to_latex(self, list_of_strings, remove_dollar_signs=False):
        """ Loop through list of strings and if any match the dict, then swap in LaTex symbol.
        """

        if isinstance(list_of_strings, str):  # if single string is given instead of list
            list_of_strings = [list_of_strings]
            string_flag = True
        else:
            string_flag = False

        list_of_strings = list_of_strings.copy()
        for n, s in enumerate(list_of_strings):  # each string in list
            for k in self.dict.keys():  # check each key in Latex dict
                if s == k:  # string contains key
                    # print(s, ',', self.dict[k])
                    list_of_strings[n] = self.dict[k]   # replace string with LaTex
                    if remove_dollar_signs:
                        list_of_strings[n] = list_of_strings[n].replace('$', '')

        if string_flag:
            list_of_strings = list_of_strings[0]

        return list_of_strings

def plot_arc_grid_map(labels=None, var_list=None, color_list=None, ax=None,
                      r_start=1.0, w=None, r_space=None, arrow=False):
    """ Plot arc grid map.

        labels: list of lists containing strings, each string is associated with its own arc mask
    """

    if r_space is None:
        r_space = 0.05 * r_start

    # Get unique variables
    if var_list is None:
        var_list = []
        for s in labels:
            var_list = var_list + s

        var_list = list(set(var_list))

    # Number of variables
    n_var = len(var_list)

    # Set colors
    if color_list is None:
        cmap = cm.get_cmap('jet')
        color_list = cmap(np.linspace(0, 1, n_var)).tolist()

    # Make grid map
    n_label = len(labels)
    var_map = pd.DataFrame(np.zeros((n_label, n_var)), columns=var_list)
    for r, lab in enumerate(labels):
        for c, v in enumerate(var_list):
            if v in lab:
                var_map.loc[r, v] = True
            else:
                var_map.loc[r, v] = False

    # Grid for arc mask
    theta_grid = np.linspace(0.0, 2*np.pi, n_label + 1)

    # Plot arcs
    for n in range(n_var):
        r = r_start + r_space*n
        on_index = np.squeeze(var_map.iloc[:, n].values)
        plot_arc_grid(theta_grid=theta_grid, on_index=on_index, r=r, w=w, theta_center=True, ax=ax,
                      theta_origin=np.pi/2, grid_res=np.pi/128, arrow=arrow,
                      alpha=1.0, lw=0.0, edgecolor=None, color=color_list[n])


def plot_arc_grid(theta_grid=None, on_index=None, r=1.0, w=None, theta_center=True, ax=None,
                  theta_origin=0.0, grid_res=np.pi/128, arrow=False, alpha=1.0, lw=0.0, edgecolor=None, color='black'):
    """ Plot arc grid.
    """

    if theta_grid is None:
        theta_grid = np.linspace(0, 2 * np.pi, 8, endpoint=True)
    else:
        theta_grid = np.array(theta_grid).squeeze()

    if on_index is None:
        on_index = np.ones_like(theta_grid)

    # Number of acrs
    n_theta = theta_grid.shape[0]

    # Set the origin
    theta_grid = theta_grid + theta_origin

    # Align theta arcs
    if theta_center:
        theta_res = np.mean(np.diff(theta_grid))
        theta_grid = theta_grid - theta_res/2

    if w is None:
        w = 0.1 *r

    # Plot arcs
    if ax is None:
        fig, ax = plt.subplots()

    for n in range(n_theta - 1):
        if on_index[n]:
            # Theta range
            N = int(np.ceil(np.abs((theta_grid[n] - theta_grid[n + 1]) / grid_res)))
            theta = np.linspace(theta_grid[n], theta_grid[n + 1], N)

            if ax.name == 'polar':
                r1 = np.ones_like(theta)
            else:
                # Inner grid
                x1 = (r - w/2) * np.cos(theta)
                y1 = (r - w/2) * np.sin(theta)

                # Outer grid
                x2 = (r + w/2) * np.cos(theta)
                y2 = (r + w/2) * np.sin(theta)

                # If arrow instead of arc
                if arrow:
                    mid = int(x2.shape[0]/2)
                    c = 1
                    x1 = x1[mid-c:mid+c]
                    y1 = y1[mid-c:mid+c]

                # All (x,y) data
                x = np.hstack((x1, np.flip(x2)))
                y = np.hstack((y1, np.flip(y2)))

                # Plot patch
                ax.fill(x, y, facecolor=color, alpha=alpha, edgecolor=edgecolor, linewidth=lw)


def plot_circle_color_grid(theta_grid=None, color_list=None, r=1.0, w=None, theta_center=True, ax=None,
                           theta_origin=0.0, grid_res=np.pi/128, arrow=False, alpha=1.0, lw=0.5, edgecolor=None):
    """ Plot colored grid.
    """

    if theta_grid is None:
        if color_list is not None:
            theta_grid = np.linspace(0, 2*np.pi, len(color_list) + 1, endpoint=True)
        else:
            theta_grid = np.linspace(0, 2 * np.pi, 10, endpoint=True)
    else:
        theta_grid = np.array(theta_grid).squeeze()

    # Number of acrs
    n_theta = theta_grid.shape[0]

    # Set the origin
    theta_grid = theta_grid + theta_origin

    # Align theta arcs
    if theta_center:
        theta_res = np.mean(np.diff(theta_grid))
        theta_grid = theta_grid - theta_res/2

    if w is None:
        w = 0.1 *r

    if color_list is None:
        cmap = cm.get_cmap('jet')
        color_list = cmap(np.linspace(0, 1, n_theta))

    # Plot arcs
    if ax is None:
        fig, ax = plt.subplots()

    for n in range(n_theta - 1):
        # Theta range
        # theta = np.arange(theta_grid[n], theta_grid[n + 1] + 0*1e-6, step=grid_res)
        N = int(np.ceil(np.abs((theta_grid[n] - theta_grid[n + 1]) / grid_res)))
        theta = np.linspace(theta_grid[n], theta_grid[n + 1], N)

        if ax.name == 'polar':
            r1 = np.ones_like(theta)
        else:
            # Inner grid
            x1 = (r - w/2) * np.cos(theta)
            y1 = (r - w/2) * np.sin(theta)

            # Outer grid
            x2 = (r + w/2) * np.cos(theta)
            y2 = (r + w/2) * np.sin(theta)

            # If arrow instead of arc
            if arrow:
                mid = int(x2.shape[0]/2)
                c = 1
                x1 = x1[mid-c:mid+c]
                y1 = y1[mid-c:mid+c]

            # All (x,y) data
            x = np.hstack((x1, np.flip(x2)))
            y = np.hstack((y1, np.flip(y2)))

            # Plot patch
            ax.fill(x, y, facecolor=color_list[n], alpha=alpha, edgecolor=edgecolor, linewidth=lw)
            # plt.plot(x1, y1, color=color_list[n])
            # plt.plot(x2, y2, color=color_list[n])


def make_color_map(color_list=None, color_proportions=None, N=256):
    """ Make a colormap from a list of colors.
    """

    if color_list is None:
        color_list = ['white', 'deepskyblue', 'mediumblue', 'yellow', 'orange', 'red', 'darkred']

    if color_proportions is None:
        color_proportions = np.linspace(0.01, 1, len(color_list) - 1)
        v = np.hstack((np.array(0.0), color_proportions))
    elif color_proportions == 'even':
        color_proportions = np.linspace(0.0, 1, len(color_list))
        v = color_proportions

    l = list(zip(v, color_list))
    cmap = LinearSegmentedColormap.from_list('rg', l, N=N)

    return cmap


def add_colorbar(fig, ax, data, cmap=None, label=None, ticks=None):
    offset_x = 0.017
    offset_y = 0.08

    cb_width = 0.75 * ax.get_position().width
    cb_height = 0.05 * ax.get_position().height

    cnorm = colors.Normalize(vmin=data.min(), vmax=data.max())
    cbax = fig.add_axes([offset_x + ax.get_position().x0, ax.get_position().y0 - offset_y, cb_width, cb_height])
    cb = fig.colorbar(cm.ScalarMappable(norm=cnorm, cmap=cmap), cax=cbax, orientation='horizontal')
    cb.ax.tick_params(labelsize=7, direction='in')
    cbax.yaxis.set_ticks_position('left')
    cb.set_label(label, labelpad=0, size=8)
    cb.ax.set_xticks(np.round(np.linspace(data.min(), data.max(), 5), 2))


def image_from_xyz(x, y, z=None, bins=100, sigma=None):
    x_min = np.min(x)
    x_max = np.max(x)
    y_min = np.min(y)
    y_max = np.max(y)
    z_min = np.min(z)
    z_max = np.max(z)

    hist, xedges, yedges = np.histogram2d(x, y, bins=bins, range=[[x_min - 1, x_max + 1], [y_min - 1, y_max + 1]])

    if z is None:  # use point density as color
        I = hist.T
    else:  # use z value as color
        I = np.zeros_like(hist)
        for bx in range(xedges.shape[0] - 1):
            x1 = xedges[bx]
            x2 = xedges[bx + 1]
            for by in range(yedges.shape[0] - 1):
                y1 = yedges[by]
                y2 = yedges[by + 1]
                xy_bin = (x >= x1) & (x < x2) & (y >= y1) & (y < y2)
                color_bin = z[xy_bin]
                if color_bin.shape[0] < 1:
                    I[by, bx] = z_min
                else:
                    I[by, bx] = np.mean(color_bin)

    if sigma is not None:
        I = scipy.ndimage.gaussian_filter(I, sigma=sigma, mode='reflect')

    return I


def plot_pulse_trajectory(df, cvar, data_range=None, arrow_size=0.008, nskip=1, dpi=100, figsize=8, cmap=None):
    # Get trajectory data
    time = df.time.values - 0.1
    x = df.xpos.values
    y = df.ypos.values
    phi = utils.wrapToPi(df.phi.values)
    phidot = df.phidot.values
    cvar_raw = df.cvar_raw.values

    # Pulse time
    startI = np.argmin(np.abs(time - 0.0))
    endI = np.argmin(np.abs(time - 0.675))
    pulse = np.arange(startI, endI)
    time_pulse = time[pulse]
    phi_pulse = phi[pulse]
    phidot_pulse = phidot[pulse]

    # Get data in range
    if data_range is None:
        data_range = (0, time.shape[0])

    index = np.arange(data_range[0], data_range[-1], 1)

    time = time[index]
    x = x[index]
    y = y[index]
    phi = phi[index]
    phidot = phidot[index]
    obsv = cvar[index]
    cvar_raw = cvar_raw[index]

    # Make figure
    fig, ax = plt.subplots(2, 2, figsize=(figsize, 0.4 * figsize), dpi=dpi,
                           gridspec_kw={
                               'width_ratios': [1.5, 1],
                               'height_ratios': [1, 1],
                               'wspace': 0.4,
                               'hspace': 0.4}
                           )

    # Plot pulse trajectory
    cmap_pulse = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    cmap_pulse = ListedColormap(cmap_pulse)
    color = 0 * np.ones_like(time)
    color[pulse] = 1.0
    color = color[index]

    plot_trajectory(x, y, phi,
                    color=color,
                    ax=ax[0, 0],
                    nskip=nskip,
                    size_radius=arrow_size,
                    colormap=cmap_pulse)

    fifi.mpl_functions.adjust_spines(ax[0, 0], [])

    # Plot color trajectory
    ax[1, 0].set_title('Observability', fontsize=8)

    if cmap is None:
        crange = 0.1
        cmap = cm.get_cmap('RdPu')
        cmap = cmap(np.linspace(crange, 1, 100))
        cmap = ListedColormap(cmap)

    color = obsv.copy()
    cnorm = (np.nanmin(color), np.nanmax(color))
    # cnorm = (0.04, 0.09)
    plot_trajectory(x, y, phi,
                    color=color,
                    ax=ax[1, 0],
                    nskip=nskip,
                    size_radius=arrow_size,
                    reverse=True,
                    colormap=cmap,
                    colornorm=cnorm)

    fifi.mpl_functions.adjust_spines(ax[1, 0], [])

    # Colorbar
    norm = colors.Normalize(vmin=0.0, vmax=1.0)
    cax = fig.add_axes([ax[1, 0].get_position().x1 - 0.25, ax[1, 0].get_position().y0 - 0.05,
                        0.4 * ax[1, 0].get_position().width, 0.075 * ax[1, 0].get_position().height])
    cb = fig.colorbar(cm.ScalarMappable(norm=norm, cmap=cmap),
                      cax=cax, orientation='horizontal', label='Observability level')
    cb.ax.tick_params(labelsize=6, direction='out')
    cb.set_label('Observability level', labelpad=0, size=7)
    cb.ax.set_xticks([0, 1])

    # Phi
    ax[0, 1].axhline(y=0, color='gray', linestyle='--', lw=0.5)
    ax[0, 1].plot(*circplot(time, phi), color='black')
    ax[0, 1].plot(*circplot(time_pulse, phi_pulse), color='red')
    ax[0, 1].set_ylabel('Course direction \n(rad)', fontsize=8)
    ax[0, 1].tick_params(axis='both', which='major', labelsize=7)
    ax[0, 1].set_xlim(time[0], time[-1])
    pi_yaxis(ax[0, 1], tickpispace=0.5, lim=(-np.pi, np.pi))
    fifi.mpl_functions.adjust_spines(ax[0, 1], ['left', 'bottom'], tick_length=3, linewidth=0.75)
    ax[0, 1].spines['bottom'].set_visible(False)
    ax[0, 1].tick_params(bottom=False, labelbottom=False)

    # Phidot
    ax[1, 1].axhline(y=0, color='gray', linestyle='--', lw=0.5)
    ax[1, 1].plot(time, np.abs(phidot), color='black')
    ax[1, 1].plot(time_pulse, np.abs(phidot_pulse), color='red')
    ax[1, 1].set_ylabel('Angular velocity \n(rad/s)', fontsize=8)
    ax[1, 1].set_xlabel('Time (s)', fontsize=8)
    ax[1, 1].tick_params(axis='both', which='major', labelsize=7)
    ax[1, 1].set_xlim(time[0], time[-1])

    cc = 'darkmagenta'
    ax_right = ax[1, 1].twinx()
    ax_right.plot(time, obsv, color=cc)
    ax_right.plot(time, cvar_raw, '.', color='dodgerblue', linewidth=1.0, markersize=2)
    ax_right.set_ylabel('Observability level', fontsize=8, color=cc)
    ax_right.tick_params(axis='both', which='major', labelsize=7)
    ax_right.tick_params(axis='y', direction='in', colors=cc)
    ax_right.spines['right'].set_color(cc)
    ax_right.set_xlim(time[0], time[-1])
    ax_right.spines[['top', 'bottom', 'left']].set_visible(False)
    ax_right.spines['right'].set_position(('data', 1.08))
    # ax_right.set_ylim(0.025, 0.125)

    # ax[1, 1].set_ylim(-5, 60)
    # ax[1, 1].set_ylim(bottom=-5)

    ax_right.set_ylim(bottom=-0.0)
    ax_right.set_ylim(top=1.0)

    ax[0, 1].set_xlim(left=-0.1)
    ax[1, 1].set_xlim(left=-0.1)

    ax[1, 1].set_ylim(0, 60)
    fifi.mpl_functions.adjust_spines(ax[1, 1], ['left', 'bottom'], tick_length=3, linewidth=0.75)
    # ax[1, 1].set_ylim(-0.1, 40)
    ax[1, 1].set_ylim(bottom=-0.1)

    fig.align_ylabels(ax[:, 1])
    plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=0.5, hspace=0.5)


def plot_trajectory(xpos, ypos, phi, color, ax=None, size_radius=None, nskip=0,
                    colormap='bone_r', colornorm=None, edgecolor='none', reverse=False):
    if color is None:
        color = phi

    color = np.array(color)

    # Set size radius
    xymean = np.mean(np.abs(np.hstack((xpos, ypos))))
    if size_radius is None:  # auto set
        xymean = 0.21 * xymean
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
    xrange = np.max([xrange, 0.02])
    yrange = ypos.max() - ypos.min()
    yrange = np.max([yrange, 0.02])

    if yrange < (size_radius / 2):
        yrange = 10

    if xrange < (size_radius / 2):
        xrange = 10

    ax.set_xlim(xpos.min() - 0.2 * xrange, xpos.max() + 0.2 * xrange)
    ax.set_ylim(ypos.min() - 0.2 * yrange, ypos.max() + 0.2 * yrange)

    # fifi.mpl_functions.adjust_spines(ax, [])


def pi_yaxis(ax=0.5, tickpispace=0.5, lim=None, real_lim=None):
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

    if lim is None:
        ax.set_ylim(-1 * np.pi, 1 * np.pi)
    else:
        ax.set_ylim(lim)

    ax.set_ylim(real_lim)


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
            phi_stiched = np.concatenate((phi_stiched[0:i], [np.nan], phi_stiched[i + 1:None]))
            t_stiched = np.concatenate((t_stiched[0:i], [np.nan], t_stiched[i + 1:None]))

    return t_stiched, phi_stiched
