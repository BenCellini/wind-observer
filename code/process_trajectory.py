
import numpy as np
import pandas as pd
import scipy
import matplotlib.pyplot as plt
from matplotlib import cm, colors
from matplotlib.colors import ListedColormap
import pynumdiff

import figurefirst as fifi
import figure_functions as ff
from utils import wrapTo2Pi, wrapToPi

pd.set_option('mode.chained_assignment', None)


class Trajectory:
    def __init__(self, traj, time_range=(-100, 900), w=0.4, zeta=0.0, norm=True, fc=20, filter_order=5):
        """ Proces trajectory data.

            Inputs
                traj: data frame with trajectory data
                time_range: (min_time, max_time) time window to analyze
                w: ambient wind magnitude
                zeta: ambient wind direction
                norm: (boolean) if True, normalize position
                fc: cutoff frequency for low-pass filter in hz
                filter_order: low-pass filter order
        """

        # Get raw trajectory & add data
        self.traj_raw = traj.copy()  # raw trajectory data
        self.traj_raw['time_seconds'] = traj['time stamp'] / 1000  # time in seconds
        self.traj_raw['pulse'] = (self.traj_raw['time stamp'] >= 0) & (self.traj_raw['time stamp'] < 680)
        self.traj_raw['w'] = w * np.ones_like(self.traj_raw['time stamp'])
        self.traj_raw['zeta'] = zeta * np.ones_like(self.traj_raw['time stamp'])
        self.traj_raw['g'] = np.sqrt((self.traj_raw.xvel ** 2) + (self.traj_raw.yvel ** 2))

        # Get trajectory window
        self.time_window = (self.traj_raw['time stamp'] >= time_range[0]) & \
                           (self.traj_raw['time stamp'] <= time_range[1])

        self.traj = self.traj_raw.loc[self.time_window]  # trajectory data for set time window
        self.traj['pulse'] = (self.traj['time stamp'] >= 0) & (self.traj['time stamp'] < 680)

        # Normalize position
        if norm:
            self.traj.x = self.traj.x - self.traj.x.iloc[0]
            self.traj.y = self.traj.y - self.traj.y.iloc[0]

        # Design low-pass filter
        self.dt = np.round(np.squeeze(np.mean(np.diff(self.traj_raw.time_seconds.values))), 5)  # sampling time
        self.fs = 1 / self.dt  # sampling frequency
        self.fc = fc  # cutoff frequency
        self.filter_b, self.filter_a = scipy.signal.butter(filter_order, self.fc / (self.fs / 2))

        # Filter heading data
        self.traj['phi'] = np.unwrap(self.traj['heading'].values.copy())
        self.traj['phi_wrap'] = wrapToPi(self.traj['phi'].values.copy())

        self.traj['phi_filt'] = scipy.signal.filtfilt(self.filter_b, self.filter_a, self.traj['phi'].values)
        self.traj['phi_filt_wrap'] = wrapToPi(self.traj['phi_filt'].values)

        # Filter velocity
        self.traj['xvel_filt'] = scipy.signal.filtfilt(self.filter_b, self.filter_a, self.traj.xvel.values)
        self.traj['yvel_filt'] = scipy.signal.filtfilt(self.filter_b, self.filter_a, self.traj.yvel.values)
        self.traj['g_filt'] = np.sqrt((self.traj.xvel_filt ** 2) + (self.traj.yvel_filt ** 2))

        # Velocity in body reference frame
        self.traj['v_para'] = self.traj['g_filt'].copy()
        self.traj['v_perp'] = 0.0 * self.traj['g_filt'].copy()

        # Calculate air velocity
        self.traj['a_para'] = self.traj['v_para'] - self.traj['w'] * np.cos(self.traj['phi_filt'] - self.traj['zeta'])
        self.traj['a_perp'] = self.traj['v_perp'] + self.traj['w'] * np.sin(self.traj['phi_filt'] - self.traj['zeta'])
        self.traj['a'] = np.sqrt(self.traj['a_para'] ** 2 + self.traj['a_perp'] ** 2)
        self.traj['gamma'] = np.arctan2(self.traj['a_perp'], self.traj['a_para'])

        # Calculate derivatives
        self.traj['gdot'] = pynumdiff.finite_difference.second_order(self.traj['g_filt'].values, self.dt)[1]
        self.traj['phidot'] = pynumdiff.finite_difference.second_order(self.traj['phi_filt'].values, self.dt)[1]
        self.traj['phi2dot'] = pynumdiff.finite_difference.second_order(self.traj['phidot'].values, self.dt)[1]

    def plot_pulse_trajectory(self, size=7.0):
        fig, ax = plt.subplots(1, 1, figsize=(size, size), dpi=100)
        ax.plot(self.traj['x'], self.traj['y'], 'k', linewidth=2)
        ax.plot(self.traj['x'][self.traj['pulse']], self.traj['y'][self.traj['pulse']], 'r', linewidth=2)
        ax.set_aspect('equal', adjustable='box')
        fifi.mpl_functions.adjust_spines(ax, [])

    def plot_heading_trajectory(self, size=7.0, color=None, arrow_size=None, nskip=0, data_range=None, cmap=None, colornorm=None):
        if arrow_size is None:
            arrow_size = 0.07 * np.mean(np.abs(np.hstack((self.traj['x'], self.traj['y']))))

        if cmap is None:
            if color is None:
                crange = 0.05
                cmap = cm.get_cmap('bone_r')
            else:
                crange = 0.1
                cmap = cm.get_cmap('RdPu')

            cmap = cmap(np.linspace(crange, 1, 1000))
            cmap = ListedColormap(cmap)

        if color is None:
            color = self.traj['time_seconds'].values
            
        x = self.traj['x'].values
        y = self.traj['y'].values
        phi = self.traj['phi_filt'].values
        
        if data_range is not None:
            index = np.arange(data_range[0], data_range[-1], 1)
            x = x[index]
            y = y[index]
            phi = phi[index]
            color = color[index]

        fig, ax = plt.subplots(1, 1, figsize=(size, size), dpi=100)
        ff.plot_trajectory(x, y, phi,
                           color=color,
                           ax=ax,
                           nskip=nskip,
                           size_radius=arrow_size,
                           colormap=cmap,
                           colornorm=colornorm)

        fifi.mpl_functions.adjust_spines(ax, [])

    def plot_velocity(self, size=5.0):
        fig, ax = plt.subplots(2, 2, figsize=(size, size), dpi=100)

        time = self.traj['time_seconds']

        ax[0, 0].plot(*ff.circplot(time, self.traj['phi_filt_wrap'].values), '-')
        ax[0, 0].plot(*ff.circplot(time, self.traj['phi_wrap'].values), '.', markersize=3)

        ax[0, 1].plot(time, self.traj['g_filt'].values, '-')
        ax[0, 1].plot(time, self.traj['g'].values, '.', markersize=3)

        ax[1, 0].plot(time, self.traj['phidot'].values)

        ax[1, 1].plot(time, self.traj['gdot'].values)

        ff.pi_yaxis(ax[0, 0], tickpispace=0.5, lim=None)

        data_labels = [r'$\phi$ (rad)', r'$g$ (m/s)', r'$\dot{\phi}$ (rad/s)', r'$\dot{g}$ (m/$s^2$)']
        p = 0
        for r in range(ax.shape[0]):
            for c in range(ax.shape[1]):
                ax[r, c].grid()
                ax[r, c].set_ylabel(data_labels[p], fontsize=10)

                if r > 0:
                    ax[r, c].set_xlabel('time (s)', fontsize=10)
                else:
                    ax[r, c].xaxis.set_tick_params(labelbottom=False)

                p = p + 1

        plt.subplots_adjust(wspace=0.5, hspace=0.1)
