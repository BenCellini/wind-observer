
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.colors import ListedColormap
from setdict import SetDict
import figurefirst as fifi
import figure_functions as ff
from observability import empirical_observability_matrix, num_jacobian
from eiso_brute import eiso_brute
from simulator import FlyWindDynamics


class eiso_fly_wind:
    def __init__(self, simulator=None, x0=None, output_mode=None, control_mode='open_loop', init_accel=True,
                 fs=10.0, T=0.3, r_para=1.0, r_perp=0.0, r_phi=0.0):
        """ Run EISO for fly-wind system with different simulation parameters

            Inputs
                simulator: simulator object
                x0: dictionary of initial states
                output_mode: output mode for simulator
                control_mode: control mode for simulator
                fs: sampling frequency [hz]
                T: time period [s]
        """

        # Simulator
        if simulator is None:
            self.simulator = FlyWindDynamics(polar_mode=False,
                                             control_mode=control_mode,
                                             update_every_step='True',
                                             output_mode=output_mode)
        else:
            self.simulator = simulator

        # Time
        self.fs = fs  # sampling frequency [hz]
        self.T = T  # time period [s]
        self.dt = 1 / self.fs  # sampling time [s]
        self.tsim = np.arange(0, T, self.dt).T  # simulation time vector [s]
        self.n = self.tsim.shape[0]

        # Default initial state
        self.params = {'v_para': 0.0, 'v_perp': 0.0, 'phi': np.pi / 3.74, 'phidot': 0.0, 'w': 0.23, 'zeta': 0.0,  # main states
                       'I': 0.1, 'm': 0.2, 'C_para': 1.74, 'C_perp': 10.31, 'C_phi': 1.48, 'd': 0.7,  # dynamics parameters
                       'km1': 1.32, 'km2': 0.0, 'km3': 1.44, 'km4': 1.0,  # motor calibration parameters
                       'ks1': 1.0, 'ks2': 1.0, 'ks3': 0.0, 'ks4': 1.0, 'ks5': 0.0, 'ks6': 1.0, 'ks7': 0.0}  # sensor calibration parameters

        # Set specified initial states
        if x0 is not None:
            SetDict().set_dict_with_overwrite(self.params, x0)

        # Controls
        self.r_para = r_para * np.ones_like(self.tsim)
        self.r_perp = r_perp * np.ones_like(self.tsim)
        self.r_phi = r_phi * np.ones_like(self.tsim)
        self.wdot = 0.0 * np.ones_like(self.tsim)
        self.zetadot = 0.0 * np.ones_like(self.tsim)

        self.usim = np.stack([self.r_para, self.r_perp, self.r_phi, self.wdot, self.zetadot], axis=1)

        # Set initial state so there is no acceleration
        if not init_accel:
            self.set_no_acceleration_initial_state()

        # Initial state list
        self.x0 = np.array(list(self.params.values()))

        # Observability
        self.Oe = np.array([0])
        self.Oe_df = pd.DataFrame(self.Oe)
        self.sim_data = []
        self.xsim = []

    def set_no_acceleration_initial_state(self):
        v_para0 = (self.params['km1'] * self.r_para[0] / self.params['C_para']) \
            + self.params['w'] * np.cos(self.params['phi'] - self.params['zeta'])

        v_perp0 = (self.params['km3'] * self.r_perp[0] / self.params['C_perp']) \
            - self.params['w'] * np.sin(self.params['phi'] - self.params['zeta'])

        SetDict().set_dict_with_overwrite(self.params, {'v_para': v_para0, 'v_perp': v_perp0})

    def simulate(self, x0=None, r_para=None, r_perp=None, r_phi=None, wdot=None, zetadot=None, init_accel=True):
        """ Run simulation

            Inputs
                x0: dictionary of initial states
                r_para: parallel control input
                r_perp: perpendicular control input
                r_phi: turning control input
                init_accel: (boolean) if False then set initial state such that there is no acceleration
        """

        # Controls
        if r_para is not None:
            if np.isscalar(r_para):
                self.r_para = r_para * np.ones_like(self.tsim)
            else:
                self.r_para = r_para.copy()

        if r_perp is not None:
            if np.isscalar(r_perp):
                self.r_perp = r_perp * np.ones_like(self.tsim)
            else:
                self.r_perp = r_perp.copy()

        if r_phi is not None:
            if np.isscalar(r_phi):
                self.r_phi = r_phi * np.ones_like(self.tsim)
            else:
                self.r_phi = r_phi.copy()

        if wdot is not None:
            if np.isscalar(wdot):
                self.wdot = wdot * np.ones_like(self.tsim)
            else:
                self.wdot = wdot.copy()

        if zetadot is not None:
            if np.isscalar(zetadot):
                self.zetadot = zetadot * np.ones_like(self.tsim)
            else:
                self.zetadot = zetadot.copy()

        self.usim = np.stack([self.r_para, self.r_perp, self.r_phi, self.wdot, self.zetadot], axis=1)

        # Set specified initial states
        if x0 is not None:
            SetDict().set_dict_with_overwrite(self.params, x0)

        # Set initial state so there is no acceleration
        if not init_accel:
            self.set_no_acceleration_initial_state()

        # Initial state list
        self.x0 = np.array(list(self.params.values()))

        # Simulate system
        sim_data, y = self.simulator.simulate(self.x0, self.tsim, self.usim)
        self.sim_data = sim_data
        self.xsim = self.simulator.x.copy()

    def observability_matrix(self, eps=1e-5, states_to_use=None):
        # Construct observability matrix
        Oe, sim_data, _ = empirical_observability_matrix(self.simulator, self.x0, self.tsim, self.usim, eps=eps)

        # Make data frame where column correspond to states
        Oe_df = pd.DataFrame(Oe, columns=self.simulator.state_names)

        # Select only columns corresponding to specific states
        if states_to_use is not None:
            Oe_df = Oe_df.loc[:, states_to_use]

        self.Oe = Oe
        self.Oe_df = Oe_df
        self.sim_data = sim_data

    def eiso(self, ej_list=None, O=None, beta=1e-6, show_n_comb=True):
        # Set O
        if O is None:
            O = self.Oe_df.values.copy()

        # Run EISO for each state
        CN_state = []
        state_names = []
        for s in ej_list:
            # Get state basis vector
            if callable(s):  # function was given
                u0 = np.array([self.sim_data['u_para'][0],
                               self.sim_data['u_perp'][0],
                               self.sim_data['u_phi'][0],
                               self.sim_data['wdot'][0],
                               self.sim_data['zetadot'][0]])

                # print(pd.DataFrame(u0).T)

                ej = num_jacobian(s, self.x0, u0=u0, eps=1e-5)  # find state basis vector based on function
                state_names.append(s.__name__)  # use function name
                # print(pd.DataFrame(ej))
            else:  # use state basis vector directly
                ej = s
                state_names.append(self.simulator.state_names[int(s)])  # get name from simulator

            # Run EISO
            CN_min, O_min, row_min, CN, rows = eiso_brute(O=O, ej=ej, beta=beta, show_n_comb=show_n_comb)

            CN_state.append(CN_min)

        CN_state_df = pd.DataFrame(CN_state, index=state_names).T

        return CN_state_df

    def plot_traj(self, cmap=None, size=5, dpi=100, arrow_size=0.05, nskip=None):
        # Plot the trajectory
        if cmap is None:
            crange = 0.1
            cmap = cm.get_cmap('bone_r')
            cmap = cmap(np.linspace(crange, 1, 100))
            cmap = ListedColormap(cmap)

        if nskip is None:
            nskip = int(self.fs / 40.0)

        # Plot
        fig, ax = plt.subplots(1, 1, figsize=(size, size), dpi=dpi)

        ff.plot_trajectory(self.sim_data['xpos'],
                           self.sim_data['ypos'],
                           self.sim_data['phi'],
                           color=self.sim_data['time'],
                           ax=ax,
                           size_radius=arrow_size,
                           nskip=nskip,
                           colormap=cmap)

        fifi.mpl_functions.adjust_spines(ax, [])
