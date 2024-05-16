
import sys
import os

sys.path.append(os.path.join(os.path.pardir, 'simulator'))

# import numpy as np
# from numpy import matlib
import pandas as pd
import pynumdiff
# import scipy
from scipy.optimize import fsolve
import matplotlib.pyplot as plt
from matplotlib import cm, colors
from matplotlib.colors import ListedColormap
import figurefirst as fifi
import figure_functions as ff

from casadi import *
import do_mpc

from setdict import SetDict
from simulator_fly import FlyWindDynamics


class MpcFlyWind:
    def __init__(self, v_para, v_perp, phi, w, zeta, x0=None, dt=0.01, n_horizon=20, r_weight=1e-9, run=True):
        """ Use Model Predictive Control (MPC) to control simulated fly-wind trajectories.

            Inputs:
                v_para: parallel velocity time-series
                v_perp: perpendicular velocity time-series
                phi: orientation time-series
                w: wind magnitude time-series
                zeta: wind direction time-series
                x0: dictionary of initial states
                dt: time step [s]
                n_horizon: # of steps in the future to run MPC
                r_weight: penalty on inputs
                run: (boolean) to run the MPC routine when creating the object
        """

        # Set set-point time series
        self.v_para = v_para.copy()
        self.v_perp = v_perp.copy()
        self.phi = phi.copy()
        self.phidot = pynumdiff.finite_difference.second_order(self.phi, dt)[1]
        self.w = w.copy()
        self.zeta = zeta.copy()

        # Default initial state
        # SI units
        m = 0.25e-6  # [kg]
        I = 5.2e-13  # [N*m*s^2] yaw mass moment of inertia: 10.1242/jeb.02369
        # I = 4.971e-12  # [N*m*s^2] yaw mass moment of inertia: 10.1242/jeb.038778
        C_phi = 27.36e-12  # [N*m*s] yaw damping: 10.1242/jeb.038778
        C_para = m / 0.170  # [N*s/m] calculate using the mass and time constant reported in 10.1242/jeb.098665
        C_perp = C_para  # assume same as C_para

        # Convert to units of mg & mm to help with scaling for ODE solver
        m = m * 1e6  # [mg]
        I = I * 1e6 * (1e3) ** 2  # [mg*mm/s^2 * mm*s^2]
        C_phi = C_phi * 1e6 * (1e3) ** 2  # [mg*mm/s^2 *m*s]
        C_para = C_para * 1e6  # [mg/s]
        C_perp = C_perp * 1e6  # [mg/s]

        self.x0 = {'v_para': self.v_para[0],
                   'v_perp': self.v_perp[0],
                   'phi': self.phi[0],
                   'phidot': self.phidot[0],
                   'w': self.w[0],
                   'zeta': self.zeta[0],
                   'I': I, 'm': m, 'C_para': C_para, 'C_perp': C_perp, 'C_phi': C_phi, 'd': 0.3,
                   'km1': 1.0, 'km2': 0.0, 'km3': 1.0, 'km4': 1.0,
                   'ks1': 1.0, 'ks2': 1.0, 'ks3': 0.0, 'ks4': 1.0, 'ks5': 0.0, 'ks6': 1.0, 'ks7': 0.0}

        # Overwrite specified initial states
        if x0 is not None:
            SetDict().set_dict_with_overwrite(self.x0, x0)

        # Store MPC parameters
        self.dt = np.round(dt, 5)
        self.fs = 1 / self.dt
        self.n_horizon = n_horizon

        # Get total # of points & simulation time
        self.n_points = np.squeeze(self.v_para).shape[0]
        self.T = (self.n_points - 1) * self.dt
        # self.tsim = np.arange(0.0, self.T + self.dt/2, self.dt)
        self.tsim = self.dt * (np.linspace(1.0, self.n_points, self.n_points) - 1)
        self.xsim = np.zeros_like(self.tsim)
        self.usim = np.zeros_like(self.tsim)

        # Define continuous-time MPC model
        self.model = do_mpc.model.Model('continuous')

        # Define state variables for MPC
        v_para = self.model.set_variable('_x', 'v_para')
        v_perp = self.model.set_variable('_x', 'v_perp')
        phi = self.model.set_variable('_x', 'phi')
        phidot = self.model.set_variable('_x', 'phidot')
        w = self.model.set_variable('_x', 'w')
        zeta = self.model.set_variable('_x', 'zeta')

        # Define set-point variables for MPC
        v_para_setpoint = self.model.set_variable(var_type='_tvp', var_name='v_para_setpoint')
        v_perp_setpoint = self.model.set_variable(var_type='_tvp', var_name='v_perp_setpoint')
        phi_setpoint = self.model.set_variable(var_type='_tvp', var_name='phi_setpoint')
        w_setpoint = self.model.set_variable(var_type='_tvp', var_name='w_setpoint')
        zeta_setpoint = self.model.set_variable(var_type='_tvp', var_name='zeta_setpoint')

        # Define input variables for MPC
        u_para = self.model.set_variable('_u', 'u_para')
        u_perp = self.model.set_variable('_u', 'u_perp')
        u_phi = self.model.set_variable('_u', 'u_phi')
        u_w = self.model.set_variable('_u', 'u_w')
        u_zeta = self.model.set_variable('_u', 'u_zeta')

        # Air velocity
        a_para = v_para - w * cos(phi - zeta)
        a_perp = v_perp + w * sin(phi - zeta)

        # Define state equations for MPC
        self.model.set_rhs('v_para', ((self.x0['km1'] * u_para - self.x0['C_para'] * a_para) / self.x0['m']) + v_perp * phidot)
        self.model.set_rhs('v_perp', ((self.x0['km3'] * u_perp - self.x0['C_perp'] * a_perp) / self.x0['m']) - v_para * phidot)
        self.model.set_rhs('phi', phidot)
        self.model.set_rhs('phidot', (self.x0['km4'] * u_phi / self.x0['I']) - (self.x0['C_phi'] * phidot / self.x0['I']) + (self.x0['km2'] * u_para / self.x0['I']))
        self.model.set_rhs('w', u_w)
        self.model.set_rhs('zeta', u_zeta)

        # Build MPC model
        self.model.setup()
        self.mpc = do_mpc.controller.MPC(self.model)

        # Set estimator & simulator
        self.estimator = do_mpc.estimator.StateFeedback(self.model)
        self.simulator = do_mpc.simulator.Simulator(self.model)

        params_simulator = {
            # Note: cvode doesn't support DAE systems.
            'integration_tool': 'idas',  # cvodes, idas
            'abstol': 1e-8,
            'reltol': 1e-8,
            't_step': self.dt
        }

        self.simulator.set_param(**params_simulator)

        # Set MPC parameters
        setup_mpc = {
            'n_horizon': self.n_horizon,
            'n_robust': 0,
            'open_loop': 0,
            't_step': self.dt,
            'state_discretization': 'collocation',
            'collocation_type': 'radau',
            'collocation_deg': 3,
            'collocation_ni': 1,
            'store_full_solution': True,

            # Use MA27 linear solver in ipopt for faster calculations:
            'nlpsol_opts': {'ipopt.linear_solver': 'mumps',  # mumps, MA27
                            'ipopt.print_level': 0,
                            'ipopt.sb': 'yes',
                            'print_time': 0,
                            }
        }

        self.mpc.set_param(**setup_mpc)

        # Set MPC objective function
        self.set_objective(case=0, r_weight=r_weight)

        # Get template's for MPC time-varying parameters
        self.mpc_tvp_template = self.mpc.get_tvp_template()
        self.simulator_tvp_template = self.simulator.get_tvp_template()

        # Set time-varying set-point functions
        self.mpc.set_tvp_fun(self.mpc_tvp_function)
        self.simulator.set_tvp_fun(self.simulator_tvp_function)

        # Setup MPC & simulator
        self.mpc.setup()
        self.simulator.setup()

        # Set variables to store MPC simulation data
        self.x_mpc = np.array([0.0, 0.0])
        self.u_mpc = np.array([0.0, 0.0])

        # Initialize system simulator
        output_mode = ['phi', 'psi', 'gamma']
        self.system = FlyWindDynamics(control_mode='open_loop', output_mode=output_mode)
        self.sim_data = {}
        self.sim_data_df = pd.DataFrame(np.array([0.0, 0.0]))

        # Replay error
        self.v_para_error = np.array([0.0, 0.0])
        self.v_perp_error = np.array([0.0, 0.0])
        self.phi_error = np.array([0.0, 0.0])
        self.error_metric = np.array(0.0)

        if run:
            # Run MPC
            self.run_mpc()

            # Replay
            self.replay()

    def set_objective(self, case=0, r_weight=1e-4):
        """ Set MCP objective function.

            Inputs:
                case: type of objective function
                r_weight: weight for control penalty
        """

        # Set stage cost
        if case == 0:
            lterm = (self.model.x['v_para'] - self.model.tvp['v_para_setpoint']) ** 2 + \
                    (self.model.x['v_perp'] - self.model.tvp['v_perp_setpoint']) ** 2 + \
                    (self.model.x['phi'] - self.model.tvp['phi_setpoint']) ** 2 + \
                    (self.model.x['w'] - self.model.tvp['w_setpoint']) ** 2 + \
                    (self.model.x['zeta'] - self.model.tvp['zeta_setpoint']) ** 2

        else:
            lterm = (self.model.x['v_para'] - self.model.tvp['v_para_setpoint']) ** 2 + \
                    (self.model.x['v_perp'] - self.model.tvp['v_perp_setpoint']) ** 2 + \
                    (self.model.x['phi'] - self.model.tvp['phi_setpoint']) ** 2 + \
                    (self.model.x['w'] - self.model.tvp['w_setpoint']) ** 2 + \
                    (self.model.x['zeta'] - self.model.tvp['zeta_setpoint']) ** 2

            print('only case 0 working now')

        # Set terminal cost same as state cost
        mterm = lterm

        # Set objective
        self.mpc.set_objective(mterm=mterm, lterm=lterm)  # objective function
        self.mpc.set_rterm(u_para=r_weight, u_perp=r_weight, u_phi=r_weight, u_w=0.0, u_zeta=0.0)  # input penalty

        # Add constraints
        # self.mpc.bounds['lower', '_x', 'phi'] = -np.pi
        # self.mpc.bounds['upper', '_x', 'phi'] = np.pi

    def mpc_tvp_function(self, t):
        """ Set the set-point function for MPC optimizer.

            Inputs:
                t: current time
        """

        # Set current step index
        k_step = int(np.round(t / self.dt))

        # Update set-point time horizon
        for n in range(self.n_horizon + 1):
            k_set = k_step + n
            if k_set >= self.n_points:  # horizon is beyond end of input data
                k_set = self.n_points - 1  # set part of horizon beyond input data to last point

            # Update each set-point over time horizon
            self.mpc_tvp_template['_tvp', n, 'v_para_setpoint'] = self.v_para[k_set]
            self.mpc_tvp_template['_tvp', n, 'v_perp_setpoint'] = self.v_perp[k_set]
            self.mpc_tvp_template['_tvp', n, 'phi_setpoint'] = self.phi[k_set]
            self.mpc_tvp_template['_tvp', n, 'w_setpoint'] = self.w[k_set]
            self.mpc_tvp_template['_tvp', n, 'zeta_setpoint'] = self.zeta[k_set]

        return self.mpc_tvp_template

    def simulator_tvp_function(self, t):
        """ Set the set-point function for MPC simulator.

            Inputs:
                t: current time
        """

        # Set current step index
        k_step = int(np.round(t / self.dt))
        if k_step >= self.n_points:  # point is beyond end of input data
            k_step = self.n_points - 1  # set point beyond input data to last point

        # Update current set-point
        self.simulator_tvp_template['v_para_setpoint'] = self.v_para[k_step]
        self.simulator_tvp_template['v_perp_setpoint'] = self.v_perp[k_step]
        self.simulator_tvp_template['phi_setpoint'] = self.phi[k_step]
        self.simulator_tvp_template['w_setpoint'] = self.w[k_step]
        self.simulator_tvp_template['zeta_setpoint'] = self.zeta[k_step]

        return self.simulator_tvp_template

    def run_mpc(self):
        # Set initial state to match 1st point from set-points
        x0 = np.array([self.v_para[0],
                       self.v_perp[0],
                       self.phi[0],
                       self.phidot[0],
                       self.w[0],
                       self.zeta[0]]).reshape(-1, 1)

        # Initial controls are 0
        u0 = np.array([0.0, 0.0, 0.0, 0.0, 0.0]).reshape(-1, 1)

        # Set initial MPC time, state, inputs
        self.mpc.t0 = np.array(0.0)
        self.mpc.x0 = x0
        self.mpc.u0 = u0
        self.mpc.set_initial_guess()

        # Set simulator MPC time, state, inputs
        self.simulator.t0 = np.array(0.0)
        self.simulator.x0 = x0
        self.simulator.set_initial_guess()

        # Initialize variables to store MPC data
        self.x_mpc = [x0]
        self.u_mpc = [u0]

        # Run simulation
        x_step = x0.copy()
        for k in range(self.n_points - 1):
            u_step = self.mpc.make_step(x_step)
            x_step = self.simulator.make_step(u_step)
            self.u_mpc.append(u_step)
            self.x_mpc.append(x_step)

        self.u_mpc = np.hstack(self.u_mpc).T
        self.x_mpc = np.hstack(self.x_mpc).T

    def replay(self, usim=None):
        """ Replay the MPC control inputs into open-loop model. Must run 'run_mpc' before this method to get controls.
        """

        # Make sure initial conditions are the same
        x0_sim = np.array(list(self.x0.values()))

        # Replay
        if usim is None: # Us MPC controls
            self.sim_data, y = self.system.simulate(x0_sim, self.tsim, self.u_mpc)
        else: # directly set controls
            self.sim_data, y = self.system.simulate(x0_sim, self.tsim, usim)

        self.sim_data_df = pd.DataFrame(self.sim_data)
        self.xsim = self.system.x.copy()
        self.usim = self.system.u.copy()

        # Compute error metrics
        self.v_para_error = self.v_para - self.sim_data['v_para']
        self.v_perp_error = self.v_perp - self.sim_data['v_perp']
        self.phi_error = self.phi - self.sim_data['phi']

        self.error_metric = np.sum(self.v_para_error ** 2) + np.sum(self.v_perp_error ** 2) + np.sum(
            self.phi_error ** 2)

        self.error_metric = self.error_metric / self.n_points  # normalize by # of points

    def plot_trajectory(self, color=None, cmap=None, size=5, dpi=100, arrow_size=None, nskip=None):
        if arrow_size is None:
            arrow_size = 0.7 * np.mean(np.abs(np.hstack((self.sim_data['xpos'], self.sim_data['ypos']))))

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

        x = self.sim_data['xpos']
        y = self.sim_data['ypos']
        phi = self.sim_data['phi']

        if color is None:
            color = self.sim_data['time']

        ff.plot_trajectory(x, y, phi,
                           color=color,
                           ax=ax,
                           size_radius=arrow_size,
                           nskip=nskip,
                           colormap=cmap)

        fifi.mpl_functions.adjust_spines(ax, [])

    def plot_setpoint_tracking(self, size=7, dpi=100, lw=2):
        """ Plot closed-loop MPC states trajectories vs the desired set-point.
        """
        fig, ax = plt.subplots(2, 1, figsize=(size, 1.2 * size), dpi=dpi)

        # Plot set-points
        ax[0].plot(self.tsim, self.v_para, 'k', lw=lw, label='set-point')
        ax[0].plot(self.tsim, self.v_perp, 'k', lw=lw)
        ax[0].plot(self.tsim, self.phi, 'k', lw=lw)
        ax[0].plot(self.tsim, self.w, 'k', lw=lw)
        ax[0].plot(self.tsim, self.zeta, 'k', lw=lw)

        # Plot MPC results
        # ax[0].plot(self.tsim, self.x_mpc[:, 0], '--', label='v_para', lw=lw)
        # ax[0].plot(self.tsim, self.x_mpc[:, 1], '--', label='v_perp', lw=lw)
        # ax[0].plot(self.tsim, self.x_mpc[:, 2], '--', label='phi', lw=lw)
        # ax[0].plot(self.tsim, self.x_mpc[:, 4], '--', label='w', lw=lw)
        # ax[0].plot(self.tsim, self.x_mpc[:, 5], '--', label='zeta', lw=lw)

        ax[0].plot(self.tsim, self.sim_data_df.v_para, '--', label='v_para', lw=lw)
        ax[0].plot(self.tsim, self.sim_data_df.v_perp, '--', label='v_perp', lw=lw)
        ax[0].plot(self.tsim, self.sim_data_df.phi, '--', label='phi', lw=lw)
        ax[0].plot(self.tsim, self.sim_data_df.w, '--', label='w', lw=lw)
        ax[0].plot(self.tsim, self.sim_data_df.zeta, '--', label='zeta', lw=lw)

        ax[0].grid()
        ax[0].legend()

        ax[0].xaxis.set_tick_params(labelbottom=False)
        ax[0].set_ylabel('States')

        # Plot MPC control inputs
        ax[1].plot(self.tsim, self.u_mpc[:, 0], '-', label='u_para', lw=lw)
        ax[1].plot(self.tsim, self.u_mpc[:, 1], '-', label='u_perp', lw=lw)
        ax[1].plot(self.tsim, (1e-3)*self.u_mpc[:, 2], '-', label='u_phi', lw=lw)
        ax[1].plot(self.tsim, self.u_mpc[:, 3], '-', label='u_w', lw=lw)
        ax[1].plot(self.tsim, self.u_mpc[:, 4], '-', label='u_zeta', lw=lw)

        ax[1].grid()
        ax[1].legend()

        ax[1].set_xlabel('Time (s)')
        ax[1].set_ylabel('Inputs')

        # for n, s, in enumerate(self.mpc.x0.keys()):
        #     # ax.plot(self.tsim, getattr(self, s), 'k')
        #     ax.plot(self.tsim, self.x_mpc[:, n], '--', label=s)


class PsiTurn:
    def __init__(self, psi, g):
        """ Calculate parallel & perpendicular velocity for a psi turn.
        """
        self.psi = psi
        self.g = g

        # v_para_guess = self.g / 2
        # v_perp_guess = self.g / 2
        # self.initial_guess = np.array([v_para_guess, -v_perp_guess])
        # # self.initial_guess = np.array([0.01, 0.01])
        # self.solution = fsolve(self.equations, self.initial_guess, xtol=0.0001)
        # self.v_para_sol = self.solution[0]
        # self.v_perp_sol = self.solution[1]

        self.v_para_sol = self.g * np.cos(self.psi)
        self.v_perp_sol = self.g * np.sin(self.psi)

        self.g_sol = np.sqrt(self.v_perp_sol**2 + self.v_para_sol**2)
        self.psi_sol = np.arctan2(self.v_perp_sol, self.v_para_sol)

    def equations(self, x):
        v_para, v_perp = x
        # eq1 = np.arctan2(v_perp, v_para) - self.psi
        eq1 = np.arctan2(v_perp, v_para) - self.psi
        eq2 = np.sqrt(v_perp**2 + v_para**2) - self.g
        return [eq1, eq2]

