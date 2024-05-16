import numpy as np
import pandas as pd
from scipy import integrate
import scipy
import matplotlib.pyplot as plt
from matplotlib import cm, colors
from matplotlib.colors import ListedColormap

from utils import list_of_dicts_to_dict_of_lists, cart2polar
from setdict import SetDict
import figurefirst as fifi
import figure_functions as ff


class FlyWindDynamics:
    """ Simulate a dynamical system modeling a flying insect flying in a 2D environment with ambient wind.
    """

    def __init__(self, control_mode='open_loop', output_mode=None):
        """ Initialize the fly-wind-dynamics simulator.
        """

        # Set state names
        #   v_para: parallel speed in fly frame
        #   v_perp: perpendicular speed in fly frame
        #   phi: heading angle in global frame
        #   phidot: angular velocity
        #   w: wind speed
        #   zeta: wind angle in global frame
        #   I: yaw inertia
        #   m: mass
        #   C_para, C_perp, C_phi: damping
        #   d: altitude
        #   km1, km2, km3, km4: motor calibration parameters
        #   ks1, ks2, ks3, ks4, ks5, ks6, ks7: sensor calibration parameters
        self.state_names = ['v_para', 'v_perp', 'phi', 'phidot', 'w', 'zeta',  # main states
                            'I', 'm', 'C_para', 'C_perp', 'C_phi', 'd',  # parameters
                            'km1', 'km2', 'km3', 'km4',  # motor calibration parameters
                            'ks1', 'ks2', 'ks3', 'ks4', 'ks5', 'ks6', 'ks7']  # sensor calibration parameters

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

        # Set Initial state names & values
        self.x0 = {'v_para': 0.0, 'v_perp': 0.0, 'phi': 0.0, 'phidot': 0.0, 'w': 0.0, 'zeta': 0.0,
                   'I': I, 'm': m, 'C_para': C_para, 'C_perp': C_perp, 'C_phi': C_phi, 'd': 0.3,
                   'km1': 1.0, 'km2': 0.0, 'km3': 1.0, 'km4': 1.0,
                   'ks1': 1.0, 'ks2': 1.0, 'ks3': 0.0, 'ks4': 1.0, 'ks5': 0.0, 'ks6': 1.0, 'ks7': 0.0}

        self.state_names = [*self.x0.keys()]

        # Set input names
        #   u_para: parallel thrust in fly frame
        #   u_perp: perpendicular thrust in fly frame
        #   u_phi: rotational torque
        #   wdot: derivative of wind speed
        #   zetadot: derivative of wind angle
        self.input_names = ['u_para', 'u_perp', 'u_phi', 'wdot', 'zetadot']  # input names

        # Set sizes
        self.n_state = len(self.state_names)  # number of states
        self.n_input = len(self.input_names)  # number of inputs

        # Initialize the open-loop inputs
        self.t = 0.0  # current time
        self.u_para = np.array(0.0)  # parallel thrust
        self.u_perp = np.array(0.0)  # perpendicular thrust
        self.u_phi = np.array(0.0)  # rotational torque
        self.wdot = np.array(0.0)  # derivative of wind speed
        self.zetadot = np.array(0.0)  # derivative of wind angle in global frame

        # Variables to store states & controls at current time-step of simulation
        self.x = None  # state data
        self.u = None  # input data
        self.y = None  # output data

        # Initialize output mode
        if output_mode is not None:
            self.set_output_mode(output_mode=output_mode)
        else:
            self.n_output = None
            self.output_mode = []
            self.output_names = []

        # Initialize variables to store simulation data
        self.t_solve = []
        self.x_solve = []
        self.dt = 0.0  # sample time
        self.xvel = 0.0  # x velocity in global frame
        self.yvel = 0.0  # y velocity in global frame
        self.xpos = 0.0  # x position in global frame
        self.ypos = 0.0  # x position in global frame
        self.sim_data = {}  # all simulation data in dictionary

        # Initialize controller mode
        #   'open-loop': directly set the inputs
        #   'hover': set the inputs such that the simulated fly doesn't move no matter what wind does
        #   'no_dynamics': all dynamics are canceled out automatically by inputs, directly set trajectory
        #   'align_phi': PD controller to control heading
        self.control_mode = control_mode

        # Initialize control gains
        self.Kp_para = 10.0  # proportional control constant for parallel speed
        self.Kp_perp = 0.0  # proportional control constant for perpendicular speed
        self.Kp_phi = 80.0  # proportional control constant for rotational speed
        self.Kd_phi = 3.0  # derivative control constant for rotational speed

        # Initialize the control commands (closed-loop reference values or open-loop)
        self.r_para = np.array(0.0)  # parallel thrust if open-loop or reference speed if closed-loop
        self.r_perp = np.array(0.0)  # perpendicular thrust if open-loop or reference speed if closed-loop
        self.r_phi = np.array(0.0)  # rotational torque if open-loop or reference rotational speed if closed-loop

    def unpack_states(self, x, flag2D=False):
        if not flag2D:
            v_para, v_perp, phi, phidot, w, zeta, I, m, C_para, C_perp, C_phi, d, \
                km1, km2, km3, km4, ks1, ks2, ks3, ks4, ks5, ks6, ks7 = x
        else:
            x = np.atleast_2d(x)
            v_para = x[:, 0]
            v_perp = x[:, 1]
            phi = x[:, 2]
            phidot = x[:, 3]
            w = x[:, 4]
            zeta = x[:, 5]

            I = x[:, 6]
            m = x[:, 7]
            C_para = x[:, 8]
            C_perp = x[:, 9]
            C_phi = x[:, 10]
            d = x[:, 11]

            km1 = x[:, 12]
            km2 = x[:, 13]
            km3 = x[:, 14]
            km4 = x[:, 15]

            ks1 = x[:, 16]
            ks2 = x[:, 17]
            ks3 = x[:, 18]
            ks4 = x[:, 19]
            ks5 = x[:, 20]
            ks6 = x[:, 21]
            ks7 = x[:, 22]

        return v_para, v_perp, phi, phidot, w, zeta, I, m, C_para, C_perp, C_phi, d, \
            km1, km2, km3, km4, ks1, ks2, ks3, ks4, ks5, ks6, ks7

    def set_output_mode(self, output_mode):
        self.output_mode = output_mode
        self.n_output = len(self.output_mode)

    def update_inputs(self, x=None, t=None, r_para=0.0, r_perp=0.0, r_phi=0.0, wdot=0.0, zetadot=0.0):
        # Set state
        if x is None:
            x = self.x

        # Set time
        if t is not None:
            self.t = t

        # Set commands
        self.r_para = np.array(r_para)
        self.r_perp = np.array(r_perp)
        self.r_phi = np.array(r_phi)

        # Calculate control inputs
        self.u_para, self.u_perp, self.u_phi = self.calculate_control_inputs(r_para, r_perp, r_phi, x)

        # Set wind
        self.wdot = wdot
        self.zetadot = zetadot

    def calculate_air_velocity(self, x, flag2D=False, w_direct=None):
        # Get states
        v_para, v_perp, phi, phidot, w, zeta, I, m, C_para, C_perp, C_phi, d, \
            km1, km2, km3, km4, ks1, ks2, ks3, ks4, ks5, ks6, ks7 = self.unpack_states(x, flag2D=flag2D)

        # If w is set directly
        if w_direct is not None:
            w = w_direct.copy()

        # Air speed in parallel & perpendicular directions
        a_para = v_para - w * np.cos(phi - zeta)
        a_perp = v_perp + w * np.sin(phi - zeta)

        # Air velocity angle & magnitude
        a = np.linalg.norm((a_perp, a_para), ord=2, axis=0)  # air velocity magnitude
        gamma = np.arctan2(a_perp, a_para)  # air velocity angle

        return a_para, a_perp, a, gamma

    def calculate_ground_velocity(self, x, flag2D=False):
        # Get states
        v_para, v_perp, phi, phidot, w, zeta, I, m, C_para, C_perp, C_phi, d, \
            km1, km2, km3, km4, ks1, ks2, ks3, ks4, ks5, ks6, ks7 = self.unpack_states(x, flag2D=flag2D)

        # Ground velocity angle & magnitude
        g = np.linalg.norm((v_perp, v_para), ord=2, axis=0)  # ground velocity magnitude
        psi = np.arctan2(v_perp, v_para)  # ground velocity angle

        return v_para, v_perp, g, psi

    def calculate_acceleration(self, x, u, flag2D=False):
        # Get states
        v_para, v_perp, phi, phidot, w, zeta, I, m, C_para, C_perp, C_phi, d, \
            km1, km2, km3, km4, ks1, ks2, ks3, ks4, ks5, ks6, ks7 = self.unpack_states(x, flag2D=flag2D)

        # Inputs
        u_para = u[0]
        u_perp = u[1]

        # Calculate air velocity
        a_para, a_perp, a, gamma = self.calculate_air_velocity(x)

        # Compute drag
        D_para = C_para * a_para
        D_perp = C_perp * a_perp

        # Acceleration
        v_para_dot = ((km1 * u_para - D_para) / m) + (v_perp * phidot)
        v_perp_dot = ((km3 * u_perp - D_perp) / m) - (v_para * phidot)

        # Acceleration velocity angle & magnitude
        q = np.linalg.norm((v_perp_dot, v_para_dot), ord=2, axis=0)  # acceleration magnitude
        alpha = np.arctan2(v_perp_dot, v_para_dot)  # acceleration angle

        return v_para_dot, v_perp_dot, q, alpha

    def set_controller_gains(self, Kp_para=10.0, Kp_perp=0.0, Kp_phi=80.0, Kd_phi=3.0):
        if Kp_para is not None:
            self.Kp_para = Kp_para  # proportional control constant for parallel speed

        if Kp_perp is not None:
            self.Kp_perp = Kp_perp  # proportional control constant for perpendicular speed

        if Kp_phi is not None:
            self.Kp_phi = Kp_phi  # proportional control constant for rotational speed

        if Kd_phi is not None:
            self.Kd_phi = Kd_phi  # derivative control constant for rotational speed

    def calculate_control_inputs(self, r_para, r_perp, r_phi, x, flag2D=False):
        # Get states
        v_para, v_perp, phi, phidot, w, zeta, I, m, C_para, C_perp, C_phi, d, \
            km1, km2, km3, km4, ks1, ks2, ks3, ks4, ks5, ks6, ks7 = self.unpack_states(x, flag2D=flag2D)

        v_para, v_perp, g, psi = self.calculate_ground_velocity(x, flag2D=flag2D)
        a_para, a_perp, a, gamma = self.calculate_air_velocity(x, flag2D=flag2D)
        dir_of_travel = phi + psi

        # Calculate control input forces/torques based on control mode & control commands
        if self.control_mode == 'open_loop':
            u_para = np.array(r_para).copy()
            u_perp = np.array(r_perp).copy()
            u_phi = np.array(r_phi).copy()

        elif self.control_mode == 'align_psi':
            u_para = self.Kp_para * (r_para - v_para)
            u_perp = self.Kp_perp * (r_perp - v_perp)
            u_phi = self.Kp_phi * (r_phi - dir_of_travel) - self.Kd_phi * phidot

        elif self.control_mode == 'align_psi_constant_v_para':
            u_para = (((C_para * a_para) - (m * v_perp * phidot)) / km1) + (m * r_para / km1)
            u_perp = 0.0
            u_phi = self.Kp_phi * (r_phi - dir_of_travel) - self.Kd_phi * phidot

        elif self.control_mode == 'align_psi_constant_g':
            u_para = (((C_para * a_para) - (m * v_perp * phidot)) / km1) + (m * r_para / km1)
            u_perp = (((C_perp * a_perp) + (m * v_para * phidot)) / km3)
            u_phi = self.Kp_phi * (r_phi - dir_of_travel) - self.Kd_phi * phidot

        elif self.control_mode == 'align_phi_constant_v_para':
            u_para = (((C_para * a_para) - (m * v_perp * phidot)) / km1) + (m * r_para / km1)
            u_perp = 0.0
            u_phi = self.Kp_phi * (r_phi - phi) - self.Kd_phi * phidot

        elif self.control_mode == 'align_phi_constant_g':
            u_para = (((C_para * a_para) - (m * v_perp * phidot)) / km1) + (m * r_para / km1)
            u_perp = (((C_perp * a_perp) + (m * v_para * phidot)) / km3)
            u_phi = self.Kp_phi * (r_phi - phi) - self.Kd_phi * phidot

        elif self.control_mode == 'align_phidot_constant_v_para':
            u_para = (((C_para * a_para) - (m * v_perp * phidot)) / km1) + (m * r_para / km1)
            u_perp = 0.0
            u_phi = self.Kp_phi * (r_phi - phidot)

        elif self.control_mode == 'align_phidot_constant_g':
            u_para = (((C_para * a_para) - (m * v_perp * phidot)) / km1) + (m * r_para / km1)
            u_perp = (((C_perp * a_perp) + (m * v_para * phidot)) / km3)
            u_phi = self.Kp_phi * (r_phi - phidot)

        elif self.control_mode == 'constant_path':
            u_para = (((C_para * a_para) - (m * v_perp * phidot)) / km1) + (m * r_para / km1)
            u_perp = self.Kp_perp * (r_perp - v_perp)
            u_phi = np.array(r_phi).copy()
            # print(u_perp)

        elif self.control_mode == 'hover':  # set thrust to cancel out wind, can add control afterwards
            u_para = (((C_para * a_para) - (m * v_perp * phidot)) / km1)
            u_perp = (((C_perp * a_perp) + (m * v_para * phidot)) / km3)
            u_phi = ((C_phi * phidot) / km4)

        elif self.control_mode == 'no_dynamics_control':  # set thrust to cancel out wind, can add control afterwards
            u_para = (((C_para * a_para) - (m * v_perp * phidot)) / km1) + (m * r_para / km1)
            u_perp = (((C_perp * a_perp) + (m * v_para * phidot)) / km3) + (m * r_perp / km3)
            u_phi = ((C_phi * phidot) / km4) + (r_phi * I / km4)

        else:
            raise Exception("'control_mode' must be set to one of available options")

        return u_para, u_perp, u_phi  # return the open-loop control inputs

    def system_ode(self, t, x):
        """ Dynamical system model.

        Inputs
            x: current states (tuple)
                v_para - parallel speed
                v_perp - perpendicular speed
                phi - orientation in global frame
                phidot - change in orientation angle
                w - wind speed
                zeta - wind angle in global frame
            t: current time

        Outputs
            xdot: derivative of states
        """

        # Get states
        v_para, v_perp, phi, phidot, w, zeta, I, m, C_para, C_perp, C_phi, d, \
            km1, km2, km3, km4, ks1, ks2, ks3, ks4, ks5, ks6, ks7 = x

        self.t_solve.append(t)
        self.x_solve.append(x)

        # Calculate air velocity
        a_para, a_perp, a, gamma = self.calculate_air_velocity(x)

        # Compute drag
        D_para = C_para * a_para
        D_perp = C_perp * a_perp
        D_phi = C_phi * phidot

        # Update controls every time ODE solver is called
        self.u_para, self.u_perp, self.u_phi = self.calculate_control_inputs(self.r_para, self.r_perp, self.r_phi, x)

        # Derivative of states
        xdot = np.array([((km1 * self.u_para - D_para) / m) + (v_perp * phidot),  # v_para_dot
                         ((km3 * self.u_perp - D_perp) / m) - (v_para * phidot),  # v_perp_dot
                         phidot,  # phidot
                         (km4 * self.u_phi / I) - (D_phi / I) + (km2 * self.u_para / I),  # phiddot
                         self.wdot,  # wdot
                         self.zetadot,  # zetadot
                         0,  # I
                         0,  # m
                         0,  # C_para
                         0,  # C_perp
                         0,  # C_phi
                         0,  # d
                         0,  # km1
                         0,  # km2
                         0,  # km3
                         0,  # km4
                         0,  # ks1
                         0,  # ks2
                         0,  # ks3
                         0,  # ks4
                         0,  # ks5
                         0,  # ks6
                         0,  # ks7
                         ])

        return xdot

    def get_data_from_states(self, x, dt, vector_flag=False):
        # Get states
        v_para, v_perp, phi, phidot, w, zeta, I, m, C_para, C_perp, C_phi, d, \
            km1, km2, km3, km4, ks1, ks2, ks3, ks4, ks5, ks6, ks7 = self.unpack_states(x, flag2D=True)

        # Calculate ground velocity
        v_para, v_perp, g, psi = self.calculate_ground_velocity(self.x, flag2D=True)

        # Optic flow
        of = g / d

        # Calculate air velocity
        a_para, a_perp, a, gamma = self.calculate_air_velocity(self.x, flag2D=True)
        dir_of_travel = phi + psi

        # Drag
        D_para = C_para * a_para
        D_perp = C_perp * a_perp
        D_phi = C_phi * phidot

        # Acceleration
        v_para_dot = ((km1 * self.u_para - D_para) / m) + (v_perp * phidot)
        v_perp_dot = ((km3 * self.u_perp - D_perp) / m) - (v_para * phidot)
        q = np.linalg.norm((v_para_dot, v_perp_dot), ord=2, axis=0)  # acceleration magnitude
        alpha = np.arctan2(v_perp_dot, v_para_dot)  # acceleration angle
        alpha = np.unwrap(alpha)

        v_para_dot_uncal = ks1 * v_para_dot
        v_perp_dot_uncal = ks2 * v_perp_dot

        # Angular acceleration
        phiddot = km4 * (self.u_phi / I) - (D_phi / I)

        # Controls in polar coordinates
        u_g, u_psi = cart2polar(self.u_para, self.u_perp)

        # Uncalibrated sensors
        phi_uncal = ks1 * phi
        gamma_uncal = (ks2 * gamma) + ks3
        psi_uncal = (ks4 * psi) + ks5
        alpha_uncal = (ks6 * alpha) + 0.0 * ks7

        # Angle units vectors
        phi_y = np.sin(phi)
        phi_x = np.cos(phi)
        psi_y = np.sin(psi)
        psi_x = np.cos(psi)
        gamma_y = np.sin(gamma)
        gamma_x = np.cos(gamma)
        alpha_y = np.sin(alpha)
        alpha_x = np.cos(alpha)
        zeta_y = np.sin(zeta)
        zeta_x = np.cos(zeta)

        phi_uncal_y = np.sin(phi_uncal)
        phi_uncal_x = np.cos(phi_uncal)
        psi_uncal_y = np.sin(psi_uncal)
        psi_uncal_x = np.cos(psi_uncal)
        gamma_uncal_y = np.sin(gamma_uncal)
        gamma_uncal_x = np.cos(gamma_uncal)
        alpha_uncal_y = np.sin(alpha_uncal)
        alpha_uncal_x = np.cos(alpha_uncal)

        # Ground velocity in global frame
        xvel = v_para * np.cos(phi) - v_perp * np.sin(phi)
        yvel = v_para * np.sin(phi) + v_perp * np.cos(phi)

        if vector_flag:
            self.xpos = integrate.cumtrapz(xvel, x=dt, initial=0)
            self.ypos = integrate.cumtrapz(yvel, x=dt, initial=0)
            self.t = dt

        else:
            # Compute change in position
            xvel_v = np.hstack((self.xvel, xvel))
            yvel_y = np.hstack((self.yvel, yvel))

            delta_xpos = scipy.integrate.trapz(xvel_v, dx=dt)
            delta_ypos = scipy.integrate.trapz(yvel_y, dx=dt)

            # New position
            self.xpos = self.xpos + delta_xpos
            self.ypos = self.ypos + delta_ypos

            # Update current time
            self.t = np.round(self.t + dt, 8)

        # Update current velocity
        self.xvel = xvel.copy()
        self.yvel = yvel.copy()

        # Collect data
        data = {'time': self.t,
                'v_para': v_para,
                'v_perp': v_perp,
                'phi': phi,
                'phidot': phidot,
                'w': w,
                'zeta': zeta,
                'I': I,
                'm': m,
                'C_para': C_para,
                'C_perp': C_perp,
                'C_phi': C_phi,
                'd': d,
                'km1': km1,
                'km2': km2,
                'km3': km3,
                'km4': km4,
                'ks1': ks1,
                'ks2': ks2,
                'ks3': ks3,
                'ks4': ks4,
                'ks5': ks5,
                'ks6': ks5,
                'ks7': ks5,
                'g': g,
                'psi': psi,
                'of': of,
                'a_para': a_para,
                'a_perp': a_perp,
                'a': a,
                'gamma': gamma,
                'v_para_dot': v_para_dot,
                'v_perp_dot': v_perp_dot,
                'v_para_dot_uncal': v_para_dot_uncal,
                'v_perp_dot_uncal': v_perp_dot_uncal,
                'phiddot': phiddot,
                'q': q,
                'alpha': alpha,
                'dir_of_travel': dir_of_travel,
                'xvel': xvel,
                'yvel': yvel,
                'xpos': self.xpos,
                'ypos': self.ypos,
                'u_para': self.u_para,
                'u_perp': self.u_perp,
                'u_phi': self.u_phi,
                'wdot': self.wdot,
                'zetadot': self.zetadot,
                'u_g': u_g,
                'u_psi': u_psi,
                'phi_uncal': phi_uncal,
                'gamma_uncal': gamma_uncal,
                'psi_uncal': psi_uncal,
                'alpha_uncal': alpha_uncal,

                'phi_y': phi_y,
                'phi_x': phi_x,
                'psi_y': psi_y,
                'psi_x': psi_x,
                'gamma_y': gamma_y,
                'gamma_x': gamma_x,
                'alpha_y': alpha_y,
                'alpha_x': alpha_x,
                'zeta_x': zeta_x,
                'zeta_y': zeta_y,

                'phi_uncal_y': phi_uncal_y,
                'phi_uncal_x': phi_uncal_x,
                'psi_uncal_y': psi_uncal_y,
                'psi_uncal_x': psi_uncal_x,
                'gamma_uncal_y': gamma_uncal_y,
                'gamma_uncal_x': gamma_uncal_x,
                'alpha_uncal_y': alpha_uncal_y,
                'alpha_uncal_x': alpha_uncal_x}

        return data

    def odeint_step(self, x0=None, dt=None, usim=None):
        """ Solve ODE for one time step"""

        if x0 is None:
            x0 = self.x.copy()
        else:
            x0 = x0.copy()

        if dt is None:
            dt = self.dt

        # Get inputs
        if usim is None:
            usim = np.zeros(self.n_input)

        usim = usim.copy()

        # Update control inputs & wind
        self.update_inputs(x0, r_para=usim[0], r_perp=usim[1], r_phi=usim[2], wdot=usim[3], zetadot=usim[4])

        # Integrate for one time step
        t_span = np.array([0, dt])
        x_solve = integrate.odeint(self.system_ode, x0, t_span, tcrit=t_span, tfirst=True)

        # Just get solution at t=dt
        self.x = x_solve[1]

        # Calculate data
        data = self.get_data_from_states(self.x.copy(), dt, vector_flag=False)

        return self.x.copy(), data

    def odeint_simulate(self, x0=None, tsim=None, usim=None):
        # Reset simulator
        self.reset()

        # Set initial states
        if x0 is None:
            x0 = np.array(list(self.x0.values()))
        elif isinstance(x0, dict):
            SetDict().set_dict_with_overwrite(self.x0, x0)
            x0 = np.array(list(self.x0.values()))
        elif isinstance(x0, list):
            x0 = np.array(x0)

        # Time step
        dt = np.mean(np.diff(tsim))

        # Run once at time 0 to get initial data
        x, data = self.odeint_step(x0=x0, dt=0.0, usim=usim[0, :])

        # Solve ODE in steps
        t_solve = [0]
        x_solve = [x]
        sim_data = [data]
        for n in range(1, tsim.shape[0]):  # for each data point in input time vector
            # Step
            x, data = self.odeint_step(x0=x, dt=dt, usim=usim[n, :])

            t_solve.append(self.t)
            x_solve.append(x)
            sim_data.append(data)

        # Concatenate state vectors & data
        t_solve = np.hstack(t_solve)
        x_solve = np.vstack(x_solve)
        sim_data = list_of_dicts_to_dict_of_lists(sim_data, make_array=True)

        # Unwrap angles
        sim_data['phi'] = np.unwrap(sim_data['phi'])
        sim_data['psi'] = np.unwrap(sim_data['psi'])
        sim_data['gamma'] = np.unwrap(sim_data['gamma'])
        sim_data['alpha'] = np.unwrap(sim_data['alpha'])

        return sim_data, x_solve, t_solve

    def reset(self, time=0.0, xpos=0.0, ypos=0.0, xvel=0.0, yvel=0.0):
        self.t = time
        self.xpos = xpos
        self.ypos = ypos
        self.xvel = xvel
        self.yvel = yvel
        self.t_solve = []
        self.x_solve = []

    def get_outputs(self, sim_data, output_mode=None):
        # Collect outputs
        if output_mode is not None:
            self.output_mode = output_mode

        if isinstance(self.output_mode, list):
            output_names = self.output_mode
        else:
            output_names = self.output_mode.split(',')

        n_output = len(output_names)
        n_point = len(sim_data['time'])
        y = np.nan * np.zeros((n_point, n_output))
        self.output_names = output_names
        for n in range(n_output):
            y[:, n] = sim_data[output_names[n].strip()]

        return y

    def simulate(self, x0, tsim, usim, output_mode=None, solver_type='step'):
        """ Simulate the system.
        """

        # Get inputs
        if isinstance(usim, dict):
            # Make dictionary to store inputs
            usim_dict = {'u_para': np.zeros_like(tsim),
                         'u_perp': np.zeros_like(tsim),
                         'u_phi': np.zeros_like(tsim),
                         'wdot': np.zeros_like(tsim),
                         'zetadot': np.zeros_like(tsim)}

            # Set inputs given, if not given then keep as 0's
            SetDict().set_dict_with_overwrite(usim_dict, usim)

            # Concatenate into array
            usim = np.stack((usim_dict['u_para'],
                             usim_dict['u_perp'],
                             usim_dict['u_phi'],
                             usim_dict['wdot'],
                             usim_dict['zetadot']), axis=1)

        # Solve
        usim[0, :] = usim[1, :]
        if solver_type == 'step':
            sim_data, x, t = self.odeint_simulate(x0, tsim, usim)
        else:
            raise Exception('solver type not valid')

        # Calculate outputs
        y = self.get_outputs(sim_data, output_mode=output_mode)

        self.x = x
        self.y = y

        self.u = np.stack((sim_data['u_para'],
                           sim_data['u_perp'],
                           sim_data['u_phi'],
                           sim_data['wdot'],
                           sim_data['zetadot']), axis=1)

        self.sim_data = pd.DataFrame(sim_data)

        return sim_data, y

    def plot_trajectory(self, cmap=None, size=5, dpi=100, arrow_size=None, nskip=0):
        if arrow_size is None:
            arrow_size = 0.7 * np.mean(np.abs(np.hstack((self.sim_data['xpos'], self.sim_data['ypos']))))

        # Plot the trajectory
        if cmap is None:
            crange = 0.1
            cmap = cm.get_cmap('bone_r')
            cmap = cmap(np.linspace(crange, 1, 100))
            cmap = ListedColormap(cmap)

        # Plot
        fig, ax = plt.subplots(1, 1, figsize=(size, size), dpi=dpi)

        ff.plot_trajectory(self.sim_data['xpos'].values,
                           self.sim_data['ypos'].values,
                           self.sim_data['phi'].values,
                           color=self.sim_data['time'],
                           ax=ax,
                           size_radius=arrow_size,
                           nskip=nskip,
                           colormap=cmap)

        fifi.mpl_functions.adjust_spines(ax, [])

    # Output functions
    def g(self, x, u=None):
        v_para, v_perp, g, psi = self.calculate_ground_velocity(x, flag2D=True)
        return g

    def psi(self, x, u=None):
        v_para, v_perp, g, psi = self.calculate_ground_velocity(x, flag2D=True)
        return psi

    def a(self, x, u=None):
        a_para, a_perp, a, gamma = self.calculate_air_velocity(x, flag2D=True, w_direct=None)
        return a

    def gamma(self, x, u=None):
        a_para, a_perp, a, gamma = self.calculate_air_velocity(x, flag2D=True, w_direct=None)
        return gamma

    def q(self, x, u=None):
        v_para_dot, v_perp_dot, q, alpha = self.calculate_acceleration(x, u, flag2D=True)
        return q

    def alpha(self, x, u=None):
        v_para_dot, v_perp_dot, q, alpha = self.calculate_acceleration(x, u, flag2D=True)
        return alpha


class FlyWindDynamics_test:
    def __init__(self):
        w = 0.4
        zeta = np.pi

        m = 0.25e-6  # [kg]
        I = 5.2e-13  # [N*m*s^2] yaw mass moment of inertia: 10.1242/jeb.02369
        C_phi = 27.36e-12  # [N*m*s] yaw damping: 10.1242/jeb.038778
        C_para = m / 0.170  # [N*s/m] calculate using the mass and time constant reported in 10.1242/jeb.098665
        C_perp = C_para  # assume same as C_para

        m = m * 1e6  # [mg]
        I = I * 1e6 * (1e3) ** 2  # [mg*mm/s^2 * mm*s^2]
        C_phi = C_phi * 1e6 * (1e3) ** 2  # [mg*mm/s^2 *m*s]
        C_para = C_para * 1e6  # [mg/s]
        C_perp = C_perp * 1e6  # [mg/s]

        self.x0 = {'v_para': 0.3, 'v_perp': 0.0, 'phi': 0.0, 'phidot': 0.0, 'w': w, 'zeta': zeta,
                   'I': I, 'm': m, 'C_para': C_para, 'C_perp': C_perp, 'C_phi': C_phi,
                   'km1': 1.0, 'km2': 0.0, 'km3': 1.0, 'km4': 1.0}

        self.x = {}

    def system_ode(self, t, x):
        # Get states
        v_para, v_perp, phi, phidot = x

        # Calculate air velocity
        a_para = v_para - self.x0['w'] * np.cos(phi - self.x0['zeta'])
        a_perp = v_perp + self.x0['w'] * np.sin(phi - self.x0['zeta'])

        # Compute drag
        D_para = self.x0['C_para'] * a_para
        D_perp = self.x0['C_perp'] * a_perp
        D_phi = self.x0['C_phi'] * phidot

        u_para = 1.0
        u_perp = 1.0
        u_phi = 10.0 * np.sin(2*np.pi*3*t)

        # Derivative of states
        xdot = np.array([((self.x0['km1'] * u_para - D_para) / self.x0['m']) + (v_perp * phidot),  # v_para_dot
                         ((self.x0['km3'] * u_perp - D_perp) / self.x0['m']) - (v_para * phidot),  # v_perp_dot
                         phidot,  # phidot
                         (self.x0['km4'] * u_phi / self.x0['I']) - (D_phi / self.x0['I']) + (self.x0['km2'] * u_para / self.x0['I'])])  # phiddot

        return xdot

    def run(self, tsim, x0):
        x = integrate.odeint(self.system_ode, x0, tsim, tcrit=tsim, tfirst=True)

        v_para = x[:, 0]
        v_perp = x[:, 1]
        phi = x[:, 2]
        phidot = x[:, 3]

        a_para = v_para - self.x0['w'] * np.cos(phi - self.x0['zeta'])
        a_perp = v_perp + self.x0['w'] * np.sin(phi - self.x0['zeta'])

        D_para = self.x0['C_para'] * a_para
        D_perp = self.x0['C_perp'] * a_perp
        D_phi = self.x0['C_phi'] * phidot

        u_para = 1.0
        u_perp = 1.0
        u_phi = 10.0 * np.sin(2*np.pi*3*tsim)

        v_para_dot = ((self.x0['km1'] * u_para - D_para) / self.x0['m']) + (v_perp * phidot)
        v_perp_dot = ((self.x0['km3'] * u_perp - D_perp) / self.x0['m']) - (v_para * phidot)

        self.x = {'v_para': v_para,
                  'v_perp': v_perp,
                  'phi': phi,
                  'phidot': phidot,
                  'v_para_dot': v_para_dot,
                  'v_perp_dot': v_perp_dot}

        return pd.DataFrame(self.x)


