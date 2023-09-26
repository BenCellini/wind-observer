
import numpy as np
# from scipy.optimize import fsolve


def empirical_observability_matrix(system, x0, tsim, usim, eps=1e-5):
    """ Empirically calculates the observability matrix O for a given system & input.

        Inputs
            system:             simulator object
            x0:                 initial state
            tsim:               simulation time
            usim:               simulation inputs
            eps:                amount to perturb initial state

        Outputs
            O:                  numerically calculated observability matrix
            sim_data            nominal trajectory simulation data
            deltay:             the difference in perturbed measurements at each time step
                                (basically O stored in a 3D array)
    """

    # Simulate once for nominal trajectory
    sim_data, y = system.simulate(x0, tsim, usim)
    n_state = system.x.shape[1]  # number of states
    n_output = y.shape[1]  # number of outputs

    # Calculate O
    w = len(tsim)  # of points in time window
    delta = eps * np.eye(n_state)  # perturbation amount for each state
    deltay = np.zeros((n_output, n_state, w))  # preallocate deltay
    for k in range(n_state):  # each state
        # Perturb initial condition in both directions
        x0plus = x0 + delta[:, k]
        x0minus = x0 - delta[:, k]

        # Simulate measurements from perturbed initial conditions
        _, yplus = system.simulate(x0plus, tsim, usim)
        _, yminus = system.simulate(x0minus, tsim, usim)

        # Calculate the numerical Jacobian & normalize by 2x the perturbation amount
        deltay[:, k, :] = np.array(yplus - yminus).T / (2 * eps)

    # Construct O by stacking the 3rd dimension of deltay along the 1st dimension, O is a (p*m x n) matrix
    O = []  # list to store datat at each time point fo O
    for j in range(w):
        O.append(deltay[:, :, j])

    O = np.vstack(O)

    return O, sim_data, deltay


def empirical_observability_matrix_sliding(system, tsim, usim, xsim, eps=1e-5,
                                           time_resolution=None, simulation_time=None):
    """ Calculate the observability matrix O for every point along a nominal state trajectory.

        Inputs
            system:             simulator object
            x0:                 initial state
            tsim:               simulation time
            usim:               simulation inputs
            eps:                amount to perturb initial state
            simulation_time:    simulation time for each calculation of O
            time_resolution:    how often to calculate O along the nominal state trajectory.

        Outputs
            O_sliding:          a list of the O's calculated at set times along the nominal state trajectory
            O_time:             the times O was computed
            O_index:            the indices where O was computed
    """

    dt = np.mean(np.diff(tsim))
    N = np.squeeze(tsim).shape[0]

    # Set simulation_time to fill space between time_resolution
    if simulation_time is None:
        simulation_time = time_resolution

    # The size of the simulation time
    simulation_index = np.round(simulation_time / dt, decimals=0).astype(int)

    # Set time_resolution to every point, if not specified
    if time_resolution is None:
        time_resolution = dt

    # If time_resolution is a vector, then use the entries as the indices to calculate O
    if not np.isscalar(time_resolution) or (time_resolution == 0):  # time_resolution is a vector
        O_index = time_resolution
    else:  # evenly space the indices to calculate O
        # Resolution on nominal trajectory to calculate O, measured in indices
        time_resolution_index = np.round(time_resolution / dt, decimals=0).astype(int)

        # All the indices to calculate O
        O_index = np.arange(0, N - simulation_index, time_resolution_index)  # indices to compute O

    O_time = O_index * np.array(dt)  # times to compute O
    n_point = len(O_index)  # # of times to calculate O

    # Calculate O for each point on nominal trajectory
    window_data = {'t': [], 'u': [], 'x': [], 'y': [], 'sim_data': []}  # where to store sliding window trajectory data
    O_sliding = []  # where to store the sliding O's
    for n in range(n_point):  # each point on trajectory
        # Start simulation at point along nominal trajectory
        x0 = np.squeeze(xsim[O_index[n], :])  # get state on trajectory & set it as the initial condition

        # Get the range to pull out time & input data for simulation
        win = np.arange(O_index[n], O_index[n] + simulation_index, 1)  # index range

        # Remove part of window if it is past the end of the nominal trajectory
        withinN = win < N
        win = win[withinN]

        # Pull out time & control inputs in window
        twin = tsim[win]  # time in window
        twin0 = twin - twin[0]  # start at 0
        uwin = usim[win, :]  # inputs in window

        # Calculate O for window
        O, sim_data, deltay = empirical_observability_matrix(system, x0, twin0, uwin, eps=eps)

        # Simulate once for nominal trajectory
        sim_data, y = system.simulate(x0, twin0, uwin)

        # Store data
        O_sliding.append(O.copy())
        window_data['t'].append(twin.copy())
        window_data['u'].append(uwin.copy())
        window_data['x'].append(system.x.copy())
        window_data['y'].append(y.copy())
        window_data['sim_data'].append(sim_data)

    return O_sliding, O_time, O_index, window_data


def rank_test(O, states=None, tol=None):
    """ Evaluate the observability of states based on the observability matrix O.

        Inputs
            O:              observability matrix
            states:         states of interest, default is all states
            tol:            tolerance for matrix rank computation singular values

        Outputs
            observability:  observability gramian
    """

    n_state = O.shape[1]

    # Get rank of O
    O_rank = np.linalg.matrix_rank(O, tol)

    # Set states to evaluate
    if states is None:
        states = np.arange(0, n_state, 1)
    else:
        states = np.array(states)

    # Evaluate each state
    observability = np.zeros(n_state)  # to store the observability of each state
    for s in states:
        # Create state vector
        state_vector = np.zeros((1, n_state))
        state_vector[0, s] = 1

        # Augment O
        O_augmented = np.vstack((O, state_vector))

        # Get rank of augmented O
        O_augmented_rank = np.linalg.matrix_rank(O_augmented, tol)

        # If ranks are equal, then the state is observable
        if O_augmented_rank == O_rank:
            observability[s] = 1

    return observability


def num_jacobian(h_x, x0, u0=None, eps=1e-5):
    """ Compute the numerical Jacobian of the function h(x) evaluated at state x0

            Inputs
                h_x: output function handle as a function of the state h(x)
                x0: initial state as 1D vector
                eps: amount to perturb initial state

            Outputs
                jx: Jacobian of the function h(x) evaluated at state x0
    """

    # Run output function for state
    y = h_x(x0, u=u0)
    n_output = y.shape[0]

    # Calculate Jacobian
    n_state = x0.shape[0]  # number of states
    delta = eps * np.eye(n_state)  # perturbation amount for each state
    deltay = np.zeros((n_output, n_state))
    for k in range(n_state):  # each state
        # Perturb state in both directions
        x0plus = x0 + delta[:, k]
        x0minus = x0 - delta[:, k]

        # Compute measurements from perturbed state
        yplus = h_x(x0plus, u=u0)
        yminus = h_x(x0minus, u=u0)

        # Calculate the numerical Jacobian
        deltay[:, k] = np.array(yplus - yminus).T

    # Normalize deltay by perturbation amount to get Jacobian
    jx0 = deltay / (2 * eps)  # normalize by 2x the perturbation amount

    return jx0
