import numpy as np
import copy
import pandas as pd


def random_segments_from_df(df, n_segment=1, segment_sizes=(10,), reset_index=False):
    """ Pull out random segments from data-frame.

        n_segment: # of segments to pull out
        segment_sizes: tuple of possible segment sizes
        reset_index: if True, then reset index
    """

    # Get random segments
    n_df_size = df.shape[0]
    n_segment_sizes = len(segment_sizes)
    segment_start_list = []
    segment_list = []
    for n in range(n_segment):  # each segment
        # Set the segment size randomly
        np.random.seed(seed=n)
        segment_size_index = np.squeeze(np.random.randint(0, high=n_segment_sizes, size=1, dtype=int))
        segment_size = int(segment_sizes[segment_size_index])

        # Set the start point randomly & make sure we don't reuse a start point
        c = 1
        np.random.seed(seed=n)
        segment_start = int(np.squeeze(np.random.randint(0, high=n_df_size - segment_size, size=1, dtype=int)))
        while segment_start in segment_start_list:
            # Random start
            np.random.seed(seed=n + c)
            segment_start = int(np.squeeze(np.random.randint(0, high=n_df_size - segment_size, size=1, dtype=int)))
            c = c + 1

            # Make sure we don't loop forever
            if c > 100:
                print('Warning: reusing random start point after 100 iterations, try reducing the # of segments')
                break

        segment_start_list.append(segment_start)  # add segment start to list
        segment = df.iloc[segment_start:(segment_start + segment_size), :]  # get segment

        # Reset index, if specified
        if reset_index:
            segment = segment.reset_index(drop=True)

        segment_list.append(segment)  # add segment start to list

    return segment_list, segment_start_list


def list_of_dicts_to_dict_of_lists(list_of_dicts, keynames=None, make_array=None):
    """ Takes a list containing dictionary with the same key names &
        converts it to a single dictionary where each key is a list.

        Inputs
            list_of_dicts:      input list
            keynames:           if None then use all the keys, otherwise set the key names here as a list of strings

        Outputs
            dict_of_lists:      output dictionary
    """

    # Get all the key names if not given as input
    if keynames is None:
        keynames = list_of_dicts[0].keys()  # use 1st dict to get key names

    # Create output dictionary
    dict_of_lists = {}
    for k in keynames:
        dict_of_lists[k] = []  # each key is a list

        # Get the values from the dictionaries & append to list in output dictionaries
        for n in list_of_dicts:
            dict_of_lists[k].append(n[k])

    if make_array is not None:
        for k in keynames:
            dict_of_lists[k] = np.hstack(dict_of_lists[k])

    return dict_of_lists


# def wrapTo2Pi(rad):
#     rad = rad % (2 * np.pi)
#     return rad


def wrapTo2Pi(rad):
    rad = copy.copy(rad)
    rad = rad % (2 * np.pi)
    return rad


def wrapToPi(rad):
    rad_wrap = copy.copy(rad)
    q = (rad_wrap < -np.pi) | (np.pi < rad_wrap)
    rad_wrap[q] = ((rad_wrap[q] + np.pi) % (2 * np.pi)) - np.pi
    return rad_wrap


def polar2cart(r, theta):
    # Transform polar to cartesian
    x = r * np.cos(theta)
    y = r * np.sin(theta)

    return x, y


def cart2polar(x, y):
    # Transform cartesian to polar
    r = np.sqrt((x ** 2) + (y ** 2))
    theta = np.arctan2(y, x)

    return r, theta


def sliding_window(df, slide=None, w=None, n_window_limit=None, seed=None, aug_column_names=None):
    """ Takes a pandas data frame with n rows, list of columns names, and a window size w.
        Then creates an augmented data frame that collects prior or future rows (in window)
        and stacks them as new columns. The augmented data frame will be size (n - w - 1) as the first/last
        w rows do not have enough data before/after them.

        Inputs
            df: pandas data frame
            slide: how much to slide the window, 1 is every point, None is random points
            w: window size
            n_window_limit: if slide=None, then this is limit for the # of random windows selected
            seed: random seed
            aug_column_names: names of the columns to augment

        Outputs
            df_aug: augmented pandas data frame.
                    new columns are named: old_name_0, old_name_1, ... , old_name_w-1
    """

    df = df.reset_index(drop=True)

    if aug_column_names is None:
        aug_column_names = df.columns

    n_points = df.shape[0]  # points in data-frame
    n_possible_start_points = n_points - w + 1  # start point needs to w points from end of data frame

    # Set the window start points
    if slide is None:  # random window start points
        # Need to set the # of random windows
        assert (n_window_limit is not None),\
            '"n_window_limit" must not be None when "slide"=None'
        assert (n_window_limit < n_possible_start_points),\
            '"n_window_limit" must be less than # of rows in data-frame minus the window size'

        if n_window_limit > 0.9*n_possible_start_points:
            print('You are using 90% of the possible start points, do you really need to be doing this randomly?')

        # Set seed
        np.random.seed(seed=seed)

        # Set random start points
        window_start = np.unique(np.random.randint(0, high=n_possible_start_points, size=n_window_limit, dtype=int))

        # Add points up until limit if there was repeated values
        n = 1
        while window_start.shape[0] < n_window_limit:
            new_start = np.random.randint(0, high=n_possible_start_points, size=1, dtype=int)
            window_start = np.hstack((window_start, new_start))
            window_start = np.unique(window_start)
            n += 1

    else:  # start from 0 and slide to set start points
        window_start = np.arange(0, n_possible_start_points, slide, dtype=int)

    # Get the window indices
    window_indices = np.nan * np.zeros((window_start.shape[0], w), dtype=int)
    for r, ws in enumerate(window_start):
        window_indices[r, :] = np.arange(ws, ws + w, 1, dtype=int)

    # Pull out data in each window
    df_list = []
    for r in range(window_indices.shape[0]):
        win = window_indices[r, :]
        df_win = df.loc[win, aug_column_names]
        df_list.append(df_win)

    # Make new column names
    new_column_names = {}
    for a in aug_column_names:
        new_column_names[a] = []

    for a in aug_column_names:  # each augmented column
        for k in range(w):  # each point in lookback window
            new_column_names[a].append(a + '_' + str(k))

    # Stack each variable time series in rows
    df_aug_list = []
    for df_window in df_list:  # each window
        df_aug = []
        for c, cname in enumerate(df_window.columns):  # each column
            var_aug = df_window.loc[:, [cname]].T
            var_aug.columns = new_column_names[cname]
            df_aug.append(var_aug)

        temp = pd.concat(df_aug, axis=1, ignore_index=False)
        df_aug = pd.DataFrame(np.concatenate(df_aug, axis=1), columns=temp.columns)
        df_aug.index = df_window.index[0:1]
        df_aug_list.append(df_aug)

    df_aug_all = pd.concat(df_aug_list, axis=0)

    return df_aug_all, df_list


def collect_offset_rows(df, aug_column_names=None, keep_column_names=None, w=1, direction='backward'):
    """ Takes a pandas data frame with n rows, list of columns names, and a window size w.
        Then creates an augmented data frame that collects prior or future rows (in window)
        and stacks them as new columns. The augmented data frame will be size (n - w - 1) as the first/last
        w rows do not have enough data before/after them.

        Inputs
            df: pandas data frame
            aug_column_names: names of the columns to augment
            keep_column_names: names of the columns to keep, but not augment
            w: lookback window size (# of rows)
            direction: get the rows from behind ('backward') or front ('forward')

        Outputs
            df_aug: augmented pandas data frame.
                    new columns are named: old_name_0, old_name_1, ... , old_name_w-1
    """

    df = df.reset_index(drop=True)

    # Default for testing
    if df is None:
        df = np.atleast_2d(np.arange(0, 11, 1, dtype=np.double)).T
        df = np.matlib.repmat(df, 1, 4)
        df = pd.DataFrame(df, columns=['a', 'b', 'c', 'd'])
        aug_column_names = ['a', 'b']
    else:  # use the input  values
        # Default is all columns
        if aug_column_names is None:
            aug_column_names = df.columns

    # Make new column names & dictionary to store data
    new_column_names = {}
    df_aug_dict = {}
    for a in aug_column_names:
        new_column_names[a] = []
        df_aug_dict[a] = []

    for a in aug_column_names:  # each augmented column
        for k in range(w):  # each point in lookback window
            new_column_names[a].append(a + '_' + str(k))

    # Augment data
    n_row = df.shape[0]  # # of rows
    n_row_train = n_row - w + 1  # # of rows in augmented data
    for a in aug_column_names:  # each column to augment
        data = df.loc[:, [a]]  # data to augment
        data = np.asmatrix(data)  # as numpy matrix
        df_aug_dict[a] = np.nan * np.ones((n_row_train, len(new_column_names[a])))  # new augmented data matrix

        # Put augmented data in new column, for each column to augment
        for i in range(len(new_column_names[a])):  # each column to augment
            if direction == 'backward':
                # Start index, starts at the lookback window size & slides up by 1 for each point in window
                startI = w - 1 - i

                # End index, starts at end of the matrix &  & slides up by 1 for each point in window
                endI = n_row - i  # end index, starts at end of matrix &

            elif direction == 'forward':
                # Start index, starts at the beginning of matrix & slides up down by 1 for each point in window
                startI = i

                # End index, starts at end of the matrix minus the window size
                # & slides down by 1 for each point in window
                endI = n_row - w + 1 + i  # end index, starts at end of matrix &

            else:
                raise Exception("direction must be 'forward' or 'backward'")

            # Put augmented data in new column
            df_aug_dict[a][:, i] = np.squeeze(data[startI:endI, :])

        # Convert data to pandas data frame & set new column names
        df_aug_dict[a] = pd.DataFrame(df_aug_dict[a], columns=new_column_names[a])

    # Combine augmented column data
    df_aug = pd.concat(list(df_aug_dict.values()), axis=1)

    # Add non-augmented data, if specified
    if keep_column_names is not None:
        for c in keep_column_names:
            if direction == 'backward':
                startI = w - 1
                endI = n_row
            elif direction == 'forward':
                startI = 0
                endI = n_row - w
            else:
                raise Exception("direction must be 'forward' or 'backward'")

            keep = df.loc[startI:endI, [c]].reset_index(drop=True)
            df_aug = pd.concat([df_aug, keep], axis=1)

    return df_aug


def log_scale_with_negatives(x, epsilon=2.0, inverse=False):
    """ Transform a set of numbers to log-scale.
        If there are negative numbers, treat them as positive but preserve the sign.
        epsilon is a positive number that is added to all values (after converted to positive)
        to prevent numbers < 1 appearing before log transform.
    """

    # Find the negative values
    x_sign = np.sign(x)  # sign of x
    x_negative_idx = x < 0  # where negative values are

    # Log-scale the positive & negative values while preserving the sign
    y = x.copy()
    y[~x_negative_idx] = np.log(epsilon + y[~x_negative_idx])
    y[x_negative_idx] = -np.log(epsilon + -y[x_negative_idx])

    # Take inverse if specified
    if inverse:
        y = 1 / y

    return y

