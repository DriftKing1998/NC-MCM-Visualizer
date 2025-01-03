import colorsys

import numpy as np


# General Functions #

def generate_equidistant_colors(n, color=None):
    """
        Generate a list of RGB colors in HSV space with equidistant hues.

        Parameters:
        - n:  int, required
            Number of colors to generate.

        Returns:
        - colors: List of RGB colors.
    """
    colors = []
    if int == type(color):
        color = int(color%3)
        for i in range(n):
            val = i / n  # value
            rgb = [val, val, val]
            rgb[color] += 2 - np.exp(val)
            colors.append(tuple(rgb))
    else:
        for i in range(n):
            hue = i / n  # hue value
            saturation = 1.0  # fully saturated
            value = 1.0  # full brightness
            rgb_color = colorsys.hsv_to_rgb(hue, saturation, value)
            colors.append(rgb_color)
    return colors


def shift_pos_by(old_positioning, new_positioning, degree, offset):
    """
        Shift positions in polar coordinates.

        Parameters:

        - old_positioning: np.ndarray, required
            Dictionary of node positions.

        - new_positioning:  np.ndarray, required
            Dictionary of new node positions will be updated

        - degree:  float, required
            Degree to shift positions.

        - offset:  float, required
            Offset distance.

        Returns:

        - new_positioning: Updated dictionary of node positions.
    """
    for node, coords in old_positioning.items():
        new_positioning[node] = (coords[0] + offset * np.cos(np.radians(degree)),
                                 coords[1] + offset * np.sin(np.radians(degree)))
    return new_positioning


# Data Processing #


def adj_matrix_ncmcm(data, cog_stat_num=3, clustering_rep=None):
    """
        Calculate the adjacency matrix and list of cognitive-behavioral states.

        Parameters:
       
        - data: Database, required
            Data from the database.

        - cog_stat_num: int, optional
            Number of cognitive states in the plot (e.g., C1, C2, C3 ...).

        - clustering_rep: int, optional
            Defines which clustering should be used (by index), otherwise best p-value is used

        Returns:
       
        - cog_beh_states: list
            List of all cognitive-behavioral states (coded as: CCBB).

        - T: np.ndarray
            Adjacency matrix for the cognitive-behavioral states.
    """
    if type(clustering_rep) is int:
        best_clustering_idx = clustering_rep
    else:
        print('Clustering was chosen according to best p-memorylessness.')
        best_clustering_idx = np.argmax(data.p_memoryless[cog_stat_num - 1, :])  # according to mr.markov himself

    C = data.xc[:, cog_stat_num - 1, best_clustering_idx].astype(int)
    b = np.unique(data.B)
    c = np.unique(C)
    T = np.zeros((len(c) * len(b), len(c) * len(b)))
    C_B_states = np.asarray([str(cs + 1) + '-' + str(bs) for cs in c for bs in b])

    for m in range(len(data.B) - 1):
        cur_sample = m
        next_sample = m + 1
        cur_state = np.where(str(C[cur_sample] + 1) + '-' + str(data.B[cur_sample]) == C_B_states)[0][0]
        next_state = np.where(str(C[next_sample] + 1) + '-' + str(data.B[next_sample]) == C_B_states)[0][0]
        T[next_state, cur_state] += 1

    # normalize T
    T = T / (len(data.B) - 1)
    T = T.T

    return T, C_B_states


def make_integer_list(input_list):
    """
        Convert a list of strings to a list of integers and create a translation list.

        Parameters:
       
        - input_list: list, required
            List of strings.

        Returns:
       
        - integer_list: list
            A list of integers corresponding to input_list.

        - translation_list: list
            A list of unique strings in input_list.
    """
    string_to_int = {}
    integer_list = []

    for s in input_list:
        if s not in string_to_int:
            string_to_int[s] = len(string_to_int)
        integer_list.append(string_to_int[s])

    translation_list = list(string_to_int.keys())

    return integer_list, translation_list


def make_windowed_data(X, B, win=15):
    """
        Create windowed data from input sequences. The format needed for BundDLeNet

        Parameters:
       
        - X: np.ndarray, required
            Input sequences.

        - B: np.ndarray, required
            Labels.

        - win: int, optional
            Window size.

        Returns:
       
        - newX: np.ndarray
            Windowed input sequences.

        - newB: np.ndarray
            Updated labels.
    """
    win += 1
    X_win = np.zeros((X.shape[0] - win + 1, win, X.shape[1]))
    for i, _ in enumerate(X_win):
        X_win[i] = X[i:i + win]
    newB = B[win - 1:]

    newX = X_win[:, :-1, :]
    return newX, newB


