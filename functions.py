import colorsys
import numpy as np

def make_integer_list(input_list):
    string_to_int = {}
    integer_list = []

    for s in input_list:
        if s not in string_to_int:
            string_to_int[s] = len(string_to_int)
        integer_list.append(string_to_int[s])

    translation_list = list(string_to_int.keys())

    return integer_list, translation_list


def generate_equidistant_colors(n):
    colors = []
    for i in range(n):
        hue = i / n  # Calculate the hue value
        saturation = 1.0  # Set saturation to 1.0 (fully saturated)
        value = 1.0  # Set value to 1.0 (full brightness)
        rgb_color = colorsys.hsv_to_rgb(hue, saturation, value)  # Convert HSV to RGB
        colors.append(rgb_color)
    return colors


def make_windowed_data(X, B, win=15):
    win += 1
    X_win = np.zeros((X.shape[0] - win + 1, win, X.shape[1]))
    for i, _ in enumerate(X_win):
        X_win[i] = X[i:i + win]
    newB = B[win - 1:]

    newX = X_win[:, :-1, :]
    return newX, newB