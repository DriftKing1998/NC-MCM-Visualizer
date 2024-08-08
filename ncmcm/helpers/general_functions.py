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

