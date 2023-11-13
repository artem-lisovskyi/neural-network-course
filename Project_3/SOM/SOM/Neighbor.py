# Neighborhood Functions
from typing import Tuple

import numpy as np


def bubble(c: Tuple[int, int], sigma: int, x_steps: np.ndarray, y_steps: np.ndarray, **ignore):
    ax = np.logical_and(x_steps > c[0] - sigma, x_steps < c[0] + sigma)
    ay = np.logical_and(y_steps > c[1] - sigma, y_steps < c[1] + sigma)
    return np.outer(ax, ay) * 1.


def gaussian(c: Tuple[int, int], sigma: int, xx: np.ndarray, yy: np.ndarray, **ignore):
    d = 2 * sigma * sigma
    ax = np.exp(-np.power(xx - xx.T[c], 2) / d)
    ay = np.exp(-np.power(yy - yy.T[c], 2) / d)
    return (ax * ay).T


def triangle(c: Tuple[int, int], sigma: int, x_steps: np.ndarray, y_steps: np.ndarray, **ignore):
    triangle_x = (-abs(c[0] - x_steps)) + sigma
    triangle_y = (-abs(c[1] - y_steps)) + sigma
    triangle_x[triangle_x < 0] = 0.
    triangle_y[triangle_y < 0] = 0.
    return np.outer(triangle_x, triangle_y)


neighborhood_functions = {
    'bubble': bubble,
    'gaussian': gaussian,
    'triangle': triangle
}
