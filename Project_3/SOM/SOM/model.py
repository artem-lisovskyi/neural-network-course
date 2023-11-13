import time
from functools import partial
from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

from SOM.Distance import distance_functions
from SOM.Neighbor import neighborhood_functions


class SOM:
    def __init__(
            self,
            size: Tuple[int, int],
            feature: int,
            learning_rate: int or float,
            max_iterations: int,
            shuffle: bool = False,
            neighbor_function: str = "bubble",
            distance_function: str = "euclidean",
            mutable_update=None,
            first_show=0
    ):
        self.shuffle = shuffle
        self.size = size
        self.feature = feature
        self.learning_rate = learning_rate
        self.max_iterations = max_iterations
        self.errors = []

        if mutable_update:
            self.mutable_update = mutable_update
        else:
            self.mutable_update = lambda origin, iteration: origin / (1 + iteration / (max_iterations / 2))
        self.weights = np.random.randn(*size, feature) * 2 - 1
        self.activation_map = np.zeros(size)
        self.x_steps, self.y_steps = np.arange(size[0]), np.arange(size[1])
        self.xx, self.yy = np.meshgrid(self.x_steps, self.y_steps)
        self.distance = distance_functions[distance_function]

        self.first_show = first_show

        self.neighborhood = partial(
            neighborhood_functions[neighbor_function],
            x_steps=self.x_steps,
            y_steps=self.y_steps,
            xx=self.xx,
            yy=self.yy
        )

        self.title = f"iter:{max_iterations}/size:{size}/lr:{learning_rate}/distance:{distance_function}/neighbor:{neighbor_function}"
        print(self.title)

    def fit(self, data: np.ndarray, num):
        batch, feature = data.shape
        assert feature == self.feature, "The data dimensions passed in during training do not match the settings."
        for _ in range(self.max_iterations):
            if self.shuffle:
                np.random.shuffle(data)
            for i, x in enumerate(tqdm(data, desc=f"Epoch {_}")):
                winner = self.get_winner(x)
                eta = self.mutable_update(self.learning_rate, i)
                g = self.neighborhood(winner, 4) * eta
                self.weights += np.einsum('ij, ijk->ijk', g, x - self.weights)

            winning_neurons = np.array([self.get_winner(x) for x in data])
            topographic_error = np.sum(np.array([np.linalg.norm(winning_neurons[i] - winning_neurons[j])
                                                 for i, j in enumerate(num)]) > 1) / len(num)
            time.sleep(0.01)
            print(f"Epoch {_} Mean Squared Error: ", self.map_error(data))
            print(f"Topographic error: {topographic_error}")
            time.sleep(0.01)

    def get_winner(self, x):
        self.activation_map = self.distance(x, self.weights)
        winner = np.unravel_index(self.activation_map.argmin(), self.size)
        return winner

    def map_error(self, data: np.ndarray):
        distance = self._distance_from_weights(data)
        coords = np.argmin(distance, axis=1)
        weights = self.weights[np.unravel_index(coords, self.size)]
        error = np.linalg.norm(data - weights, axis=1).mean()
        self.errors.append(error)
        return error

    def _distance_from_weights(self, data):
        weights_flat = self.weights.reshape(-1, self.weights.shape[2])
        input_data_sq = np.power(data, 2).sum(axis=1, keepdims=True)
        weights_flat_sq = np.power(weights_flat, 2).sum(axis=1, keepdims=True)
        cross_term = np.dot(data, weights_flat.T)
        return np.sqrt(-2 * cross_term + input_data_sq + weights_flat_sq.T)
