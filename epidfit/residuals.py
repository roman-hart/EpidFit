import numpy as np


def mse(y_observed: np.array, y_predicted: np.array):
    return np.mean((y_observed - y_predicted) ** 2)


def mae(y_observed: np.array, y_predicted: np.array):
    return np.mean(np.abs(y_observed - y_predicted))


def medae(y_observed: np.array, y_predicted: np.array):
    return np.median(np.abs(y_observed - y_predicted))
