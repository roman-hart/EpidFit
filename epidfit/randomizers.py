import numpy as np


def randomize_all_normal(parameter_name: str, value: float, sd: float = 0.1):
    return np.random.normal(value, value * sd)


def randomize_all_uniform(parameter_name: str, value: float, sd: float = 0.1):
    return np.random.uniform(value * (1 - sd), value * (1 + sd))


def randomize_all_exponential(parameter_name: str, value: float, sd: float = 0.1):
    scale = value * sd
    return value + np.random.exponential(scale=scale) - scale


def randomize_beta_normal(parameter_name: str, value: float, sd: float = 0.1):
    if parameter_name == 'beta':
        return np.random.normal(value, value * sd)
    return value
