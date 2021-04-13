import numpy as np


def sigmoid(x):
    return 1.0 / (1 + np.exp(-x))


def derivative_sigmoid(x):
    return x * (1 - x)
