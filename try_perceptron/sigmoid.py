import numpy as np


def sigmoid(x):
    return 1.0 / (1 + np.exp(-x))


def derivative_sigmoid(sig_val):
    return sig_val * (1 - sig_val)
