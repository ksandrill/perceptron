import timeit

import numpy as np
from perceptron import Perceptron

EPOCH = 5000


def main():
    decepticon = Perceptron(2, 3, 3, 1)
    data = (
        (np.array([0.1, 0.1]), np.array([0.0])),
        (np.array([0.2, 0.4]), np.array([0.0])),
        (np.array([0.4, 0.5]), np.array([0.0])),
        (np.array([0.3, 0.9]), np.array([0.0])),
        (np.array([0.4, 0.7]), np.array([0.0])),

        (np.array([0.9, 0.1]), np.array([1.0])),
        (np.array([0.8, 0.4]), np.array([1.0])),
        (np.array([0.75, 0.5]), np.array([1.0])),
        (np.array([0.63, 0.9]), np.array([1.0])),
        (np.array([0.54, 0.7]), np.array([1.0])),
        (np.array([0.59, 0.7]), np.array([1.0])),
    )

    for i in range(EPOCH):
        for j in range(len(data)):
            decepticon.input_layer = data[j][0]
            decepticon.feed_forward()
            decepticon.back_prop(data[j][1])
    for i in range(len(data)):
        decepticon.input_layer = data[i][0]
        decepticon.feed_forward()
        print("in: ", decepticon.input_layer, " out: ", decepticon.output_layer)


if __name__ == '__main__':
    time = timeit.timeit(main, number=1)
    print("time: ", time, "sec")
