import numpy as np
import sigmoid_staff as ss

ETA = 0.5


def _feed_forward_layout_step(layer: np.ndarray, input_to_layer: np.ndarray, layer_weights: np.ndarray):
    layer_len = len(layer)
    for neurone in range(layer_len):
        # neurone activation
        layer[neurone] = ss.sigmoid(np.dot(input_to_layer, layer_weights[neurone]))


def _calc_hidden_layer_delta(layer: np.array, next_layer: np.array,
                             next_layer_weights: np.array,
                             next_layer_delta: np.array) -> np.array:
    layer_len = len(layer)
    next_layer_len = len(next_layer)
    hidden_layer_delta = np.empty(shape=[layer_len])
    for k in range(layer_len):
        sum_ = 0.0
        for j in range(next_layer_len):
            sum_ += next_layer_weights[j][k] * next_layer_delta[j]
        hidden_layer_delta[k] = ss.derivative_sigmoid(layer[k]) * sum_
    return hidden_layer_delta


def _calc_hidden_layer_weight_correction(layer: np.array, layer_delta_array: np.array,
                                         prev_layer: np.array) -> np.array:
    layer_len = layer.shape[0]
    prev_layer_len = prev_layer.shape[0]
    hidden_layer_weight = np.empty(shape=[layer_len, prev_layer_len])
    for i in range(layer_len):
        for j in range(prev_layer_len):
            hidden_layer_weight[i][j] = -ETA * prev_layer[j] * layer_delta_array[i]
    return hidden_layer_weight


class Perceptron:
    def __init__(self, input_n: int, first_l_n: int, second_l_n: int, output_n: int):
        self.input_layer = np.empty(shape=input_n, dtype=np.float32)
        self.first_layer_weights = np.random.uniform(low=-0.1, high=0.1, size=(first_l_n, input_n))
        self.first_layer = np.empty(shape=first_l_n, dtype=np.float32)
        self.second_layer_weights = np.random.uniform(low=-0.1, high=0.1, size=(second_l_n, first_l_n))
        self.second_layer = np.empty(shape=second_l_n)
        self.output_layer_weights = np.random.uniform(low=-0.1, high=0.1, size=(output_n, second_l_n))
        self.output_layer = np.empty(shape=output_n, dtype=np.float32)

    def print_weights(self):
        print("first layer weights:")
        print(self.first_layer_weights)
        print("second layer weights:")
        print(self.second_layer_weights)
        print("output layer weights:")
        print(self.output_layer_weights)

    def print_layers(self):
        print("input layer:")
        print(self.input_layer)
        print("first layer:")
        print(self.first_layer)
        print("second layer:")
        print(self.second_layer)
        print("output layer:")
        print(self.output_layer)

    def back_prop(self, target: np.array):
        # print("back_prop")
        len_out = len(self.output_layer)
        error_array = np.array([self.output_layer[i] - target[i] for i in range(len_out)])
        output_layer_delta_array = np.array(
            [error_array[i] * ss.derivative_sigmoid(self.output_layer[i]) for i in range(len_out)])
        output_layer_weight_correction = self._calc_output_layer_weight_correction(output_layer_delta_array, len_out,
                                                                                   self.second_layer)
        second_layer_delta_array = _calc_hidden_layer_delta(self.second_layer, self.output_layer,
                                                            self.output_layer_weights,
                                                            output_layer_delta_array)
        second_layer_weight_correction = _calc_hidden_layer_weight_correction(self.second_layer,
                                                                              second_layer_delta_array,
                                                                              self.first_layer)
        first_layer_delta_array = _calc_hidden_layer_delta(self.first_layer, self.second_layer,
                                                           self.second_layer_weights, second_layer_delta_array)
        first_layer_weight_correction = _calc_hidden_layer_weight_correction(self.first_layer, first_layer_delta_array,
                                                                             self.input_layer)

        self.first_layer_weights += first_layer_weight_correction
        self.second_layer_weights += second_layer_weight_correction
        self.output_layer_weights += output_layer_weight_correction

    def _calc_output_layer_weight_correction(self, output_layer_delta_array: np.array, len_out: int,
                                             prev_layer: np.array) -> np.array:
        output_weight_correction = np.empty(shape=self.output_layer_weights.shape, dtype=np.float32)
        prev_layer_len = prev_layer.shape[0]
        for neurone in range(len_out):
            for i in range(prev_layer_len):
                output_weight_correction[neurone][i] = -ETA * output_layer_delta_array[neurone] * prev_layer[
                    i]
        return output_weight_correction

    def feed_forward(self):
        # feed forward for first layer
        _feed_forward_layout_step(self.first_layer, self.input_layer, self.first_layer_weights)
        # feed forward for second layer
        _feed_forward_layout_step(self.second_layer, self.first_layer, self.second_layer_weights)
        # feed forward for second layer
        _feed_forward_layout_step(self.output_layer, self.second_layer, self.output_layer_weights)
        # print("first layer: ", self.first_layer)
        # print("second layer: ", self.second_layer)
        # print("last layer: ", self.output_layer)
