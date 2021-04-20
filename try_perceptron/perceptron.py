from layer import Layer
import numpy as np


class Perceptron:
    def __init__(self, input_size: int, first_layer_size: int, second_layer_size: int, output_layer_size: int):
        self.input_layer = np.zeros(shape=input_size, dtype=np.float32)
        self.first_hidden_layer = Layer(first_layer_size, input_size)
        self.second_hidden_layer = Layer(second_layer_size, first_layer_size)
        self.output_layer = Layer(output_layer_size, second_layer_size)

    def __repr__(self):
        return "input_layer:\n" + repr(self.input_layer) + "\nfirst_hidden_layer:\n" + repr(
            self.first_hidden_layer) + "\n second_hidden_layer:\n" + repr(
            self.second_hidden_layer) + "\n output_layer:\n" + repr(self.output_layer)

    def init_weights(self, low: float, high: float):
        self.first_hidden_layer.init_weights(low, high)
        self.second_hidden_layer.init_weights(low, high)
        self.output_layer.init_weights(low, high)

    def train(self, epoch: int, lr: float, data: list):
        iterations = len(data)
        for i in range(epoch):
            for j in range(iterations):
                self.input_layer = data[j][0]
                self.feed_forward()
                self.back_prop(data[j][1], lr)

    def get_results(self, data: list) -> list:
        iterations = len(data)
        answer_list = []
        for i in range(iterations):
            print("input: ", data[i][0], end=" ")
            self.input_layer = data[i][0]
            self.feed_forward()
            print("output: ", self.output_layer.layer_output, end=" ")
            answer_list.append(self.output_layer.layer_output[0])
            print("real_output: ", data[i][1])
        return answer_list

    def feed_forward(self):
        self.first_hidden_layer.activate_neurones(self.input_layer)
        self.second_hidden_layer.activate_neurones(self.first_hidden_layer.layer_output)
        self.output_layer.activate_neurones(self.second_hidden_layer.layer_output)

    def back_prop(self, target: np.array, lr: float):
        error = self.output_layer.layer_output - target
        output_layer_delta = self._calc_output_layer_delta(error)
        output_layer_weights_corr = self._calc_layer_weight_corr(output_layer_delta,
                                                                 self.second_hidden_layer.layer_output)
        second_layer_delta = self._calc_hidden_layer_delta(self.second_hidden_layer.layer_output, output_layer_delta,
                                                           self.output_layer.layer_weights)
        second_layer_weights_corr = self._calc_layer_weight_corr(second_layer_delta,
                                                                 self.first_hidden_layer.layer_output)
        first_layer_delta = self._calc_hidden_layer_delta(self.first_hidden_layer.layer_output, second_layer_delta,
                                                          self.second_hidden_layer.layer_weights)
        first_layer_weights_corr = self._calc_layer_weight_corr(first_layer_delta, self.input_layer)
        self.output_layer.layer_weights -= lr * output_layer_weights_corr
        self.second_hidden_layer.layer_weights -= lr * second_layer_weights_corr
        self.first_hidden_layer.layer_weights -= lr * first_layer_weights_corr

    def _calc_output_layer_delta(self, error: np.array) -> np.array:
        return np.array(error * self.output_layer.layer_output * (1.0 - self.output_layer.layer_output))

    def _calc_layer_weight_corr(self, layer_delta: np.array, layer_input: np.array) -> np.array:
        layer_len = layer_delta.shape[0]
        layer_weight_len = layer_input.shape[0]
        layer_weight_corr = np.zeros(shape=[layer_len, layer_weight_len], dtype=np.float32)
        for neurone in range(layer_len):
            for weight in range(layer_weight_len):
                layer_weight_corr[neurone][weight] = layer_delta[neurone] * layer_input[weight]
        return layer_weight_corr

    def _calc_hidden_layer_delta(self, next_layer_input: np.array, next_layer_delta: np.array,
                                 next_layer_weights: np.array):
        cur_delta = next_layer_delta @ next_layer_weights
        return next_layer_input * (1.0 - next_layer_input) * cur_delta
