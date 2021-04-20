import numpy as np
import sigmoid as sig


class Layer:
    def __init__(self, neurone_count: int, weights_count: int):
        self.layer_output = np.zeros(shape=neurone_count, dtype=np.float32)
        self.layer_weights = np.zeros(shape=[neurone_count, weights_count], dtype=np.float32)

    def init_weights(self, low: float, high: float):
        self.layer_weights = np.random.uniform(low=low, high=high, size=self.layer_weights.shape)

    def __repr__(self):
        return " weights:\n" + repr(self.layer_weights) + "\n" + " output:" + "\n" + repr(self.layer_output)

    def activate_neurones(self, input_layer: np.ndarray):
        layer_size = self.layer_output.shape[0]
        for neurone in range(layer_size):
            neurone_activation = sig.sigmoid(np.dot(input_layer, self.layer_weights[neurone]))
            self.layer_output[neurone] = neurone_activation if not np.isnan(neurone) else 0.0
