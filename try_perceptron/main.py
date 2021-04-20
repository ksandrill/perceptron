import timeit
import numpy as np
import perceptron
import plotly.graph_objs as go
import plotly.subplots as sub

input_layer_size = 1
first_hidden_layer_size = 30
second_hidden_layer_size = 20
output_layer_size = 1


def main():
    nn = perceptron.Perceptron(input_layer_size, first_hidden_layer_size, second_hidden_layer_size, output_layer_size)
    nn.init_weights(-1, 1)
    data = [(np.array([x]), np.array([(np.sin(x) + 1.0) / 2.0])) for x in np.linspace(-np.pi, np.pi, 50, endpoint=True)]
    nn.train(15000, 0.05, data)
    output = nn.get_results(data)
    fig = sub.make_subplots(rows=1, cols=2)
    fig.add_trace(
        go.Scatter(x=list(map(lambda x: x[0][0], data)), y=list(map(lambda y: y[1][0], data))),
        row=1, col=1
    )

    fig.add_trace(
        go.Scatter(x=list(map(lambda x: x[0][0], data)), y=output),
        row=1, col=2
    )

    fig.show()


if __name__ == "__main__":
    time = timeit.timeit(main, number=1)
    print("time: ", time, "sec")
