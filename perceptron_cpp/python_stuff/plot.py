import subprocess

import plotly.graph_objs as go


def read_data_from_file(filepath: str) -> list:
    data_list = []
    with open(filepath, "r") as f:
        for line in f:
            data_list.append(float(line))
    return data_list


def exec_low_level_slave(path: str):
    proc = subprocess.Popen([path])
    proc.wait()


def draw_graph(x_list: list, y_list: list, name: str, x_axis_name: str, y_axis_name: str, color: str):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=x_list, y=y_list, name=name,
                             line=dict(color=color)))
    fig.update_traces(showlegend=True)
    fig.update_layout(legend_orientation="h", xaxis_title=x_axis_name, yaxis_title=y_axis_name)
    fig.show()


def main():
    exec_low_level_slave("../cmake-build-release/perceptron_cpp.exe")
    average_error_list = read_data_from_file("../cmake-build-release/error.txt")
    average_error_list_len = len(average_error_list)
    draw_graph(x_list=[i for i in range(1, average_error_list_len)], y_list=average_error_list, name="avg_mse",
               x_axis_name="epoch", y_axis_name="avg_mse", color="red")


if __name__ == "__main__":
    main()
