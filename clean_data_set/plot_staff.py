import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


def plot_histogram(column: pd.Series, title: str):
    plt.title(title)
    column.hist(bins=1 + int(np.log2(column.shape[0])), density=True, grid=True)
    column.plot.kde()
    quant = np.nanquantile(column, q=[0.25, 0.75])
    low = quant[0] - 1.5 * (quant[1] - quant[0])
    high = quant[1] + 1.5 * (quant[1] - quant[0])
    plt.axvline(low, color='red')
    plt.axvline(high, color='red')
    plt.show()


def plot_all_histograms(data_frame: pd.DataFrame):
    for i in data_frame.columns:
        plot_histogram(data_frame[i], i)
