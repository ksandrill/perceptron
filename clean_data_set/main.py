import matplotlib.pyplot as plt

import data_frame_clean as dfc
import plot_staff
from black_magic import *
from corr_mat_util import get_leave_only_one
import numpy as np


def get_kgf_intervals(data_frame: pd.DataFrame) -> List[tuple[float, float]]:
    kgf_column = data_frame["КГФ"]
    sorted_kgf = sorted(kgf_column.unique())
    bins = int(1 + np.log2(len(sorted_kgf)))
    min_col = sorted_kgf[0]
    max_col = sorted_kgf[len(sorted_kgf) - 1]
    width = (max_col - min_col) / bins
    return [(min_col + width * i, min_col + width * (i + 1)) for i in range(bins)]


def find_interval(value: float, kgf_intervals: List[tuple[float, float]]):
    for i in kgf_intervals:
        if i[0] <= value < i[1]:
            return i
    return np.NAN


def get_class_set(data_frame: pd.DataFrame, kgf_intervals: List[tuple[float, float]]):
    classes: List[tuple[float, float], float] = []
    for index, row in data_frame.iterrows():
        value = (find_interval(data_frame["КГФ"].loc[index], kgf_intervals), data_frame["G_total"].loc[index])
        daflag = True
        for i in classes:
            if i[0] == value[0] or np.all(np.isnan(i[0])) and np.all(np.isnan(value[0])):
                if i[1] == value[1] or np.isnan(i[1]) and np.isnan(value[1]):
                    daflag = False
                    break
        if daflag:
            classes.append(value)

    return classes


def main():
    # data_frame = dfc.get_data_frame()
    # dfc.save_csv(data_frame, 'data_set/clean.csv')
    data_frame = pd.read_csv('data_set/clean.csv')
    pd.set_option('display.max_rows', None)
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', None)
    # print(data_frame.columns)
    #plot_staff.plot_all_histograms(data_frame)
    data_frame = dfc.delete_all_outliers(data_frame)
    dfc.delete_small_uniques(data_frame)
    dfc.delete_big_nan(data_frame)
    kgf_intervals = get_kgf_intervals(data_frame)
    class_list = (get_class_set(data_frame, kgf_intervals))
    gain_ratio_dict = get_features_gain_ratio(data_frame, class_list, kgf_intervals)
    gain_ratio_dict = dict(sorted(gain_ratio_dict.items(), key=lambda item: item[1]))
    kill_one = get_leave_only_one(data_frame)
    print(gain_ratio_dict)
    for dic in kill_one:
        if dic[0] not in gain_ratio_dict or dic[1] not in gain_ratio_dict:
            continue
        left_gain_ratio = gain_ratio_dict[dic[0]]
        right_gain_ratio = gain_ratio_dict[dic[1]]
        if left_gain_ratio > right_gain_ratio:
            print(dic[1], " was killed")
            data_frame.drop((dic[1]), axis=1, inplace=True)
            gain_ratio_dict.pop(dic[1])

        else:
            print(dic[0], " was killed")
            data_frame.drop((dic[0]), axis=1, inplace=True)
            gain_ratio_dict.pop(dic[0])


    for i in gain_ratio_dict:
        print("feature: ", i, " gain_ratio: ", gain_ratio_dict[i])
    #data_frame.to_excel("neural_magic.xlsx")
    plt.barh(list(gain_ratio_dict.keys()), width=list(gain_ratio_dict.values()), color="red")
    plt.show()
    print(class_list)


if __name__ == "__main__":
    main()
