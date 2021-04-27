from typing import List

import pandas as pd
from main import find_interval
import numpy as np


def count_freq(data_frame: pd.DataFrame, classes, kgf_intervals):
    freq_dict = {cur_cl: 0 for cur_cl in classes}
    for index, row in data_frame.iterrows():
        value = (find_interval(data_frame["КГФ"].loc[index], kgf_intervals), data_frame["G_total"].loc[index])
        for i in freq_dict.keys():
            if i[0] != i[0] and value[0] != value[0] or value[0] == i[0]:
                if i[1] != i[1] and value[1] != value[1] or value[1] == i[1]:
                    freq_dict[i] += 1
                    break
    # for i in freq_dict:
    #     print(i, freq_dict[i])
    return freq_dict


def get_avg_info(data_frame: pd.DataFrame, classes,
                 kgf_intervals) -> float:
    freq_dict = count_freq(data_frame, classes, kgf_intervals)
    freq_val_list = list(freq_dict.values())
    info = 0.0
    T = data_frame.shape[0]
    for freq_Ti in freq_val_list:
        if freq_Ti > 0:
            aux = freq_Ti / T
            info += aux * np.log2(aux)
    return -info


def get_feature_and_split_info(data_frame: pd.DataFrame, feature_name: str, classes,
                               kgf_intervals) -> (float, float):
    uniq_list = data_frame[feature_name].unique()
    power = data_frame.shape[0]
    feature_info = 0.0
    split_info = 0.0
    for i in uniq_list:
        new_data_frame = data_frame[np.isnan(data_frame[feature_name])] if np.isnan(i) else data_frame[
            data_frame[feature_name] == i]
        avg_info = get_avg_info(new_data_frame, classes, kgf_intervals)
        power_i = new_data_frame.shape[0]
        power_relation = power_i / power
        feature_info += power_relation * avg_info
        split_info += power_relation * np.log2(power_relation)
    #print(feature_name + " ", feature_info, split_info)
    return feature_info, -split_info


def gain_ratio(avg_info: float, feature_info: float, split_info: float) -> float:
    return (avg_info - feature_info) / split_info


def get_features_gain_ratio(data_frame: pd.DataFrame, classes, kgf_intervals) -> dict:
    features = data_frame.columns[0:-2]
    frame_avg_info = get_avg_info(data_frame, classes, kgf_intervals)
    #print(frame_avg_info, " avg info")
    gain_ratio_dict = {}
    for feature in features:
        feature_info, split_info = get_feature_and_split_info(data_frame, feature, classes, kgf_intervals)
        gain_ratio_dict[feature] = gain_ratio(frame_avg_info, feature_info, split_info)
    return gain_ratio_dict
