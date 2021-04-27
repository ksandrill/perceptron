from typing import List
import pandas as pd
import numpy as np


def get_data_frame() -> pd.DataFrame:
    data_frame = pd.read_excel("data_set/ID_data_mass_18122012.xlsx", sheet_name=1)
    clean_data_frame(data_frame)

    return data_frame


def save_csv(data_frame: pd.DataFrame, path: str):
    data_frame.to_csv(path, index=False)


def clean_data_frame(data_frame: pd.DataFrame):
    data_frame.iloc[0, 0] = data_frame.iloc[1, 0]
    data_frame.iloc[0, 1] = "Дата"
    data_frame.columns = data_frame.iloc[0]
    data_frame.rename(columns={'Pлин': 'Рлин'}, inplace=True)
    data_frame.rename(columns={'Дебит гааз': 'Дебит газа'}, inplace=True)
    data_frame.columns = get_fixed_repeats(list(data_frame.columns))
    data_frame.drop([0], inplace=True)
    data_frame.drop([1], inplace=True)
    data_frame["КГФ_1"] = data_frame["КГФ_1"].multiply(1000)
    data_frame["КГФ"] = data_frame["КГФ"].combine_first(data_frame["КГФ_1"])
    data_frame.drop(columns="КГФ_1", inplace=True)
    data_frame.drop(columns="№", inplace=True)
    data_frame.drop(columns="Дата", inplace=True)
    data_frame.dropna(subset=["КГФ", "G_total"], how='all', inplace=True)
    data_frame.replace(to_replace=["не спускался", '-'], value=np.NAN, inplace=True)


def get_fixed_repeats(columns: List[str]) -> List[str]:
    repeated = {}
    fixed_column_names = []
    for column in columns:
        if column in fixed_column_names:
            if column not in repeated:
                repeated[column] = 1
            else:
                repeated[column] += 1
            column = column + "_" + str(repeated[column])
        fixed_column_names.append(column)
    return fixed_column_names


def delete_small_uniques(data_frame):
    del_list = []
    print("delete small uniques:")
    for i in data_frame.columns:
        if data_frame[i].nunique(dropna=True) == 1:
            del_list.append(i)
            print(" delete :", i)
    for i in del_list:
        data_frame.drop(i, axis=1, inplace=True)


def delete_outliers(data_frame: pd.DataFrame, col_name: str):
    column = data_frame[col_name]
    quant = np.nanquantile(column, q=[0.25, 0.75])
    low = quant[0] - 1.5 * (quant[1] - quant[0])
    high = quant[1] + 1.5 * (quant[1] - quant[0])
    for index, row in data_frame.iterrows():
        aux = data_frame[col_name].loc[index]
        if not (low < aux < high) and np.isnan(data_frame["G_total"].loc[index]):
            data_frame.drop(index, inplace=True)
    return data_frame


def delete_all_outliers(data_frame: pd.DataFrame):
    for i in ['Рлин', 'Рлин_1', 'Туст', 'Тна шлейфе', 'Тзаб', 'Дебит ст. конд.', 'Дебит кон нестабильный',
              'Рпл. Тек (Карноухов)', 'Удельная плотность газа ']:
        data_frame = delete_outliers(data_frame, i)
    return data_frame


def delete_big_nan(data_frame: pd.DataFrame):
    nan_count_series = data_frame.isnull().sum(axis=0)
    column_size = data_frame.shape[0]
    print("Nan percentage  big delete:")
    for i, v in nan_count_series.iteritems():
        print(" ", i, end=" ")
        print(v / column_size * 100)
        if v / column_size * 100 >= 50 and i != "G_total":
            data_frame.drop(i, axis=1, inplace=True)
            print(" delete " + i)
