import numpy as np
import pandas as pd

COR_EDGE = 0.7
DELTA = 0.3


# sns.heatmap(corr_mat, xticklabels=range(len(headers)), yticklabels=1, linewidths=.5)


def get_high_correlated(so):
    saved_items = []  # highly correlated
    for i in so.items():
        if abs(i[1]) >= COR_EDGE and "КГФ" not in i[0] and "G_total" not in i[0]:
            saved_items.append(i[0])
    print(saved_items)
    return saved_items


def check_cor_utility(highly_correlated, corr_mat):
    attributes = corr_mat.columns.values

    print("\n===\n")

    leave_only_one = []

    for (h1, h2) in highly_correlated:
        need_both = False
        for other in attributes:
            if other == h1 or other == h2:
                continue

            corr_h1_other = corr_mat[other][h1] if np.isnan(corr_mat[h1][other]) else corr_mat[h1][other]
            corr_h2_other = corr_mat[h2][other] if np.isnan(corr_mat[other][h2]) else corr_mat[other][h2]

            if np.isnan(corr_h2_other) or np.isnan(corr_h1_other):
                continue

            if abs(corr_h2_other - corr_h1_other) >= DELTA:
                need_both = True
                break

        if not need_both:
            leave_only_one.append((h1, h2))
            print(f'"{h1}" and "{h2}" twins, kill one with the lower gain ratio')

    print("\n===\n")

    return leave_only_one


def get_leave_only_one(data_frame: pd.DataFrame) -> list[tuple[str, str]]:
    cor = data_frame.corr()
    cor.values[np.tril_indices(len(cor))] = np.nan
    so = cor.unstack().sort_values(kind="quicksort")
    so = so[~np.isnan(so)]
    saved_items = get_high_correlated(so)
    # print(saved_items)
    # print(len(saved_items))
    return check_cor_utility(saved_items, cor)

    # sns.heatmap(cor, xticklabels=data_frame.shape[1], yticklabels=1, linewidths=.5)
