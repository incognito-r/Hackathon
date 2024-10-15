import pandas as pd
import numpy as np


def fillna_proportional(df, column):
    vcount = df[column].value_counts()
    proportion = vcount / vcount.sum()
    missing_count = df[column].isna().sum()
    to_fill_count = (proportion * missing_count).round().astype(int)

    while to_fill_count.sum() != missing_count:
        if to_fill_count.sum() < missing_count:
            to_fill_count[to_fill_count.idxmax()] += 1
        elif to_fill_count.sum() > missing_count:
            to_fill_count[to_fill_count.idxmin()] -= 1

    na_indices = df[df[column].isna()].index
    fill_values = np.concatenate([np.repeat(index, count)
                                 for index, count in to_fill_count.items()])
    np.random.shuffle(fill_values)
    df.loc[na_indices, column] = fill_values

    return df
