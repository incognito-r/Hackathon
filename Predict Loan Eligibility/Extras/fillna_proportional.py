import pandas as pd
import numpy as np

def fillna_proportional(df, column):
    """
    Fills missing values in the specified column of the DataFrame 
    using proportional distribution based on existing value counts.

    Parameters:
    df (pd.DataFrame): The DataFrame containing the data.
    column (str): The name of the column with missing values to be filled.

    Returns:
    pd.DataFrame: The DataFrame with missing values filled.
    """
    # Step 1: Get value counts and proportions (excluding NaN)
    vcount = df[column].value_counts()
    proportion = vcount / vcount.sum()

    # Step 2: Calculate the number of missing values
    missing_count = df[column].isna().sum()

    # Step 3: Calculate how many missing values should go into each category (based on proportions)
    to_fill_count = (proportion * missing_count).round().astype(int)

    # Step 4: Adjust for rounding discrepancies
    while to_fill_count.sum() != missing_count:
        if to_fill_count.sum() < missing_count:
            to_fill_count[to_fill_count.idxmax()] += 1
        elif to_fill_count.sum() > missing_count:
            to_fill_count[to_fill_count.idxmin()] -= 1

    # Step 5: Get the indices of missing values (NaN)
    na_indices = df[df[column].isna()].index

    # Step 6: Create a list of values to fill the NaNs
    fill_values = np.concatenate([np.repeat(index, count) for index, count in to_fill_count.items()])

    # Step 7: Shuffle the fill values to randomize the assignment
    np.random.shuffle(fill_values)

    # Step 8: Assign the shuffled fill values to the missing values in the DataFrame
    df.loc[na_indices, column] = fill_values

    return df

# Example usage:
# df = fillna_proportional(df, 'Dependents')
