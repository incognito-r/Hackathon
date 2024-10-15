import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

# Step 1: Preprocessing Function


def preprocess(df, treat_na=True, treat_outliers=True, train=True,
               lower_quantile=0.01, upper_quantile=0.99):

    def drop_missing_columns(df, keep_threshold=0.3, drop_threshold=0.5):
        missing_percentage = df.isnull().mean()

        columns_to_drop = missing_percentage[missing_percentage >
                                             drop_threshold].index.tolist()
        df = df.drop(columns=columns_to_drop)

        for col in missing_percentage.index:
            if keep_threshold < missing_percentage[col] <= drop_threshold:
                user_input = input(f"Column '{col}' has {missing_percentage[col]:.2%} missing values. "
                                   f"Do you want to keep (K), drop (D), or treat (T) this column? ").strip().lower()
                if user_input == 'd':
                    df = df.drop(columns=[col])
                elif user_input == 't':
                    if df[col].dtype in ['float64', 'int64']:
                        df[col] = df[col].fillna(df[col].median())
                    else:
                        df[col] = df[col].fillna(df[col].mode()[0])

        return df

    # Step 1: Drop columns with excessive missing values
    df = drop_missing_columns(df)

    # Step 2: Identify numeric and categorical columns
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = df.select_dtypes(exclude=[np.number]).columns.tolist()

    # Step 3: Normalize string categorical data
    for col in cat_cols:
        df[col] = df[col].str.lower()  # Convert to lowercase for consistency

    # Step 4: Handle missing values if specified
    if treat_na:
        df[num_cols] = df[num_cols].fillna(df[num_cols].median(
            numeric_only=True))  # Fill numeric with median

        for col in cat_cols:
            if col in df.columns and not df[col].isnull().all():
                # Fill categorical with mode
                df[col] = df[col].fillna(df[col].mode()[0])

    # Step 5: Identify and map binary columns
    binary_columns = []

    for col in num_cols:
        if df[col].nunique() == 2:
            binary_columns.append(col)
            df[col] = df[col].map(lambda x: 1 if x ==
                                  1 else 0)  # Map to 1 and 0

    for col in cat_cols:
        if df[col].nunique() == 2:
            binary_columns.append(col)
            df[col] = df[col].map(lambda x: 1 if x ==
                                  df[col].unique()[1] else 0)

    # Step 6: Outlier treatment if specified
    if treat_outliers:
        def cap_outliers(series, lower_q, upper_q):
            lower_bound = series.quantile(lower_q)
            upper_bound = series.quantile(upper_q)
            # Cap the outliers
            return series.clip(lower=lower_bound, upper=upper_bound)

        for col in num_cols:
            if col not in binary_columns:  # Skip binary columns
                df[col] = cap_outliers(df[col], lower_quantile, upper_quantile)

    # Step 7: Scaling numeric data
    scaler = StandardScaler()  # Initialize the scaler
    df[num_cols] = scaler.fit_transform(df[num_cols])  # Scale numeric columns

    # Return processed DataFrame, scaler, and binary column names
    return df, scaler, binary_columns
