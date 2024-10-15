from xgboost import XGBClassifier
import joblib
from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np
import os


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


def preprocess(df, scaler=None, treat_na=True, treat_outliers=True,
               lower_quantile=0.01, upper_quantile=0.99, train=True):

    def drop_missing_columns(df, drop_threshold=0.4):
        missing_percentage = df.isnull().mean()
        columns_to_drop = missing_percentage[missing_percentage >
                                             drop_threshold].index.tolist()
        df = df.drop(columns=columns_to_drop)
        return df

    # Rename columns for df_train_X and df_test_X to integers
    df.columns = range(df.shape[1])

    # Step 1: Drop columns with excessive missing values
    df = drop_missing_columns(df)

    # Step 2: Identify numeric and categorical columns
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = df.select_dtypes(exclude=[np.number]).columns.tolist()

    # Step 3: Normalize string categorical data
    # Convert to lowercase for consistency
    df[cat_cols] = df[cat_cols].apply(lambda x: x.str.lower())

    # Step 4: Handle missing values if specified using proportional fill
    if treat_na:
        for col in num_cols + cat_cols:  # Apply to both numeric and categorical columns
            df = fillna_proportional(df, col)

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
    if scaler is None and train:
        scaler = StandardScaler()
        # Fit and transform during training
        df[num_cols] = scaler.fit_transform(df[num_cols])
    elif scaler is not None:
        # Only transform during prediction
        df[num_cols] = scaler.transform(df[num_cols])

    return (df, scaler) if train else df


def train_model(df_train_X, df_train_Y):

    def save_model(model, model_path='./model/model.pkl'):
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        joblib.dump(model, model_path)
        print(f"Model saved to {model_path}")

    def save_scaler(scaler, scaler_path='./model/scaler.pkl'):
        joblib.dump(scaler, scaler_path)
        print(f"Scaler saved to {scaler_path}")

    # Preprocess training data and get the scaler
    df_train_clean, scaler = preprocess(
        df_train_X, treat_na=True, treat_outliers=True, lower_quantile=0.05, upper_quantile=0.95
    )

    # Extract feature matrix and target labels
    X_train = df_train_clean.iloc[:, 1:]  # Exclude ID column
    y_train = df_train_Y['target']

    # Initialize and train logistic regression model
    # model = LogisticRegression()
    model = XGBClassifier(n_estimators=100, learning_rate=0.1,
                          max_depth=3, random_state=42)

    try:
        model.fit(X_train, y_train)
        print("Model trained successfully!")
        save_model(model)
        save_scaler(scaler)

    except Exception as e:
        print(f"An error occurred during model training: {e}")
        raise

    return model, scaler  # Return both the model and the scaler


def predictor(df_test_X, model, scaler):

    if df_test_X.empty:
        raise ValueError("Input DataFrame is empty.")

    df_test_X.columns = range(df_test_X.shape[1])

    IDs = df_test_X.iloc[:, 0]  # Assuming the first column is ID

    input_clean = preprocess(df_test_X, treat_na=True,
                             treat_outliers=True, train=False)

    if input_clean.shape[1] < 2:
        raise ValueError(
            "Input DataFrame must have at least one feature column for prediction.")

    num_cols = input_clean.select_dtypes(include=[np.number]).columns.tolist()
    input_clean[num_cols] = scaler.transform(input_clean[num_cols])

    predictions = model.predict(
        input_clean.iloc[:, 1:])  # Exclude the ID column

    return pd.DataFrame({'ID': IDs, 'Predicted': predictions})
