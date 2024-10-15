from preprocess import preprocess
import pandas as pd


def predictor(input_df, model, scaler):

    # Check if the input DataFrame is empty
    if input_df.empty:
        raise ValueError("Input DataFrame is empty.")

    # Ensure the ID column is present
    if 0 not in input_df.columns:
        raise ValueError(
            "Input DataFrame must contain an ID column at index 0.")

    # Extract the ID column
    IDs = input_df.iloc[:, 0]  # Assuming the first column is ID

    # Preprocess the input data
    input_clean, scaler, _ = preprocess(
        input_df, treat_na=True, treat_outliers=True, train=False)

    # Check if there are any features to predict
    if input_clean.shape[1] < 2:
        raise ValueError(
            "Input DataFrame must have at least one feature column for prediction.")

    # Make predictions using the trained model
    predictions = model.predict(
        input_clean.iloc[:, 1:])  # Exclude the ID column

    # Return a DataFrame with the ID and predicted values
    return pd.DataFrame({
        'ID': IDs,
        'Predicted': predictions
    })
