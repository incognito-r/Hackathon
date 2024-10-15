from preprocess import preprocess
import os
import pandas as pd

# Step 2: Model Training Function

from sklearn.linear_model import LogisticRegression
from sklearn.exceptions import NotFittedError
import joblib


def train_model(df_train_X, df_train_Y):

    def save_model(model, scaler, model_path='./model/model.pkl', scaler_path='./model/scaler.pkl'):
        model_dir = os.path.dirname(model_path)
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)
            print(f"Directory '{model_dir}' created.")

        joblib.dump(model, model_path)
        joblib.dump(scaler, scaler_path)
        print(f"Model saved to {model_path} \nScaler saved to {scaler_path}")

    # Preprocess training data
    df_train_clean, scaler, _ = preprocess(
        df_train_X, treat_na=True, treat_outliers=True, lower_quantile=0.05, upper_quantile=0.95
    )

    # Extract feature matrix and target labels
    X_train = df_train_clean.iloc[:, 1:]  # Exclude ID column
    # Ensure that 'target' column exists in df_train_Y
    y_train = df_train_Y['target']

    # Initialize and train logistic regression model
    model = LogisticRegression()

    try:
        model.fit(X_train, y_train)
        print("Model trained successfully!")

        # Save the model and scaler
        save_model(model, scaler)

    except Exception as e:
        print(f"An error occurred during model training: {e}")
        raise

    return model, scaler
