import pandas as pd
import os
from sklearn.preprocessing import LabelEncoder

LABEL_COLUMN = "Churn"

def load_data(file_path="dataset.csv"):
    print("📂 Loading dataset from:", os.path.abspath(file_path))

    data = pd.read_csv(file_path)

    print("📊 Columns found:")
    print(list(data.columns))

    if LABEL_COLUMN not in data.columns:
        raise ValueError("❌ 'Churn' column not found in dataset")

    # Drop customer ID if present
    if "customerID" in data.columns:
        data = data.drop("customerID", axis=1)

    # Handle missing values
    data = data.dropna()

    X = data.drop(LABEL_COLUMN, axis=1)
    y = data[LABEL_COLUMN].map({"Yes": 1, "No": 0})

    # Encode categorical features
    for column in X.columns:
        if X[column].dtype == "object":
            encoder = LabelEncoder()
            X[column] = encoder.fit_transform(X[column])

    print("✅ Dataset loaded & processed")
    print("🔹 Total samples:", data.shape[0])
    print("🔹 Features:", X.shape[1])
    print("🔹 Churn distribution:")
    print(y.value_counts())

    return X, y
