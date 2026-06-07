import os
import pandas as pd
import numpy as np
import joblib

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import (
    StandardScaler,
    OneHotEncoder
)

from sklearn.ensemble import RandomForestRegressor

from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    r2_score
)

# ==========================================================
# FILES
# ==========================================================

MODEL_FILE = "model.pkl"
PIPELINE_FILE = "pipeline.pkl"

# ==========================================================
# BUILD PIPELINE
# ==========================================================

def build_pipeline(num_cols, cat_cols):

    num_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])

    cat_pipeline = Pipeline([
        ("onehot",
         OneHotEncoder(
             handle_unknown="ignore"
         ))
    ])

    full_pipeline = ColumnTransformer([
        ("num", num_pipeline, num_cols),
        ("cat", cat_pipeline, cat_cols)
    ])

    return full_pipeline

# ==========================================================
# TRAINING
# ==========================================================

if not os.path.exists(MODEL_FILE):

    print("Training Model...")

    # --------------------------------------
    # Load Dataset
    # --------------------------------------

    df = pd.read_csv(
        "Gurugram_housing_data.csv"
    )

    # --------------------------------------
    # Clean Price
    # --------------------------------------

    df["Price"] = (
        df["Price"]
        .astype(str)
        .str.replace(",", "", regex=False)
        .str.replace("₹", "", regex=False)
        .str.strip()
    )

    df["Price"] = pd.to_numeric(
        df["Price"],
        errors="coerce"
    )

    # --------------------------------------
    # Clean Rate per sqft
    # --------------------------------------

    df["Rate per sqft"] = (
        df["Rate per sqft"]
        .astype(str)
        .str.replace(",", "", regex=False)
        .str.replace("₹", "", regex=False)
        .str.strip()
    )

    df["Rate per sqft"] = pd.to_numeric(
        df["Rate per sqft"],
        errors="coerce"
    )

    # --------------------------------------
    # Remove Invalid Rows
    # --------------------------------------

    df = df.dropna(
        subset=["Price"]
    )

    # --------------------------------------
    # Verify
    # --------------------------------------

    print("\nColumn Types")
    print(df[["Price", "Rate per sqft"]].dtypes)

    # --------------------------------------
    # Features & Target
    # --------------------------------------

    X = df.drop(
        "Price",
        axis=1
    )

    y = df["Price"]

    # --------------------------------------
    # Train Test Split
    # --------------------------------------

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42
    )

    # --------------------------------------
    # Save Test Set
    # --------------------------------------

    X_test.to_csv(
        "input.csv",
        index=False
    )

    test_copy = X_test.copy()

    test_copy["Price"] = y_test

    test_copy.to_csv(
        "test_copy.csv",
        index=False
    )

    # --------------------------------------
    # Column Identification
    # --------------------------------------

    num_cols = X_train.select_dtypes(
        include=["int64", "float64"]
    ).columns.tolist()

    cat_cols = X_train.select_dtypes(
        include=["object"]
    ).columns.tolist()

    print("\nNumeric Columns")
    print(num_cols)

    print("\nCategorical Columns")
    print(cat_cols)

    # --------------------------------------
    # Build Pipeline
    # --------------------------------------

    pipeline = build_pipeline(
        num_cols,
        cat_cols
    )

    X_train_prepared = pipeline.fit_transform(
        X_train
    )

    # --------------------------------------
    # Model
    # --------------------------------------

    model = RandomForestRegressor(
        n_estimators=300,
        random_state=42,
        n_jobs=-1
    )

    print("\nTarget dtype:", y_train.dtype)

    model.fit(
        X_train_prepared,
        y_train
    )

    # --------------------------------------
    # Save Model & Pipeline
    # --------------------------------------

    joblib.dump(
        model,
        MODEL_FILE
    )

    joblib.dump(
        pipeline,
        PIPELINE_FILE
    )

    print("Training Complete")

# ==========================================================
# INFERENCE
# ==========================================================

print("\nRunning Inference...")

model = joblib.load(
    MODEL_FILE
)

pipeline = joblib.load(
    PIPELINE_FILE
)

input_data = pd.read_csv(
    "input.csv"
)

# clean same way

input_data["Rate per sqft"] = (
    input_data["Rate per sqft"]
    .astype(str)
    .str.replace(",", "", regex=False)
)

input_data["Rate per sqft"] = pd.to_numeric(
    input_data["Rate per sqft"],
    errors="coerce"
)

input_prepared = pipeline.transform(
    input_data
)

predictions = model.predict(
    input_prepared
)

input_data["Predicted_Price"] = predictions

input_data.to_csv(
    "output.csv",
    index=False
)

print("Inference Complete")

# ==========================================================
# EVALUATION
# ==========================================================

print("\nEvaluating Model...")

actual_data = pd.read_csv(
    "test_copy.csv"
)

actual_data["Price"] = (
    actual_data["Price"]
    .astype(str)
    .str.replace(",", "", regex=False)
    .str.strip()
)

actual_data["Price"] = pd.to_numeric(
    actual_data["Price"],
    errors="coerce"
)

predicted_data = pd.read_csv(
    "output.csv"
)

y_true = actual_data["Price"]

y_pred = predicted_data[
    "Predicted_Price"
]

rmse = np.sqrt(
    mean_squared_error(
        y_true,
        y_pred
    )
)

mae = mean_absolute_error(
    y_true,
    y_pred
)

r2 = r2_score(
    y_true,
    y_pred
)

mape = np.mean(
    np.abs(
        (y_true - y_pred)
        / y_true
    )
) * 100

accuracy = 100 - mape

print("\nModel Performance")
print("-" * 40)

print(f"RMSE      : {rmse:,.2f}")
print(f"MAE       : {mae:,.2f}")
print(f"R² Score  : {r2:.4f}")
print(f"MAPE      : {mape:.2f}%")
print(f"Accuracy  : {accuracy:.2f}%")

comparison = pd.DataFrame({
    "Actual Price": y_true,
    "Predicted Price": y_pred
})

print("\nSample Predictions")
print(comparison.head(10))