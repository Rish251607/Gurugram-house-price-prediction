# ==========================================================
# HOUSE PRICE PREDICTION - GURUGRAM DATASET
# ==========================================================

import pandas as pd
import numpy as np
import joblib

from sklearn.model_selection import (
    train_test_split,
    cross_val_score
)

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import (
    OneHotEncoder,
    StandardScaler
)

from sklearn.ensemble import RandomForestRegressor

from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    r2_score
)

# ==========================================================
# LOAD DATA
# ==========================================================

df = pd.read_csv("Gurugram_housing_data.csv")

# ==========================================================
# CLEAN PRICE (TARGET COLUMN)
# ==========================================================

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

# ==========================================================
# CLEAN RATE PER SQFT
# ==========================================================

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

# ==========================================================
# REMOVE INVALID ROWS
# ==========================================================

df = df.dropna(subset=["Price"])

print("Price dtype:", df["Price"].dtype)
print("Rate per sqft dtype:", df["Rate per sqft"].dtype)

# ==========================================================
# FEATURES & TARGET
# ==========================================================

X = df.drop("Price", axis=1)
y = df["Price"]

# ==========================================================
# TRAIN TEST SPLIT
# ==========================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)

# ==========================================================
# NUMERIC & CATEGORICAL COLUMNS
# ==========================================================

num_cols = X_train.select_dtypes(
    include=["int64", "float64"]
).columns.tolist()

cat_cols = X_train.select_dtypes(
    include=["object"]
).columns.tolist()

# ==========================================================
# IMPUTATION
# ==========================================================

num_imputer = SimpleImputer(strategy="median")

X_train_num = pd.DataFrame(
    num_imputer.fit_transform(X_train[num_cols]),
    columns=num_cols,
    index=X_train.index
)

X_test_num = pd.DataFrame(
    num_imputer.transform(X_test[num_cols]),
    columns=num_cols,
    index=X_test.index
)

# ==========================================================
# SCALING
# ==========================================================

scaler = StandardScaler()

X_train_num = pd.DataFrame(
    scaler.fit_transform(X_train_num),
    columns=num_cols,
    index=X_train.index
)

X_test_num = pd.DataFrame(
    scaler.transform(X_test_num),
    columns=num_cols,
    index=X_test.index
)

# ==========================================================
# ONE HOT ENCODING
# ==========================================================

encoder = OneHotEncoder(
    handle_unknown="ignore",
    sparse_output=False
)

X_train_cat = pd.DataFrame(
    encoder.fit_transform(X_train[cat_cols]),
    columns=encoder.get_feature_names_out(cat_cols),
    index=X_train.index
)

X_test_cat = pd.DataFrame(
    encoder.transform(X_test[cat_cols]),
    columns=encoder.get_feature_names_out(cat_cols),
    index=X_test.index
)

# ==========================================================
# FINAL DATASET
# ==========================================================

X_train_prepared = pd.concat(
    [X_train_num, X_train_cat],
    axis=1
)

X_test_prepared = pd.concat(
    [X_test_num, X_test_cat],
    axis=1
)

print("Train Shape :", X_train_prepared.shape)
print("Test Shape  :", X_test_prepared.shape)

# ==========================================================
# MODEL
# ==========================================================

rf = RandomForestRegressor(
    n_estimators=300,
    max_depth=None,
    random_state=42,
    n_jobs=-1
)

# ==========================================================
# CROSS VALIDATION
# ==========================================================

cv_scores = cross_val_score(
    rf,
    X_train_prepared,
    y_train,
    cv=5,
    scoring="r2",
    n_jobs=-1
)

print("\nCross Validation R² Scores")
print(cv_scores)

print(
    "\nMean CV R²:",
    round(cv_scores.mean(), 4)
)

# ==========================================================
# TRAIN FINAL MODEL
# ==========================================================

rf.fit(
    X_train_prepared,
    y_train
)

# ==========================================================
# SAVE MODEL
# ==========================================================

joblib.dump(rf, "model.pkl")
joblib.dump(num_imputer, "imputer.pkl")
joblib.dump(scaler, "scaler.pkl")
joblib.dump(encoder, "encoder.pkl")

# ==========================================================
# PREDICTION
# ==========================================================

predictions = rf.predict(
    X_test_prepared
)

# ==========================================================
# EVALUATION
# ==========================================================

rmse = np.sqrt(
    mean_squared_error(
        y_test,
        predictions
    )
)

mae = mean_absolute_error(
    y_test,
    predictions
)

r2 = r2_score(
    y_test,
    predictions
)

mape = np.mean(
    np.abs(
        (y_test - predictions) / y_test
    )
) * 100

accuracy = 100 - mape

# ==========================================================
# RESULTS
# ==========================================================

print("\n==============================")
print("MODEL PERFORMANCE")
print("==============================")

print(f"RMSE      : {rmse:,.2f}")
print(f"MAE       : {mae:,.2f}")
print(f"R² Score  : {r2:.4f}")
print(f"MAPE      : {mape:.2f}%")
print(f"Accuracy  : {accuracy:.2f}%")

# ==========================================================
# ACTUAL VS PREDICTED
# ==========================================================

comparison = pd.DataFrame({
    "Actual Price": y_test.values,
    "Predicted Price": predictions
})

print("\nSample Predictions")
print(comparison.head(10))