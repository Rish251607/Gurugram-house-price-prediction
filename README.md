# Gurugram House Price Prediction using Machine Learning

##  Project Overview

This project predicts residential property prices in Gurugram using Machine Learning techniques. The objective is to build an accurate house price prediction model by performing data cleaning, preprocessing, model comparison, training, and evaluation.

The project demonstrates both:

- Traditional Machine Learning Workflow (Without Pipeline)
- Production-Ready Workflow (Using Scikit-Learn Pipeline)

The final model uses **Random Forest Regressor**, selected after comparing multiple machine learning algorithms.

---

##  Features

- Data Cleaning and Preprocessing
- Missing Value Handling
- Feature Scaling
- One-Hot Encoding
- Model Comparison
- Random Forest Regression
- Scikit-Learn Pipeline
- Model Persistence using Joblib
- Performance Evaluation
- Prediction on New Data

---

## Project Structure

```
gurugram-house-price-prediction/
│
├── Gurugram_housing_data.csv
├── Main_without_pipeline.py
├── Main_with_pipeline.py
├── model_comparision.py
├── screenshot.png
├── README.md
└── requirements.txt
```

---

##  Dataset

Dataset Used:

```
Gurugram_housing_data.csv
```

The dataset contains housing-related information such as:

- Property Location
- Property Type
- Area
- Rate per sqft
- Other Housing Features
- Price (Target Variable)

---

## Data Preprocessing

### Price Cleaning

The following cleaning operations were performed:

- Removed ₹ symbol
- Removed commas
- Converted values to numeric format
- Removed invalid records

### Rate per sqft Cleaning

- Removed ₹ symbol
- Removed commas
- Converted values to numeric format

### Missing Value Handling

Numerical columns:

- Median Imputation

Categorical columns:

- One-Hot Encoding

### Feature Scaling

- StandardScaler

---

##  Model Comparison

Three machine learning models were evaluated:

| Model | Purpose |
|---------|---------|
| Linear Regression | Baseline Model |
| Decision Tree Regressor | Tree-Based Model |
| Random Forest Regressor | Ensemble Model |

Evaluation Metrics:

- RMSE
- MAE
- R² Score

Random Forest Regressor achieved the best performance and was selected as the final model.

---

##  Final Model

```python
RandomForestRegressor(
    n_estimators=300,
    random_state=42,
    n_jobs=-1
)
```

---

##  Pipeline Implementation

The project includes a production-ready implementation using:

- Pipeline
- ColumnTransformer

Benefits:

- Cleaner Code
- Reusable Preprocessing
- Easier Deployment
- Reduced Data Leakage
- Better Maintainability

---

##  Files Description

### 1. Main_without_pipeline.py

Traditional Machine Learning implementation.

Includes:

- Manual Imputation
- Manual Scaling
- Manual Encoding
- Model Training
- Model Evaluation

---

### 2. Main_with_pipeline.py

Production-ready implementation using:

- Pipeline
- ColumnTransformer
- Automated Preprocessing
- Model Saving
- Inference Workflow

---

### 3. model_comparision.py

Compares:

- Linear Regression
- Decision Tree Regressor
- Random Forest Regressor

Selects the best model based on evaluation metrics.

---

##  Model Performance

| Metric | Value |
|----------|----------|
| RMSE | 4,207,513.13 |
| MAE | 438,538.17 |
| R² Score | 0.9947 |
| MAPE | 1.39% |
| Accuracy | 98.61% |

---

##  Output Screenshot

Add your project screenshot here:

```markdown
![Project Output](screenshot.png)
```

---

##  Technologies Used

- Python
- Pandas
- NumPy
- Scikit-Learn
- Joblib

---

##  How to Run

### Clone Repository

```bash
git clone https://github.com/YOUR_USERNAME/gurugram-house-price-prediction.git
```

### Move into Project Folder

```bash
cd gurugram-house-price-prediction
```

### Install Dependencies

```bash
pip install pandas numpy scikit-learn joblib
```

### Run Model Comparison

```bash
python model_comparision.py
```

### Run Without Pipeline Version

```bash
python Main_without_pipeline.py
```

### Run With Pipeline Version

```bash
python Main_with_pipeline.py
```

---

## Future Improvements

- Hyperparameter Tuning
- XGBoost
- LightGBM
- Streamlit Dashboard
- Web Deployment
- Feature Selection
- Advanced Feature Engineering

---

## Author

**Rishabh Sharma**

Associate Analyst | Data Analyst | Machine Learning Enthusiast

LinkedIn: https://www.linkedin.com/in/rishabh-sharma-551624258/

---
