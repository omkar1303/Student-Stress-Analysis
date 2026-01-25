
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler, StandardScaler
from sklearn.metrics import accuracy_score, classification_report, f1_score
from imblearn.over_sampling import SMOTE
import re

# 1. Load Data
print("Loading data...")
try:
    df = pd.read_csv('google_form_950_responses.csv')
    print(f"Loaded {len(df)} original samples.")
except FileNotFoundError:
    print("Error: 'google_form_950_responses.csv' not found.")
    exit()

# Load Synthetic Data
try:
    df_syn = pd.read_csv('synthetic_student_data.csv')
    print(f"Loaded {len(df_syn)} synthetic samples.")
    
    # Concatenate
    df = pd.concat([df, df_syn], axis=0, ignore_index=True)
    print(f"Total samples after merging: {len(df)}")
except FileNotFoundError:
    print("Warning: 'synthetic_student_data.csv' not found. Proceeding with original data only.")

# 2. Data Cleaning & Preprocessing
print("Preprocessing data...")
# Renaming columns
df.rename(columns={
    "Certification Course": "certification",
    "Gender": "gender",
    "Department": "dep",
    "Height(CM)": "height",
    "Weight(KG)": "weight",
    "10th Mark": "mark10th",
    "12th Mark": "mark12th",
    "college mark": "collegemark",
    "daily studing time": "studytime",
    "prefer to study in": "prefertime",
    "salary expectation": "salexpect",
    "Do you like your degree?": "likedegree",
    "willingness to pursue a career based on their degree": "carrer_willing",
    "social medai & video": "smtime",
    "Travelling Time": "travel",
    "Stress Level": "stress",
    "Financial Status": "financial",
    "part-time job": "parttime"
}, inplace=True)

# Ordinal Encoding for Target
ordinal_mapping = {
    "Low": 0,
    "Medium": 1,
    "High": 2
}
if 'stress' in df.columns:
    df["Stress"] = df["stress"].map(ordinal_mapping)
    df = df.drop(["stress"], axis=1)

df.dropna(inplace=True)

# Feature Separation
X = df.drop("Stress", axis=1)
y = df["Stress"]

columns_to_encode = ['certification', 'gender', 'dep', 'hobbies', 'prefertime', 'likedegree', 'financial', 'parttime']
columns_to_minmax = ['height', 'weight', 'mark10th', 'mark12th', 'collegemark', 'studytime']
columns_to_std = ['salexpect']

# Ensure numeric columns
for col in columns_to_minmax + columns_to_std:
    if col in X.columns:
        X[col] = pd.to_numeric(X[col], errors='coerce')
X.fillna(0, inplace=True)

# Feature Engineering / Encoding
print("Encoding features...")
X = pd.get_dummies(X, columns=columns_to_encode, drop_first=True)
# Handle any remaining object columns
X = pd.get_dummies(X, drop_first=True)

# Scaling
print("Scaling features...")
scaler_minmax = MinMaxScaler()
minmax_cols = [c for c in columns_to_minmax if c in X.columns]
if minmax_cols:
    X[minmax_cols] = scaler_minmax.fit_transform(X[minmax_cols])

scaler_std = StandardScaler()
std_cols = [c for c in columns_to_std if c in X.columns]
if std_cols:
    X[std_cols] = scaler_std.fit_transform(X[std_cols])

# Sanitize Column Names for XGBoost
print("Sanitizing column names...")
regex = re.compile(r"\[|\]|<", re.IGNORECASE)
X.columns = [regex.sub("_", str(col)) for col in X.columns]

# 3. Handling Class Imbalance (SMOTE)
print("Handling class imbalance with SMOTE...")
print("Original class distribution:", y.value_counts().to_dict())

smote = SMOTE(random_state=42)
X_res, y_res = smote.fit_resample(X, y)

print("Resampled class distribution:", y_res.value_counts().to_dict())

# 4. Train/Test Split
X_train, X_test, y_train, y_test = train_test_split(X_res, y_res, test_size=0.2, random_state=42)

# 5. Model Training (XGBoost)
print("Training XGBoost Classifier...")
xgb_clf = xgb.XGBClassifier(
    objective='multi:softmax', 
    num_class=3, 
    random_state=42,
    eval_metric='mlogloss'
)

xgb_clf.fit(X_train, y_train)
y_pred_xgb = xgb_clf.predict(X_test)

print("\n--- XGBoost (Default) ---")
print("Accuracy:", accuracy_score(y_test, y_pred_xgb))
print(classification_report(y_test, y_pred_xgb))

# 6. Hyperparameter Tuning (GridSearch)
print("Performing Grid Search for XGBoost...")
param_grid = {
    'n_estimators': [100, 200],
    'max_depth': [3, 5, 7],
    'learning_rate': [0.05, 0.1, 0.2],
    'subsample': [0.8, 1.0]
}

grid_search = GridSearchCV(
    estimator=xgb.XGBClassifier(objective='multi:softmax', num_class=3, random_state=42, eval_metric='mlogloss'),
    param_grid=param_grid,
    scoring='accuracy',
    cv=3,
    n_jobs=-1,
    verbose=1
)

grid_search.fit(X_train, y_train)

print("\n--- Best XGBoost Parameters ---")
print(grid_search.best_params_)

best_model = grid_search.best_estimator_
y_pred_best = best_model.predict(X_test)

print("\n--- XGBoost (Tuned) ---")
print("Accuracy:", accuracy_score(y_test, y_pred_best))
print(classification_report(y_test, y_pred_best))

# Save Model
best_model.save_model("student_stress_model.json")
print("\nModel saved to 'student_stress_model.json'")
