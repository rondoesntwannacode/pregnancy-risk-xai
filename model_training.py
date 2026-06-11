import pandas as pd
import numpy as np
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score

from xgboost import XGBClassifier

# =====================================================
# LOAD DATASET
# =====================================================

df = pd.read_csv("dataa.csv")

# =====================================================
# CLEAN COLUMN NAMES
# =====================================================

df.columns = df.columns.str.strip().str.replace(" ", "")

print("Dataset Columns:")
print(df.columns)

# =====================================================
# TARGET COLUMN
# =====================================================

target_col = "RiskLevel"

# =====================================================
# CHECK UNIQUE TARGET VALUES
# =====================================================

print("\nOriginal Target Values:")
print(df[target_col].unique())

# =====================================================
# CLEAN TARGET LABELS
# =====================================================

df[target_col] = (
    df[target_col]
    .astype(str)
    .str.strip()
    .str.lower()
)

# =====================================================
# LABEL MAPPING
# =====================================================

mapping = {
    'low': 0,
    'high': 1,
    '0': 0,
    '1': 1
}
# IMPORTANT FIX
df[target_col] = df[target_col].map(mapping)

# =====================================================
# REMOVE INVALID ROWS
# =====================================================

print("\nMissing Target Values:")
print(df[target_col].isnull().sum())

df = df.dropna(subset=[target_col])

# =====================================================
# CONVERT TARGET TO INTEGER
# =====================================================

df[target_col] = df[target_col].astype(int)

# =====================================================
# FEATURES AND LABELS
# =====================================================

X = df.drop(columns=[target_col])
y = df[target_col]

# =====================================================
# HANDLE MISSING VALUES
# =====================================================

X = X.fillna(X.mean())

# =====================================================
# NORMALIZATION
# =====================================================

scaler = MinMaxScaler()

X_scaled = scaler.fit_transform(X)

# =====================================================
# TRAIN TEST SPLIT
# =====================================================

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled,
    y,
    test_size=0.3,
    stratify=y,
    random_state=42
)

# =====================================================
# XGBOOST MODEL
# =====================================================

model = XGBClassifier(
    eval_metric='mlogloss',
    random_state=42
)

# =====================================================
# TRAIN MODEL
# =====================================================

model.fit(X_train, y_train)

# =====================================================
# PREDICTION
# =====================================================

predictions = model.predict(X_test)

# =====================================================
# ACCURACY
# =====================================================

accuracy = accuracy_score(y_test, predictions)

print(f"\nModel Accuracy: {accuracy:.4f}")

# =====================================================
# SAVE FILES
# =====================================================

joblib.dump(model, "xgb_model.pkl")
joblib.dump(scaler, "scaler.pkl")
joblib.dump(X.columns.tolist(), "feature_names.pkl")

print("\nModel and scaler saved successfully.")