# =============== XGBoost V2b - Feature Engineering Implementation ===============
# Dropped: education_level, employment_status, income_level, ethnicity
# ----------------------------------------------------------------
# Created features (based on EDA insights and domain knowledge):
# feature 1 = bp_ratio = systolic_bp / diastolic_bp
# feature 2 = pulse_pressure = systolic_bp - diastolic_bp
# feature 3 = map = diastolic_bp + (1/3) * pulse_pressure
# feature 4 = chol_ratio = ldl_cholesterol / hdl_cholesterol
# feature 5 = chol_diff = ldl_cholesterol - hdl_cholesterol
# feature 6 = bmi_squared = bmi ** 2
# feature 7 = bmi_age = bmi * age
# feature 8 = age_squared = age ** 2
# feature 9 = elderly = 1 if age >= 65 else 0
# feature 10 = obese = 1 if bmi >= 30 else 0
# ---------------------------------------------------------------
# Total 30 features (20 original + 10 new) for modeling
# ================================================================

#%% Importing Libraries
import numpy as np
import pandas as pd
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent / "xgb_data"
OUTPUT_DIR = Path(__file__).resolve().parent / "xgb_data"

# ---------- Load data ----------
train = pd.read_csv(f"{DATA_DIR}/xgb_train.csv")
test = pd.read_csv(f"{DATA_DIR}/xgb_test.csv")

print(f"Original train shape: {train.shape}, Test shape: {test.shape}")

train_fe = train.copy()
test_fe = test.copy()

#%% ---------- Create New Features ----------
TARGET = "diagnosed_diabetes"

# Blood Pressure features
train_fe['bp_ratio'] = train_fe['systolic_bp'] / train_fe['diastolic_bp']
test_fe['bp_ratio'] = test_fe['systolic_bp'] / test_fe['diastolic_bp']

train_fe['pulse_pressure'] = train_fe['systolic_bp'] - train_fe['diastolic_bp']
test_fe['pulse_pressure'] = test_fe['systolic_bp'] - test_fe['diastolic_bp']

train_fe['map'] = train_fe['diastolic_bp'] + (1/3) * train_fe['pulse_pressure']
test_fe['map'] = test_fe['diastolic_bp'] + (1/3) * test_fe['pulse_pressure']

# Cholesterol features
train_fe['chol_ratio'] = train_fe['ldl_cholesterol'] / train_fe['hdl_cholesterol']
test_fe['chol_ratio'] = test_fe['ldl_cholesterol'] / test_fe['hdl_cholesterol']

train_fe['chol_diff'] = train_fe['ldl_cholesterol'] - train_fe['hdl_cholesterol']
test_fe['chol_diff'] = test_fe['ldl_cholesterol'] - test_fe['hdl_cholesterol']

# BMI features
train_fe['bmi_squared'] = train_fe['bmi'] ** 2
test_fe['bmi_squared'] = test_fe['bmi'] ** 2

train_fe['bmi_age'] = train_fe['bmi'] * train_fe['age']
test_fe['bmi_age'] = test_fe['bmi'] * train_fe['age']

# Age features
train_fe['age_squared'] = train_fe['age'] ** 2
test_fe['age_squared'] = test_fe['age'] ** 2

train_fe['elderly'] = (train_fe['age'] >= 65).astype(int)
test_fe['elderly'] = (test_fe['age'] >= 65).astype(int)

# Obesity flag
train_fe['obese'] = (train_fe['bmi'] >= 30).astype(int)
test_fe['obese'] = (test_fe['bmi'] >= 30).astype(int)

print(f"Created 10 new features")

#%% ---------- Drop Weak Features ----------
features_to_drop = [
    'education_level',
    'employment_status',
    'income_level',
    'ethnicity'
]

train_fe = train_fe.drop(columns=features_to_drop, errors='ignore')
test_fe = test_fe.drop(columns=features_to_drop, errors='ignore')

print(f"Dropped: {features_to_drop}")

#%% ---------- Save ----------
train_fe.to_csv(OUTPUT_DIR / "train_ft.csv", index=False)
test_fe.to_csv(OUTPUT_DIR / "test_ft.csv", index=False)

print(f"Saved: train_ft.csv ({train_fe.shape})")
print(f"Saved: test_ft.csv ({test_fe.shape})")