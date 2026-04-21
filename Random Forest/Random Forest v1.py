# ══════════════════════════════════════════════════════════════════
# Random Forest Classifier
# File: random_forest.py
# ══════════════════════════════════════════════════════════════════

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix, roc_auc_score)

# ══════════════════════════════════════════════════════════════════
# PATHS
# ══════════════════════════════════════════════════════════════════
TRAIN_PATH      = r"C:\Kaggle Diabetes ML Competition\Dataset\train.csv"
TEST_PATH       = r"C:\Kaggle Diabetes ML Competition\Dataset\test.csv"
SUBMISSION_PATH = r"C:\Kaggle Diabetes ML Competition\Dataset\result 2.csv"

# ══════════════════════════════════════════════════════════════════
# ENCODING MAPS
# (Reused from logistic regression — do not change these)
# ══════════════════════════════════════════════════════════════════
GENDER_MAP = {
    'Male'  : 0,
    'Female': 1
}
SMOKING_MAP = {
    'Never'  : 0,
    'Former' : 1,
    'Current': 2
}
EMPLOYMENT_MAP = {
    'Unemployed'   : 0,
    'Employed'     : 1,
    'Self-Employed': 2
}
EDUCATION_MAP = {
    'Highschool'  : 0,
    'Some College': 1,
    'Bachelors'   : 2,
    'Masters'     : 3,
    'PhD'         : 4
}
INCOME_MAP = {
    'Low'   : 0,
    'Middle': 1,
    'High'  : 2
}

# ══════════════════════════════════════════════════════════════════
# PREPROCESSING FUNCTION
# (Reused and improved from logistic regression)
# ══════════════════════════════════════════════════════════════════
def preprocess(df):
    df = df.copy()

    # ── Drop columns not needed ───────────────────────────────────
    df.drop(columns=['id', 'diagnosed_diabetes'], errors='ignore', inplace=True)

    # ── Fix column name mismatch found in test.csv ────────────────
    # train uses 'total_cholesterol', test uses 'cholesterol_total'
    df.rename(columns={'cholesterol_total': 'total_cholesterol'}, inplace=True)

    # ── Feature Engineering ───────────────────────────────────────
    df['age_bmi_interaction']        = df['age'] * df['bmi']
    df['cholesterol_ratio']          = df['total_cholesterol'] / df['hdl_cholesterol']
    df['ldl_hdl_ratio']              = df['ldl_cholesterol'] / df['hdl_cholesterol']
    df['bp_interaction']             = df['systolic_bp'] * df['diastolic_bp']
    df['bp_pulse_pressure']          = df['systolic_bp'] - df['diastolic_bp']
    df['activity_sleep_score']       = df['physical_activity_minutes_per_week'] * df['sleep_hours_per_day']
    df['screen_sleep_ratio']         = df['screen_time_hours_per_day'] / (df['sleep_hours_per_day'] + 1e-9)
    df['bmi_waist_interaction']      = df['bmi'] * df['waist_to_hip_ratio']
    df['triglycerides_hdl_ratio']    = df['triglycerides'] / df['hdl_cholesterol']

    # ── Ordinal Encoding ──────────────────────────────────────────
    df['gender']            = df['gender'].map(GENDER_MAP)
    df['smoking_status']    = df['smoking_status'].map(SMOKING_MAP)
    df['employment_status'] = df['employment_status'].map(EMPLOYMENT_MAP)
    df['education_level']   = df['education_level'].map(EDUCATION_MAP)
    df['income_level']      = df['income_level'].map(INCOME_MAP)

    # ── One-Hot Encoding ──────────────────────────────────────────
    df = pd.get_dummies(df, columns=['ethnicity'])

    return df

# ══════════════════════════════════════════════════════════════════
# STEP 1: LOAD DATA
# ══════════════════════════════════════════════════════════════════
print("=" * 60)
print("STEP 1: Loading data...")
train_df = pd.read_csv(TRAIN_PATH)
test_df  = pd.read_csv(TEST_PATH)

print(f"✅ Train shape : {train_df.shape}")
print(f"✅ Test shape  : {test_df.shape}")

test_ids = test_df['id'].copy()
y        = train_df['diagnosed_diabetes'].copy()

# ══════════════════════════════════════════════════════════════════
# STEP 2: PREPROCESS
# ══════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("STEP 2: Preprocessing...")

X              = preprocess(train_df)
test_processed = preprocess(test_df)

# Align test columns to exactly match training columns
for col in X.columns:
    if col not in test_processed.columns:
        test_processed[col] = 0
test_processed = test_processed[X.columns]

print(f"✅ Train features shape : {X.shape}")
print(f"✅ Test features shape  : {test_processed.shape}")
print(f"✅ Target distribution  :\n{y.value_counts()}")

# ══════════════════════════════════════════════════════════════════
# STEP 3: IMPUTE MISSING VALUES
# ══════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("STEP 3: Imputing missing values...")

imputer      = SimpleImputer(strategy='median')
X_imputed    = pd.DataFrame(imputer.fit_transform(X), columns=X.columns)
test_imputed = pd.DataFrame(imputer.transform(test_processed), columns=X.columns)

print("✅ Imputing done")

# ══════════════════════════════════════════════════════════════════
# STEP 4: SCALE FEATURES
# ══════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("STEP 4: Scaling features...")

scaler      = StandardScaler()
X_scaled    = scaler.fit_transform(X_imputed)
test_scaled = scaler.transform(test_imputed)

print("✅ Scaling done")

# ══════════════════════════════════════════════════════════════════
# STEP 5: TRAIN / VALIDATION SPLIT
# ══════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("STEP 5: Splitting train/validation...")

X_train, X_val, y_train, y_val = train_test_split(
    X_scaled, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print(f"✅ Train size      : {X_train.shape[0]}")
print(f"✅ Validation size : {X_val.shape[0]}")

# ══════════════════════════════════════════════════════════════════
# STEP 6: TRAIN RANDOM FOREST
# ══════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("STEP 6: Training Random Forest... (this may take a minute)")

model = RandomForestClassifier(
    n_estimators=200,
    max_depth=15,
    min_samples_split=5,
    min_samples_leaf=2,
    class_weight='balanced',
    random_state=42,
    n_jobs=-1           # uses all CPU cores to speed up
)

model.fit(X_train, y_train)
print("✅ Training done")

# ══════════════════════════════════════════════════════════════════
# STEP 7: EVALUATE ON VALIDATION SET
# ══════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("STEP 7: Evaluating...")

y_pred       = model.predict(X_val)
y_pred_proba = model.predict_proba(X_val)[:, 1]

print(f"\n  Accuracy : {accuracy_score(y_val, y_pred):.4f}")
print(f"  ROC-AUC  : {roc_auc_score(y_val, y_pred_proba):.4f}")
print(f"\nClassification Report:\n{classification_report(y_val, y_pred)}")
print(f"Confusion Matrix:\n{confusion_matrix(y_val, y_pred)}")

# ── Top 15 Feature Importances ────────────────────────────────────
importance_df = pd.DataFrame({
    'feature'   : X_imputed.columns,
    'importance': model.feature_importances_
}).sort_values('importance', ascending=False)

print(f"\nTop 15 Important Features:")
print(importance_df.head(15).to_string(index=False))

# ══════════════════════════════════════════════════════════════════
# STEP 8: PREDICT ON TEST SET & SAVE SUBMISSION
# ══════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("STEP 8: Predicting on test set & saving submission...")

test_preds = model.predict(test_scaled)

submission = pd.DataFrame({
    'id'                : test_ids.values,
    'diagnosed_diabetes': test_preds
})

submission.to_csv(SUBMISSION_PATH, index=False)

print(f"✅ Submission saved to: {SUBMISSION_PATH}")
print(submission.head(10))
print(f"\nTotal predictions  : {submission.shape[0]}")
print(f"Class distribution :\n{submission['diagnosed_diabetes'].value_counts()}")
print("=" * 60)