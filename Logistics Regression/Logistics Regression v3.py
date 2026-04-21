# ══════════════════════════════════════════════════════════════════
# Logistic Regression — V3: Multicollinearity Reduction + L1
# File: V3 logistics regression.py
#
# Key changes vs V2:
#   - Removed: ldl_hdl_ratio, non_hdl_cholesterol, atherogenic_index,
#               log_triglycerides, log_alcohol, overweight, bmi_category,
#               bmi_waist_interaction, middle_aged, senior, age_risk_band,
#               bp_interaction
#   - Raw columns dropped AFTER ratios are derived:
#               triglycerides, cholesterol_total, hdl_cholesterol, ldl_cholesterol
#   - Model: penalty='l1', C=0.1 (SAGA supports L1 natively)
#   - Removed n_jobs=-1 (deprecated in sklearn 1.8)
# ══════════════════════════════════════════════════════════════════

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (classification_report, confusion_matrix,
                              roc_auc_score, RocCurveDisplay)

# ══════════════════════════════════════════════════════════════════
# STEP 1: LOAD DATA
# ══════════════════════════════════════════════════════════════════
df = pd.read_csv(r"C:\Kaggle Diabetes ML Competition\Dataset\train.csv")
print(df.shape)
print(df.dtypes)
print(df.head())

# ══════════════════════════════════════════════════════════════════
# STEP 2: BASIC CLEANING
# ══════════════════════════════════════════════════════════════════
df.drop(columns=['id'], inplace=True)

df[df.select_dtypes('float64').columns] = df.select_dtypes('float64').apply(pd.to_numeric, downcast='float')
df[df.select_dtypes('int64').columns]   = df.select_dtypes('int64').apply(pd.to_numeric, downcast='integer')
print(f"Memory usage: {df.memory_usage(deep=True).sum() / 1e6:.1f} MB")

print(df.isnull().sum())
print(f"Duplicates: {df.duplicated().sum()}")
df.drop_duplicates(inplace=True)

# ══════════════════════════════════════════════════════════════════
# STEP 3: OUTLIER CAPPING
# Raw columns are capped BEFORE derived features are computed from them
# ══════════════════════════════════════════════════════════════════
continuous_cols = [
    'age', 'alcohol_consumption_per_week', 'physical_activity_minutes_per_week',
    'diet_score', 'sleep_hours_per_day', 'screen_time_hours_per_day',
    'bmi', 'waist_to_hip_ratio', 'systolic_bp', 'diastolic_bp',
    'heart_rate', 'cholesterol_total', 'hdl_cholesterol',
    'ldl_cholesterol', 'triglycerides'
]

def cap_outliers(df, col):
    Q1    = df[col].quantile(0.25)
    Q3    = df[col].quantile(0.75)
    IQR   = Q3 - Q1
    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR
    df[col] = df[col].clip(lower=lower, upper=upper)
    return df

for col in continuous_cols:
    df = cap_outliers(df, col)

# ══════════════════════════════════════════════════════════════════
# STEP 4: ENCODING
# NOTE: Must happen before FE so gender is 0/1 for low_hdl / waist_risk
# ══════════════════════════════════════════════════════════════════
df['gender'] = df['gender'].map({'Female': 0, 'Male': 1})

edu_map = {'Highschool': 1, 'Some College': 2, 'Bachelor': 3, 'Graduate': 4}
df['education_level'] = df['education_level'].map(edu_map)

income_map = {
    'Lower': 1, 'Lower-Middle': 2, 'Middle': 3,
    'Upper-Middle': 4, 'Upper': 5
}
df['income_level'] = df['income_level'].map(income_map)

smoking_map = {'Never': 0, 'Former': 1, 'Current': 2}
df['smoking_status'] = df['smoking_status'].map(smoking_map)

df['employment_status'] = df['employment_status'].map(
    lambda x: 1 if x == 'Employed' else 0
)

df = pd.get_dummies(df, columns=['ethnicity'], drop_first=True)

print(df.dtypes)
print(df.head())

# ══════════════════════════════════════════════════════════════════
# STEP 5: FEATURE ENGINEERING (V3 — Multicollinearity Reduced)
# ══════════════════════════════════════════════════════════════════

# ── A: Lipid Ratios ───────────────────────────────────────────────
# KEPT:    cholesterol_ratio       — primary lipid health summary
# KEPT:    triglycerides_hdl_ratio — best single insulin resistance proxy
# DROPPED: ldl_hdl_ratio           — collinear with cholesterol_ratio
# DROPPED: non_hdl_cholesterol     — linear combo of total and HDL
# DROPPED: atherogenic_index       — log(TG/HDL), collinear with TG/HDL ratio
df['cholesterol_ratio']       = df['cholesterol_total'] / df['hdl_cholesterol']
df['triglycerides_hdl_ratio'] = df['triglycerides']     / df['hdl_cholesterol']

# ── B: Blood Pressure ─────────────────────────────────────────────
# KEPT:    pulse_pressure, mean_arterial_pressure, high_bp_flag
# DROPPED: bp_interaction — systolic * diastolic is collinear with both raw values
df['pulse_pressure']         = df['systolic_bp'] - df['diastolic_bp']
df['mean_arterial_pressure'] = df['diastolic_bp'] + (df['pulse_pressure'] / 3)
df['high_bp_flag']           = (
    (df['systolic_bp'] >= 130) | (df['diastolic_bp'] >= 80)
).astype(int)

# ── C: Metabolic Risk Flags ───────────────────────────────────────
# KEPT:    obese, high_triglycerides, sleep_deprived, high_alcohol,
#          low_hdl, waist_risk
# DROPPED: overweight — BMI >= 25 is collinear with obese (BMI >= 30);
#          obese already captures the clinically meaningful threshold
df['obese']              = (df['bmi'] >= 30).astype(int)
df['high_triglycerides'] = (df['triglycerides'] > 150).astype(int)
df['sleep_deprived']     = (df['sleep_hours_per_day'] < 6).astype(int)
df['high_alcohol']       = (df['alcohol_consumption_per_week'] > 14).astype(int)
df['low_hdl'] = (
    ((df['gender'] == 1) & (df['hdl_cholesterol'] < 40)) |
    ((df['gender'] == 0) & (df['hdl_cholesterol'] < 50))
).astype(int)
df['waist_risk'] = (
    ((df['gender'] == 1) & (df['waist_to_hip_ratio'] > 0.90)) |
    ((df['gender'] == 0) & (df['waist_to_hip_ratio'] > 0.85))
).astype(int)

# ── D: Metabolic Syndrome Score ───────────────────────────────────
# Composite count of simultaneous risk factors (>= 3 = Metabolic Syndrome)
df['metabolic_risk_score'] = (
    df['obese']                    +
    df['high_triglycerides']       +
    df['low_hdl']                  +
    df['high_bp_flag']             +
    df['hypertension_history']     +
    df['family_history_diabetes']  +
    df['sleep_deprived']           +
    df['waist_risk']
)

# ── E: BMI Features ───────────────────────────────────────────────
# KEPT:    bmi (raw, already in dataset), obese (computed in C)
# DROPPED: bmi_category        — ordinal discretization of a continuous variable
# DROPPED: bmi_waist_interaction — collinear; waist_risk already captures this
# DROPPED: overweight            — collinear with obese (handled in C above)
# No new features computed in this section

# ── F: Age Features ───────────────────────────────────────────────
# KEPT:    age (raw), age_squared (captures non-linear risk acceleration)
# DROPPED: middle_aged, senior, age_risk_band
#          — three discretizations of the same continuous variable
df['age_squared'] = df['age'] ** 2

# ── G: Interaction Features ───────────────────────────────────────
# All kept — each combines two independent signals into a genuinely new one
df['age_bmi_interaction']   = df['age']   * df['bmi']
df['hypertension_x_family'] = df['hypertension_history'] * df['family_history_diabetes']
df['age_family_risk']       = df['age']   * df['family_history_diabetes']
df['obese_family_risk']     = df['obese'] * df['family_history_diabetes']
df['trig_hdl_bmi']          = df['triglycerides_hdl_ratio'] * df['bmi']

# ── H: Lifestyle Features ─────────────────────────────────────────
df['sedentary_index'] = df['screen_time_hours_per_day'] / (
    df['physical_activity_minutes_per_week'] / 60 + 1
)
df['activity_diet_score'] = df['physical_activity_minutes_per_week'] * df['diet_score']
df['lifestyle_score'] = (
      df['diet_score']
    + (df['physical_activity_minutes_per_week'] / 300)
    - (df['sedentary_index'] * 0.5)
    - (df['sleep_deprived']  * 2)
    - (df['high_alcohol']    * 2)
)

# ── I: Log Transforms ─────────────────────────────────────────────
# DROPPED: log_triglycerides — collinear with triglycerides_hdl_ratio
# DROPPED: log_alcohol       — collinear with alcohol_consumption_per_week
# Section intentionally empty

# ── J: Drop Raw Source Columns ────────────────────────────────────
# These raw columns have been fully replaced by clinically superior ratios.
# Keeping them alongside their derived ratios was the primary cause
# of sign-flipping in V2 coefficients.
# NOTE: high_triglycerides and low_hdl (binary flags) are retained above
#       because they encode clinical threshold information that a ratio alone misses.
RAW_TO_DROP = ['triglycerides', 'cholesterol_total', 'hdl_cholesterol', 'ldl_cholesterol']
df.drop(columns=RAW_TO_DROP, inplace=True)

print(f"\nTotal features after engineering: {df.shape[1]}")

# ══════════════════════════════════════════════════════════════════
# STEP 6: SPLIT X AND y
# ══════════════════════════════════════════════════════════════════
X = df.drop(columns=['diagnosed_diabetes'])
y = df['diagnosed_diabetes']

print(f"Features: {X.shape[1]}")
print(f"Class balance:\n{y.value_counts(normalize=True)}")

# ══════════════════════════════════════════════════════════════════
# STEP 7: TRAIN / TEST SPLIT
# ══════════════════════════════════════════════════════════════════
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.1,
    random_state=42,
    stratify=y
)

# ══════════════════════════════════════════════════════════════════
# STEP 8: IMPUTE MISSING VALUES
# ══════════════════════════════════════════════════════════════════
imputer = SimpleImputer(strategy='median')

X_train = pd.DataFrame(imputer.fit_transform(X_train), columns=X_train.columns)
X_test  = pd.DataFrame(imputer.transform(X_test),      columns=X_test.columns)

print(f"NaNs in X_train: {X_train.isnull().sum().sum()}")
print(f"NaNs in X_test:  {X_test.isnull().sum().sum()}")

# ══════════════════════════════════════════════════════════════════
# STEP 9: SCALE FEATURES
# ══════════════════════════════════════════════════════════════════
scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled  = scaler.transform(X_test)

# ══════════════════════════════════════════════════════════════════
# STEP 10: TRAIN MODEL
#
# penalty='l1'  — Lasso regularization: drives redundant coefficients
#                 to exactly zero, acting as automatic feature selection
#                 on any collinearity that survived manual pruning
# C=0.1         — Moderately strong regularization; increase toward 1.0
#                 if training metrics suggest underfitting
# solver='saga' — Only solver that supports L1 on large datasets
# n_jobs removed — Deprecated in sklearn 1.8, no longer has any effect
# ══════════════════════════════════════════════════════════════════
model = LogisticRegression(
    penalty='l1',
    solver='saga',
    C=0.3,
    max_iter=1000,
    class_weight='balanced',
    random_state=42
)
model.fit(X_train_scaled, y_train)

# ══════════════════════════════════════════════════════════════════
# STEP 11: EVALUATE
# ══════════════════════════════════════════════════════════════════
y_pred      = model.predict(X_test_scaled)
y_pred_prob = model.predict_proba(X_test_scaled)[:, 1]

print(classification_report(y_test, y_pred, target_names=['No Diabetes', 'Diabetes']))
print(f"AUC-ROC: {roc_auc_score(y_test, y_pred_prob):.4f}")

cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Predicted 0', 'Predicted 1'],
            yticklabels=['Actual 0', 'Actual 1'])
plt.title('Confusion Matrix')
plt.tight_layout()
plt.show()

RocCurveDisplay.from_predictions(y_test, y_pred_prob)
plt.title('ROC Curve — V3 (L1 Regularised)')
plt.show()

# Coefficient table — signs should now align with clinical logic
coef_df = pd.DataFrame({
    'Feature'    : X.columns,
    'Coefficient': model.coef_[0]
}).sort_values('Coefficient', key=abs, ascending=False)

print(coef_df.head(15))

# L1 diagnostic: features driven to zero had no independent signal
zeroed = coef_df[coef_df['Coefficient'] == 0.0]
print(f"\nFeatures zeroed by L1 ({len(zeroed)} total): {zeroed['Feature'].tolist()}")

coef_df.head(15).plot(kind='barh', x='Feature', y='Coefficient', legend=False)
plt.title('Top 15 Feature Coefficients (V3 — L1 Regularised)')
plt.axvline(0, color='black', linewidth=0.8)
plt.tight_layout()
plt.show()

# ══════════════════════════════════════════════════════════════════
# STEP 12: PREDICT ON TEST.CSV & SAVE SUBMISSION
# ══════════════════════════════════════════════════════════════════
print("\nLoading test.csv...")
test_df  = pd.read_csv(r"C:\Kaggle Diabetes ML Competition\Dataset\test.csv")
test_ids = test_df['id'].copy()
test_df.drop(columns=['id'], inplace=True)

# ── Encoding (identical maps and order as training) ───────────────
test_df['gender']            = test_df['gender'].map({'Female': 0, 'Male': 1})
test_df['education_level']   = test_df['education_level'].map(edu_map)
test_df['income_level']      = test_df['income_level'].map(income_map)
test_df['smoking_status']    = test_df['smoking_status'].map(smoking_map)
test_df['employment_status'] = test_df['employment_status'].map(
    lambda x: 1 if x == 'Employed' else 0
)
test_df = pd.get_dummies(test_df, columns=['ethnicity'], drop_first=True)

# ── Feature Engineering (identical to training) ───────────────────

# A: Lipid Ratios
test_df['cholesterol_ratio']       = test_df['cholesterol_total'] / test_df['hdl_cholesterol']
test_df['triglycerides_hdl_ratio'] = test_df['triglycerides']     / test_df['hdl_cholesterol']

# B: Blood Pressure
test_df['pulse_pressure']         = test_df['systolic_bp'] - test_df['diastolic_bp']
test_df['mean_arterial_pressure'] = test_df['diastolic_bp'] + (test_df['pulse_pressure'] / 3)
test_df['high_bp_flag']           = (
    (test_df['systolic_bp'] >= 130) | (test_df['diastolic_bp'] >= 80)
).astype(int)

# C: Metabolic Risk Flags
test_df['obese']              = (test_df['bmi'] >= 30).astype(int)
test_df['high_triglycerides'] = (test_df['triglycerides'] > 150).astype(int)
test_df['sleep_deprived']     = (test_df['sleep_hours_per_day'] < 6).astype(int)
test_df['high_alcohol']       = (test_df['alcohol_consumption_per_week'] > 14).astype(int)
test_df['low_hdl'] = (
    ((test_df['gender'] == 1) & (test_df['hdl_cholesterol'] < 40)) |
    ((test_df['gender'] == 0) & (test_df['hdl_cholesterol'] < 50))
).astype(int)
test_df['waist_risk'] = (
    ((test_df['gender'] == 1) & (test_df['waist_to_hip_ratio'] > 0.90)) |
    ((test_df['gender'] == 0) & (test_df['waist_to_hip_ratio'] > 0.85))
).astype(int)

# D: Metabolic Syndrome Score
test_df['metabolic_risk_score'] = (
    test_df['obese']                   +
    test_df['high_triglycerides']      +
    test_df['low_hdl']                 +
    test_df['high_bp_flag']            +
    test_df['hypertension_history']    +
    test_df['family_history_diabetes'] +
    test_df['sleep_deprived']          +
    test_df['waist_risk']
)

# E: No new features (bmi and obese already captured above)

# F: Age Features
test_df['age_squared'] = test_df['age'] ** 2

# G: Interaction Features
test_df['age_bmi_interaction']   = test_df['age']   * test_df['bmi']
test_df['hypertension_x_family'] = test_df['hypertension_history'] * test_df['family_history_diabetes']
test_df['age_family_risk']       = test_df['age']   * test_df['family_history_diabetes']
test_df['obese_family_risk']     = test_df['obese'] * test_df['family_history_diabetes']
test_df['trig_hdl_bmi']          = test_df['triglycerides_hdl_ratio'] * test_df['bmi']

# H: Lifestyle Features
test_df['sedentary_index'] = test_df['screen_time_hours_per_day'] / (
    test_df['physical_activity_minutes_per_week'] / 60 + 1
)
test_df['activity_diet_score'] = test_df['physical_activity_minutes_per_week'] * test_df['diet_score']
test_df['lifestyle_score'] = (
      test_df['diet_score']
    + (test_df['physical_activity_minutes_per_week'] / 300)
    - (test_df['sedentary_index'] * 0.5)
    - (test_df['sleep_deprived']  * 2)
    - (test_df['high_alcohol']    * 2)
)

# I: No log transforms (dropped in V3)

# J: Drop raw source columns — same list as training
test_df.drop(columns=RAW_TO_DROP, inplace=True)

# ── Match training columns EXACTLY ────────────────────────────────
train_columns = X_train.columns
for col in train_columns:
    if col not in test_df.columns:
        test_df[col] = 0
test_df = test_df[train_columns]
print(f"✅ Columns matched — Shape: {test_df.shape}")

# ── Impute (transform only — no fit) ──────────────────────────────
test_df = pd.DataFrame(
    imputer.transform(test_df),
    columns=test_df.columns
)

# ── Scale (transform only — no fit) ───────────────────────────────
test_scaled = scaler.transform(test_df)

# ── Predict ───────────────────────────────────────────────────────
test_preds = model.predict(test_scaled)

# ── Save Submission ───────────────────────────────────────────────
submission = pd.DataFrame({
    'id'                : test_ids.values,
    'diagnosed_diabetes': test_preds
})
submission.to_csv(
    r"C:\Kaggle Diabetes ML Competition\Dataset\1_v3.csv",
    index=False
)
print("✅ File saved!")
print(submission.head(10))
print(f"Total predictions: {submission.shape[0]}")

# ── Prediction distribution check ─────────────────────────────────
# In V2 the first 10 rows were all 1s — a sign of majority-class bias.
# A healthy distribution should roughly mirror the ~62/38 training split.
pred_dist = submission['diagnosed_diabetes'].value_counts(normalize=True)
print(f"\nPrediction distribution:\n{pred_dist}")