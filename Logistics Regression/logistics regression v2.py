# ══════════════════════════════════════════════════════════════════
# Logistic Regression — Improved Feature Engineering
# File: logistics regression.py
# ══════════════════════════════════════════════════════════════════

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer                    # moved to top
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
# NOTE: Encoding MUST happen before feature engineering because
#       low_hdl and waist_risk flags need gender already as 0/1
# ══════════════════════════════════════════════════════════════════

# Gender: Female=0, Male=1
df['gender'] = df['gender'].map({'Female': 0, 'Male': 1})

# Education (ordinal)
edu_map = {'Highschool': 1, 'Some College': 2, 'Bachelor': 3, 'Graduate': 4}
df['education_level'] = df['education_level'].map(edu_map)

# Income (ordinal)
income_map = {
    'Lower': 1, 'Lower-Middle': 2, 'Middle': 3,
    'Upper-Middle': 4, 'Upper': 5
}
df['income_level'] = df['income_level'].map(income_map)

# Smoking (ordinal)
smoking_map = {'Never': 0, 'Former': 1, 'Current': 2}
df['smoking_status'] = df['smoking_status'].map(smoking_map)

# Employment (binary)
df['employment_status'] = df['employment_status'].map(
    lambda x: 1 if x == 'Employed' else 0
)

# Ethnicity (one-hot — drop_first avoids multicollinearity for LR)
df = pd.get_dummies(df, columns=['ethnicity'], drop_first=True)

print(df.dtypes)
print(df.head())

# ══════════════════════════════════════════════════════════════════
# STEP 5: FEATURE ENGINEERING (IMPROVED)
# ══════════════════════════════════════════════════════════════════

# ── A: Lipid Ratios & Cardiovascular ─────────────────────────────
# Cholesterol ratio: total/HDL — higher = worse lipid profile
df['cholesterol_ratio']       = df['cholesterol_total'] / df['hdl_cholesterol']

# LDL/HDL ratio: standard atherogenic marker
df['ldl_hdl_ratio']           = df['ldl_cholesterol'] / df['hdl_cholesterol']

# Non-HDL: total minus the "good" cholesterol
df['non_hdl_cholesterol']     = df['cholesterol_total'] - df['hdl_cholesterol']

# Triglycerides/HDL ratio: THE single best proxy for insulin resistance
# A ratio > 3.0 is strongly associated with metabolic syndrome & T2D
df['triglycerides_hdl_ratio'] = df['triglycerides'] / df['hdl_cholesterol']

# Atherogenic Index of Plasma (AIP) = log(TG/HDL)
# Clinically validated cardiovascular & T2D risk marker
df['atherogenic_index']       = np.log(df['triglycerides'] / (df['hdl_cholesterol'] + 1e-9))

# ── B: Blood Pressure ─────────────────────────────────────────────
# Pulse pressure: gap between systolic and diastolic
df['pulse_pressure']          = df['systolic_bp'] - df['diastolic_bp']

# Mean Arterial Pressure: better single BP summary than either alone
df['mean_arterial_pressure']  = df['diastolic_bp'] + (df['pulse_pressure'] / 3)

# BP interaction (magnitude of combined pressure)
df['bp_interaction']          = df['systolic_bp'] * df['diastolic_bp']

# Stage 1 Hypertension flag per ACC/AHA 2017 guidelines
df['high_bp_flag']            = (
    (df['systolic_bp'] >= 130) | (df['diastolic_bp'] >= 80)
).astype(int)

# ── C: Metabolic Risk Flags ───────────────────────────────────────
# Obesity (BMI >= 30) and overweight (BMI >= 25)
df['obese']              = (df['bmi'] >= 30).astype(int)
df['overweight']         = (df['bmi'] >= 25).astype(int)

# High triglycerides (clinical threshold: > 150 mg/dL)
df['high_triglycerides'] = (df['triglycerides'] > 150).astype(int)

# Sleep deprivation (< 6 hours)
df['sleep_deprived']     = (df['sleep_hours_per_day'] < 6).astype(int)

# Heavy alcohol use (> 14 drinks/week is the high-risk threshold)
df['high_alcohol']       = (df['alcohol_consumption_per_week'] > 14).astype(int)

# Gender-adjusted HDL threshold (Male: < 40 mg/dL | Female: < 50 mg/dL)
df['low_hdl'] = (
    ((df['gender'] == 1) & (df['hdl_cholesterol'] < 40)) |
    ((df['gender'] == 0) & (df['hdl_cholesterol'] < 50))
).astype(int)

# Gender-adjusted waist-to-hip risk (Male: > 0.90 | Female: > 0.85)
df['waist_risk'] = (
    ((df['gender'] == 1) & (df['waist_to_hip_ratio'] > 0.90)) |
    ((df['gender'] == 0) & (df['waist_to_hip_ratio'] > 0.85))
).astype(int)

# ── D: Metabolic Syndrome Score ───────────────────────────────────
# Count of simultaneous risk factors present
# A score >= 3 is the clinical definition of Metabolic Syndrome
# and strongly predicts T2D onset
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
# WHO BMI classification:
# 0=Underweight, 1=Normal, 2=Overweight, 3=Obese I, 4=Obese II+
df['bmi_category'] = pd.cut(
    df['bmi'],
    bins=[0, 18.5, 25.0, 30.0, 35.0, np.inf],
    labels=[0, 1, 2, 3, 4]
).astype(float)

# Central obesity compound: both BMI and waist-to-hip signal abdominal fat
df['bmi_waist_interaction'] = df['bmi'] * df['waist_to_hip_ratio']

# ── F: Age Features ───────────────────────────────────────────────
# T2D risk accelerates non-linearly with age — squared term captures this
df['age_squared']   = df['age'] ** 2

# Binary flags for clinical age risk windows
df['middle_aged']   = ((df['age'] >= 45) & (df['age'] < 65)).astype(int)
df['senior']        = (df['age'] >= 65).astype(int)

# Ordinal age band: finer-grained than binary flags
df['age_risk_band'] = pd.cut(
    df['age'],
    bins=[0, 35, 45, 55, 65, np.inf],
    labels=[0, 1, 2, 3, 4]
).astype(float)

# ── G: Interaction Features ───────────────────────────────────────
# Age × BMI: being older AND heavier compounds T2D risk significantly
df['age_bmi_interaction']   = df['age']   * df['bmi']

# Hypertension × family history: known clinical risk amplifier
df['hypertension_x_family'] = df['hypertension_history'] * df['family_history_diabetes']

# Age × family history: genetic risk becomes more expressed with age
df['age_family_risk']       = df['age']   * df['family_history_diabetes']

# Obesity × family history: genetic predisposition amplified by obesity
df['obese_family_risk']     = df['obese'] * df['family_history_diabetes']

# Triglycerides/HDL × BMI: triple metabolic risk signal
df['trig_hdl_bmi']          = df['triglycerides_hdl_ratio'] * df['bmi']

# ── H: Lifestyle Features ─────────────────────────────────────────
# Sedentary index: high screen time relative to physical activity
df['sedentary_index'] = df['screen_time_hours_per_day'] / (
    df['physical_activity_minutes_per_week'] / 60 + 1
)

# Activity × diet synergy: both being healthy compounds the benefit
df['activity_diet_score'] = df['physical_activity_minutes_per_week'] * df['diet_score']

# Composite lifestyle health score:
# positive habits add points, risky habits subtract points
df['lifestyle_score'] = (
      df['diet_score']
    + (df['physical_activity_minutes_per_week'] / 300)
    - (df['sedentary_index'] * 0.5)
    - (df['sleep_deprived']  * 2)
    - (df['high_alcohol']    * 2)
)

# ── I: Log Transforms for Skewed Features ─────────────────────────
# Logistic regression assumes approximate linearity in log-odds space
# Log-transforming right-skewed features helps meet this assumption
df['log_triglycerides'] = np.log1p(df['triglycerides'])
df['log_alcohol']       = np.log1p(df['alcohol_consumption_per_week'])

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
# ══════════════════════════════════════════════════════════════════
model = LogisticRegression(
    solver='saga',
    max_iter=1000,
    class_weight='balanced',
    n_jobs=-1,
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
plt.title('ROC Curve')
plt.show()

coef_df = pd.DataFrame({
    'Feature'    : X.columns,
    'Coefficient': model.coef_[0]
}).sort_values('Coefficient', key=abs, ascending=False)

print(coef_df.head(15))

coef_df.head(15).plot(kind='barh', x='Feature', y='Coefficient', legend=False)
plt.title('Top 15 Feature Coefficients')
plt.axvline(0, color='black', linewidth=0.8)
plt.tight_layout()
plt.show()

# ══════════════════════════════════════════════════════════════════
# STEP 12: PREDICT ON TEST.CSV & SAVE SUBMISSION
# (Encoding first, then feature engineering — same order as training)
# ══════════════════════════════════════════════════════════════════
print("\nLoading test.csv...")
test_df  = pd.read_csv(r"C:\Kaggle Diabetes ML Competition\Dataset\test.csv")
test_ids = test_df['id'].copy()
test_df.drop(columns=['id'], inplace=True)

# ── Encoding (SAME maps and order as training) ────────────────────
test_df['gender']            = test_df['gender'].map({'Female': 0, 'Male': 1})
test_df['education_level']   = test_df['education_level'].map(edu_map)
test_df['income_level']      = test_df['income_level'].map(income_map)
test_df['smoking_status']    = test_df['smoking_status'].map(smoking_map)
test_df['employment_status'] = test_df['employment_status'].map(
    lambda x: 1 if x == 'Employed' else 0
)
test_df = pd.get_dummies(test_df, columns=['ethnicity'], drop_first=True)

# ── Feature Engineering (SAME as training — all 9 sections) ──────

# A: Lipid Ratios
test_df['cholesterol_ratio']       = test_df['cholesterol_total'] / test_df['hdl_cholesterol']
test_df['ldl_hdl_ratio']           = test_df['ldl_cholesterol']   / test_df['hdl_cholesterol']
test_df['non_hdl_cholesterol']     = test_df['cholesterol_total'] - test_df['hdl_cholesterol']
test_df['triglycerides_hdl_ratio'] = test_df['triglycerides'] / test_df['hdl_cholesterol']
test_df['atherogenic_index']       = np.log(test_df['triglycerides'] / (test_df['hdl_cholesterol'] + 1e-9))

# B: Blood Pressure
test_df['pulse_pressure']         = test_df['systolic_bp'] - test_df['diastolic_bp']
test_df['mean_arterial_pressure'] = test_df['diastolic_bp'] + (test_df['pulse_pressure'] / 3)
test_df['bp_interaction']         = test_df['systolic_bp'] * test_df['diastolic_bp']
test_df['high_bp_flag']           = (
    (test_df['systolic_bp'] >= 130) | (test_df['diastolic_bp'] >= 80)
).astype(int)

# C: Metabolic Risk Flags
test_df['obese']              = (test_df['bmi'] >= 30).astype(int)
test_df['overweight']         = (test_df['bmi'] >= 25).astype(int)
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

# E: BMI Features
test_df['bmi_category'] = pd.cut(
    test_df['bmi'],
    bins=[0, 18.5, 25.0, 30.0, 35.0, np.inf],
    labels=[0, 1, 2, 3, 4]
).astype(float)
test_df['bmi_waist_interaction'] = test_df['bmi'] * test_df['waist_to_hip_ratio']

# F: Age Features
test_df['age_squared']   = test_df['age'] ** 2
test_df['middle_aged']   = ((test_df['age'] >= 45) & (test_df['age'] < 65)).astype(int)
test_df['senior']        = (test_df['age'] >= 65).astype(int)
test_df['age_risk_band'] = pd.cut(
    test_df['age'],
    bins=[0, 35, 45, 55, 65, np.inf],
    labels=[0, 1, 2, 3, 4]
).astype(float)

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

# I: Log Transforms
test_df['log_triglycerides'] = np.log1p(test_df['triglycerides'])
test_df['log_alcohol']       = np.log1p(test_df['alcohol_consumption_per_week'])

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
    r"C:\Kaggle Diabetes ML Competition\Dataset\1_v2.csv",
    index=False
)
print("✅ File saved!")
print(submission.head(10))
print(f"Total predictions: {submission.shape[0]}")