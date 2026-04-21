# ══════════════════════════════════════════════════════════════════
# Random Forest Classifier - Hyperparameter Tuning
# ══════════════════════════════════════════════════════════════════

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix, roc_auc_score)

# ══════════════════════════════════════════════════════════════════
# PATHS
# ══════════════════════════════════════════════════════════════════
TRAIN_PATH      = r"C:\Kaggle Diabetes ML Competition\Dataset\train.csv"
TEST_PATH       = r"C:\Kaggle Diabetes ML Competition\Dataset\test.csv"
SUBMISSION_PATH = r"C:\Kaggle Diabetes ML Competition\Dataset\updated result 2.csv"

# ══════════════════════════════════════════════════════════════════
# ENCODING MAPS & PREPROCESS FUNCTION (Unchanged)
# ══════════════════════════════════════════════════════════════════
GENDER_MAP = {'Male': 0, 'Female': 1}
SMOKING_MAP = {'Never': 0, 'Former': 1, 'Current': 2}
EMPLOYMENT_MAP = {'Unemployed': 0, 'Employed': 1, 'Self-Employed': 2}
EDUCATION_MAP = {'Highschool': 0, 'Some College': 1, 'Bachelors': 2, 'Masters': 3, 'PhD': 4}
INCOME_MAP = {'Low': 0, 'Middle': 1, 'High': 2}

def preprocess(df):
    df = df.copy()
    df.drop(columns=['id', 'diagnosed_diabetes'], errors='ignore', inplace=True)
    df.rename(columns={'cholesterol_total': 'total_cholesterol'}, inplace=True)

    df['age_bmi_interaction']        = df['age'] * df['bmi']
    df['cholesterol_ratio']          = df['total_cholesterol'] / df['hdl_cholesterol']
    df['ldl_hdl_ratio']              = df['ldl_cholesterol'] / df['hdl_cholesterol']
    df['bp_interaction']             = df['systolic_bp'] * df['diastolic_bp']
    df['bp_pulse_pressure']          = df['systolic_bp'] - df['diastolic_bp']
    df['activity_sleep_score']       = df['physical_activity_minutes_per_week'] * df['sleep_hours_per_day']
    df['screen_sleep_ratio']         = df['screen_time_hours_per_day'] / (df['sleep_hours_per_day'] + 1e-9)
    df['bmi_waist_interaction']      = df['bmi'] * df['waist_to_hip_ratio']
    df['triglycerides_hdl_ratio']    = df['triglycerides'] / df['hdl_cholesterol']

    df['gender']            = df['gender'].map(GENDER_MAP)
    df['smoking_status']    = df['smoking_status'].map(SMOKING_MAP)
    df['employment_status'] = df['employment_status'].map(EMPLOYMENT_MAP)
    df['education_level']   = df['education_level'].map(EDUCATION_MAP)
    df['income_level']      = df['income_level'].map(INCOME_MAP)

    df = pd.get_dummies(df, columns=['ethnicity'])
    return df

# ══════════════════════════════════════════════════════════════════
# STEPS 1-5: LOAD, PREPROCESS, IMPUTE, SCALE, SPLIT
# ══════════════════════════════════════════════════════════════════
print("Loading data...")
train_df = pd.read_csv(TRAIN_PATH)
test_df  = pd.read_csv(TEST_PATH)
test_ids = test_df['id'].copy()
y        = train_df['diagnosed_diabetes'].copy()

print("Preprocessing...")
X              = preprocess(train_df)
test_processed = preprocess(test_df)

for col in X.columns:
    if col not in test_processed.columns:
        test_processed[col] = 0
test_processed = test_processed[X.columns]

print("Imputing and Scaling...")
imputer      = SimpleImputer(strategy='median')
X_imputed    = pd.DataFrame(imputer.fit_transform(X), columns=X.columns)
test_imputed = pd.DataFrame(imputer.transform(test_processed), columns=X.columns)

# Note: We keep the scaler here just to keep the pipeline stable, though RF doesn't strictly need it.
scaler      = StandardScaler()
X_scaled    = scaler.fit_transform(X_imputed)
test_scaled = scaler.transform(test_imputed)

X_train, X_val, y_train, y_val = train_test_split(X_scaled, y, test_size=0.2, random_state=42, stratify=y)

# ══════════════════════════════════════════════════════════════════
# STEP 6: RANDOMIZED SEARCH HYPERPARAMETER TUNING
# ══════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("STEP 6: Commencing Hyperparameter Search...")
print("GRAB A COFFEE: This will test 10 combinations and may take 10-30 minutes.")

# 1. Define the grid of parameters to test
param_grid = {
    'n_estimators': [100, 200, 300, 500],          # Number of trees
    'max_depth': [10, 15, 20, 25, None],           # Maximum depth of each tree
    'min_samples_split': [2, 5, 10, 20],           # Min samples needed to split a node
    'min_samples_leaf': [1, 2, 4, 10]              # Min samples allowed in a leaf
}

# 2. Setup the Base Model
rf_base = RandomForestClassifier(
    class_weight='balanced',
    random_state=42,
    n_jobs=-1
)

# 3. Setup the Randomized Search
rf_random = RandomizedSearchCV(
    estimator=rf_base,
    param_distributions=param_grid,
    n_iter=10,                 # Test exactly 10 random combinations from the grid
    cv=3,                      # 3-Fold Cross Validation
    scoring='roc_auc',         # Optimize specifically for Kaggle's AUC score
    verbose=2,                 # Print progress text to console
    random_state=42,
    n_jobs=1                   # Keep to 1 here so it doesn't fight the model's n_jobs=-1
)

# 4. Run the search
rf_random.fit(X_train, y_train)

print("\n✅ Tuning Complete!")
print(f"🏆 Best Parameters Found: {rf_random.best_params_}")

# 5. Extract the winning model
best_model = rf_random.best_estimator_

# ══════════════════════════════════════════════════════════════════
# STEP 7: EVALUATE THE BEST MODEL
# ══════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("STEP 7: Evaluating Best Model...")

y_pred       = best_model.predict(X_val)
y_pred_proba = best_model.predict_proba(X_val)[:, 1]

print(f"\n  Accuracy : {accuracy_score(y_val, y_pred):.4f}")
print(f"  ROC-AUC  : {roc_auc_score(y_val, y_pred_proba):.4f}")
print(f"\nClassification Report:\n{classification_report(y_val, y_pred)}")

# ══════════════════════════════════════════════════════════════════
# STEP 8: PREDICT AND SAVE
# ══════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("STEP 8: Predicting on test set & saving submission...")

test_preds = best_model.predict(test_scaled)

submission = pd.DataFrame({
    'id'                : test_ids.values,
    'diagnosed_diabetes': test_preds
})

submission.to_csv(SUBMISSION_PATH, index=False)
print(f"✅ Submission saved to: {SUBMISSION_PATH}")
print("=" * 60)