# ══════════════════════════════════════════════════════════════════
# Random Forest Classifier — V2: V3 Feature Engineering Applied
# ══════════════════════════════════════════════════════════════════
#
# Key changes vs Original RF:
#   - Replaced preprocess() with inline V3-aligned pipeline
#   - Added IQR-based outlier capping before feature derivation
#   - Updated encoding maps to match V3 Logistic Regression exactly
#   - Removed cholesterol_total rename (uses original column name)
#   - Added metabolic risk flags: obese, high_triglycerides, sleep_deprived,
#     high_alcohol, low_hdl, waist_risk
#   - Added metabolic_risk_score composite feature
#   - Added age_squared, hypertension_x_family, age_family_risk,
#     obese_family_risk, trig_hdl_bmi
#   - Added lifestyle features: sedentary_index, activity_diet_score,
#     lifestyle_score
#   - Dropped raw lipid source columns post-ratio derivation
#   - Removed ldl_hdl_ratio, bmi_waist_interaction, bp_interaction (collinear)
#   - Removed StandardScaler (RF is scale-invariant; provides no benefit)
#   - Combined train+test for get_dummies to guarantee identical columns
#   - Imputer now fit on X_train only; transforms val/test separately
#   - Expanded hyperparameter grid: added max_features and max_samples
#   - Increased n_iter from 10 to 20
#   - Added feature importance table and plot
# ══════════════════════════════════════════════════════════════════

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix, roc_auc_score, RocCurveDisplay)

# ══════════════════════════════════════════════════════════════════
# PATHS
# ══════════════════════════════════════════════════════════════════
TRAIN_PATH      = r"C:\Kaggle Diabetes ML Competition\Dataset\train.csv"
TEST_PATH       = r"C:\Kaggle Diabetes ML Competition\Dataset\test.csv"
SUBMISSION_PATH = r"C:\Kaggle Diabetes ML Competition\Dataset\2_v3.csv"

# ══════════════════════════════════════════════════════════════════
# CONSTANTS
# ══════════════════════════════════════════════════════════════════

# Raw lipid columns replaced by ratios — dropped after derivation (same as V3 LR)
RAW_TO_DROP = ['triglycerides', 'cholesterol_total', 'hdl_cholesterol', 'ldl_cholesterol']

# Encoding maps — aligned exactly with V3 Logistic Regression
EDU_MAP     = {'Highschool': 1, 'Some College': 2, 'Bachelor': 3, 'Graduate': 4}
INCOME_MAP  = {'Lower': 1, 'Lower-Middle': 2, 'Middle': 3, 'Upper-Middle': 4, 'Upper': 5}
SMOKING_MAP = {'Never': 0, 'Former': 1, 'Current': 2}

# Continuous columns capped before any feature derivation
CONTINUOUS_COLS = [
    'age', 'alcohol_consumption_per_week', 'physical_activity_minutes_per_week',
    'diet_score', 'sleep_hours_per_day', 'screen_time_hours_per_day',
    'bmi', 'waist_to_hip_ratio', 'systolic_bp', 'diastolic_bp',
    'heart_rate', 'cholesterol_total', 'hdl_cholesterol',
    'ldl_cholesterol', 'triglycerides'
]

# ══════════════════════════════════════════════════════════════════
# STEP 1: LOAD DATA
# ══════════════════════════════════════════════════════════════════
print("Loading data...")
train_df = pd.read_csv(TRAIN_PATH)
test_df  = pd.read_csv(TEST_PATH)

print(f"Train shape: {train_df.shape}")
print(f"Test shape:  {test_df.shape}")

# ══════════════════════════════════════════════════════════════════
# STEP 2: BASIC CLEANING
# ══════════════════════════════════════════════════════════════════
print("\nCleaning...")

# Preserve test IDs before any modification
test_ids = test_df['id'].copy()
test_df.drop(columns=['id'], inplace=True)
train_df.drop(columns=['id'], inplace=True)

# Downcast for memory efficiency
train_df[train_df.select_dtypes('float64').columns] = (
    train_df.select_dtypes('float64').apply(pd.to_numeric, downcast='float')
)
train_df[train_df.select_dtypes('int64').columns] = (
    train_df.select_dtypes('int64').apply(pd.to_numeric, downcast='integer')
)
print(f"Memory usage: {train_df.memory_usage(deep=True).sum() / 1e6:.1f} MB")

# Deduplicate while keeping target aligned (target is still in df here)
print(f"Duplicates: {train_df.duplicated().sum()}")
train_df.drop_duplicates(inplace=True)
train_df.reset_index(drop=True, inplace=True)

# Separate target AFTER dedup so indices stay aligned
y = train_df['diagnosed_diabetes'].copy()
train_df.drop(columns=['diagnosed_diabetes'], inplace=True)

# ══════════════════════════════════════════════════════════════════
# STEP 3: OUTLIER CAPPING
# IQR-based; raw columns are capped BEFORE any derived features
# are computed from them to avoid propagating outliers into ratios.
#
# Note: train and test are capped independently here for simplicity,
# consistent with V3 LR. A strict no-leakage approach would save
# training IQR bounds and apply them to the test set.
# ══════════════════════════════════════════════════════════════════
def cap_outliers(df, col):
    Q1    = df[col].quantile(0.25)
    Q3    = df[col].quantile(0.75)
    IQR   = Q3 - Q1
    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR
    df[col] = df[col].clip(lower=lower, upper=upper)
    return df

print("Capping outliers...")
for col in CONTINUOUS_COLS:
    if col in train_df.columns:
        train_df = cap_outliers(train_df, col)
    if col in test_df.columns:
        test_df = cap_outliers(test_df, col)

# ══════════════════════════════════════════════════════════════════
# STEP 4: ENCODING
# Must happen before feature engineering so that gender is 0/1
# when low_hdl and waist_risk flags are computed.
# ══════════════════════════════════════════════════════════════════
print("Encoding categorical features...")

for df in [train_df, test_df]:
    df['gender']            = df['gender'].map({'Female': 0, 'Male': 1})
    df['education_level']   = df['education_level'].map(EDU_MAP)
    df['income_level']      = df['income_level'].map(INCOME_MAP)
    df['smoking_status']    = df['smoking_status'].map(SMOKING_MAP)
    df['employment_status'] = df['employment_status'].map(
        lambda x: 1 if x == 'Employed' else 0
    )

# Combine train+test for get_dummies so both DataFrames are guaranteed
# to have identical ethnicity dummy columns — no manual col=0 patching needed.
n_train  = len(train_df)
combined = pd.concat([train_df, test_df], ignore_index=True)
combined = pd.get_dummies(combined, columns=['ethnicity'], drop_first=True)
train_df = combined.iloc[:n_train].reset_index(drop=True)
test_df  = combined.iloc[n_train:].reset_index(drop=True)

# ══════════════════════════════════════════════════════════════════
# STEP 5: FEATURE ENGINEERING (Aligned with V3 Logistic Regression)
# ══════════════════════════════════════════════════════════════════
def engineer_features(df):
    df = df.copy()

    # ── A: Lipid Ratios ───────────────────────────────────────────
    # KEPT:    cholesterol_ratio       — primary lipid health summary
    # KEPT:    triglycerides_hdl_ratio — best single insulin resistance proxy
    # DROPPED: ldl_hdl_ratio           — collinear with cholesterol_ratio
    df['cholesterol_ratio']       = df['cholesterol_total'] / df['hdl_cholesterol']
    df['triglycerides_hdl_ratio'] = df['triglycerides']     / df['hdl_cholesterol']

    # ── B: Blood Pressure ─────────────────────────────────────────
    # DROPPED: bp_interaction — systolic * diastolic collinear with both raw values
    df['pulse_pressure']         = df['systolic_bp'] - df['diastolic_bp']
    df['mean_arterial_pressure'] = df['diastolic_bp'] + (df['pulse_pressure'] / 3)
    df['high_bp_flag']           = (
        (df['systolic_bp'] >= 130) | (df['diastolic_bp'] >= 80)
    ).astype(int)

    # ── C: Metabolic Risk Flags ───────────────────────────────────
    # DROPPED: bmi_waist_interaction — collinear; waist_risk already captures this
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

    # ── D: Metabolic Syndrome Score ───────────────────────────────
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

    # ── F: Age Features ───────────────────────────────────────────
    # age_squared captures non-linear risk acceleration with age
    df['age_squared'] = df['age'] ** 2

    # ── G: Interaction Features ───────────────────────────────────
    df['age_bmi_interaction']   = df['age']   * df['bmi']
    df['hypertension_x_family'] = df['hypertension_history'] * df['family_history_diabetes']
    df['age_family_risk']       = df['age']   * df['family_history_diabetes']
    df['obese_family_risk']     = df['obese'] * df['family_history_diabetes']
    df['trig_hdl_bmi']          = df['triglycerides_hdl_ratio'] * df['bmi']

    # ── H: Lifestyle Features ─────────────────────────────────────
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

    # ── J: Drop Raw Source Columns ────────────────────────────────
    # Ratios are clinically superior; keeping raw columns alongside
    # their derived ratios causes redundancy and inflates feature count.
    df.drop(columns=RAW_TO_DROP, inplace=True)

    return df

print("Engineering features...")
train_df = engineer_features(train_df)
test_df  = engineer_features(test_df)

print(f"Total features after engineering: {train_df.shape[1]}")
print(f"\nClass balance:\n{y.value_counts(normalize=True)}")

# ══════════════════════════════════════════════════════════════════
# STEP 6: TRAIN / VALIDATION SPLIT
# ══════════════════════════════════════════════════════════════════
X_train, X_val, y_train, y_val = train_test_split(
    train_df, y,
    test_size=0.1,
    random_state=42,
    stratify=y
)

print(f"\nX_train shape: {X_train.shape}")
print(f"X_val shape:   {X_val.shape}")

# ══════════════════════════════════════════════════════════════════
# STEP 7: IMPUTE MISSING VALUES
# Fit on X_train only; transform val and test to prevent data leakage.
# The original RF code fit the imputer on the full training set before
# the split, which leaks validation statistics into the training pipeline.
# ══════════════════════════════════════════════════════════════════
print("\nImputing missing values...")
imputer = SimpleImputer(strategy='median')
X_train = pd.DataFrame(imputer.fit_transform(X_train), columns=X_train.columns)
X_val   = pd.DataFrame(imputer.transform(X_val),       columns=X_val.columns)
test_df = pd.DataFrame(imputer.transform(test_df),     columns=test_df.columns)

print(f"NaNs in X_train: {X_train.isnull().sum().sum()}")
print(f"NaNs in X_val:   {X_val.isnull().sum().sum()}")

# StandardScaler intentionally omitted.
# Random Forest splits on feature thresholds; the absolute scale of a feature
# has no effect on split quality or tree structure.

# ══════════════════════════════════════════════════════════════════
# STEP 8: RANDOMIZED SEARCH HYPERPARAMETER TUNING
# ══════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("STEP 8: Commencing Hyperparameter Search...")
print("⏱  n_iter=20, cv=3 — estimated 30-60 minutes depending on hardware.")

param_grid = {
    'n_estimators'     : [100, 200, 300, 500],
    'max_depth'        : [10, 15, 20, 25, None],
    'min_samples_split': [2, 5, 10, 20],
    'min_samples_leaf' : [1, 2, 4, 10],
    # max_features controls how many features each tree considers at each split.
    # 'sqrt' is the RF default; 'log2' creates sparser, more diverse trees;
    # 0.5 gives each tree access to half the feature set.
    'max_features'     : ['sqrt', 'log2', 0.5],
    # max_samples controls row subsampling per tree (bootstrap sample size).
    # Values below 1.0 increase tree diversity and reduce overfitting.
    'max_samples'      : [0.7, 0.8, 0.9, None],
}

rf_base = RandomForestClassifier(
    class_weight='balanced',
    random_state=42,
    n_jobs=-1
)

rf_random = RandomizedSearchCV(
    estimator=rf_base,
    param_distributions=param_grid,
    n_iter=20,              # Increased from 10 for better grid coverage
    cv=3,
    scoring='roc_auc',
    verbose=2,
    random_state=42,
    n_jobs=1                # Keep 1 here to avoid fighting model's n_jobs=-1
)

rf_random.fit(X_train, y_train)

print("\n✅ Tuning Complete!")
print(f"🏆 Best Parameters: {rf_random.best_params_}")
print(f"🏆 Best CV AUC-ROC: {rf_random.best_score_:.4f}")

best_model = rf_random.best_estimator_

# ══════════════════════════════════════════════════════════════════
# STEP 9: EVALUATE THE BEST MODEL
# ══════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("STEP 9: Evaluating Best Model on Validation Set...")

y_pred       = best_model.predict(X_val)
y_pred_proba = best_model.predict_proba(X_val)[:, 1]

print(f"\n  Accuracy : {accuracy_score(y_val, y_pred):.4f}")
print(f"  ROC-AUC  : {roc_auc_score(y_val, y_pred_proba):.4f}")
print(f"\nClassification Report:\n"
      f"{classification_report(y_val, y_pred, target_names=['No Diabetes', 'Diabetes'])}")

cm = confusion_matrix(y_val, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Predicted 0', 'Predicted 1'],
            yticklabels=['Actual 0', 'Actual 1'])
plt.title('Confusion Matrix — RF V2')
plt.tight_layout()
plt.show()

RocCurveDisplay.from_predictions(y_val, y_pred_proba)
plt.title('ROC Curve — RF V2 (Tuned)')
plt.show()

# Feature Importance — RF's equivalent of the LR coefficient table.
# Gini importance measures average impurity reduction across all trees.
# Unlike LR coefficients, these are always positive and sign-free.
feat_imp = pd.DataFrame({
    'Feature'   : X_train.columns,
    'Importance': best_model.feature_importances_
}).sort_values('Importance', ascending=False).reset_index(drop=True)

print(f"\nTop 15 Feature Importances:\n{feat_imp.head(15).to_string(index=False)}")

feat_imp.head(20).sort_values('Importance').plot(
    kind='barh', x='Feature', y='Importance',
    legend=False, figsize=(10, 8)
)
plt.title('Top 20 Feature Importances — RF V2')
plt.xlabel('Gini Importance')
plt.tight_layout()
plt.show()

# ══════════════════════════════════════════════════════════════════
# STEP 10: PREDICT ON TEST.CSV & SAVE SUBMISSION
# ══════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("STEP 10: Predicting on test set & saving submission...")

test_preds = best_model.predict(test_df)

submission = pd.DataFrame({
    'id'                : test_ids.values,
    'diagnosed_diabetes': test_preds
})

submission.to_csv(SUBMISSION_PATH, index=False)
print(f"✅ Submission saved to: {SUBMISSION_PATH}")
print(submission.head(10))
print(f"\nTotal predictions: {submission.shape[0]}")

pred_dist = submission['diagnosed_diabetes'].value_counts(normalize=True)
print(f"\nPrediction distribution:\n{pred_dist}")
print("=" * 60)''