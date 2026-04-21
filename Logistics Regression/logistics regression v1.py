import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (classification_report, confusion_matrix,
                              roc_auc_score, RocCurveDisplay)

df = pd.read_csv(r"C:\Kaggle Diabetes ML Competition\Dataset\train.csv")
print(df.shape)
print(df.dtypes)
print(df.head())

df.drop(columns=['id'], inplace=True)

df[df.select_dtypes('float64').columns] = df.select_dtypes('float64').apply(pd.to_numeric, downcast='float')
df[df.select_dtypes('int64').columns]   = df.select_dtypes('int64').apply(pd.to_numeric, downcast='integer')

print(f"Memory usage: {df.memory_usage(deep=True).sum() / 1e6:.1f} MB")

print(df.isnull().sum())
print(f"Duplicates: {df.duplicated().sum()}")
df.drop_duplicates(inplace=True)

continuous_cols = [
    'age', 'alcohol_consumption_per_week', 'physical_activity_minutes_per_week',
    'diet_score', 'sleep_hours_per_day', 'screen_time_hours_per_day',
    'bmi', 'waist_to_hip_ratio', 'systolic_bp', 'diastolic_bp',
    'heart_rate', 'cholesterol_total', 'hdl_cholesterol',
    'ldl_cholesterol', 'triglycerides'
]

def cap_outliers(df, col):
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR
    df[col] = df[col].clip(lower=lower, upper=upper)
    return df

for col in continuous_cols:
    df = cap_outliers(df, col)

    # Gender: Female=0, Male=1
df['gender'] = df['gender'].map({'Female': 0, 'Male': 1})

# Education has a natural order
edu_map = {
    'Highschool': 1,
    'Some College': 2,
    'Bachelor': 3,
    'Graduate': 4
}
df['education_level'] = df['education_level'].map(edu_map)

# Income has a natural order
income_map = {
    'Lower': 1,
    'Lower-Middle': 2,
    'Middle': 3,
    'Upper-Middle': 4,
    'Upper': 5
}
df['income_level'] = df['income_level'].map(income_map)

# Smoking status has an intensity order
smoking_map = {'Never': 0, 'Former': 1, 'Current': 2}
df['smoking_status'] = df['smoking_status'].map(smoking_map)

# Employment: simplify to binary (employed vs not)
df['employment_status'] = df['employment_status'].map(
    lambda x: 1 if x == 'Employed' else 0
)

# Ethnicity has no ranking, so one-hot encode it
df = pd.get_dummies(df, columns=['ethnicity'], drop_first=True)
# drop_first=True avoids multicollinearity (dummy variable trap)

print(df.dtypes)
print(df.head())

# --- Cardiovascular risk indicators ---
# Pulse pressure: gap between systolic and diastolic
df['pulse_pressure'] = df['systolic_bp'] - df['diastolic_bp']

# Cholesterol ratio: higher = worse lipid profile
df['cholesterol_ratio'] = df['cholesterol_total'] / df['hdl_cholesterol']

# Non-HDL cholesterol: total minus the "good" cholesterol
df['non_hdl_cholesterol'] = df['cholesterol_total'] - df['hdl_cholesterol']

# --- Lifestyle composite ---
# Low activity + high screen time is a risky combination
df['sedentary_index'] = df['screen_time_hours_per_day'] / (
    df['physical_activity_minutes_per_week'] / 60 + 1  # +1 avoids division by zero
)

# --- Interaction features ---
# Older + heavier BMI is a known diabetes risk amplifier
df['age_bmi_interaction'] = df['age'] * df['bmi']

# Hypertension combined with family history
df['hypertension_x_family'] = df['hypertension_history'] * df['family_history_diabetes']

# --- BMI risk flag ---
# BMI >= 30 is clinically defined as obese
df['obese'] = (df['bmi'] >= 30).astype(int)

# --- Sleep deprivation flag ---
df['sleep_deprived'] = (df['sleep_hours_per_day'] < 6).astype(int)

X = df.drop(columns=['diagnosed_diabetes'])
y = df['diagnosed_diabetes']

print(f"Features: {X.shape[1]}")
print(f"Class balance:\n{y.value_counts(normalize=True)}")


X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.1,   
    random_state=42,
    stratify=y
)

from sklearn.impute import SimpleImputer

# Fit ONLY on training data to prevent leakage
imputer = SimpleImputer(strategy='median')

X_train = pd.DataFrame(
    imputer.fit_transform(X_train),
    columns=X_train.columns
)
X_test = pd.DataFrame(
    imputer.transform(X_test),   # transform only — no fit
    columns=X_test.columns
)

# Verify no NaNs remain
print(f"NaNs in X_train: {X_train.isnull().sum().sum()}")
print(f"NaNs in X_test:  {X_test.isnull().sum().sum()}")


scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled  = scaler.transform(X_test)   # transform only — no fit


model = LogisticRegression(
    solver='saga',           
    max_iter=1000,
    class_weight='balanced',
    n_jobs=-1,               
    random_state=42
)

model.fit(X_train_scaled, y_train)


y_pred      = model.predict(X_test_scaled)
y_pred_prob = model.predict_proba(X_test_scaled)[:, 1]  # probability of class 1

# --- Core metrics ---
print(classification_report(y_test, y_pred, target_names=['No Diabetes', 'Diabetes']))

# --- AUC-ROC ---
auc = roc_auc_score(y_test, y_pred_prob)
print(f"AUC-ROC: {auc:.4f}")

# --- Confusion Matrix ---
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Predicted 0', 'Predicted 1'],
            yticklabels=['Actual 0', 'Actual 1'])
plt.title('Confusion Matrix')
plt.tight_layout()
plt.show()

# --- ROC Curve ---
RocCurveDisplay.from_predictions(y_test, y_pred_prob)
plt.title('ROC Curve')
plt.show()


coef_df = pd.DataFrame({
    'Feature': X.columns,
    'Coefficient': model.coef_[0]
}).sort_values('Coefficient', key=abs, ascending=False)

print(coef_df.head(15))

# Visualize
coef_df.head(15).plot(kind='barh', x='Feature', y='Coefficient', legend=False)
plt.title('Top 15 Feature Coefficients')
plt.axvline(0, color='black', linewidth=0.8)
plt.tight_layout()
plt.show()


# ── STEP: Load test.csv ───────────────────────────────────────────
print("Loading test.csv...")
test_df = pd.read_csv(r"C:\Kaggle Diabetes ML Competition\Dataset\test.csv")
test_ids = test_df['id']
test_df.drop(columns=['id'], inplace=True)

# ── STEP: Rename mismatched columns ───────────────────────────────
test_df.rename(columns={'cholesterol_total': 'total_cholesterol'}, inplace=True)

# ── STEP: Feature Engineering ─────────────────────────────────────
print("Feature engineering...")
test_df['age_bmi_interaction'] = test_df['age'] * test_df['bmi']
test_df['cholesterol_ratio']   = test_df['total_cholesterol'] / test_df['hdl_cholesterol']
# paste ALL your other engineered features here
print("✅ Feature engineering done")

# ── STEP: Ordinal Encoding (SAME maps as training) ─────────────────
print("Ordinal encoding...")

gender_map = {'Male': 0, 'Female': 1}
smoking_map = {'Never': 0, 'Former': 1, 'Current': 2}  # use your exact map
employment_map = {'Unemployed': 0, 'Employed': 1, 'Self-Employed': 2}  # use your exact map
education_map = {'Highschool': 0, 'Some College': 1, 'Bachelors': 2, 'Masters': 3, 'PhD': 4}  # use your exact map
income_map = {'Low': 0, 'Middle': 1, 'High': 2}  # use your exact map

test_df['gender']            = test_df['gender'].map(gender_map)
test_df['smoking_status']    = test_df['smoking_status'].map(smoking_map)
test_df['employment_status'] = test_df['employment_status'].map(employment_map)
test_df['education_level']   = test_df['education_level'].map(education_map)
test_df['income_level']      = test_df['income_level'].map(income_map)

print("✅ Ordinal encoding done")

# ── STEP: One-Hot Encoding ─────────────────────────────────────────
print("One-hot encoding...")
test_df = pd.get_dummies(test_df, columns=['ethnicity'])
print("✅ One-hot encoding done")

# ── STEP: Match training columns EXACTLY ──────────────────────────
print("Matching training columns...")
train_columns = X_train.columns

for col in train_columns:
    if col not in test_df.columns:
        test_df[col] = 0

test_df = test_df[train_columns]
print(f"✅ Columns matched — Shape: {test_df.shape}")

# ── STEP: Imputing ────────────────────────────────────────────────
print("Imputing...")
test_df = pd.DataFrame(
    imputer.transform(test_df),
    columns=test_df.columns
)
print("✅ Imputing done")

# ── STEP: Scaling ─────────────────────────────────────────────────
print("Scaling...")
test_scaled = scaler.transform(test_df)
print("✅ Scaling done")

# ── STEP: Predicting ──────────────────────────────────────────────
print("Predicting...")
test_preds = model.predict(test_scaled)
print("✅ Predictions done")

# ── STEP: Saving CSV ──────────────────────────────────────────────
print("Saving submission.csv...")
submission = pd.DataFrame({
    'id': test_ids.values,
    'diagnosed_diabetes': test_preds
})
submission.to_csv(
    r"C:\Kaggle Diabetes ML Competition\Dataset\results 1.csv",
    index=False
)
print("✅ File saved!")
print(submission.head(10))
print(f"Total predictions: {submission.shape[0]}")