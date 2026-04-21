# =============== XGBoost V2a - EDA & Feature Analysis ===============
# Exploratory Data Analysis and Feature Engineering for XGBoost model
# - Analyze feature distributions and correlations with target

#%% Importing Libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

DATA_DIR = Path(__file__).resolve().parent / "xgb_data"
OUTPUT_DIR = Path(__file__).resolve().parent / "feature_engi_outputs"

# ---------- Load data ----------
train = pd.read_csv(f"{DATA_DIR}/xgb_train.csv")
test = pd.read_csv(f"{DATA_DIR}/xgb_test.csv")

print(f"Train shape: {train.shape}, Test shape: {test.shape}")
print(f"Columns:\n{train.columns.tolist()}")
print(f"\nTarget distribution:\n{train['diagnosed_diabetes'].value_counts()}")

#%% ---------- 1. Missing Values ----------
print("\nMissing values in train:")
print(train.isnull().sum()[train.isnull().sum() > 0])

#%% ---------- 2. Correlation with Target ----------
TARGET = "diagnosed_diabetes"
numeric_cols = train.select_dtypes(include=[np.number]).columns
correlations = train[numeric_cols].corr()[TARGET].sort_values(ascending=False)
print("\nCorrelation with target:")
print(correlations)

#%% ---------- 3. Visualizations ----------
# 3.1 Correlation Heatmap
plt.figure(figsize=(12, 10))
top_features = correlations.head(15).index
sns.heatmap(train[top_features].corr(), annot=True, cmap='coolwarm', fmt='.2f')
plt.title("Top 15 Feature Correlations")
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "eda_heatmap.png", dpi=150)
plt.show()

# 3.2 Distribution Plots (auto-detect numeric features)
numeric_features = train.select_dtypes(include=[np.number]).columns.tolist()
numeric_features.remove(TARGET)
key_features = numeric_features[:6] if len(numeric_features) >= 6 else numeric_features

fig, axes = plt.subplots(2, 3, figsize=(15, 10))
axes = axes.ravel()

for i, feature in enumerate(key_features):
    for label, color in [(0, 'blue'), (1, 'red')]:
        subset = train[train[TARGET] == label][feature].dropna()
        axes[i].hist(subset, bins=30, alpha=0.5, color=color, label=f'Diabetes={label}')
        axes[i].axvline(subset.mean(), color=color, linestyle='dashed', alpha=0.7)
    axes[i].set_title(f'{feature} (corr={correlations[feature]:.3f})')
    axes[i].legend()
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "eda_distributions.png", dpi=150)
plt.show()

#%% ---------- 4. Test New Feature Combinations ----------
print("\nTesting candidate features:")
test_features = {}

if 'systolic_bp' in train.columns and 'diastolic_bp' in train.columns:
    test_features['bp_ratio'] = train['systolic_bp'] / train['diastolic_bp']
    test_features['pulse_pressure'] = train['systolic_bp'] - train['diastolic_bp']
    test_features['map'] = train['diastolic_bp'] + (1/3) * test_features['pulse_pressure']

if 'ldl_cholesterol' in train.columns and 'hdl_cholesterol' in train.columns:
    test_features['chol_ratio'] = train['ldl_cholesterol'] / train['hdl_cholesterol']
    test_features['chol_diff'] = train['ldl_cholesterol'] - train['hdl_cholesterol']

if 'bmi' in train.columns:
    test_features['bmi_squared'] = train['bmi'] ** 2
    if 'age' in train.columns:
        test_features['bmi_age'] = train['bmi'] * train['age']

if 'age' in train.columns:
    test_features['age_squared'] = train['age'] ** 2
    test_features['elderly'] = (train['age'] >= 65).astype(int)

if 'bmi' in train.columns:
    test_features['obese'] = (train['bmi'] >= 30).astype(int)

for name, feature in test_features.items():
    corr = feature.corr(train[TARGET])
    print(f"  {name}: {corr:.4f}")

#%% ---------- 5. Feature Importance (Random Forest) ----------
print("\nRandom Forest Feature Importance:")
X_temp = train.drop(columns=[TARGET, 'id'], errors='ignore')
categorical_cols = X_temp.select_dtypes(include=['object']).columns
X_temp = pd.get_dummies(X_temp, columns=categorical_cols, drop_first=True)
y_temp = train[TARGET]

X_tr, X_va, y_tr, y_va = train_test_split(X_temp, y_temp, test_size=0.2, random_state=42)

rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
rf.fit(X_tr, y_tr)

feature_importance = pd.DataFrame({
    'feature': X_temp.columns,
    'importance': rf.feature_importances_
}).sort_values('importance', ascending=False)

print(feature_importance.head(15).to_string(index=False))

plt.figure(figsize=(10, 8))
sns.barplot(data=feature_importance.head(15), y='feature', x='importance')
plt.title("Top 15 Features by Importance")
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "eda_feature_importance.png", dpi=150)
plt.show()