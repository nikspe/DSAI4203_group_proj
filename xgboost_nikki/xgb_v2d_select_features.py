# =============== XGBoost V2c - Feature Selection ===============
# Remove low-importance features based on Random Forest
# Keep only top 15 features by importance
# no creation of features
# ----------------------------------------------------

#%% Importing Libraries
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

DATA_DIR = Path(__file__).resolve().parent / "xgb_data"
OUTPUT_DIR = Path(__file__).resolve().parent / "xgb_data"

# ---------- Load data ----------
train = pd.read_csv(f"{DATA_DIR}/xgb_train.csv")
test = pd.read_csv(f"{DATA_DIR}/xgb_test.csv")

print(f"Original train shape: {train.shape}, Test shape: {test.shape}")

#%% ---------- Calculate Feature Importance ----------
TARGET = "diagnosed_diabetes"

X_temp = train.drop(columns=[TARGET, 'id'], errors='ignore')
# Handle categorical columns
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

print("\nTop 20 features by importance:")
print(feature_importance.head(20).to_string(index=False))

#%% ---------- Select Top Features ----------
# Keep top 15 features
top_n = 15
top_features = feature_importance.head(top_n)['feature'].tolist()

print(f"\nKeeping top {top_n} features:")
for f in top_features:
    print(f"  - {f}")

#%% ---------- Prepare Data with Selected Features ----------
# Keep 'id' for submission
id_column = test['id'] if 'id' in test.columns else None

# Select features (excluding id from feature list)
feature_cols = [f for f in top_features if f != 'id' and f in train.columns]

X_train = train[feature_cols].copy()
y_train = train[TARGET]

# For test, keep id + selected features
if id_column is not None:
    X_test = test[['id'] + [f for f in feature_cols if f in test.columns]].copy()
else:
    X_test = test[[f for f in feature_cols if f in test.columns]].copy()

#%% ---------- Save ----------
train_selected = X_train.copy()
train_selected[TARGET] = y_train
test_selected = X_test.copy()

train_selected.to_csv(OUTPUT_DIR / "train_selected.csv", index=False)
test_selected.to_csv(OUTPUT_DIR / "test_selected.csv", index=False)


print(f"Saved: train_selected.csv ({train_selected.shape})")
print(f"Saved: test_selected.csv ({test_selected.shape})")
print(f"Target column '{TARGET}' is in train file only")