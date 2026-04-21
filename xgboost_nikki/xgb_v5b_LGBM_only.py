# =============== XGBoost V5b - LightGBM Only ===============
# LightGBM with best params from Optuna (same as XGBoost)
# Using selected features + weight=0.8
# ----------------------------------------------------

#%% Importing Libraries
import numpy as np
import pandas as pd
from pathlib import Path
import lightgbm as lgb
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score, roc_auc_score
import warnings
warnings.filterwarnings('ignore')

DATA_DIR = Path(__file__).resolve().parent / "xgb_data"
OUTPUT_DIR = Path(__file__).resolve().parent / "xgb_output"

#%% ---------- Load data ----------
train = pd.read_csv(DATA_DIR / "train_selected.csv")
test = pd.read_csv(DATA_DIR / "test_selected.csv")

TARGET = "diagnosed_diabetes"
X = train.drop(columns=[TARGET])
y = train[TARGET]
test_ids = test['id'].copy()
X_test = test.drop(columns=['id'])

print(f"X shape: {X.shape}, X_test shape: {X_test.shape}")

#%% ---------- LightGBM params ----------
lgb_params = {
    'n_estimators': 318,
    'max_depth': 5,
    'learning_rate': 0.15,
    'subsample': 0.65,
    'colsample_bytree': 0.94,
    'reg_alpha': 1.65,
    'reg_lambda': 1.84,
    'scale_pos_weight': 0.8,
    'random_state': 42,
    'verbose': -1
}

#%% ---------- Cross Validation ----------
kf = KFold(n_splits=5, shuffle=True, random_state=42)
oof_proba = np.zeros(len(X))
test_proba = np.zeros(len(X_test))
auc_scores = []

for fold, (tr_idx, va_idx) in enumerate(kf.split(X), 1):
    print(f"\n--- Fold {fold} ---")
    
    X_tr, X_va = X.iloc[tr_idx], X.iloc[va_idx]
    y_tr, y_va = y.iloc[tr_idx], y.iloc[va_idx]
    
    model = lgb.LGBMClassifier(**lgb_params)
    model.fit(X_tr, y_tr)
    
    va_proba = model.predict_proba(X_va)[:, 1]
    oof_proba[va_idx] = va_proba
    test_proba += model.predict_proba(X_test)[:, 1] / kf.n_splits
    
    auc = roc_auc_score(y_va, va_proba)
    auc_scores.append(auc)
    print(f"Fold {fold}: AUC={auc:.5f}")

# ---------- Final CV Score ----------
cv_auc = roc_auc_score(y, oof_proba)
accuracy = accuracy_score(y, oof_proba.round())

print("\n=== CV Results ===")
print(f"Overall CV AUC: {cv_auc:.5f}")
print(f"Overall Accuracy: {accuracy:.4f}")
print(f"AUC std: {np.std(auc_scores):.5f}")

# ---------- Confusion Matrix ----------
print("\nConfusion Matrix:")
print(pd.crosstab(y, oof_proba.round(), rownames=['Actual'], colnames=['Predicted']))

# ---------- Classification Report ----------
from sklearn.metrics import classification_report
print("\nClassification Report:")
print(classification_report(y, oof_proba.round(), target_names=['No Diabetes', 'Diabetes']))

#%% ---------- Submission ----------
submission = pd.DataFrame({
    "id": test_ids,
    "diagnosed_diabetes": test_proba
})

submission.to_csv(OUTPUT_DIR / "xgb_v5b_lightgbm.csv", index=False)
print(f"\nSaved: xgb_v5b_lightgbm.csv to {OUTPUT_DIR}")