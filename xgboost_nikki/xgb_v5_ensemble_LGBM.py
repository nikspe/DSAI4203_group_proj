# =============== XGBoost V5 - Ensemble with LightGBM ===============
# Combining XGBoost and LightGBM for better generalization
# Using best params from Optuna (v4) + LightGBM with same settings
#
# === CV Results ===
# XGBoost CV AUC: 0.72512 (+/- 0.00076)
# LightGBM CV AUC: 0.72697 (+/- 0.00077)
# Ensemble CV AUC: 0.72697 (+/- 0.00074)
#
# Confusion Matrix (Ensemble):
# Predicted       0       1
# Actual                   
# 0.0        144744  118949
# 1.0        105384  330923
# ----------------------------------------------------

#%% Importing Libraries
import numpy as np
import pandas as pd
from pathlib import Path
import xgboost as xgb
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

#%% ---------- Best params from Optuna ----------
best_params = {
    'n_estimators': 318,
    'max_depth': 5,
    'learning_rate': 0.15,
    'subsample': 0.65,
    'colsample_bytree': 0.94,
    'reg_alpha': 1.65,
    'reg_lambda': 1.84,
    'scale_pos_weight': 0.8,
    'random_state': 42,
    'eval_metric': 'logloss'
}

# LightGBM params (matching XGBoost where possible)
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

#%% ---------- Cross Validation with Ensemble ----------
kf = KFold(n_splits=5, shuffle=True, random_state=42)
oof_proba_xgb = np.zeros(len(X))
oof_proba_lgb = np.zeros(len(X))
test_proba_xgb = np.zeros(len(X_test))
test_proba_lgb = np.zeros(len(X_test))

auc_scores_xgb = []
auc_scores_lgb = []
auc_scores_ensemble = []

for fold, (tr_idx, va_idx) in enumerate(kf.split(X), 1):
    print(f"\n--- Fold {fold} ---")
    
    X_tr, X_va = X.iloc[tr_idx], X.iloc[va_idx]
    y_tr, y_va = y.iloc[tr_idx], y.iloc[va_idx]
    
    # XGBoost
    xgb_model = xgb.XGBClassifier(**best_params)
    xgb_model.fit(X_tr, y_tr)
    xgb_va_proba = xgb_model.predict_proba(X_va)[:, 1]
    oof_proba_xgb[va_idx] = xgb_va_proba
    test_proba_xgb += xgb_model.predict_proba(X_test)[:, 1] / kf.n_splits
    auc_xgb = roc_auc_score(y_va, xgb_va_proba)
    
    # LightGBM
    lgb_model = lgb.LGBMClassifier(**lgb_params)
    lgb_model.fit(X_tr, y_tr)
    lgb_va_proba = lgb_model.predict_proba(X_va)[:, 1]
    oof_proba_lgb[va_idx] = lgb_va_proba
    test_proba_lgb += lgb_model.predict_proba(X_test)[:, 1] / kf.n_splits
    auc_lgb = roc_auc_score(y_va, lgb_va_proba)
    
    # Ensemble (average)
    ensemble_va_proba = (xgb_va_proba + lgb_va_proba) / 2
    auc_ensemble = roc_auc_score(y_va, ensemble_va_proba)
    
    auc_scores_xgb.append(auc_xgb)
    auc_scores_lgb.append(auc_lgb)
    auc_scores_ensemble.append(auc_ensemble)
    
    print(f"  XGB AUC: {auc_xgb:.5f}, LGB AUC: {auc_lgb:.5f}, Ensemble AUC: {auc_ensemble:.5f}")

# ---------- Final CV Scores ----------
cv_auc_xgb = roc_auc_score(y, oof_proba_xgb)
cv_auc_lgb = roc_auc_score(y, oof_proba_lgb)
ensemble_proba = (oof_proba_xgb + oof_proba_lgb) / 2
cv_auc_ensemble = roc_auc_score(y, ensemble_proba)

print("\n=== CV Results ===")
print(f"XGBoost CV AUC: {cv_auc_xgb:.5f} (+/- {np.std(auc_scores_xgb):.5f})")
print(f"LightGBM CV AUC: {cv_auc_lgb:.5f} (+/- {np.std(auc_scores_lgb):.5f})")
print(f"Ensemble CV AUC: {cv_auc_ensemble:.5f} (+/- {np.std(auc_scores_ensemble):.5f})")

#%% ---------- Confusion Matrix (Ensemble) ----------
ensemble_pred = (ensemble_proba >= 0.5).astype(int)
print("\nConfusion Matrix (Ensemble):")
print(pd.crosstab(y, ensemble_pred, rownames=['Actual'], colnames=['Predicted']))

#%% ---------- Submission ----------
final_test_proba = (test_proba_xgb + test_proba_lgb) / 2
submission = pd.DataFrame({
    "id": test_ids,
    "diagnosed_diabetes": final_test_proba
})

submission.to_csv(OUTPUT_DIR / "xgb_v5_ensemble.csv", index=False)
print(f"\nSaved: xgb_v5_ensemble.csv to {OUTPUT_DIR}")