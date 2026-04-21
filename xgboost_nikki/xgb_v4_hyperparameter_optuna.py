# =============== XGBoost V4 - Optuna Hyperparameter Tuning ===============
# Using selected features with fixed scale_pos_weight = 0.8
# 
# Best 1-AUC: 0.27484
# Best params: {'n_estimators': 318, 'max_depth': 5, 'learning_rate': 0.1500270814569286, 'subsample': 0.6500010853530851, 'colsample_bytree': 0.943793096281652, 'reg_alpha': 1.6492644510439611, 'reg_lambda': 1.8423459508094788}
# 
#  === CV Results ===
#   Overall CV Accuracy: 0.67829
#   Overall CV AUC: 0.72515
#   Accuracy std: 0.00071
#   AUC std: 0.00078
# ----------------------------------------------------

#%% Importing Libraries
import numpy as np
import pandas as pd
from pathlib import Path
import xgboost as xgb
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score, roc_auc_score
import optuna
import warnings
warnings.filterwarnings('ignore')

DATA_DIR = Path(__file__).resolve().parent / "xgb_data"
OUTPUT_DIR = Path(__file__).resolve().parent / "xgb_output"

#%% ---------- Load data (selected features) ----------
train = pd.read_csv(DATA_DIR / "train_selected.csv")
test = pd.read_csv(DATA_DIR / "test_selected.csv")

TARGET = "diagnosed_diabetes"
X = train.drop(columns=[TARGET])
y = train[TARGET]
X_test = test.drop(columns=['id'])

print(f"X shape: {X.shape}, X_test shape: {X_test.shape}")
print(f"Class distribution:\n{y.value_counts()}")

#%% ---------- Optuna Hyperparameter Tuning ----------
def objective(trial):
    params = {
        "n_estimators": trial.suggest_int("n_estimators", 100, 500),
        "max_depth": trial.suggest_int("max_depth", 3, 10),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3),
        "subsample": trial.suggest_float("subsample", 0.6, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
        "reg_alpha": trial.suggest_float("reg_alpha", 0, 2),
        "reg_lambda": trial.suggest_float("reg_lambda", 0, 2),
        "scale_pos_weight": 0.8,  # FIXED from grid search
        "random_state": 42,
        "eval_metric": "logloss"
    }
    
    model = xgb.XGBClassifier(**params)
    
    cv = KFold(n_splits=5, shuffle=True, random_state=42)
    auc_scores = []
    
    for tr_idx, va_idx in cv.split(X):
        X_tr, X_va = X.iloc[tr_idx], X.iloc[va_idx]
        y_tr, y_va = y.iloc[tr_idx], y.iloc[va_idx]
        
        model.fit(X_tr, y_tr)
        pred_proba = model.predict_proba(X_va)[:, 1]
        auc_scores.append(roc_auc_score(y_va, pred_proba))
    
    return 1 - np.mean(auc_scores)  # minimize 1 - AUC

print("Starting Optuna hyperparameter tuning...")
study = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(seed=42))
study.optimize(objective, n_trials=30, timeout=1800)

print("\n=== Hyperparameter Tuning Results ===")
print(f"Best 1-AUC: {study.best_value:.5f}")
print(f"Best params: {study.best_params}")

best_params = study.best_params.copy()
best_params.update({
    "scale_pos_weight": 0.8,
    "random_state": 42,
    "eval_metric": "logloss"
})

#%% ---------- Cross Validation with Optimized Parameters ----------
kf = KFold(n_splits=5, shuffle=True, random_state=42)
oof_pred = np.zeros(len(X))
oof_proba = np.zeros(len(X))
test_proba = np.zeros(len(X_test))

acc_scores = []
auc_scores = []

for fold, (tr_idx, va_idx) in enumerate(kf.split(X), 1):
    print(f"\n--- Fold {fold} ---")
    
    X_tr, X_va = X.iloc[tr_idx], X.iloc[va_idx]
    y_tr, y_va = y.iloc[tr_idx], y.iloc[va_idx]
    
    model = xgb.XGBClassifier(**best_params)
    model.fit(X_tr, y_tr)
    
    y_va_pred = model.predict(X_va)
    y_va_proba = model.predict_proba(X_va)[:, 1]
    
    oof_pred[va_idx] = y_va_pred
    oof_proba[va_idx] = y_va_proba
    test_proba += model.predict_proba(X_test)[:, 1] / kf.n_splits
    
    acc = accuracy_score(y_va, y_va_pred)
    auc = roc_auc_score(y_va, y_va_proba)
    
    acc_scores.append(acc)
    auc_scores.append(auc)
    
    print(f"Fold {fold}: Accuracy={acc:.5f}, AUC={auc:.5f}")

# ---------- Final CV Score ----------
cv_accuracy = accuracy_score(y, oof_pred)
cv_auc = roc_auc_score(y, oof_proba)

print("\n=== CV Results ===")
print(f"Overall CV Accuracy: {cv_accuracy:.5f}")
print(f"Overall CV AUC: {cv_auc:.5f}")
print(f"Accuracy std: {np.std(acc_scores):.5f}")
print(f"AUC std: {np.std(auc_scores):.5f}")

#%% ---------- Confusion Matrix ----------
print("\nConfusion Matrix:")
print(pd.crosstab(y, oof_pred, rownames=['Actual'], colnames=['Predicted']))

#%% ---------- Submission ----------
test_ids = test['id'].copy()
submission = pd.DataFrame({
    "id": test_ids,
    "diagnosed_diabetes": test_proba
})

submission.to_csv(OUTPUT_DIR / "xgb_v4_optuna.csv", index=False)
print(f"\nSaved: xgb_v4_optuna.csv to {OUTPUT_DIR}")