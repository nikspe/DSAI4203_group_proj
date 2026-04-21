# =============== XGBoost V1 ===============
# Baseline model with basic parameters and 5-fold CV (for generalization)
# Base Model
# Overall CV AUC = 0.72299, Accuracy = 0.6819
# ----------------------------------------------------

#%% Importing Libraries & Setup
import numpy as np
import pandas as pd
from pathlib import Path
import xgboost as xgb
from sklearn.model_selection import KFold
from sklearn import metrics

DATA_DIR = Path(__file__).resolve().parent / "xgb_data"
OUTPUT_DIR = Path(__file__).resolve().parent / "xgb_output"

# ---------- Load data ----------
train = pd.read_csv(f"{DATA_DIR}/xgb_train.csv")
test = pd.read_csv(f"{DATA_DIR}/xgb_test.csv")

#%% -------- Prepare Features -----------
TARGET = "diagnosed_diabetes"

X = train.drop(columns=[TARGET])
y = train[TARGET] 

X_test = test.drop(columns=[TARGET], errors="ignore")

print(f"Training: {X.shape}, Test: {X_test.shape}")
print(f"Class distribution:\n{y.value_counts()}")

#%% ---------- Cross Validation ----------
kf = KFold(n_splits=5, shuffle=True, random_state=42)
oof_pred = np.zeros(len(X))  # Out-of-fold predictions for training data
test_pred = np.zeros(len(X_test))  # Ensemble predictions for test data

for fold, (tr_idx, va_idx) in enumerate(kf.split(X), 1):
    print(f"\n--- Fold {fold} ---")
    
    # Split data
    X_tr, X_va = X.iloc[tr_idx], X.iloc[va_idx]
    y_tr, y_va = y[tr_idx], y[va_idx]
    
    model = xgb.XGBClassifier(  
        n_estimators=100,
        learning_rate=0.3,  
        max_depth=6,        
        subsample=1.0,      
        random_state=42,
        eval_metric='logloss'
    )
    model.fit(X_tr, y_tr)
    
    # Store out-of-fold predictions (probabilities for class 1)
    oof_pred[va_idx] = model.predict_proba(X_va)[:, 1]  # Changed: get probability
    
    # Add to ensemble test predictions (probability)
    test_pred += model.predict_proba(X_test)[:, 1] / kf.n_splits  # Changed
    
    # Calculate fold performance - CHANGED metrics
    # Use AUC instead of RMSE
    auc = metrics.roc_auc_score(y_va, oof_pred[va_idx])
    print(f"Fold {fold}: AUC={auc:.5f}")

# ---------- Final CV Score ----------
# CHANGED: Use classification metrics
cv_auc = metrics.roc_auc_score(y, oof_pred)
accuracy = metrics.accuracy_score(y, oof_pred.round())  # Round probabilities to 0/1
print(f"Overall CV AUC = {cv_auc:.5f}, Accuracy = {accuracy:.4f}")

# Also print confusion matrix
print("\nConfusion Matrix:")
print(metrics.confusion_matrix(y, oof_pred.round()))


#%% ---------- Submission to CSV ----------
submission = pd.DataFrame({
    "id": test["id"], 
    "diagnosed_diabetes": test_pred  
})

# %%
submission.to_csv(OUTPUT_DIR / "xgb_v1_Kfold.csv", index=False)
print("Saved xgb_v1_Kfold.csv to", OUTPUT_DIR)
# %%
