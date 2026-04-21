# =============== XGBoost V0.1 Lightly tuned ===============
# Baseline model with simple train-test split (no CV)
# Lightly tuned parameters (lower learning rate -> prevent overfitting, slower learning)
# Base Model for comparison with CV versions
#
# Actual No Diabetes (52,739):    [20,616 ✓]  [32,123 ✗]  ← 39% correct (worse)
# Actual Has Diabetes (87,261):   [12,824 ✗]  [74,437 ✓]  ← 85% correct (slightly better)
# ----------------------------------------------------

#%% Importing Libraries & Setup
import numpy as np
import pandas as pd
from pathlib import Path
import xgboost as xgb
from sklearn.model_selection import train_test_split
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

#%% ---------- Simple Train-Test Split (replaces CV) ----------
# Split data into train and validation sets (80/20 split)
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"\nTrain size: {len(X_train)}, Validation size: {len(X_val)}")

# Train the model
model = xgb.XGBClassifier(  
    n_estimators=100, 
    learning_rate=0.08, 
    gamma=0, 
    subsample=0.75,
    colsample_bytree=1,
    max_depth=7,
    random_state=42,
    eval_metric='logloss' 
)
model.fit(X_train, y_train)

# Get predictions on validation set
val_pred_proba = model.predict_proba(X_val)[:, 1]
val_pred_binary = model.predict(X_val)

# Get predictions on test set
test_pred = model.predict_proba(X_test)[:, 1]

# ---------- Validation Metrics ----------
val_auc = metrics.roc_auc_score(y_val, val_pred_proba)
accuracy = metrics.accuracy_score(y_val, val_pred_binary)
print(f"\nValidation AUC = {val_auc:.5f}, Accuracy = {accuracy:.4f}")

# Confusion Matrix
print("\nConfusion Matrix:")
print(metrics.confusion_matrix(y_val, val_pred_binary))

# Also print classification report
print("\nClassification Report:")
print(metrics.classification_report(y_val, val_pred_binary))

#%% ---------- Submission to CSV ----------
submission = pd.DataFrame({
    "id": test["id"], 
    "diagnosed_diabetes": test_pred  
})

# %%
submission.to_csv(OUTPUT_DIR / "xgb_v0.1_light_tuned.csv", index=False)
print("\nSaved xgb_v0.1_light_tuned.csv to", OUTPUT_DIR)
# %%