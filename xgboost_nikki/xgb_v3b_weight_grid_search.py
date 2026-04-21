# =============== XGBoost V3.1 Weight Grid Search ===============
# Testing different scale_pos_weight values to find optimal balance
# Testing weights: 0.8, 1.0, 1.2, 1.4, 1.655
#
# Best for balanced recall: weight=0.8 (balanced recall=0.651)
# Best for AUC: weight=0.8 (AUC=0.72325)
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
train = pd.read_csv(f"{DATA_DIR}/train_selected.csv")
test = pd.read_csv(f"{DATA_DIR}/test_selected.csv")

#%% -------- Prepare Features -----------
# Extract ids before dropping
test_ids = test['id'].copy()

# Prepare features
TARGET = "diagnosed_diabetes"
X = train.drop(columns=[TARGET])
y = train[TARGET]
X_test = test.drop(columns=['id'])

print(f"X shape: {X.shape}, X_test shape: {X_test.shape}")
print(f"Columns match: {X.columns.tolist() == X_test.columns.tolist()}")
#%% ---------- Grid Search Parameters ----------
# Different scale_pos_weight values to test
weights_to_test = [0.8, 1.0, 1.2, 1.4, 1.655]
results_summary = []

#%% ---------- Cross Validation for Each Weight ----------
for weight in weights_to_test:
    print("\n" + "="*60)
    print(f"Testing scale_pos_weight = {weight}")
    print("="*60)
    
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    oof_pred = np.zeros(len(X))
    test_pred = np.zeros(len(X_test))
    fold_auc_scores = []
    
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
            scale_pos_weight=weight,     
            random_state=42,
            eval_metric='logloss'
        )
        model.fit(X_tr, y_tr)
        
        # Store out-of-fold predictions
        oof_pred[va_idx] = model.predict_proba(X_va)[:, 1]
        test_pred += model.predict_proba(X_test)[:, 1] / kf.n_splits
        
        # Calculate fold performance
        auc = metrics.roc_auc_score(y_va, oof_pred[va_idx])
        fold_auc_scores.append(auc)
        print(f"Fold {fold}: AUC={auc:.5f}")
    
    # Calculate final metrics for this weight
    cv_auc = metrics.roc_auc_score(y, oof_pred)
    accuracy = metrics.accuracy_score(y, oof_pred.round())
    
    # Calculate per-class recall
    recall_0 = metrics.recall_score(y, oof_pred.round(), pos_label=0)
    recall_1 = metrics.recall_score(y, oof_pred.round(), pos_label=1)
    
    print(f"\nResults for weight={weight}:")
    print(f"  Mean CV AUC = {np.mean(fold_auc_scores):.5f} (+/- {np.std(fold_auc_scores):.5f})")
    print(f"  Overall AUC = {cv_auc:.5f}")
    print(f"  Accuracy = {accuracy:.4f}")
    print(f"  Recall Class 0 (No Diabetes) = {recall_0:.3f}")
    print(f"  Recall Class 1 (Diabetes) = {recall_1:.3f}")
    
    # Store results
    results_summary.append({
        'weight': weight,
        'auc': cv_auc,
        'accuracy': accuracy,
        'recall_0': recall_0,
        'recall_1': recall_1,
        'std': np.std(fold_auc_scores)
    })
    
    # Save submission for this weight
    submission = pd.DataFrame({
        "id": test["id"], 
        "diagnosed_diabetes": test_pred  
    })
    submission.to_csv(OUTPUT_DIR / f"xgb_v3.1_weight_grid_search.csv", index=False)

#%% ---------- Print Summary of All Weights ----------
print("\n" + "="*60)
print("GRID SEARCH SUMMARY")
print("="*60)
print(f"{'Weight':<10} {'AUC':<8} {'Acc':<8} {'Recall0':<10} {'Recall1':<10} {'Std':<8}")
print("-"*60)

for r in results_summary:
    print(f"{r['weight']:<10} {r['auc']:.5f} {r['accuracy']:.4f} {r['recall_0']:.3f} {' '*5} {r['recall_1']:.3f} {' '*5} {r['std']:.5f}")

# Find best weight by balanced metric (average of recall0 and recall1)
for r in results_summary:
    r['balanced_recall'] = (r['recall_0'] + r['recall_1']) / 2

best_balanced = max(results_summary, key=lambda x: x['balanced_recall'])
best_auc = max(results_summary, key=lambda x: x['auc'])

print(f"\nBest for balanced recall: weight={best_balanced['weight']} (balanced recall={best_balanced['balanced_recall']:.3f})")
print(f"Best for AUC: weight={best_auc['weight']} (AUC={best_auc['auc']:.5f})")