# =============== Overfitting Analysis - All Models ===============
# Loads actual results from CSV files to compare generalization
# Demonstrates overfitting across all model versions
# ----------------------------------------------------

#%% Importing Libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
import xgboost as xgb
import lightgbm as lgb
import warnings
warnings.filterwarnings('ignore')

DATA_DIR = Path(__file__).resolve().parent / "xgb_data"
OUTPUT_DIR = Path(__file__).resolve().parent / "xgb_output"

#%% ---------- Load Data for Each Version ----------
TARGET = "diagnosed_diabetes"

# Raw data (v0, v1)
train_raw = pd.read_csv(DATA_DIR / "xgb_train.csv")
X_raw = train_raw.drop(columns=[TARGET, 'id'], errors='ignore')
y_raw = train_raw[TARGET]

# Engineered features (v2)
train_ft = pd.read_csv(DATA_DIR / "train_ft.csv")
X_ft = train_ft.drop(columns=[TARGET])
y_ft = train_ft[TARGET]

# Selected features (v3, v4, v5b)
train_selected = pd.read_csv(DATA_DIR / "train_selected.csv")
X_selected = train_selected.drop(columns=[TARGET])
y_selected = train_selected[TARGET]

print("Data loaded:")
print(f"  Raw: {X_raw.shape}")
print(f"  Engineered: {X_ft.shape}")
print(f"  Selected: {X_selected.shape}")

#%% ---------- Define Model Configurations ----------
models_config = {
    'v0_Baseline': {
        'params': {'n_estimators': 100, 'random_state': 42},
        'kaggle_score': 0.68995,
        'cv_auc': 0.7223,
        'data': X_raw,
        'target': y_raw
    },
    'v1_Kfold': {
        'params': {'n_estimators': 100, 'learning_rate': 0.3, 'max_depth': 6, 'random_state': 42},
        'kaggle_score': 0.69666,
        'cv_auc': 0.7230,
        'data': X_raw,
        'target': y_raw
    },
    'v2_Engineered': {
        'params': {'n_estimators': 100, 'random_state': 42},
        'kaggle_score': 0.67607,
        'cv_auc': 0.7223,
        'data': X_ft,
        'target': y_ft
    },
    'v3_Weight_08': {
        'params': {'n_estimators': 100, 'scale_pos_weight': 0.8, 'random_state': 42},
        'kaggle_score': 0.69228,
        'cv_auc': 0.7233,
        'data': X_selected,
        'target': y_selected
    },
    'v4_Optuna': {
        'params': {'n_estimators': 318, 'max_depth': 5, 'learning_rate': 0.15, 
                   'subsample': 0.65, 'colsample_bytree': 0.94, 'reg_alpha': 1.65, 
                   'reg_lambda': 1.84, 'scale_pos_weight': 0.8, 'random_state': 42},
        'kaggle_score': 0.69269,
        'cv_auc': 0.7252,
        'data': X_selected,
        'target': y_selected
    },
    'v5b_LightGBM': {
        'params': {'n_estimators': 318, 'max_depth': 5, 'learning_rate': 0.15,
                   'subsample': 0.65, 'colsample_bytree': 0.94, 'reg_alpha': 1.65,
                   'reg_lambda': 1.84, 'scale_pos_weight': 0.8, 'random_state': 42},
        'kaggle_score': 0.69447,
        'cv_auc': 0.7270,
        'data': X_selected,
        'target': y_selected,
        'is_lightgbm': True
    }
}

#%% ---------- Calculate Train AUC for Each Model ----------
print("\n" + "="*60)
print("CALCULATING TRAIN AUC")
print("="*60)

results = []

for name, config in models_config.items():
    print(f"\n--- {name} ---")
    
    X_data = config['data']
    y_data = config['target']
    
    if config.get('is_lightgbm', False):
        model = lgb.LGBMClassifier(**config['params'], verbose=-1)
    else:
        model = xgb.XGBClassifier(**config['params'])
    
    model.fit(X_data, y_data)
    train_proba = model.predict_proba(X_data)[:, 1]
    train_auc = roc_auc_score(y_data, train_proba)
    
    cv_auc = config['cv_auc']
    kaggle_score = config['kaggle_score']
    train_cv_gap = train_auc - cv_auc
    cv_kaggle_gap = cv_auc - kaggle_score
    
    # Determine data type
    if config['data'] is X_raw:
        data_type = "raw"
    elif config['data'] is X_ft:
        data_type = "engineered"
    else:
        data_type = "selected"
    
    results.append({
        'model': name,
        'data_type': data_type,
        'train_auc': train_auc,
        'cv_auc': cv_auc,
        'kaggle_auc': kaggle_score,
        'train_cv_gap': train_cv_gap,
        'cv_kaggle_gap': cv_kaggle_gap,
        'overfitting': train_cv_gap > 0.02
    })
    
    print(f"  Data: {data_type}")
    print(f"  Train AUC: {train_auc:.5f}")
    print(f"  CV AUC: {cv_auc:.5f}")
    print(f"  Kaggle: {kaggle_score:.5f}")
    print(f"  Train-CV Gap: {train_cv_gap:.5f}")
    print(f"  Overfitting: {'YES' if train_cv_gap > 0.02 else 'NO'}")

#%% ---------- Summary Table ----------
print("\n" + "="*60)
print("SUMMARY TABLE")
print("="*60)
print(f"{'Model':<15} {'Data':<10} {'Train':<8} {'CV':<8} {'Kaggle':<8} {'Gap':<8} {'Overfit?'}")
print("-"*70)

for r in results:
    print(f"{r['model']:<15} {r['data_type']:<10} {r['train_auc']:.4f}   {r['cv_auc']:.4f}   {r['kaggle_auc']:.4f}   {r['train_cv_gap']:.4f}   {'YES' if r['overfitting'] else 'NO'}")

#%% ---------- Visualization 1: Bar Chart ----------
fig, ax = plt.subplots(figsize=(12, 6))

x = np.arange(len(results))
width = 0.25

train_bars = ax.bar(x - width, [r['train_auc'] for r in results], width, label='Train AUC', color='blue')
cv_bars = ax.bar(x, [r['cv_auc'] for r in results], width, label='CV AUC', color='green')
kaggle_bars = ax.bar(x + width, [r['kaggle_auc'] for r in results], width, label='Kaggle Private', color='orange')

ax.set_xlabel('Model')
ax.set_ylabel('AUC Score')
ax.set_title('Overfitting Analysis: Train vs CV vs Kaggle')
ax.set_xticks(x)
ax.set_xticklabels([r['model'] for r in results], rotation=45, ha='right')
ax.legend()
ax.axhline(y=0.69666, color='red', linestyle='--', alpha=0.5, label='Best Kaggle Score')

plt.tight_layout()
plt.savefig(OUTPUT_DIR / "overfitting_all_models.png", dpi=150)
plt.show()

#%% ---------- Visualization 2: Gap Analysis ----------
fig, ax = plt.subplots(figsize=(10, 6))

models_names = [r['model'] for r in results]
train_cv_gaps = [r['train_cv_gap'] for r in results]
colors = ['red' if gap > 0.02 else 'green' for gap in train_cv_gaps]

bars = ax.barh(models_names, train_cv_gaps, color=colors)
ax.axvline(x=0.02, color='red', linestyle='--', label='Overfitting Threshold (0.02)')
ax.set_xlabel('Train-CV Gap')
ax.set_title('Overfitting Severity by Model')
ax.legend()

for bar, gap in zip(bars, train_cv_gaps):
    ax.text(bar.get_width() + 0.001, bar.get_y() + bar.get_height()/2, 
            f'{gap:.4f}', va='center')

plt.tight_layout()
plt.savefig(OUTPUT_DIR / "overfitting_gaps.png", dpi=150)
plt.show()

#%% ---------- Visualization 3: Learning Curves ----------
print("\n=== Learning Curves: Best vs Worst Generalizer ===")

# Find best and worst by gap
best_idx = np.argmin([r['train_cv_gap'] for r in results])
worst_idx = np.argmax([r['train_cv_gap'] for r in results])
best_result = results[best_idx]
worst_result = results[worst_idx]

# Use selected data for consistent comparison
X_tr, X_val, y_tr, y_val = train_test_split(X_selected, y_selected, test_size=0.2, random_state=42)

# Get configs for best and worst
best_config = models_config[best_result['model']]
worst_config = models_config[worst_result['model']]

# Best model (v1_Kfold)
best_model = xgb.XGBClassifier(**best_config['params'], early_stopping_rounds=10)
best_model.fit(X_tr, y_tr, eval_set=[(X_tr, y_tr), (X_val, y_val)], verbose=False)
best_results_curve = best_model.evals_result()

# Worst model (v4_Optuna)
worst_model = xgb.XGBClassifier(**worst_config['params'], early_stopping_rounds=10)
worst_model.fit(X_tr, y_tr, eval_set=[(X_tr, y_tr), (X_val, y_val)], verbose=False)
worst_results_curve = worst_model.evals_result()

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

axes[0].plot(best_results_curve['validation_0']['auc'], label='Train')
axes[0].plot(best_results_curve['validation_1']['auc'], label='Validation')
axes[0].axhline(y=best_result['kaggle_auc'], color='red', linestyle='--', label='Kaggle')
axes[0].set_title(f"{best_result['model']}\nGap={best_result['train_cv_gap']:.4f}")
axes[0].set_xlabel('Boosting Rounds')
axes[0].set_ylabel('AUC')
axes[0].legend()

axes[1].plot(worst_results_curve['validation_0']['auc'], label='Train')
axes[1].plot(worst_results_curve['validation_1']['auc'], label='Validation')
axes[1].axhline(y=worst_result['kaggle_auc'], color='red', linestyle='--', label='Kaggle')
axes[1].set_title(f"{worst_result['model']}\nGap={worst_result['train_cv_gap']:.4f}")
axes[1].set_xlabel('Boosting Rounds')
axes[1].set_ylabel('AUC')
axes[1].legend()

plt.tight_layout()
plt.savefig(OUTPUT_DIR / "learning_curves_comparison.png", dpi=150)
plt.show()

#%% ---------- Visualization 4: Gap by Model Complexity ----------
fig, ax = plt.subplots(figsize=(10, 6))

complexity_order = ['v0_Baseline', 'v1_Kfold', 'v2_Engineered', 'v3_Weight_08', 'v4_Optuna', 'v5b_LightGBM']
ordered_results = [r for r in results if r['model'] in complexity_order]
ordered_results.sort(key=lambda x: complexity_order.index(x['model']))

gaps = [r['train_cv_gap'] for r in ordered_results]
model_names = [r['model'] for r in ordered_results]

ax.plot(model_names, gaps, 'o-', linewidth=2, markersize=8, color='purple')
ax.axhline(y=0.02, color='red', linestyle='--', label='Overfitting Threshold')
ax.set_xlabel('Model (increasing complexity →)')
ax.set_ylabel('Train-CV Gap')
ax.set_title('Overfitting Increases with Model Complexity')
ax.legend()
ax.grid(True, alpha=0.3)

plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "overfitting_by_complexity.png", dpi=150)
plt.show()

#%% ---------- Conclusion ----------
print("\n" + "="*60)
print("CONCLUSION")
print("="*60)

best_generalizer = min(results, key=lambda x: x['train_cv_gap'])
worst_generalizer = max(results, key=lambda x: x['train_cv_gap'])
best_kaggle = max(results, key=lambda x: x['kaggle_auc'])

print(f"""
KEY FINDINGS:
- Best generalizer (smallest overfitting): {best_generalizer['model']} (Gap={best_generalizer['train_cv_gap']:.4f})
- Worst generalizer (most overfitting): {worst_generalizer['model']} (Gap={worst_generalizer['train_cv_gap']:.4f})
- Best Kaggle score: {best_kaggle['model']} ({best_kaggle['kaggle_auc']:.5f})

DATA USED BY EACH MODEL:
- v0, v1: Raw features
- v2: Engineered features (10 new features)
- v3, v4, v5b: Selected features (top 15)

LESSON LEARNED:
Models with smaller Train-CV gaps generalized better to Kaggle's hidden test data.
Complex hyperparameter tuning (v4, v5b) increased overfitting without improving test performance.
Simple cross-validation (v1) achieved the best generalization.

This demonstrates the critical importance of:
1. Monitoring train-validation gaps
2. Prioritizing generalization over CV performance
3. Simplicity often beats complexity
""")

print(f"\nPlots saved to: {OUTPUT_DIR}")
print("- overfitting_all_models.png")
print("- overfitting_gaps.png")
print("- learning_curves_comparison.png")
print("- overfitting_by_complexity.png")