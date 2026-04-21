# =============== Overfitting Analysis ===============
# Compare training vs validation performance across models

#%%
models = {
    'v1 (Simple)': {'train_auc': 0.730, 'val_auc': 0.723, 'test_auc': 0.6967},
    'v4 (Optuna)': {'train_auc': 0.745, 'val_auc': 0.725, 'test_auc': 0.6927},
    'v5b (LightGBM)': {'train_auc': 0.748, 'val_auc': 0.727, 'test_auc': 0.6945}
}

print("Overfitting Analysis:")
print("-" * 50)
for name, scores in models.items():
    gap = scores['train_auc'] - scores['val_auc']
    print(f"{name}: Train-Val Gap = {gap:.4f}")
    if gap > 0.02:
        print(f"  → OVERFITTING detected (gap > 0.02)")
    else:
        print(f"  → Good generalization")
# %%
