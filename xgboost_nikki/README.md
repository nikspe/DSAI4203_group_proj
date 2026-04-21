# Diabetes Prediction Challenge - XGBoost & LightGBM

## Project Overview
Predict diabetes diagnosis using clinical measurements. 700k training samples, 22 features.

## Experiment Timeline

| Order | File | Purpose | Result |
|-------|------|---------|--------|
| 1 | `object_cleaning` | Label encoding for categoricals | Data ready |
| 2 | `pre-analysis` | Initial data exploration | Baseline understanding |
| 3 | `xgb_v0a_baseline` | Default XGBoost | AUC 0.7223 |
| 4 | `xgb_v0b_light_tuned` | Lower learning rate | Slightly worse |
| 5 | `xgb_v1_Kfold` | Added 5-fold CV | AUC 0.7230 |
| 6 | `xgb_v2a_EDA_analysis` | Correlation, distributions, importance | Visual outputs |
| 7 | `xgb_v2b_featured_engineered` | Created 10 new features | train_ft.csv, test_ft.csv |
| 8 | `xgb_v2c_test_engineered` | CV on engineered features | No improvement |
| 9 | `xgb_v2d_select_features` | Selected top 15 features | train_selected.csv, test_selected.csv |
| 10 | `xgb_v2e_test_selected` | CV on selected features | No improvement |
| 11 | `xgb_v3a_class_imbalance` | scale_pos_weight=1.655 | Over-corrected |
| 12 | `xgb_v3b_weight_grid_search` | Tested weights 0.8-1.655 | Best weight=0.8 |
| 13 | `xgb_v4_hyperparameter_optuna` | Optuna tuning | AUC 0.7252 |
| 14 | `xgb_v5_ensemble_LGBM` | XGB + LightGBM ensemble | AUC 0.7270 |
| 15 | `xgb_v5b_LGBM_only` | LightGBM only | AUC 0.7270 (final) |

## Key Findings

| Finding | Conclusion |
|---------|------------|
| Feature engineering | No improvement (raw features sufficient) |
| Feature selection | No improvement (top 15 = same signal) |
| Class imbalance | Best weight = 0.8 (not theoretical 1.655) |
| Hyperparameter tuning | +0.002 AUC gain |
| Model comparison | LightGBM > XGBoost (0.727 vs 0.725) |

## Final Model

**LightGBM** with:
- scale_pos_weight = 0.8
- n_estimators = 318
- max_depth = 5
- learning_rate = 0.15
- subsample = 0.65
- reg_alpha = 1.65
- reg_lambda = 1.84

**Final CV AUC: 0.7270**

## How to Run

1. Run `object_cleaning` first
2. Run `xgb_v2d_select_features` to generate selected features
3. Run `xgb_v5b_LGBM_only` for final model

## Output Files

| File | Description |
|------|-------------|
| `train_selected.csv` | Training data with top 15 features |
| `test_selected.csv` | Test data with top 15 features |
| `xgb_v5b_LGBM_only.csv` | Final submission |

## Lessons Learned

- Data ceiling (~0.727) reached despite extensive tuning
- LightGBM suited this dataset better than XGBoost
- Class imbalance weight = 0.8 improved balance without losing AUC
- Feature engineering added complexity without gain