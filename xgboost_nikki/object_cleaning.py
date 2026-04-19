# =============== Extra Data Cleaning ===============
# Data preprocessing for XGBoost model
# Handling categorical variables with label-encoding (cause only <=5 categories)
# ----------------------------------------------------

#%% Importing Libraries and Dataset
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from pathlib import Path

# ---------- Load data ----------
DATA_DIR = Path("../data")
OUT_DIR = Path(__file__).resolve().parent / "xgb_data"

train = pd.read_csv(DATA_DIR / "train.csv")
test = pd.read_csv(DATA_DIR / "test.csv")

# %% -------- Preprocessing -----------

# Preprocessing Function
def xgboost_preprocess(df):
    df = df.copy()
    
    # Handle object columns
    obj_cols = df.select_dtypes(include=["object"]).columns
    
    for col in obj_cols:
        df[col] = df[col].fillna("None")
        
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))
    
    return df

# Combine datasets
combined = pd.concat([train, test], ignore_index=True)

# Apply to combined data
combined_clean = xgboost_preprocess(combined)

# Split processed datasets (train and test)
train_clean = combined_clean[:len(train)]  # First part = training data
test_clean = combined_clean[len(train):]   # Second part = test data

#%% ---------- Process to CSV -------------
out_path = OUT_DIR / "xgb_train.csv"
train_clean.to_csv(out_path, index=False)
print("Saved xgb_train.csv to", out_path)

out_path = OUT_DIR / "xgb_test.csv"
test_clean.to_csv(out_path, index=False)
print("Saved xgb_test.csv to", out_path)