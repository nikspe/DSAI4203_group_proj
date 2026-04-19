# =============== Pre-analysis: Data Exploration ===============
# Target distribution: diagnosed_diabetes
#  1.0    0.623
#  0.0    0.377
# ----------------------------------------------------

#%% Importing Libraries & Setup
import numpy as np
import pandas as pd
from pathlib import Path

DATA_DIR = Path("../data")
OUTPUT_DIR = Path("../data_cleaned")

#%% Importing Data
train = pd.read_csv(f"{DATA_DIR}/train.csv")
test = pd.read_csv(f"{DATA_DIR}/test.csv")
sub = pd.read_csv(f"{DATA_DIR}/sample_submission.csv")

#%% Separate features and target
TARGET = "diagnosed_diabetes"

X = train.drop(columns=[TARGET])
y = train[TARGET]

print("Train shape :", X.shape)
print("Test shape  :", test.shape)
print("\nTarget distribution:")
print(y.value_counts(normalize=True).round(3))

#%% checking number of unique values in object columns
print(train.select_dtypes(include=["object"]).nunique())

#%% count NA values in each column
na_counts = train.isna().sum()
print(na_counts[na_counts > 0].sort_values(ascending=False))

#%% Overall data info
train.info()