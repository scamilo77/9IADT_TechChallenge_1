import numpy as np
import pandas as pd
from typing import Tuple

import pandas as pd
import os

PROCESSED_DATA_DIR = "data/processed"

# ------------------------------------------------
#   Diabetes Dataset
#-------------------------------------------------

DIABETES_ZERO_AS_MISSING = [
    "Glucose", 
    "BloodPressure", 
    "SkinThickness", 
    "Insulin", 
    "BMI"
]


def clean_diabetes_data(df: pd.DataFrame) -> pd.DataFrame:

    if not isinstance(df, pd.DataFrame):
        raise TypeError("Input must be a pandas DataFrame.")
    
    df = df.copy(deep=True)

    # Substitui zeros por NaN nas colunas relevantes
    for col in DIABETES_ZERO_AS_MISSING:
        df[col] = df[col].replace(0, np.nan)

    # Preenche NaN com a mediana da coluna
    for col in DIABETES_ZERO_AS_MISSING:
        median_value = df[col].median()
        df[col].fillna(median_value, inplace=True)


    dir = os.path.join(PROCESSED_DATA_DIR, "diabetes-data-set")
    os.makedirs(dir, exist_ok=True)
    df.to_csv(os.path.join(PROCESSED_DATA_DIR, "diabetes-data-set", "diabetes.csv"), index=False)
    return df   

def split_diabetes_data(df: pd.DataFrame, target_column: str) -> Tuple[pd.DataFrame, pd.Series]:
    X = df.drop(columns=["Outcome"])
    y = df["Outcome"]
    return X, y

# ------------------------------------------------
#   Breast Cancer Dataset
#-------------------------------------------------

def clean_breast_cancer_data(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy(deep=True)

    if "id" in df.columns:
        df.drop(columns=["id"])

    df = df.dropna(axis=1, how="all")

    if "diagnosis" in df.columns:
        df["diagnosis"] = df["diagnosis"].map({"M": 1, "B": 0})


    dir = os.path.join(PROCESSED_DATA_DIR, "breast-cancer-wisconsin-data")
    os.makedirs(dir, exist_ok=True)
    df.to_csv(os.path.join(PROCESSED_DATA_DIR, "breast-cancer-wisconsin-data", "data.csv"), index=False)
    return df

def split_breast_cancer_data(df: pd.DataFrame, target_column: str) -> Tuple[pd.DataFrame, pd.Series]:
    X = df.drop(columns=["diagnosis"])
    y = df["diagnosis"]
    return X, y
