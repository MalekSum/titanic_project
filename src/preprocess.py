# src/preprocess.py

import seaborn as sns
import pandas as pd
import numpy as np

from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from scipy.stats import zscore


def load_data():
    return sns.load_dataset("titanic")


def eda(df):
    print(df.head())
    print(df.info())
    print(df.describe())


def clean_data(df):

    df = df.drop_duplicates()

    df = df.drop(columns=[
        "deck", "embark_town", "alive",
        "who", "adult_male", "alone", "class"
    ])

    df["age"] = df["age"].fillna(df["age"].median())
    df["fare"] = df["fare"].fillna(df["fare"].median())
    df["embarked"] = df["embarked"].fillna(df["embarked"].mode()[0])

    df["sex"] = df["sex"].str.lower().str.strip()
    df["embarked"] = df["embarked"].str.lower().str.strip()

    return df


def feature_engineering(df):

    df["family_size"] = df["sibsp"] + df["parch"] + 1
    df["is_alone"] = (df["family_size"] == 1).astype(int)

    df = df[(np.abs(zscore(df["fare"])) < 3)]

    Q1 = df["age"].quantile(0.25)
    Q3 = df["age"].quantile(0.75)
    IQR = Q3 - Q1

    df = df[
        (df["age"] >= Q1 - 1.5 * IQR) &
        (df["age"] <= Q3 + 1.5 * IQR)
    ]

    return df


def build_preprocessor():

    num_cols = ["age", "sibsp", "parch", "fare", "family_size"]
    cat_cols = ["sex", "embarked"]

    numeric_pipeline = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])

    categorical_pipeline = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(drop="first"))
    ])

    return ColumnTransformer(transformers=[
        ("num", numeric_pipeline, num_cols),
        ("cat", categorical_pipeline, cat_cols),
    ])
