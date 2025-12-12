# src/preprocess.py

import seaborn as sns
import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

from .utils import cap_outliers_iqr


def load_data():
    return sns.load_dataset("titanic")


def eda(df):
    print(df.head())
    print(df.info())
    print(df.describe())


def clean_data(df):
    df = df.drop_duplicates()

    to_drop = ["deck", "embark_town", "alive", "who", "adult_male", "alone"]
    df = df.drop(columns=to_drop)

    df["age"] = df["age"].fillna(df["age"].median())
    df["embarked"] = df["embarked"].fillna(df["embarked"].mode()[0])
    df["fare"] = df["fare"].fillna(df["fare"].median())

    df["embarked"] = df["embarked"].astype(str)

    return df


def feature_engineering(df):

    df["family_size"] = df["sibsp"] + df["parch"] + 1
    df["is_alone"] = (df["family_size"] == 1).astype(int)

    df["title"] = df["class"].astype(str)

    df["fare"] = cap_outliers_iqr(df["fare"])

    return df


def build_preprocessor():

    num_cols = ["age", "sibsp", "parch", "fare", "family_size"]
    cat_cols = ["sex", "embarked", "class"]

    numeric_pipeline = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])

    categorical_pipeline = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore"))
    ])

    return ColumnTransformer(transformers=[
        ("num", numeric_pipeline, num_cols),
        ("cat", categorical_pipeline, cat_cols),
    ])
