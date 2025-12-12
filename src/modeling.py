# src/modeling.py

from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline


def build_logreg_model(preprocessor):
    return Pipeline(steps=[
        ("preprocess", preprocessor),
        ("classifier", LogisticRegression(max_iter=500))
    ])


def build_knn_model(preprocessor, k=5):
    return Pipeline(steps=[
        ("preprocess", preprocessor),
        ("classifier", KNeighborsClassifier(n_neighbors=k))
    ])
