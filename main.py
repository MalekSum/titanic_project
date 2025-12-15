import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

from sklearn.model_selection import train_test_split

from src.preprocess import load_data, eda, clean_data, feature_engineering, build_preprocessor
from src.modeling import build_logreg_model, build_knn_model
from src.evaluate import evaluate_model


def main():

    df = load_data()
    eda(df)

    df = clean_data(df)
    df = feature_engineering(df)

    # VISUALIZATION

    plt.figure()
    df["age"].hist(bins=30)
    plt.title("Age Distribution")
    plt.show()

    plt.figure()
    df["fare"].hist(bins=30)
    plt.title("Fare Distribution")
    plt.show()

    plt.figure()
    sns.boxplot(x=df["age"])
    plt.title("Age Boxplot")
    plt.show()

    plt.figure()
    sns.boxplot(x=df["fare"])
    plt.title("Fare Boxplot")
    plt.show()

    plt.figure()
    sns.barplot(x="sex", y="survived", data=df)
    plt.title("Survival Rate by Sex")
    plt.show()

    pivot = df.pivot_table(
        values="fare",
        index="sex",
        columns="embarked",
        aggfunc="mean"
    )
    plt.figure()
    sns.heatmap(pivot, annot=True, cmap="coolwarm")
    plt.title("Average Fare by Sex and Embarked")
    plt.show()

    melted = df.melt(
        id_vars="survived",
        value_vars=["age", "fare"],
        var_name="Feature",
        value_name="Value"
    )
    plt.figure()
    sns.boxplot(x="Feature", y="Value", hue="survived", data=melted)
    plt.title("Age & Fare by Survival")
    plt.show()

    # CORRELATION

    corr = df.corr(numeric_only=True)

    plt.figure(figsize=(8,6))
    sns.heatmap(corr, annot=True, cmap="coolwarm")
    plt.title("Correlation Heatmap")
    plt.show()

    # MODELING

    X = df.drop(columns=["survived"])
    y = df["survived"]

    preprocessor = build_preprocessor()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    logreg = build_logreg_model(preprocessor)
    logreg.fit(X_train, y_train)
    evaluate_model(logreg, X_test, y_test, "Logistic Regression")

    knn = build_knn_model(preprocessor)
    knn.fit(X_train, y_train)
    evaluate_model(knn, X_test, y_test, "KNN (k=5)")


if __name__ == "__main__":
    main()
