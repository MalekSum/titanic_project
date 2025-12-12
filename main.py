# main.py

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

from sklearn.model_selection import train_test_split

from src.preprocess import load_data, eda, clean_data, feature_engineering, build_preprocessor
from src.modeling import build_logreg_model, build_knn_model
from src.evaluate import evaluate_model


def main():

    print("\n=== 1) Load dataset ===")
    df = load_data()

    print("\n=== 2) EDA ===")
    eda(df)

    print("\n=== 3) Cleaning ===")
    df = clean_data(df)

    print("\n=== 4) Feature Engineering ===")
    df = feature_engineering(df)

    target = "survived"
    X = df.drop(columns=[target])
    y = df[target]

    print("\n=== 5) Preprocessing pipelines ===")
    preprocessor = build_preprocessor()

    print("\n=== 6) Train/test split ===")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    print("\n=== 7) Logistic Regression ===")
    logreg = build_logreg_model(preprocessor)
    logreg.fit(X_train, y_train)

    lr_acc, lr_pred = evaluate_model(logreg, X_test, y_test, "Logistic Regression")

    print("\n=== 8) KNN ===")
    knn = build_knn_model(preprocessor)
    knn.fit(X_train, y_train)

    knn_acc, knn_pred = evaluate_model(knn, X_test, y_test, "KNN (k=5)")

    # ====================================================
    # COMPARISON
    # ====================================================

    print("\n=== MODEL COMPARISON ===\n")
    print(f"Logistic Regression Accuracy : {lr_acc:.4f}")
    print(f"KNN Accuracy                : {knn_acc:.4f}")

    if lr_acc > knn_acc:
        print("\n→ Final Choice: Logistic Regression performs better overall.")
    elif knn_acc > lr_acc:
        print("\n→ Final Choice: KNN performs better overall.")
    else:
        print("\n→ Both models have identical accuracy.")

    # ====================================================
    # SCATTER PLOTS
    # ====================================================

    comparison_df = X_test.copy()
    comparison_df["True"] = y_test.values
    comparison_df["LogReg_Pred"] = lr_pred
    comparison_df["KNN_Pred"] = knn_pred

    # Logistic Regression Plot
    plt.figure(figsize=(12,4))
    plt.scatter(comparison_df.index, comparison_df["True"], label="True", alpha=0.6)
    plt.scatter(comparison_df.index, comparison_df["LogReg_Pred"], marker="x", label="LogReg")
    plt.title("True vs Logistic Regression Predictions")
    plt.xlabel("Sample Index")
    plt.ylabel("Survived")
    plt.legend()
    plt.show()

    # KNN Plot
    plt.figure(figsize=(12,4))
    plt.scatter(comparison_df.index, comparison_df["True"], label="True", alpha=0.6)
    plt.scatter(comparison_df.index, comparison_df["KNN_Pred"], marker="x", label="KNN")
    plt.title("True vs KNN Predictions")
    plt.xlabel("Sample Index")
    plt.ylabel("Survived")
    plt.legend()
    plt.show()

    # Combined
    plt.figure(figsize=(12,4))
    plt.scatter(comparison_df.index, comparison_df["True"], label="True", alpha=0.6)
    plt.scatter(comparison_df.index, comparison_df["LogReg_Pred"], label="LogReg")
    plt.scatter(comparison_df.index, comparison_df["KNN_Pred"], marker="^", label="KNN")
    plt.title("True vs LogReg & KNN Predictions")
    plt.xlabel("Sample Index")
    plt.ylabel("Survived")
    plt.legend()
    plt.show()


if __name__ == "__main__":
    main()
